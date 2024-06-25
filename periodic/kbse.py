#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Sylvia Bintrim <sjb2225@columbia.edu>
#

'''
static screening TDA-BSE with kpt sampling
Density fitting, frozen orbitals.
'''
import time
import numpy as np
from pyscf import lib
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import scf as pbcscf
from pyscf.lib import logger
from pyscf import __config__
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.tdscf import krhf
einsum = lib.einsum
    
REAL_EIG_THRESHOLD = getattr(__config__, 'pbc_tdscf_rhf_TDDFT_pick_eig_threshold', 1e-3)#-3 in tdscf
# Low excitation filter to avoid numerical instability
POSITIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)
deg_eia_thresh = getattr(__config__, 'tdscf_rhf_TDDFT_deg_eia_thresh', 1e-3)
MO_BASE = getattr(__config__, 'MO_BASE', 1)

def kernel(bse, nstates=None, orbs=None, verbose=logger.NOTE):
    '''static screening BSE excitation energies

    Returns:
        A list :  converged, number of states, excitation energies, eigenvectors
    '''
#    assert(isinstance(bse.mf, dft.rks.RKS) or isinstance(bse.mf, dft.rks_symm.SymAdaptedRKS))
    
    log = logger.Logger(bse.stdout, bse.verbose)

    #When we freeze GW orbitals,
    #we still output zeros for the frozen orbs AND 
    #any non-orbs orbitals
    if orbs is None:
        orbs = [x for x in range(bse.mf_nmo)]
    
    if len(orbs) > sum([x != 0 for x in bse.gw_e[0]]):
        logger.warn(bse, 'BSE orbs must be a subset of GW orbs!')
        raise RuntimeError

    orbs = [orb for orb in orbs if bse.gw_e[0][orb] != 0]
    
    nocc = sum([x < bse.mf_nocc for x in orbs])
    nmo = len(orbs)
    
    nkpts = bse.nkpts
    nvir = nmo - nocc
    
    
    matvec, diag = bse.gen_matvec(orbs)
    
    size = nocc*nvir
    if not bse.TDA:
        size *= 2
    
    if nstates is None:
        nstates = len(orbs)
        
    guess, nstates = bse.get_init_guess(nstates=nstates, orbs=orbs, diag=diag)
    
    nroots = nstates

    def precond(r, e0, x0):
        return r/(e0-diag+1e-8)

    if bse.TDA:
        eig = lib.davidson1
    else: 
        eig = lib.davidson_nosym1
    
    # GHF or customized RHF/UHF may be of complex type
    real_system = (bse._scf.mo_coeff[0].dtype == np.double) #gamma_point(bse._scf.kpts) and
                       
    
    def pickeig(w, v, nroots, envs):
        real_idx = np.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                              (w.real > POSITIVE_EIG_THRESHOLD))[0]
        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
    
    precision = bse._scf.cell.precision * 1e-2
    conv, e, xy = eig(matvec, guess, precond, pick=pickeig,
                       tol=bse.conv_tol, max_cycle=bse.max_cycle,
                       max_space=bse.max_space, nroots=nroots, 
                       fill_heff=krhf.purify_krlyov_heff(precision, 0, log), verbose=log)
    #TODO: not exactly compatible with PySCF TDA xy format,
    #since the latter has kshifts
    #so PySCF basically has an extra xy index for kshift that I don't have
    xy =   [(xi[:nocc*nvir*nkpts].reshape(nkpts, nocc, nvir)*np.sqrt(.5), 0) for xi in xy]
    #xy =   [xi[:nocc*nvir*nkpts].reshape(nkpts, nocc, nvir)*np.sqrt(.5) for xi in xy]
        
    if bse.verbose >= logger.INFO:
        np.set_printoptions(threshold=nocc*nvir)
        logger.debug(bse, '  BSE excitation energies =\n%s', e.real)
        for n, en, vn, convn in zip(range(nroots), e, xy, conv):
            logger.info(bse, '  BSE root %d E = %.16g eV  conv = %s',
                        n, en*27.2114, convn)
    #TODO: bse.e is also not compatible with PySCF TDA e format,
    #since the latter has kshifts
    return conv, nstates, e, xy

def matvec(bse, r, qkLij, qeps_body_inv, all_kidx, orbs):
    '''matrix-vector multiplication'''
   
    nocc = sum([x < bse.mf_nocc for x in orbs])
    nmo = len(orbs)
    kpts = bse.kpts
    nkpts = bse.nkpts
    nvir = nmo - nocc
    
    kptlist = range(bse.nkpts)
    nklist = len(kptlist) 
    
    gw_e_occ = bse.gw_e[:,bse.mf_nocc-nocc:bse.mf_nocc]
    gw_e_vir = bse.gw_e[:,bse.mf_nocc:bse.mf_nocc+nvir]
    
    Loo = qkLij[:,:,:, :nocc, :nocc]
    Lov = qkLij[:,:,:, :nocc, nocc:]
    Lvv = qkLij[:,:,:, nocc:, nocc:]
    
    r1 = r[:nkpts*nocc*nvir].copy().reshape(nkpts, nocc, nvir)
    
    #for A
    Hr1 = einsum('ka,kia->kia', gw_e_vir, r1) - einsum('ki,kia->kia', gw_e_occ, r1)
    
    for k in range(nklist):
        kn = kptlist[k]
        for kL in range(nkpts):
            # Find km that conserves with kn and kL
            km = all_kidx[kL][kn]
            if bse.CIS:
                Hr1[kn,:] -= (1/nkpts) * einsum('Pij, Pab, jb->ia', Loo[kL,kn,:].conj(), Lvv[kL,kn,:], r1[km,:])
            else:
                Hr1[kn,:] -= (1/nkpts) * einsum('Pij, PQ, Qab, jb->ia', Loo[kL,kn,:].conj(),\
                                            qeps_body_inv[kL], Lvv[kL,kn,:], r1[km,:])
        if bse.singlet: 
            #kL is (0,0,0) 
            #should be already shifted back to 0 if shifted kmesh
            Hr1[kn,:] += (2/nkpts) * einsum('Qia, qQjb,qjb->ia', Lov[0,kn].conj(), Lov[0], r1)
    if bse.TDA:
        return Hr1.ravel()
    else:
        raise NotImplementedError
#        r2 = r[nocc*nvir:].copy().reshape(nocc,nvir)
#
#        #for B
#        Hr1 -= einsum('Pib, PQ, Qja, jb->ia', eris.Lov, i_tilde, eris.Lov, r2)
#        if bse.singlet:
#            Hr1 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r2)
#
#        #for -A
#        Hr2 = einsum('a,ia->ia', gw_e_vir, r2) - einsum('i,ia->ia', gw_e_occ, r2)
#        Hr2 -= einsum('Pji, PQ, Qab, jb->ia', eris.Loo, i_tilde, eris.Lvv, r2)
#        if bse.singlet:
#            Hr2 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r2)
#
#        #for -B
#        Hr2 -= einsum('Pib, PQ, Qja, jb->ia', eris.Lov, i_tilde, eris.Lov, r1)
#        if bse.singlet:
#            Hr2 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r1)
#
#        return np.hstack((Hr1.ravel(), -Hr2.ravel()))


from pyscf.pbc.gw.krgw_ac import get_rho_response
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
def make_imds(gw, orbs):
    mf_nocc = gw._scf.mol.nelectron//2
    nocc = sum([x < mf_nocc for x in orbs])
    nmo = len(orbs)
    nvir = nmo - nocc
    
    mo_energy = np.array(gw._scf.mo_energy)
    mo_coeff = np.array(gw._scf.mo_coeff)
    nao = np.shape(gw._scf.mo_coeff)[-1]
    nkpts = gw.nkpts
    kpts = gw.kpts
    mydf = gw.with_df

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]
    
    qkLij = []
    qeps_body_inv = []
    all_kidx = []
    for kL in range(nkpts):
        ints_batch_t0 = time.process_time()
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts),dtype=np.int64)
        kidx_r = np.zeros((nkpts),dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    logger.debug(gw, "Read Lpq (kL: %s / %s, ki: %s, kj: %s)"%(kL+1, nkpts, i, j))
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign \
                            in mydf.sr_loop([kpti, kptj], max_memory=0.2*gw._scf.max_memory, compact=False):
                        Lpq.append(LpqR+LpqI*1.0j)
                    # support uneqaul naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1,nao*nao)
                    tao = []
                    ao_loc = None
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                    Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                    Lij.append(Lij_out.reshape(-1,nao,nao))
        Lij = np.asarray(Lij)
        naux = Lij.shape[1]
        qkLij.append(Lij[:,:, mf_nocc-nocc:mf_nocc+nvir, mf_nocc-nocc:mf_nocc+nvir])
        all_kidx.append(kidx)
        print('integral batch', time.process_time()-ints_batch_t0)

        # body dielectric matrix eps_body
        #static screening for BSE
        t0 = time.process_time()
        Pi = get_rho_response(gw, 0.0, mo_energy, Lij, kL, kidx)
        print('get_rho_response', time.process_time()-t0)

        t0 = time.process_time() 
        eps_body_inv = np.linalg.inv(np.eye(naux)-Pi)
        print('eps_body_inv', time.process_time()-t0)
        qeps_body_inv.append(eps_body_inv)
        
    return np.array(qkLij), qeps_body_inv, all_kidx
    
def _get_e_ia(mo_energy, mo_occ):
    e_ia = []
    nkpts = len(mo_occ)
    kconserv = np.arange(nkpts)
    for k in range(nkpts):
        kp = kconserv[k]
        moeocc = mo_energy[k][mo_occ[k] > 1e-6]
        moevir = mo_energy[kp][mo_occ[kp] < 1e-6]
        e_ia.append( -moeocc[:,None] + moevir )
    return e_ia

class BSE(krhf.TDA):
    '''static screening BSE

    Attributes:
        TDA : bool
            Whether to use the Tamm-Dancoff approximation to the BSE.  Default is True.
        singlet : bool
            Whether the excited state is a singlet or triplet.  Default is True.    
    Saved results:
        converged : bool
        nstates : int
        es : list
            BSE eigenvalues (excitation energies)
        vs : list
            BSE eigenvectors 
    '''
    _keys = {
        'frozen', 'mol', 'with_df',
        'kpts', 'nkpts', 'mo_energy', 'mo_coeff', 'mo_occ',
    }
    
    def __init__(self, gw, TDA=True, singlet=True,  CIS=False, mo_coeff=None, mo_occ=None):
        assert(isinstance(gw._scf, pbcdft.krks.KRKS) or isinstance(gw._scf, pbcscf.krhf.KRHF) or isinstance(gw._scf, pbcdft.krks_symm.SymAdaptedKRKS))
        if CIS:
            assert isinstance(gw._scf, pbcscf.krhf.KRHF)
        if mo_coeff  is None: mo_coeff  = gw._scf.mo_coeff
        if mo_occ    is None: mo_occ    = gw._scf.mo_occ
        
        self.gw = gw
        self.mf_nmo = np.shape(mo_coeff)[-1]
        self.mf_nocc = gw._scf.mol.nelectron//2
        self.gw_nmo = gw.nmo
        self.gw_nocc = gw.nocc
        self.mf = gw._scf
        self.mol = gw._scf.mol
        self._scf = gw._scf
        self.nkpts = gw.nkpts
        self.kpts = gw.kpts
        if CIS:
            self.gw_e = np.array(gw._scf.mo_energy)
        else:
            self.gw_e = gw.mo_energy
        self.mo_energy = gw._scf.mo_energy
        self.verbose = self.mol.verbose
        self.stdout = gw.stdout
        self.max_memory = gw.max_memory

        # DF-KGW must use GDF integrals
        if getattr(self.mf, 'with_df', None):
            self.with_df = self.mf.with_df
        else:
            raise NotImplementedError

        max_space = getattr(__config__, 'tdscf_rhf_TDA_max_space', 50)
        max_cycle = getattr(__config__, 'tdscf_rhf_TDA_max_cycle', 100)
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', 1e-6)
        self.nstates = getattr(__config__, 'tdscf_rhf_TDA_nstates', 3)

        self.frozen = gw.frozen
        self.TDA = TDA
        self.singlet = singlet
        self.CIS = CIS
        
        
##################################################
# don't modify the following attributes, they are not input options
        self.conv = False
        self.e = None
        self.xy = None
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        # self.nstates = None
        # self._nocc = None
        # self._nmo = None
        self._keys = set(self.__dict__.keys())
        
    def dump_flags(self):
        log = logger.new_logger(self, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('MF nocc = %d, nvir = %d', self.mf_nocc, self.mf_nmo - self.mf_nocc)
        log.info('GW nocc = %d, nvir = %d', self.gw_nocc, self.gw_nmo - self.gw_nocc)
        if self.frozen is not None:
            log.info('frozen = %s', self.frozen)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        logger.info(self, 'BSE within the TDA = %s', self.TDA)
        if getattr(self, 'TDA') is False:
            logger.warn(self, 'non-TDA BSE may not always converge (triplet instability problem).')
        logger.info(self, 'singlet = %s', self.singlet)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        return self
    
    def kernel(self, nstates=None, orbs=None):
        nmo = self.mf_nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        if orbs is None:
            orbs = [x for x in range(self.mf_nmo)]
        mem_incore = 2*nkpts*nmo**2*naux * 16/1e6 + 2*nkpts**2*len(orbs)**2*naux * 16/1e6
        print('mem_incore', mem_incore)
        mem_now = lib.current_memory()[0]
        print('mem_now', mem_now)
        if (mem_incore + mem_now > 0.99*self.max_memory):
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError        

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        self.conv, self.nstates, self.e, self.xy = kernel(self, nstates=nstates, orbs=orbs)
        logger.timer(self, 'BSE', *cput0)
        return self.conv, self.nstates, self.e, self.xy

    matvec = matvec
    make_imds = make_imds
    
    
    def vector_size(self, orbs):
       '''size of the vector'''
       # nocc = self.nocc
       nocc = sum([x < self.mf_nocc for x in orbs])
       nmo = len(orbs)
       nvir = nmo - nocc
       nkpts = self.nkpts
       if self.TDA:
           return nkpts*nocc*nvir
       else: 
           return 2*nkpts*nocc*nvir
    
    def gen_matvec(self, orbs):
    
        imds_t0 = time.process_time()
        qkLij, qeps_body_inv, all_kidx = make_imds(self.gw, orbs)
        print('imds total time', time.process_time()-imds_t0)
        
        diag = self.get_diag(qkLij[0,:], qeps_body_inv[0], orbs)
        matvec = lambda xs: [self.matvec(x, qkLij, qeps_body_inv, all_kidx, orbs) for x in xs]
        return matvec, diag

    def get_diag(self, kLij, eps_body_inv, orbs):
        nocc = sum([x < self.mf_nocc for x in orbs])
        nmo = len(orbs)
        nvir = nmo - nocc
        nkpts = self.nkpts
        
        '''
        Loo = kLij[:,:, :nocc, :nocc]
        Lov = kLij[:,:, :nocc, nocc:]
        Lvv = kLij[:,:, nocc:, nocc:]
        
        gw_e_occ = self.gw_e[:,self.mf_nocc-nocc:self.mf_nocc]
        gw_e_vir = self.gw_e[:,self.mf_nocc:self.mf_nocc+nvir]
        
        diag = np.zeros((nkpts, nocc, nvir), dtype='complex128')
        for i in range(nocc):
            for a in range(nvir):
                diag[:,i,a] += gw_e_vir[:,a] - gw_e_occ[:,i]
                if self.CIS:
                    diag[:,i,a] -= (1/nkpts)*einsum('kP, kP->k', Loo[:,:,i,i], Lvv[:,:,a,a].conj())
                else:
                    diag[:,i,a] -= (1/nkpts)*einsum('kP, PQ, kQ->k', Loo[:,:,i,i], eps_body_inv, Lvv[:,:,a,a].conj())
                if self.singlet:
                    diag[:,i,a] += (2/nkpts)*einsum('kP,kP->k', Lov[:,:,i,a], Lov[:,:,i,a].conj())
        diag = diag.ravel()'''
        #no need to use the more expensive diag

        mo_occ = [self._scf.mo_occ[k][self.mf_nocc-nocc:self.mf_nocc+nvir] for k in range(nkpts)]
        e_ia = _get_e_ia(self.gw_e[:,self.mf_nocc-nocc:self.mf_nocc+nvir], mo_occ)
        diag = np.hstack([x.ravel() for x in e_ia]).ravel()

        if self.TDA:
            return diag
        else: 
            raise NotImplementedError
            # return np.hstack((diag, -diag))
        
    #def get_init_guess(self, nstates, orbs, diag=None):        
    #    idx = diag.argsort()
    #    size = self.vector_size(orbs)
    #    dtype = getattr(diag, 'dtype', self.mo_coeff[0].dtype)
    #    nroots = nstates
    #    guess = []
    #    for i in idx[:nroots]:
    #        g = np.zeros(size, dtype)
    #        g[i] = 1.0
    #        guess.append(g)
    #    return guess, nroots
    
    def get_init_guess(self, nstates, orbs, diag=None):
        nocc = sum([x < self.mf_nocc for x in orbs])
        nmo = len(orbs)
        nvir = nmo - nocc
        mo_occ = [self._scf.mo_occ[k][self.mf_nocc-nocc:self.mf_nocc+nvir] for k in range(self.nkpts)]
        mo_energy = self.gw_e[:,self.mf_nocc-nocc:self.mf_nocc+nvir]
        e_ia = np.concatenate( [x.reshape(-1) for x in
                                   _get_e_ia(mo_energy, mo_occ)] )
        nov = e_ia.size
        nstates = min(nstates, nov)
        e_threshold = np.sort(e_ia)[nstates-1]
        e_threshold += deg_eia_thresh

        idx = np.where(e_ia <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        return np.array(x0, dtype='complex128'), nstates

    # @property
    # def nocc(self):
    #     return self.get_nocc()
    # @nocc.setter
    # def nocc(self, n):
    #     self._nocc = n

    # @property
    # def nmo(self):
    #     return self.get_nmo()
    # @nmo.setter
    # def nmo(self, n):
    #     self._nmo = n

    # get_nocc = get_nocc
    # get_nmo = get_nmo
    # get_frozen_mask = get_frozen_mask

if __name__ == '__main__':
    from pyscf.pbc import gto as pbcgto
    # from pyscf.pbc import dft as pbcdft
    from pyscf.pbc.tools import pyscf_ase, lattice
    import bse
    from pyscf.pbc.gw import krgw_ac
    
    ##########################################################
    # same as molecular BSE for large cell size with cell.dim = 0. 
    # Any difference is exactly due to the GW not matching.
    cell = pbcgto.Cell()
    cell.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    cell.a = np.eye(3)
    cell.basis = 'gth-tzvp'
    cell.pseudo = 'gth-pbe'
    cell.dimension = 0
    cell.build()
    
    fnl = 'pbe0'
    
    kdensity = 1
    kmesh = [kdensity, kdensity, kdensity]
    scaled_center=[0.0, 0.0, 0.0]
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    #must have exxdiv=None
    mymf = pbcdft.KRKS(cell, kpts, exxdiv=None).density_fit()
    mymf.xc = fnl
    ekrhf = mymf.kernel()
    
    nocc = cell.nelectron//2
    nmo = np.shape(mymf.mo_energy)[-1]
    nvir = nmo-nocc
    gw_orbs = range(nmo-2)#range(nmo)
    bse_orbs = range(nmo-4)
    _nstates = 6
    
    mygw = krgw_ac.KRGWAC(mymf)
    mygw.linearized = True
    mygw.ac = 'pade'
    # without finite size corrections
    mygw.fc = False
    mygw.kernel(orbs=gw_orbs)
    kgw_e = mygw.mo_energy
    gw_nocc = mygw.nocc
    gw_vir = sum([e>0 for e in kgw_e[0]]) - gw_nocc
    sorted_kgw_gaps = sorted([27.2114*(a-i) for a in kgw_e[0][nocc:nocc+gw_vir] for i in kgw_e[0][nocc-gw_nocc:nocc]])
    
    mybse = BSE(mygw, TDA=True, singlet=True)
    conv, excitations, ekS, xy = mybse.kernel(nstates=_nstates, orbs=bse_orbs)
    
    mybse.singlet=False
    conv, excitations, ekT, xy = mybse.kernel(nstates=_nstates, orbs=bse_orbs)

    
    #molecule
    mol = cell.to_mol()
    # mol.build()
    
    from pyscf import dft
    mf = dft.RKS(mol).density_fit()
    mf.xc = fnl
    mf.kernel()

    from pyscf import gw
    mygw = gw.GW(mf, freq_int='ac')
    mygw.kernel(orbs=gw_orbs)
    gw_e = mygw.mo_energy
    sorted_gw_gaps = sorted([27.2114*(a-i) for a in gw_e[nocc:nocc+gw_vir] for i in gw_e[nocc-gw_nocc:nocc]])

    mybse = bse.BSE(mygw, TDA=True)
    conv, excitations, eS, xy = mybse.kernel(nstates=_nstates, orbs=bse_orbs)
   
    mybse.singlet=False
    conv, excitations, eT, xy = mybse.kernel(nstates=_nstates, orbs=bse_orbs)
    
    for i in range(_nstates):
        assert(abs(27.2114*(ekS[i] - eS[i])) < 0.005+abs(sorted_kgw_gaps[i]-sorted_gw_gaps[i]))
        print(str(i)+' singlet agrees to within 0.005 eV of GW error')    
        assert(abs(27.2114*(ekT[i] - eT[i])) < 0.005+abs(sorted_kgw_gaps[i]-sorted_gw_gaps[i]))
        print(str(i)+' triplet agrees to within 0.005 eV of GW error')  
        
    print('BSE for a large cell with cell.dim = 0 is the same as molecular BSE \
          for singlets and triplets!')
    
    ##########################################################
    # Cannot check that a supercell has same excitations as unit cell with corresponding kmesh:
    # cannot target just Gamma pt excitations
    
    ##########################################################
    # same as HF-based kCIS when replace GW energies with MF ones and turn off screening
    # Candidate formula of solid: c, si, sic, bn, bp, aln, alp, mgo, mgs, lih, lif, licl
    #WARNING: Have to modify pbc bse code to pass this test
    formula = 'c'
    ase_atom = lattice.get_ase_atom(formula)
    cell = pbcgto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell
    cell.unit = 'B'
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-hf'
    cell.build()
      
    # kmesh = [2,2,2]
    kmesh = [2,1,1]
    scaled_center=[0.0, 0.0, 0.0]
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    #must have exxdiv=None
    mymf =  pbcscf.KRHF(cell, kpts, exxdiv=None).density_fit()
    #must be HF to compare!
    ekrhf = mymf.kernel()
      
    nocc = cell.nelectron//2
    nmo = np.shape(mymf.mo_energy)[-1]
    nvir = nmo-nocc
      
    mygw = krgw_ac.KRGWAC(mymf)
      
    _nstates = 12
    #sometimes need to ask for many excitations for them to agree
      
    mybse = BSE(mygw, TDA=True, singlet=True, CIS=True)
    conv, excitations, ekS, xy = mybse.kernel(nstates = _nstates)
      
    mybse.singlet=False
    conv, excitations, ekT, xy = mybse.kernel(nstates = _nstates)
      
    mypbctd = mymf.TDA()
    TDAeS = mypbctd.run(nstates = _nstates).e
    TDAeT = mypbctd.run(nstates = _nstates, singlet = False).e
      
      
    for i in range(_nstates):
        assert(abs(27.2114*(ekS[i] - TDAeS[0][i])) < 0.0005)
        print(str(i)+' singlet agrees to within 0.0005 eV')    
        assert(abs(27.2114*(ekT[i] - TDAeT[0][i])) < 0.0005)
        print(str(i)+' triplet agrees to within 0.0005 eV')  
      
    print('TDA matches HF-based BSE without screening and with gw_e = mf.mo_energy \
    \n for singlets and triplets with kpt sampling!')
