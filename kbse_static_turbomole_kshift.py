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
static screening BSE with iterative diagonalization, 
with or without TDA, singlet or triplet excitations. 
Density-fitted. Turbomole-style.
Periodic.
Frozen orbitals.
Requires Nk^2 * Naux * Nmo^2 memory (sim. to hybrid f'nl)
'''

import numpy as np
from pyscf import lib
from pyscf.pbc import dft
from pyscf.lib import logger
from pyscf import __config__
from pyscf.pbc.lib.kpts_helper import get_kconserv_ria

einsum = lib.einsum
    
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-3)
# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)
# MO_BASE = getattr(__config__, 'MO_BASE', 1)

def kernel(bse, nstates=None, orbs=None, kshift_lst=None, kconserv=None, verbose=logger.NOTE):
    '''static screening BSE excitation energies

    Returns:
        A list :  converged, number of states, excitation energies, eigenvectors
    '''
    #mf must be DFT; for HF use xc = 'hf'
#    assert(isinstance(bse.mf, dft.rks.RKS) or isinstance(bse.mf, dft.rks_symm.SymAdaptedRKS))
    
    log = logger.Logger(bse.stdout, bse.verbose)

    if orbs is None:
        orbs = [x for x in range(bse.mf_nmo)]
    if kshift_lst is None:
        kshift_lst = bse.kshift_lst
    if kconserv is None:
        kconserv = bse.kconserv
    
    orbs_nocc = sum([x < bse.mf_nocc for x in orbs])
    if orbs_nocc > bse.gw_nocc:
        orbs = [bse.mf_nocc - x for x in range(bse.gw_nocc, 0, -1)] + list(orbs)[orbs_nocc:]
    orbs_vir = sum([x > bse.mf_nocc for x in orbs])
    if orbs_vir > bse.gw_nmo - bse.gw_nocc:
        orbs = list(orbs)[:bse.gw_nocc] + [x for x in range(bse.gw_nocc, bse.gw_nmo)]
    
    nocc = sum([x < bse.mf_nocc for x in orbs])
    nmo = len(orbs)

    if nmo > sum([x != 0 for x in bse.gw_e[0]]):
        logger.warn(bse, 'BSE orbs must be a subset of GW orbs!')
        raise RuntimeError

    nkpts = bse.nkpts
    # kpts = bse.kpts
    nvir = nmo - nocc
    
    if nstates is None:
        nstates = len(orbs)
     
    # GHF or customized RHF/UHF may be of complex type
    real_system = (bse._scf.mo_coeff[0].dtype == np.double)
    
    if bse.TDA:
        eig = lib.davidson1
    else: 
        raise NotImplementedError
        # eig = lib.davidson_nosym1
    
    converged = []
    es = []
    xys = []
    for i,kshift in enumerate(kshift_lst):
        kconserv = kconserv[kshift]

        matvec, diag = bse.gen_matvec(orbs, kshift)
        
        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        guess,nroots = bse.get_init_guess(nstates, orbs, kshift, diag=diag)
        
        def pickeig(w, v, nroots, envs):
            real_idx = np.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > POSTIVE_EIG_THRESHOLD))[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
        conv, e, x1 = eig(matvec, guess, precond, pick=pickeig,
                           tol=bse.conv_tol, max_cycle=bse.max_cycle,
                           max_space=bse.max_space, nroots=nroots, verbose=log)
        converged.append( conv)
        es.append( e )
        # 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        #TODO: non-TDA
        xy = [(xi.reshape(nkpts, nocc, nvir)*np.sqrt(.5), 0) for xi in x1]
        xys.append(xy)
        
        if bse.verbose >= logger.INFO:
            np.set_printoptions(threshold=nocc*nvir)
            logger.debug(bse, 'for kshift %d: BSE excitation energies =\n%s', i, e.real)
            for n, en, vn, convn in zip(range(nroots), e, xy, conv):
                logger.info(bse, '  BSE root %d E = %.16g eV  conv = %s',
                            n, en*27.2114, convn)
    return converged, es, xys
    
def matvec(bse, r, qkLij, qeps_body_inv, all_kidx_r, orbs, kshift):
    '''matrix-vector multiplication'''
   
    nocc = sum([x < bse.mf_nocc for x in orbs])
    nmo = len(orbs)
    nkpts = bse.nkpts
    nvir = nmo - nocc
    
    kptlist = range(bse.nkpts)
    nklist = len(kptlist) 
    
    #WARNING: Change back! TDA
    gw_e = np.asarray(bse.gw._scf.mo_energy)[:,bse.mf_nocc-nocc:bse.mf_nocc+nvir]
    # gw_e = bse.gw_e[:,bse.mf_nocc-nocc:bse.mf_nocc+nvir]
    mo_occ = np.hstack((np.ones((nkpts, nocc)), np.zeros((nkpts, nvir))))
    kconserv = bse.kconserv[kshift]
    gw_ia = np.asarray([x for x in
                               _get_e_ia(gw_e, mo_occ, kconserv)], dtype='complex128')
    
    Loo = qkLij[:,:,:,:nocc, :nocc]
    Lov = qkLij[:,:,:,:nocc, nocc:]
    Lvv = qkLij[:,:,:,nocc:, nocc:]
    
    r1 = r[:nkpts*nocc*nvir].copy().reshape(nkpts, nocc, nvir)
    
    #for A
    Hr1 = einsum('kia,kia->kia', gw_ia, r1)
    
    for kL in range(nkpts):
        for k in range(nklist):
            kn = kptlist[k]
            # Find km that conserves with kn and kL (-km+kn+kL=G)
            km = all_kidx_r[kL][kn]
            # WARNING: Change back! TDA
            Hr1[kn,:] -= (1/nkpts) * einsum('Pij, Pab, jb->ia', Loo[kL,kn,:].conj(), Lvv[kL,kn,:], r1[km,:])
            # Hr1[kn,:] -= (1/nkpts) * einsum('Pij, PQ, Qab, jb->ia', Loo[kL,kn,:].conj(), qeps_body_inv[kL], Lvv[kL,kn,:], r1[km,:])
    if bse.singlet:
        #kL is (0,0,0) 
        #should be already shifted back to 0 if shifted kmesh
        for k in range(nklist):
            kn = kptlist[k]
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
def make_imds(gw, orbs, kshift):
    mo_energy = np.array(gw._scf.mo_energy)[:,orbs] #mf mo_energy
    mo_coeff = np.array(gw._scf.mo_coeff)[:,:,orbs]
    nmo = len(orbs)
    nao = np.shape(gw._scf.mo_coeff)[-1]
    nkpts = gw.nkpts
    kpts = gw.kpts
    mydf = gw.with_df

    #if shifted, center is no longer Gamma (needs to be for conserv check)
    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]
    
    qkLij = []
    qeps_body_inv = []
    all_kidx_r = []
    for kL in range(nkpts):
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts),dtype=np.int64)
        kidx_r = np.zeros((nkpts),dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                #given kshift q': k -> k+q'
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kshift] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    logger.debug(gw, "Read Lpq (kL: %s / %s, ki: %s, kj: %s)"%(kL+1, nkpts, i, j))
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign \
                            in mydf.sr_loop([kpti, kptj], max_memory=0.1*gw._scf.max_memory, compact=False):
                        Lpq.append(LpqR+LpqI*1.0j)
                    # support uneqaul naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1,nao*nao)
                    tao = []
                    ao_loc = None
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                    Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                    Lij.append(Lij_out.reshape(-1,nmo,nmo))
        Lij = np.asarray(Lij)
        naux = Lij.shape[1]
        qkLij.append(Lij)
        all_kidx_r.append(kidx_r)
        
        # body dielectric matrix eps_body
        #static screening for BSE
        Pi = get_rho_response(gw, 0.0, mo_energy, Lij, kL, kidx)
        eps_body_inv = np.linalg.inv(np.eye(naux)-Pi)
        qeps_body_inv.append(eps_body_inv)
    return np.array(qkLij), qeps_body_inv, all_kidx_r
  
#TODO: This only works for TDA
def _get_e_ia(mo_energy, mo_occ, kconserv=None):
    e_ia = []
    nkpts = len(mo_occ)
    if kconserv is None: kconserv = np.arange(nkpts)
    for k in range(nkpts):
        kp = kconserv[k]
        moeocc = mo_energy[k][mo_occ[k] > 1e-6]
        moevir = mo_energy[kp][mo_occ[kp] < 1e-6]
        e_ia.append( -moeocc[:,None] + moevir )
    return e_ia

  
from pyscf.pbc.tdscf import krhf
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
        'kpts', 'nkpts', 'mo_energy', 'mo_coeff', 'mo_occ', 'kconserv', 'kshift_lst'}
    
    def __init__(self, gw, TDA=True, singlet=True,  kshift_lst=None, mo_coeff=None, mo_occ=None):
        assert(isinstance(gw._scf, dft.krks.KRKS) or isinstance(gw._scf, dft.krks_symm.SymAdaptedKRKS))
        if mo_coeff  is None: mo_coeff  = gw._scf.mo_coeff
        if mo_occ    is None: mo_occ    = gw._scf.mo_occ
        if kshift_lst is None: kshift_lst = [0]
        
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
        self.gw_e = gw.mo_energy
        self.mo_energy = gw._scf.mo_energy
        self.verbose = self.mol.verbose
        self.stdout = gw.stdout
        self.max_memory = gw.max_memory
        
        self.kconserv = get_kconserv_ria(self.mf.cell, self.kpts)
        self.kshift_lst = kshift_lst

        # DF-KGW must use GDF integrals
        if getattr(self.mf, 'with_df', None):
            self.with_df = self.mf.with_df
        else:
            raise NotImplementedError

        self.max_space = getattr(__config__, 'eom_rccsd_EOM_max_space', 20)
        self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', 1e-7)
        self.nstates = getattr(__config__, 'tdscf_rhf_TDA_nstates', 3)

        self.frozen = gw.frozen
        self.TDA = TDA
        self.singlet = singlet
        
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
            log.info('GW frozen orbs = %s', self.frozen)
        log.info('kshift_lst = %s', self.kshift_lst)
        log.info('deg_eia_thresh = %.3e', self.deg_eia_thresh)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        logger.info(self, 'BSE within the TDA = %s', self.TDA)
        if getattr(self, 'TDA') is False:
            logger.warn(self, 'non-TDA BSE may not always converge (triplet instability problem).')
        logger.info(self, 'singlet = %s', self.singlet)
        return self
    
    def kernel(self, nstates=None, orbs=None, kshift_lst=None, kconserv=None):
        nmo = self.gw_nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (2*nkpts**2*nmo**2*naux) * 16/1e6
        mem_now = lib.current_memory()[0]
        if (mem_incore + mem_now > 0.99*self.max_memory):
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError        

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        self.conv, self.e, self.xy = \
            kernel(self, nstates=nstates, orbs=orbs, kshift_lst=kshift_lst, kconserv=kconserv)
        logger.timer(self, 'BSE', *cput0)
        return self.conv, self.e, self.xy

    matvec = matvec
    make_imds = make_imds
    _get_e_ia = _get_e_ia
    
    def gen_matvec(self, orbs, kshift):
    
        qkLij, qeps_body_inv, all_kidx_r = make_imds(self.gw, orbs, kshift)
        
        diag = self.get_diag(qkLij[0,:], qeps_body_inv[0], orbs, kshift)
        matvec = lambda xs: [self.matvec(x, qkLij, qeps_body_inv, all_kidx_r, orbs, kshift) for x in xs]
        return matvec, diag
       
    def get_init_guess(self, nstates, orbs, kshift, diag=None):
        nocc = sum([x < self.mf_nocc for x in orbs])
        nmo = len(orbs)
        nvir = nmo - nocc
        nkpts = self.nkpts
        
        if self.TDA:
            size = nkpts*nocc*nvir
        else: 
            size = 2*nkpts*nocc*nvir
        
        idx = diag.argsort()
        dtype = getattr(diag, 'dtype', self.mo_coeff[0].dtype)
        nroots = nstates
        guess = []
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess, nroots

    def get_diag(self, kLij, eps_body_inv, orbs, kshift):
        nocc = sum([x < self.mf_nocc for x in orbs])
        nmo = len(orbs)
        nvir = nmo - nocc
        nkpts = self.nkpts
        
        # #WARNING: Change back! TDA
        gw_e = np.asarray(self.mo_energy)[:,self.mf_nocc-nocc:self.mf_nocc+nvir]
        # gw_e = self.gw_e[:,self.mf_nocc-nocc:self.mf_nocc+nvir]
        mo_occ = np.hstack((np.ones((nkpts, nocc)), np.zeros((nkpts, nvir))))
        kconserv = self.kconserv[kshift]
        gw_ia = [x for x in _get_e_ia(gw_e, mo_occ, kconserv)]
        
        Loo = kLij[:,:,:nocc, :nocc]
        Lov = kLij[:,:,:nocc, nocc:]
        Lvv = kLij[:,:,nocc:, nocc:]

        diag = np.asarray(gw_ia.copy(), dtype='complex128') #np.zeros((nkpts, nocc, nvir), dtype='complex128')
        for i in range(nocc):
            for a in range(nvir):
                # diag[:,i,a] -= einsum('kP, PQ, kQ->k', Loo[:,:,i,i].conj(), eps_body_inv, Lvv[:,:,a,a])
                #WARNING: Change back! TDA
                diag[:,i,a] -= einsum('kP, kP->k', Loo[:,:,i,i].conj(), Lvv[:,:,a,a])
                if self.singlet:
                    diag[:,i,a] += 2*einsum('kP,kP->k', Lov[:,:,i,a].conj(), Lov[:,:,i,a])
        diag = diag.ravel()
        if self.TDA:
            return diag
        else: 
            raise NotImplementedError
            # return np.hstack((diag, -diag))

if __name__ == '__main__':
    from pyscf.pbc import gto as pbcgto
    from pyscf import gto
    from pyscf.pbc.tools import pyscf_ase, lattice
    import bse_static_turbomole_for_gwac_frozen as bse
    
    
    ##########################################################
    # same as kTDA when replace GW energies with MF ones and turn off screening (not just at Gamma pt)
    # Candidate formula of solid: c, si, sic, bn, bp, aln, alp, mgo, mgs, lih, lif, licl
    #WARNING: Have to modify pbc code to pass this test
    
    from pyscf.pbc.gw import krgw_ac
    #kpt sampling
    formula = 'c'
    ase_atom = lattice.get_ase_atom(formula)
    cell = pbcgto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell
    cell.unit = 'B'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-hf'
    cell.build()
    
    kmesh = [2,1,1]
    scaled_center=[0.0, 0.0, 0.0]
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    #must have exxdiv=None
    mymf = dft.KRKS(cell, kpts, exxdiv=None).density_fit()
    mymf.xc = 'hf' #must be hf to compare to TDA!
    ekrhf = mymf.kernel()
    
    nocc = cell.nelectron//2
    nmo = np.shape(mymf.mo_energy)[-1]
    nvir = nmo-nocc
    
    mygw = krgw_ac.KRGWAC(mymf)
    mygw.linearized = True
    mygw.ac = 'pade'
    # without finite size corrections
    mygw.fc = False
    mygw.kernel()
    
    _nstates = 6
    _kshift_lst = [0,1]
    _kshift = 1

    #WARNING: Must change gw_e to mf.mo_energy and remove screening!
    mybse = BSE(mygw, TDA=True, singlet=True, kshift_lst = _kshift_lst)
    conv, ekS, xy = mybse.kernel(nstates = _nstates)

    mybse.singlet=False
    conv, ekT, xy = mybse.kernel(nstates = _nstates)
    
    mypbctd = mymf.TDA(kshift_lst = _kshift_lst)
    print('PySCF kTDA singlets')
    TDAeS = mypbctd.run(nstates = _nstates).e
    print('PySCF kTDA triplets')
    TDAeT = mypbctd.run(nstates = _nstates, singlet = False).e
    
    print('BSE code kTDA', 27.2114*(ekS[_kshift]), 27.2114*(ekT[_kshift]))
    print('Pyscf code TkDA', 27.2114*(TDAeS[_kshift]), 27.2114*(TDAeT[_kshift]))
    
    for i in range(_nstates):
        assert(abs(27.2114*(ekS[_kshift][i] - TDAeS[_kshift][i])) < 1e-6)
        print(str(i)+' singlet agrees to within 1e-6 eV')    
        assert(abs(27.2114*(ekT[_kshift][i] - TDAeT[_kshift][i])) < 1e-6)
        print(str(i)+' triplet agrees to within 1e-6 eV')  
    
    print('TDA matches HF-based BSE without screening and with gw_e = mf.mo_energy \
          \n for singlets and triplets with kpt sampling!')
    
    
    import sys
    sys.exit()
    
    ##########################################################
    # same as molecular BSE for large cell size with cell.dim = 0. 
    # Any difference is exactly due to the GW not matching.
    
    # Candidate formula of solid: c, si, sic, bn, bp, aln, alp, mgo, mgs, lih, lif, licl
    #kpt sampling
    cell = pbcgto.Cell()
    cell.atom = '''He 0 0 0;
                He 1.4 0 0'''
    cell.a = np.eye(3)*50
    cell.unit = 'B'
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.dimension = 0
    cell.build()
    
    kdensity = 1
    kmesh = [kdensity, kdensity, kdensity]
    scaled_center=[0.0, 0.0, 0.0]
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    #must have exxdiv=None
    mymf = dft.KRKS(cell, kpts, exxdiv=None).density_fit()
    mymf.xc = 'pbe'
    ekrhf = mymf.kernel()
    
    nocc = cell.nelectron//2
    nmo = np.shape(mymf.mo_energy)[-1]
    nvir = nmo-nocc
    
    gw_orbs = range(nmo)#range(nmo)
    bse_orbs = range(nmo)
    _nstates = 5
    
    from pyscf.pbc.gw import krgw_ac
    import krgw_ac_frozen as krgw_ac
    mygw = krgw_ac.KRGWAC(mymf)
    mygw.linearized = True
    mygw.ac = 'pade'
    # without finite size corrections
    mygw.fc = False
    mygw.kernel(orbs=gw_orbs)
    kgw_e = mygw.mo_energy
    gw_nocc = mygw.nocc
    gw_vir = mygw.nmo - gw_nocc
    sorted_kgw_gaps = sorted([27.2114*(a-i) for a in kgw_e[0][nocc:nocc+gw_vir] for i in kgw_e[0][nocc-gw_nocc:nocc]])
    
    mybse = BSE(mygw, TDA=True, singlet=True)
    conv,  ekS, xy = mybse.kernel(nstates=_nstates, orbs=bse_orbs)
    
    mybse.singlet=False
    conv,  ekT, xy = mybse.kernel(nstates=_nstates, orbs=bse_orbs)
    
    print(27.2114*ekS[0], 27.2114*ekT[0])
    
    #molecule
    mol = cell.to_mol()
    mol.build()

    from pyscf import dft
    mf = dft.RKS(mol).density_fit()
    mf.xc = 'pbe'
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
    
    print(27.2114*ekS[0], 27.2114*ekT[0])
    for i in range(_nstates):
        assert(abs(27.2114*(ekS[0][i] - eS[i])) < 0.005+abs(sorted_kgw_gaps[i]-sorted_gw_gaps[i]))
        print(str(i)+' singlet agrees to within 0.005 eV')    
        assert(abs(27.2114*(ekT[0][i] - eT[i])) < 0.005+abs(sorted_kgw_gaps[i]-sorted_gw_gaps[i]))
        print(str(i)+' triplet agrees to within 0.005 eV')  
        
    print('BSE for a large cell with cell.dim = 0 is the same as molecular BSE \
          for singlets and triplets!')


    ##########################################################
    # kBSE are same for 2x2x2 supercell and 2x2x2 kpt sampling
    # Candidate formula of solid: c, si, sic, bn, bp, aln, alp, mgo, mgs, lih, lif, licl
    from pyscf.pbc import dft
    #kpt sampling
    formula = 'c'
    ase_atom = lattice.get_ase_atom(formula)
    cell = pbcgto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell
    cell.unit = 'B'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build(verbose=2)
    
    kmesh = [2,1,1]
    scaled_center=[0.0, 0.0, 0.0]
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    #must have exxdiv=None
    mymf = dft.KRKS(cell, kpts, exxdiv=None).density_fit()
    # from pyscf.pbc import df
    # gdf = df.GDF(cell, kpts)
    # mymf.with_df = gdf
    mymf.xc = 'lda'
    ekrhf = mymf.kernel()
    
    nocc = cell.nelectron//2
    nmo = np.shape(mymf.mo_energy)[-1]
    nvir = nmo-nocc
    
    mygw = krgw_ac.KRGWAC(mymf)
    mygw.linearized = True
    mygw.ac = 'pade'
    # without finite size corrections
    mygw.fc = False
    mygw.kernel()
    
    _nstates = 5
    
    bse = BSE(mygw, TDA=True, singlet=True)#, kshift_lst=[0,1])
    conv,  ekS, xy = bse.kernel()
    
    bse.singlet=False
    conv, ekT, xy = bse.kernel()
    
    #supercell
    from pyscf.pbc.tools.pbc import super_cell 
    cell = super_cell(cell, [2,1,1])
    cell.build()
    
    kdensity = 1
    kmesh = [kdensity, kdensity, kdensity]
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    #must have exxdiv=None
    mymf = dft.KRKS(cell, kpts, exxdiv=None).density_fit()
    mymf.xc = 'lda'
    ekrhf = mymf.kernel()
    
    nocc = cell.nelectron//2
    nmo = np.shape(mymf.mo_energy)[-1]
    nvir = nmo-nocc
    
    mygw = krgw_ac.KRGWAC(mymf)
    mygw.linearized = True
    mygw.ac = 'pade'
    # without finite size corrections
    mygw.fc = False
    mygw.kernel()
    
    bse = BSE(mygw, TDA=True, singlet=True)
    conv,  eS, xy = bse.kernel()

    bse.singlet=False
    conv,  eT, xy = bse.kernel()
    
    for i in range(_nstates):
        assert(abs(27.2114*(ekS[0][i] - eS[0][i])) < 0.005)
        print(str(i)+' singlet matches to within 0.005 eV')
        assert(abs(27.2114*(ekT[0][i] - eT[0][i])) < 0.005)
        print(str(i)+' triplet matches to within 0.005 eV')
    
    print('BSE supercell at Gamma pt matches BSE unit cell with kpt sampling \
          \n for singlets and triplets!')