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
'''

import time
# from functools import reduce
import numpy as np
from pyscf import lib
from pyscf.pbc import dft, scf
from pyscf.lib import logger
from pyscf import __config__
# import math
# from pyscf.scf import hf_symm
# from pyscf.data import nist

einsum = lib.einsum

from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask
    
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)
MO_BASE = getattr(__config__, 'MO_BASE', 1)

def kernel(bse, nstates=None, verbose=logger.NOTE):
    '''static screening BSE excitation energies

    Returns:
        A list :  converged, number of states, excitation energies, eigenvectors
    '''
    #mf must be DFT; for HF use xc = 'hf'
#    assert(isinstance(bse.mf, dft.rks.RKS) or isinstance(bse.mf, dft.rks_symm.SymAdaptedRKS))
    assert(bse.frozen == 0 or bse.frozen is None)
    
    # cput0 = (time.time(), time.perf_counter())
    log = logger.Logger(bse.stdout, bse.verbose)
    # if bse.verbose >= logger.WARN:
    #     bse.check_sanity()
    bse.dump_flags()
    
    nocc = bse.nocc
    nmo = bse.nmo
    nkpts = bse.nkpts
    # kpts = bse.kpts
    nvir = nmo - nocc
    
    if nstates is None: nstates = 1
    
    matvec, diag = bse.gen_matvec()
    
    size = nocc*nvir
    if not bse.TDA:
        size *= 2
    
    guess, nstates = bse.get_init_guess(nroots=nstates, diag=diag)
        
    nroots = nstates

    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    if bse.TDA:
        eig = lib.davidson1
    else: 
        eig = lib.davidson_nosym1
    
    # GHF or customized RHF/UHF may be of complex type
    real_system = (bse._scf.mo_coeff[0].dtype == np.double)
    
    def pickeig(w, v, nroots, envs):
        real_idx = np.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                              (w.real > POSTIVE_EIG_THRESHOLD))[0]
        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
    conv, e, xy = eig(matvec, guess, precond, pick=pickeig,
                       tol=bse.conv_tol, max_cycle=bse.max_cycle,
                       max_space=bse.max_space, nroots=nroots, verbose=log)
    xy =   [(xi[:nocc*nvir*nkpts].reshape(nkpts, nocc, nvir)*np.sqrt(.5), 0) for xi in xy]
        
    if bse.verbose >= logger.INFO:
        np.set_printoptions(threshold=nocc*nvir)
        logger.debug(bse, '  BSE excitation energies =\n%s', e.real)
        for n, en, vn, convn in zip(range(nroots), e, xy, conv):
            logger.info(bse, '  BSE root %d E = %.16g eV  conv = %s',
                        n, en*27.2114, convn)
        # log.timer('BSE', *cput0)
    return conv, nstates, e, xy
    
def matvec(bse, r, qkLij, qeps_body_inv, all_kidx_r):
    '''matrix-vector multiplication'''
   
    nocc = bse.nocc
    nmo = bse.nmo
    nkpts = bse.nkpts
    nvir = nmo - nocc
    
    kptlist = range(bse.nkpts)
    nklist = len(kptlist)
    
    gw_e = bse.gw_e #TDA: np.asarray(bse.gw._scf.mo_energy)
    gw_e_occ = gw_e[:,:nocc]
    gw_e_vir = gw_e[:,nocc:]
    
    # qkLij, qeps_body_inv, all_kidx_r = make_imds(bse.gw)
    Loo = qkLij[:,:,:,:nocc, :nocc]
    Lov = qkLij[:,:,:,:nocc, nocc:]
    Lvv = qkLij[:,:,:,nocc:, nocc:]
    
    r1 = r[:nkpts*nocc*nvir].copy().reshape(nkpts, nocc, nvir)
    
    #for A
    Hr1 = einsum('ka,kia->kia', gw_e_vir, r1) - einsum('ki,kia->kia', gw_e_occ, r1)
    
    for kL in range(nkpts):
        for k in range(nklist):
            kn = kptlist[k]
            # Find km that conserves with kn and kL (-km+kn+kL=G)
            km = all_kidx_r[kL][kn]
            Hr1[kn,:] -= (1/nkpts) * einsum('Pij, PQ, Qab, jb->ia', Loo[kL,kn,:].conj(), qeps_body_inv[kL], Lvv[kL,kn,:], r1[km,:])
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

#TODO: I don't know if we want to store for all kpts kL or make on the fly
#For memory, will probably need to take the slower approach and make the integrals
#on the fly
from pyscf.pbc.gw.krgw_ac import get_rho_response
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
def make_imds(gw):
    mo_energy = np.array(gw._scf.mo_energy) #mf mo_energy
    mo_coeff = np.array(gw._scf.mo_coeff)
    # nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    mydf = gw.with_df

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
                            in mydf.sr_loop([kpti, kptj], max_memory=0.1*gw._scf.max_memory, compact=False):
                        Lpq.append(LpqR+LpqI*1.0j)
                    # support uneqaul naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1,nmo**2)
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
            
    return np.asarray(qkLij), qeps_body_inv, all_kidx_r
    
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
        'kpts', 'nkpts', 'mo_energy', 'mo_coeff', 'mo_occ',
    }
    
    def __init__(self, gw, frozen=0, TDA=True, singlet=True,  mo_coeff=None, mo_occ=None):
        assert(isinstance(gw._scf, dft.krks.KRKS) or isinstance(gw._scf, dft.krks_symm.SymAdaptedKRKS))
        if mo_coeff  is None: mo_coeff  = gw._scf.mo_coeff
        if mo_occ    is None: mo_occ    = gw._scf.mo_occ
        
        self.gw = gw
        self.mf = gw._scf
        self.mol = gw._scf.mol
        self._scf = gw._scf
        self.gw_e = gw.mo_energy
        self.mo_energy = gw._scf.mo_energy
        self.nkpts = gw.nkpts
        self.kpts = gw.kpts
        self.verbose = self.mol.verbose
        self.stdout = gw.stdout
        self.max_memory = gw.max_memory

        #TODO: implement frozen orbs (since kGW needs them)
        if frozen > 0:
            raise NotImplementedError
        self.frozen = frozen

        # DF-KGW must use GDF integrals
        if getattr(self.mf, 'with_df', None):
            self.with_df = self.mf.with_df
        else:
            raise NotImplementedError

        self.max_space = getattr(__config__, 'eom_rccsd_EOM_max_space', 20)
        self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', 1e-7)

        self.frozen = frozen
        self.TDA = TDA
        self.singlet = singlet
        
##################################################
# don't modify the following attributes, they are not input options
        self.conv = False
        self.e = None
        self.xy = None
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self.nstates = None
        self._nmo = None
        self._keys = set(self.__dict__.keys())
        
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
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
        return self
    
    def kernel(self, nstates=None):
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (2*nkpts**2*nmo**2*naux) * 16/1e6
        mem_now = lib.current_memory()[0]
        if (mem_incore + mem_now > 0.99*self.max_memory):
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError        

        self.conv, self.nstates, self.e, self.xy = kernel(self, nstates=nstates)
    
        return self.conv, self.nstates, self.e, self.xy

    matvec = matvec
    make_imds = make_imds
    
    
    def vector_size(self):
       '''size of the vector'''
       nocc = self.nocc
       nvir = self.nmo - nocc
       nkpts = self.nkpts
       if self.TDA:
           return nkpts*nocc*nvir
       else: 
           return 2*nkpts*nocc*nvir
    
    def gen_matvec(self):
        # nmo = self.nmo
        # nocc = self.nocc
        # nvir = nmo - nocc
    
        qkLij, qeps_body_inv, all_kidx_r = make_imds(self.gw)
        
        diag = self.get_diag(qkLij[0,:], qeps_body_inv[0])
        matvec = lambda xs: [self.matvec(x, qkLij, qeps_body_inv, all_kidx_r) for x in xs]
        return matvec, diag

    def get_diag(self, kLij, eps_body_inv):
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        nkpts = self.nkpts
        
        gw_e = np.asarray(self.mo_energy)
        # gw_e = self.gw_e
        gw_e_occ = gw_e[:,:nocc]
        gw_e_vir = gw_e[:,nocc:]
        
        # Loo = qkLij[:,:,:,:nocc,:nocc]
        # Lov = qkLij[:,:,:,:nocc,nocc:]
        # Lvv = qkLij[:,:,:,nocc:,nocc:]
        Loo = kLij[:,:,:nocc,:nocc]
        Lov = kLij[:,:,:nocc,nocc:]
        Lvv = kLij[:,:,nocc:,nocc:]
        
        diag = np.zeros((nkpts, nocc, nvir), dtype='complex128')
        for i in range(nocc):
            for a in range(nvir):
                diag[:,i,a] += gw_e_vir[:,a] - gw_e_occ[:,i]
                diag[:,i,a] -= einsum('kP, PQ, kQ->k', Loo[:,:,i,i].conj(), eps_body_inv, Lvv[:,:,a,a])
                if self.singlet:
                    diag[:,i,a] += 2*einsum('kP,kP->k', Lov[:,:,i,a].conj(), Lov[:,:,i,a])
        diag = diag.ravel()
        if self.TDA:
            return diag
        else: 
            return np.hstack((diag, -diag))
        
    def get_init_guess(self, nroots=1, diag=None):
        idx = diag.argsort()
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', self.mo_coeff[0].dtype)
        nroots = min(nroots, size)
        guess = []
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess, nroots


    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc.tools import pyscf_ase, lattice
    
    ##############################
    # Create a "Cell"
    ##############################
    
    cell = gto.Cell()
    # Candidate formula of solid: c, si, sic, bn, bp, aln, alp, mgo, mgs, lih, lif, licl
    formula = 'c'
    ase_atom = lattice.get_ase_atom(formula)
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell
    cell.unit = 'B'
    # cell.atom = [
    #         ['H', (0,0,0)],
    #         ['H', (1.4,0,0)]]
    # cell.a = [[40,0,0],[0,40,0],[0,0,40]]
    # cell.pseudo = 'gth-pade'
    cell.basis = 'gth-szv'
    cell.verbose = 7
    cell.build()

    ##############################
    #  K-point SCF
    ##############################
    kdensity = 1
    kmesh = [kdensity, kdensity, kdensity]
    scaled_center=[0.0, 0.0, 0.0]
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    #must have exxdiv=None
    mymf = dft.KRKS(cell, kpts, exxdiv=None).density_fit()
    mymf.xc = 'hf'
    mymf.conv_tol = 1e-12
    ekrhf = mymf.kernel()
    
    nocc = cell.nelectron//2
    nmo = np.shape(mymf.mo_energy)[-1]
    nvir = nmo-nocc
    
    #GW must start with GDF, not FFT (default)
    from pyscf.pbc.gw import krgw_ac
    mygw = krgw_ac.KRGWAC(mymf)
    mygw.linearized = True
    mygw.ac = 'pade'
    # without finite size corrections
    mygw.fc = False
    mygw.kernel()
    print('GW energies (eV)', mygw.mo_energy*27.2114)
    
    bse = BSE(mygw, TDA=True, singlet=True)
    conv, excitations, e, xy = bse.kernel(nstates=7)

    bse.singlet=False
    conv, excitations, e, xy = bse.kernel(nstates=7)

    # mypbctd = mymf.TDA()
    # mypbctd.run(nstates=7)
    # mypbctd.run(nstates=7, singlet=False)
    
    #TODO:
    #     k-point TDSCF solutions can have non-zero momentum transfer between particle and hole.
    # This can be controlled by `td.kshift_lst`. By default, kshift_lst = [0] and only the
    # zero-momentum transfer solution (i.e., 'vertical' in k-space) will be solved, 