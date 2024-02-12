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
static screening BSE with iterative diagonalization, with or without TDA, singlet or triplet excitations. 
Density-fitted. Turbomole-style.
!NOTE: since exact gw cannot use DF integrals, this code will not work for exact gw...
only AC and maybe CD GW.
'''

import numpy as np
from pyscf import lib, gw, symm, dft
from pyscf.lib import logger
from pyscf import __config__

einsum = lib.einsum
    
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)
# MO_BASE = getattr(__config__, 'MO_BASE', 1)

def kernel(bse, nstates=None, orbs=None, verbose=logger.NOTE):
    '''static screening BSE excitation energies

    Returns:
        A list :  converged, number of states, excitation energies, eigenvectors
    '''
    #mf must be DFT; for HF use xc = 'hf'
    assert(isinstance(bse.mf, dft.rks.RKS) or isinstance(bse.mf, dft.rks_symm.SymAdaptedRKS))
    # assert(bse.frozen == 0 or bse.frozen is None)
    
    log = logger.Logger(bse.stdout, bse.verbose)
    #TODO: checking sanity based on the parent class may not make sense here.
    #not sure.
    if bse.verbose >= logger.WARN:
        bse.check_sanity()
    
    if orbs is None:
        orbs = [x for x in range(bse.mf_nmo)]
    
    orbs_nocc = sum([x < bse.mf_nocc for x in orbs])
    if orbs_nocc > bse.gw_nocc:
        orbs = [bse.mf_nocc - x for x in range(bse.gw_nocc, 0, -1)] + list(orbs)[orbs_nocc:]
    orbs_vir = sum([x > bse.mf_nocc for x in orbs])
    if orbs_vir > bse.gw_nmo - bse.gw_nocc:
        orbs = list(orbs)[:bse.gw_nocc] + [x for x in range(bse.gw_nocc, bse.gw_nmo)]
   
    nmo = len(orbs)
    nocc = sum([x < bse.mf_nocc for x in orbs])
    nvir = nmo - nocc
    
    if nmo > sum([x != 0 for x in bse.gw_e]):
        logger.warn(bse, 'BSE orbs must be a subset of GW orbs!')
        raise RuntimeError
    
    if nstates is None: nstates = len(orbs)
    
    matvec, diag = bse.gen_matvec(orbs)
    
    size = nocc*nvir
    if not bse.TDA:
        size *= 2
    
    # guess = bse.get_init_guess(nstates=nstates) #for non-TDA BSE (weird)
    guess, nstates = bse.get_init_guess(nstates=nstates, orbs=orbs, diag=diag)
        
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
    xy =   [(xi[:nocc*nvir].reshape(nocc, nvir)*np.sqrt(.5), 0) for xi in xy]
        
    if bse.verbose >= logger.INFO:
        np.set_printoptions(threshold=nocc*nvir)
        logger.debug(bse, '  BSE excitation energies =\n%s', e.real)
        for n, en, vn, convn in zip(range(nroots), e, xy, conv):
            logger.info(bse, '  BSE root %d E = %.16g eV  conv = %s',
                        n, en*27.2114, convn)
    return conv, nstates, e, xy
    
def matvec(bse, r, eris, i_tilde, orbs):
    '''matrix-vector multiplication'''
   
    nocc = sum([x < bse.mf_nocc for x in orbs])
    nmo = len(orbs)
    nvir = nmo - nocc
    
    gw_e = bse.gw_e
    gw_e_occ = gw_e[bse.mf_nocc-nocc:bse.mf_nocc]
    gw_e_vir = gw_e[bse.mf_nocc:bse.mf_nocc+nvir]
    
    r1 = r[:nocc*nvir].copy().reshape(nocc,nvir)
    #for A
    Hr1 = einsum('a,ia->ia', gw_e_vir, r1) - einsum('i,ia->ia', gw_e_occ, r1)
    Hr1 -= einsum('Pji, PQ, Qab, jb->ia', eris.Loo, i_tilde, eris.Lvv, r1)
#    Hr1 -= einsum('Pji, Pab, jb->ia', eris.Loo, R_tilde.vv, r1)
    if bse.singlet:
        Hr1 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r1)
    
    if bse.TDA:
        return Hr1.ravel()
    
    else:
        r2 = r[nocc*nvir:].copy().reshape(nocc,nvir)
    
        #for B
        Hr1 -= einsum('Pib, PQ, Qja, jb->ia', eris.Lov, i_tilde, eris.Lov, r2)
#        Hr1 -= einsum('Pib, Pja, jb->ia', eris.Lov, R_tilde.ov, r2)
        if bse.singlet:
            Hr1 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r2)
        
        #for -A
        Hr2 = einsum('a,ia->ia', gw_e_vir, r2) - einsum('i,ia->ia', gw_e_occ, r2)
        Hr2 -= einsum('Pji, PQ, Qab, jb->ia', eris.Loo, i_tilde, eris.Lvv, r2)
#        Hr2 -= einsum('Pji, Pab, jb->ia', eris.Loo, R_tilde.vv, r2)
        if bse.singlet:
            Hr2 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r2) 
        
        #for -B
        Hr2 -= einsum('Pib, PQ, Qja, jb->ia', eris.Lov, i_tilde, eris.Lov, r1)
#        Hr2 -= einsum('Pib, Pja, jb->ia', eris.Lov, R_tilde.ov, r1)
        if bse.singlet:
            Hr2 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r1)
            
        return np.hstack((Hr1.ravel(), -Hr2.ravel()))

from types import SimpleNamespace
#import scipy.linalg as linalg
#def make_R_tilde(mf, eris):
#    QP_diff = (mf.mo_energy[:nocc, None]-mf.mo_energy[None, nocc:]).T
#    i_mat = 4*einsum('Pia,Qia,ai->PQ', eris.Lov, eris.Lov, 1./QP_diff)
#    c = np.linalg.cholesky(np.eye(np.shape(i_mat)[0])-i_mat)
#
#    R_tilde = SimpleNamespace()
#    R_tilde_ov = linalg.solve_triangular(c, eris.Lov, lower=True)
#    R_tilde.ov = linalg.solve_triangular(c, R_tilde_ov, trans='T', lower=True)
#
#    R_tilde_vv = linalg.solve_triangular(c, eris.Lvv, lower=True)
#    R_tilde.vv = linalg.solve_triangular(c, R_tilde_vv, trans='T', lower=True)
#
#    return R_tilde

from pyscf.ao2mo import _ao2mo
def make_imds(gw, orbs):
    mf_nocc = gw._scf.mol.nelectron//2
    mo_coeff = np.array(gw._scf.mo_coeff)[:,orbs]
    nocc = sum([x < mf_nocc for x in orbs])
    nmo = len(orbs)
    nvir = nmo - nocc
  
    
    with_df = gw.with_df
 
    naux = with_df.get_naoaux()

    Loo = np.empty((naux,nocc,nocc))
    Lov = np.empty((naux,nocc,nvir))
    Lvv = np.empty((naux,nvir,nvir))
    mo = np.asarray(mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    for k, eri1 in enumerate(with_df.loop()):
        Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1-p0,nmo,nmo)
        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
        Lvv[p0:p1] = Lpq[:,nocc:,nocc:]

    eris = SimpleNamespace()
    eris.Loo = Loo
    eris.Lov = Lov
    eris.Lvv = Lvv

    mo_energy = gw._scf.mo_energy[np.ix_(orbs)]
    QP_diff = (mo_energy[:nocc, None]-mo_energy[None, nocc:]).T
    i_mat = 4*einsum('Pia,Qia,ai->PQ', eris.Lov, eris.Lov, 1./QP_diff)
    i_tilde = np.linalg.inv(np.eye(np.shape(i_mat)[0])-i_mat)
#    R_tilde = make_R_tilde(bse.mf, eris)
    
    return eris, i_tilde
    
from pyscf.tdscf import rhf
class BSE(rhf.TDA):
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
        'frozen', 'mol', 'with_df', 'mo_energy', 'mo_coeff', 'mo_occ'
    }
    def __init__(self, gw, TDA=True, singlet=True,  mo_coeff=None, mo_occ=None):
        assert(isinstance(gw._scf, dft.rks.RKS) or isinstance(gw._scf, dft.rks_symm.SymAdaptedRKS))
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
        self.gw_e = gw.mo_energy
        self.verbose = self.mol.verbose
        self.stdout = gw._scf.stdout
        self.max_memory = gw._scf.max_memory

        self.max_space = getattr(__config__, 'eom_rccsd_EOM_max_space', 20)
        self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', 1e-7)
        self.nstates = getattr(__config__, 'tdscf_rhf_TDA_nstates', 3)

        self.frozen = gw.frozen
        self.TDA = TDA
        self.singlet = singlet
        
        self.wfnsym = None
        
        # DF-GW can use GDF integrals
        # EXACT GW cannot use GDF integrals
        if getattr(self.mf, 'with_df', None):
            self.with_df = self.mf.with_df
        else:
            raise NotImplementedError
        
##################################################
# don't modify the following attributes, they are not input options
        self.conv = False
        self.e = None
        self.xy = None
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        # self._nocc = None
        # self.nstates = None
        # self._nmo = None
        self._keys = set(self.__dict__.keys())
        
        
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
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
        if self.singlet is None:
            log.info('nstates = %d', self.nstates)
        elif self.singlet:
            log.info('nstates = %d singlet', self.nstates)
        else:
            log.info('nstates = %d triplet', self.nstates)
        return self
    
    matvec = matvec
    make_imds = make_imds
    
    def kernel(self, nstates=None, orbs=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        self.conv, self.nstates, self.e, self.xy = kernel(self, nstates=nstates, orbs=orbs)
        logger.timer(self, 'BSE', *cput0)
        return self.conv, self.nstates, self.e, self.xy
    
    def gen_matvec(self, orbs):
        eris, i_tilde = make_imds(self.gw, orbs)
        diag = self.get_diag(eris, i_tilde, orbs)
        matvec = lambda xs: [self.matvec(x, eris, i_tilde, orbs) for x in xs]
        return matvec, diag

    def get_diag(self, eris, i_tilde, orbs):
        nocc = sum([x < self.mf_nocc for x in orbs])
        nmo = len(orbs)
        nvir = nmo - nocc
        
        gw_e_occ = self.gw_e[self.mf_nocc-nocc:self.mf_nocc]
        gw_e_vir = self.gw_e[self.mf_nocc:self.mf_nocc+nvir]
        
        diag = np.zeros((nocc,nvir))
        for i in range(nocc):
            for a in range(nvir):
                diag[i,a] += gw_e_vir[a] - gw_e_occ[i]
                diag[i,a] -= einsum('P, PQ, Q', eris.Loo[:,i,i], i_tilde, eris.Lvv[:,a,a])
#                diag[i,a] -= np.dot(eris.Loo[:,i,i], R_tilde.vv[:,a,a])
                if self.singlet:
                    diag[i,a] += 2*np.dot(eris.Lov[:,i,a], eris.Lov[:,i,a])
        diag = diag.ravel()
        if self.TDA:
            return diag
        else: 
            return np.hstack((diag, -diag))
    
    #TODO: this guess works for TDA=False, but not with frozen orbs. 
    # def get_init_guess(self, nstates=None, gw_e=None):
       
    #     if nstates is None: nstates = self.nstates
       
    #     if gw_e is None: gw_e = self.gw_e
    #     print('self.mf_nocc:self.mf_nocc+nvir', self.mf_nocc,self.mf_nocc+nvir)
    #     print('self.mf_nocc-nocc:self.mf_nocc', self.mf_nocc-nocc,self.mf_nocc)
    #     print('GW occ', gw_e[self.mf_nocc-nocc:self.mf_nocc, None])
    #     print('GW vir', gw_e[None,self.mf_nocc:self.mf_nocc+nvir])
    #     Ediff = gw_e[None,self.mf_nocc:self.mf_nocc+nvir] - gw_e[self.mf_nocc-nocc:self.mf_nocc, None]
    #     print('np.shape(Ediff)', np.shape(Ediff))
    #     e_ia = np.hstack([x.ravel() for x in Ediff])
    #     e_ia_max = e_ia.max()
    #     nov = e_ia.size
    #     nstates = min(nstates, nov)
    #     e_threshold = min(e_ia_max, e_ia[np.argsort(e_ia)[nstates-1]])
    #     # Handle degeneracy, include all degenerated states in initial guess
    #     e_threshold += 1e-6

    #     idx = np.where(e_ia <= e_threshold)[0]
    #     x0 = np.zeros((idx.size, nov))
    #     for i, j in enumerate(idx):
    #         x0[i, j] = 1  # Koopmans' excitations
    #     guess = x0
    #     if not self.TDA:
    #         guess = np.hstack((guess,0*guess))
        
    #     # guess = BSE.init_guess(self, self.mf)
    #     return guess
    
    def vector_size(self, orbs):
        '''size of the vector'''
        nocc = sum([x < self.mf_nocc for x in orbs])
        nmo = len(orbs)
        nvir = nmo - nocc
        if self.TDA:
            return nocc*nvir
        else: 
            return 2*nocc*nvir
       
    def get_init_guess(self, nstates, orbs, diag=None):
        idx = diag.argsort()
        size = self.vector_size(orbs)
        dtype = getattr(diag, 'dtype', self.mo_coeff[0].dtype)
        nroots = min(nstates, size)
        guess = []
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess, nroots

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
     from pyscf import gto
     #doi: 10.1063/5.0023168
     mol = gto.Mole(unit='A')
     mol.atom = [['O',(0.0000, 0.0000, 0.0000)],
             ['H', (0.7571, 0.0000, 0.5861)],
             ['H', (-0.7571, 0.0000, 0.5861)]]
     mol.basis = 'aug-cc-pVTZ'
     mol.symmetry = True
     mol.build()
     formula = 'water'

     mf = dft.RKS(mol).density_fit()
     mf.xc = 'hf'
     mf.kernel()

     nocc = mol.nelectron//2
     nmo = mf.mo_energy.size
     nvir = nmo-nocc

     mygw = gw.GW(mf, frozen=0, freq_int='ac')
     mygw.kernel(orbs=range(nmo))
     gw_e = mygw.mo_energy
     print('gw_e', gw_e)

     #Technically, should be TDA=False to compare with lit values,
     #but the final two singlet exc. converged to something weird (weird symm too),
     #so I had to set TDA=True.
     bse = BSE(mygw, TDA=True, singlet=True)
     bse.verbose = 9
     conv, excitations, e, xy = bse.kernel(nstates=4, orbs=range(nmo))
     print(e*27.2114)
     assert(abs(27.2114*bse.e[0] - 8.09129) < 5*1e-2)
     assert(abs(27.2114*bse.e[1] - 9.78553) < 5*1e-2)
     assert(abs(27.2114*bse.e[2] - 10.41702)  < 5*1e-2)
     # bse.analyze()
     print('BSE singlet matches lit')

     bse.singlet=False
     conv, excitations, e, xy = bse.kernel(nstates=4, orbs=range(nmo))
     print(e*27.2114)
     assert(abs(27.2114*bse.e[0] - 7.61802) < 5*1e-2)
     assert(abs(27.2114*bse.e[1] - 9.59825) < 5*1e-2)
     assert(abs(27.2114*bse.e[2] - 9.79518) < 5*1e-2)
     # bse.analyze()
     print('BSE triplet matches lit')
