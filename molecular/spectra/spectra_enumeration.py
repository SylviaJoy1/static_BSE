#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:05:09 2023

@author: sylviabintrim
"""

import sys
sys.path.append('/Users/sylviabintrim/Desktop/Desktop - Sylvia’s MacBook Pro/BSE_with_kpts/static_BSE/molecular')  

import numpy as np
from scipy.constants import physical_constants

ha_2_ev = 1/physical_constants["electron volt-hartree relationship"][0]
n_states=15
spectral_width=0.1

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def run_spectral_analysis(mol, xc="lda"):

    # Ground State DFT
    mf = dft.RKS(mol, xc=xc).run()

    # Excited State DFT
    mytd = tddft.TDDFT(mf)
    mytd.nstates = n_states
    mytd.max_space = 100
    mytd.max_cycle = 200
    mytd.kernel();
    mytd.analyze()
    osc_strengths = mytd.oscillator_strength()[:n_states-5]
  
    # Convolve lineshapes to make spectra
    energies_ev = mytd.e[:n_states-5]*ha_2_ev
    x_range_tddft = np.linspace(energies_ev.min()*0.9, energies_ev.max()*1.1, num=1000)
    intensity_tddft = np.zeros(x_range_tddft.size)

    for e, f in zip(energies_ev, osc_strengths):
        intensity_tddft += gaussian(x_range_tddft, e, spectral_width) * f

    # Rough Normalization
    dx = (x_range_tddft[-1] - x_range_tddft[0])/x_range_tddft.size
    area = (intensity_tddft*dx).sum()
    intensity_tddft /= area
  
    # BSE
    mygw = gw.GW(mf, freq_int='ac')
    mygw.kernel()
    formula = 'oxazole'
    mf.with_df = df.DF(mol)
    mf.with_df._cderi_to_save = formula+'.h5'
    mybse = BSE(mygw, TDA=True, singlet=True)
    conv, excitations, e, xy = mybse.kernel(nstates=n_states)
    mybse.analyze()
    osc_strengths = mybse.oscillator_strength()[:n_states-5]

    # Convolve lineshapes to make spectra
    energies_ev = mybse.e[:n_states-5]*ha_2_ev
    x_range_bse = np.linspace(energies_ev.min()*0.9, energies_ev.max()*1.1, num=1000)
    intensity_bse = np.zeros(x_range_bse.size)

    for e, f in zip(energies_ev, osc_strengths):
        intensity_bse += gaussian(x_range_bse, e, spectral_width) * f

    # Rough Normalization
    dx = (x_range_bse[-1] - x_range_bse[0])/x_range_bse.size
    area = (intensity_bse*dx).sum()
    intensity_bse /= area
   
    return x_range_tddft, x_range_bse, intensity_tddft, intensity_bse

from functools import reduce
def get_dipole_mo(bse, pblock="occ", qblock="vir"):
    """[summary]
    Args:
        eom ([type]): [description]
        pblock (str, optional): 'occ', 'vir', 'all'. Defaults to "occ".
        qblock (str, optional): 'occ', 'vir', 'all'. Defaults to "vir".

    Returns:
        np.ndarray: shape (naxis, pblock_length, qblock_length)
    """
    # TODO figure out why 'cint1e_ipovlp_cart' causes shape mismatch for `ip_ao` and `mo_coeff` when basis='gth-dzvp'.
    # Meanwhile, let's use 'cint1e_ipovlp_sph' or 'int1e_ipovlp' because they seems to be fine.
    nocc = bse.mf_nocc
    nmo = bse.mf_nmo
    dtype = 'complex128'
    scf = bse.gw._scf

    # int1e_ipovlp gives overlap gradients, i.e. d/dr
    # To get momentum operator, use (-i) * int1e_ipovlp
    ip_ao = scf.mol.intor('cint1e_ipovlp_sph', comp=3)
    ip_ao = np.asarray(ip_ao, dtype=dtype).transpose(0,1,2)  # with shape (naxis, nmo, nmo)
    ip_ao *= -1j

    mo_coeff = scf.mo_coeff
    # mo_energy = scf.mo_energy
    # for x in range(3):
    #     for k in range(nkpts):
    #         print(f"\naxis:{x}, kpt:{k}, dipole AO diagonals:{ip_ao[x,k].diagonal()}")

    # I.p matrix in MO basis (only the occ-vir block)
    def get_range(key):
        if key in ["occ", "all"]:
            start = 0
            end = nocc if key == "occ" else nmo
        elif key == "vir":
            start = nocc
            end = nmo
        return start, end
    pstart, pend = get_range(pblock)
    qstart, qend = get_range(qblock)
    plen = pend - pstart
    qlen = qend - qstart

    ip_mo = np.empty((3, plen, qlen), dtype=dtype)
   
    pmo = mo_coeff[:, pstart:pend]
    qmo = mo_coeff[:, qstart:qend]
    for x in range(3):
        ip_mo[x] = reduce(np.dot, (pmo.T.conj(), ip_ao[x], qmo))
    
    # eia = \epsilon_a - \epsilon_i
    # p_mo_e = [mo_energy[k][pstart:pend] for k in range(nkpts)]
    # q_mo_e = [mo_energy[k][qstart:qend] for k in range(nkpts)]

    # epq = np.empty((nkpts, plen, qlen), dtype=mo_energy[0].dtype)
    # for k in range(nkpts):
    #     epq[k] = p_mo_e[k][:,None] - q_mo_e[k]

    # dipole in MO basis = -I p(p,q) / (\epsilon_p - \epsison_q)
    dipole = np.empty((3, plen, qlen), dtype=ip_mo.dtype)
    for x in range(3):
        #TODO check: should be 1 or -1 * (\epsilon_p - \epsison_q)
        # dipole[x] = -1. * ip_mo[x] / epq

        # switch to pure momentum operator (is it the velocity gauge in dipole approximation?)
        dipole[x] = ip_mo[x]

    return dipole

def optical_absorption_singlet(mol, nexc=n_states, xc="lda"):
    """Compute full CCSD spectra.

    Args:
        eom ([type]): [description]
        scan ([type]): [description]
        eta ([type]): [description]
        tol ([type], optional): [description]. Defaults to 1e-5.
        maxiter (int, optional): [description]. Defaults to 500.
        imds ([type], optional): [description]. Defaults to None.
    """
    mf = dft.RKS(mol, xc=xc).run()
    mygw = gw.GW(mf, freq_int='ac')
    mygw.kernel()
    formula = 'oxazole'
    mf.with_df = df.DF(mol)
    mf.with_df._cderi_to_save = formula+'.h5'
    mybse = BSE(mygw, TDA=True, singlet=True)
    conv, excitations, es, xys = mybse.kernel(nstates=n_states)
    energies_ev = es[:n_states-5]*ha_2_ev
    
    #xkia dipole matrix elements in MO basis
    dipole = get_dipole_mo(mybse)

    # Convolve lineshapes to make spectra
    x_range_bse = np.linspace(energies_ev.min()*0.9, energies_ev.max()*1.1, num=1000)
    intensity_bse = np.zeros((3, len(x_range_bse)))
    
    for x in range(3):
        for e, xy in zip(energies_ev , xys[:n_states-5]):
            f = abs(np.sum(dipole[x]*xy[0]))**2 #0 to take just TDA part
            intensity_bse[x,:] += gaussian(x_range_bse, e, spectral_width) * f

    intensity_bse = np.sum(intensity_bse, axis=0)/x_range_bse**2

    # # Rough Normalization
    dx = (x_range_bse[-1] - x_range_bse[0])/x_range_bse.size
    area = (intensity_bse*dx).sum()
    intensity_bse /= area

    return x_range_bse, intensity_bse

if __name__ == '__main__':
  # abs spectrum, based on code from James Smith
  #https://github.com/jamesETsmith/2022_simons_collab_pyscf_workshop/blob/main/demos/05_Excited_States.ipynb
     from pyscf import gto, dft, gw, tddft, df
     from bse_static_turbomole_for_gwac_frozen import BSE
    
     mol = gto.M(atom="""  C      1.0701      0.4341     -0.0336
   C      0.8115     -0.9049     -0.1725
   C     -0.6249     -1.0279     -0.0726
   H      1.9842      1.0231     -0.0364
   H      1.5156     -1.7176     -0.3255
   H     -1.2289     -1.9326     -0.1322
   O     -0.0873      1.1351      0.1422
   N     -1.1414      0.1776      0.1122""",
   verbose=9)
     
     xc = 'LDA'
     x_range_tddft, x_range_bse, intensity_tddft, intensity_bse = run_spectral_analysis(mol, xc=xc)
     x_range_bse, intensity_bse = optical_absorption_singlet(mol)
     
     import matplotlib.pyplot as plt
     ax = plt.figure(figsize=(5, 6), dpi=100).add_subplot()
     ax.plot(x_range_tddft, intensity_tddft, label='TDDFT@'+xc)
     ax.plot(x_range_bse, intensity_bse, label='BSE@'+xc)
     plt.legend(loc='best')
     plt.savefig('mol_enumerated_spectrum_correct_norm.png')