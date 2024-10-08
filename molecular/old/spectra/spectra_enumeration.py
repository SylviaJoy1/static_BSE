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

au2ev = 27.2114
ha_2_ev = 1/physical_constants["electron volt-hartree relationship"][0]
n_states=100
spectral_width=0.3
FWHM = 0.2#5.0/au2ev #0.2/27.21139

def gaussian(x, mu, sig=FWHM):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def lorentzian(x, mean, fwhm=FWHM):
    eta = fwhm/2
    return 1.0/np.pi * eta / ((x-mean)**2 + eta**2)

def JamesSmithSpectra(mytd, mybse, gauge='length'):

    mytd.kernel()
    osc_strengths = mytd.oscillator_strength(gauge=gauge)[:n_states-5]
  
    # Convolve lineshapes to make spectra
    energies_ev = mytd.e[:n_states-5]*ha_2_ev
    x_range_tddft = np.linspace(energies_ev.min()*0.9, energies_ev.max()*1.1, num=1000)
    intensity_tddft = np.zeros(x_range_tddft.size)

    print('energies_ev', energies_ev)

    for e, f in zip(energies_ev, osc_strengths):
        intensity_tddft +=  gaussian(x_range_tddft, e) * f #lorentzian(x_range_tddft, e) * f

    # Rough Normalization
    # dx = (x_range_tddft[-1] - x_range_tddft[0])/x_range_tddft.size
    # area = (intensity_tddft*dx).sum()
    # intensity_tddft /= area
  
    conv, excitations, es, xys = mybse.kernel(nstates=n_states)
    osc_strengths = mybse.oscillator_strength(gauge=gauge)[:n_states-5]

    # Convolve lineshapes to make spectra
    energies_ev = mybse.e[:n_states-5]*ha_2_ev
    x_range_bse = np.linspace(energies_ev.min()*0.9, energies_ev.max()*1.1, num=1000)
    intensity_bse = np.zeros(x_range_bse.size)

    for e, f in zip(energies_ev, osc_strengths):
        intensity_bse += gaussian(x_range_bse, e) * f #lorentzian(x_range_bse, e) * f

    
    # Rough Normalization
    # dx = (x_range_bse[-1] - x_range_bse[0])/x_range_bse.size
    # area = (intensity_bse*dx).sum()
    # intensity_bse /= area
   
    return x_range_tddft, x_range_bse, intensity_tddft, intensity_bse

def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return np.einsum('z,zr->r', charges, coords)/charges.sum()

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
    with mol.with_common_orig(_charge_center(mol)):
        # ip_ao = scf.mol.intor_symmetric('int1e_r', comp=3) #xyz components
        ip_ao = scf.mol.intor('int1e_ipovlp_sph', comp=3) #doesnt give right answer
    ip_ao = np.asarray(ip_ao, dtype=dtype)#.transpose(0,1,2)  # with shape (naxis, nmo, nmo)
    ip_ao *= -1j #to get rid of the i prefactor in the integral

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

def XiaoStyleSpectra(mybse, nexc=n_states):
    conv, excitations, es, xys = mybse.kernel(nstates=nexc)
    energies_ev = es[:n_states-5]*ha_2_ev
    
    #xia dipole matrix elements in MO basis
    dipole = get_dipole_mo(mybse)

    # Convolve lineshapes to make spectra
    x_range_bse = np.linspace(energies_ev.min()*0.9, energies_ev.max()*1.1, num=1000)
    intensity_bse = np.zeros((3, len(x_range_bse)))
    
    for x in range(3):
        for e, xy in zip(energies_ev , xys[:n_states-5]):
            f = abs(np.sum(dipole[x]*xy[0]))**2 #0 to take just TDA part
            intensity_bse[x,:] += lorentzian(x_range_bse, e) * f#gaussian(x_range_bse, e, spectral_width) * f

    intensity_bse = (2/3)*np.sum(intensity_bse, axis=0)/x_range_bse#/x_range_bse**2

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
    
   #   mol = gto.M(atom="""  C      1.0701      0.4341     -0.0336
   # C      0.8115     -0.9049     -0.1725
   # C     -0.6249     -1.0279     -0.0726
   # H      1.9842      1.0231     -0.0364
   # H      1.5156     -1.7176     -0.3255
   # H     -1.2289     -1.9326     -0.1322
   # O     -0.0873      1.1351      0.1422
   # N     -1.1414      0.1776      0.1122""",
   # verbose=9, basis = 'gth-dzvp')
     
     xc = 'PBE'
     
   #   # Ground State DFT
   #   mf = dft.RKS(mol, xc=xc).run()
   
     mol = gto.Mole()
     mol.build(
         atom = """  C      1.0701      0.4341     -0.0336
         C      0.8115     -0.9049     -0.1725
          C     -0.6249     -1.0279     -0.0726
          H      1.9842      1.0231     -0.0364
          H      1.5156     -1.7176     -0.3255
          H     -1.2289     -1.9326     -0.1322
          O     -0.0873      1.1351      0.1422
          N     -1.1414      0.1776      0.1122""",
         basis = 'gth-dzvp',
          symmetry = True,
     )
     
     mf = dft.RKS(mol)
     mf.xc = xc
     mf.kernel()

     # Excited State DFT
     mytd = tddft.TDDFT(mf)
     mytd.nstates = n_states
     mytd.max_space = 100
     mytd.max_cycle = 200
     
     # BSE
     mygw = gw.GW(mf, freq_int='ac')
     mygw.kernel()
     print('gw mo energies', mygw.mo_energy)
     formula = 'oxa'
     mf.with_df = df.DF(mol)
     mf.with_df._cderi_to_save = formula+'.h5'
     mybse = BSE(mygw, TDA=True, singlet=True)
    
     
     l_x_range_tddft, l_x_range_bse, l_intensity_tddft, l_intensity_bse = JamesSmithSpectra(mytd, mybse)
     v_x_range_tddft, v_x_range_bse, v_intensity_tddft, v_intensity_bse = JamesSmithSpectra(mytd, mybse, gauge='velocity')
     # x_range_bse, intensity_bse = XiaoStyleSpectra(mybse)
     
     import matplotlib.pyplot as plt
     ax = plt.figure(figsize=(5, 6), dpi=100).add_subplot()
     ax.plot(l_x_range_tddft, l_intensity_tddft, label='length gauge, TDDFT@'+xc, color='blue')
     ax.plot(v_x_range_tddft, v_intensity_tddft, label='velocity gauge, TDDFT@'+xc, color='black', linestyle='--')
     # ax.plot(x_range_bse, intensity_bse, label='BSE@GW@'+xc, color='red')
     # plt.ylim([0,1.5])
     plt.xlim([0,12])
     plt.xlabel("Energy (eV)")
     plt.ylabel("Intensity")
     plt.legend(loc='best')
     plt.savefig('oxazole_enumerated_spectrum_JETS_gauges.png')