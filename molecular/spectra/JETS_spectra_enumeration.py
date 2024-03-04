#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:05:09 2023

@author: sylviabintrim
"""

import sys
sys.path.append('/Users/sylviabintrim/Desktop/Desktop - Sylvia’s MacBook Pro/BSE_with_kpts/static_BSE/molecular')  


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
   verbose=3)

     import numpy as np
     from scipy.constants import physical_constants

     ha_2_ev = 1/physical_constants["electron volt-hartree relationship"][0]

     def gaussian(x, mu, sig):
         return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

     def run_spectral_analysis(mol, xc="lda"):
         n_states=15
         spectral_width=0.1

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
         conv, excitations, e, xy = mybse.kernel(nstates=n_states);
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

     xc = 'LDA'
     x_range_tddft, x_range_bse, intensity_tddft, intensity_bse = run_spectral_analysis(mol, xc=xc)
    
     import matplotlib.pyplot as plt
     ax = plt.figure(figsize=(5, 6), dpi=100).add_subplot()
     ax.plot(x_range_tddft, intensity_tddft, label='TDDFT@'+xc)
     ax.plot(x_range_bse, intensity_bse, label='BSE@'+xc)
     plt.legend(loc='best')
     plt.savefig('mol_enumerated_spectrum.png')