#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:05:09 2023

@author: sylviabintrim
"""

import sys
sys.path.append('/Users/sylviabintrim/Desktop/Desktop - Sylvia’s MacBook Pro/BSE_with_kpts/static_BSE/periodic')

from functools import reduce

def get_dipole_mo(eom, pblock="occ", qblock="vir"):
    """[summary]

    TODO add kshift argument.

    Args:
        eom ([type]): [description]
        pblock (str, optional): 'occ', 'vir', 'all'. Defaults to "occ".
        qblock (str, optional): 'occ', 'vir', 'all'. Defaults to "vir".

    Returns:
        np.ndarray: shape (naxis, nkpts, pblock_length, qblock_length)
    """
    # TODO figure out why 'cint1e_ipovlp_cart' causes shape mismatch for `ip_ao` and `mo_coeff` when basis='gth-dzvp'.
    # Meanwhile, let's use 'cint1e_ipovlp_sph' or 'int1e_ipovlp' because they seems to be fine.
    kpts = eom.kpts
    nkpts = len(kpts)
    nocc = eom.mf_nocc
    nmo = eom.mf_nmo
    dtype = 'complex128'
    scf = eom.gw._scf

    # int1e_ipovlp gives overlap gradients, i.e. d/dr
    # To get momentum operator, use (-i) * int1e_ipovlp
    ip_ao = scf.cell.pbc_intor('cint1e_ipovlp_sph', kpts=kpts, comp=3)
    ip_ao = np.asarray(ip_ao, dtype=dtype).transpose(1,0,2,3)  # with shape (naxis, nkpts, nmo, nmo)
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

    ip_mo = np.empty((3, nkpts, plen, qlen), dtype=dtype)
    for k in range(nkpts):
        pmo = mo_coeff[k][:, pstart:pend]
        qmo = mo_coeff[k][:, qstart:qend]
        for x in range(3):
            ip_mo[x, k] = reduce(np.dot, (pmo.T.conj(), ip_ao[x, k], qmo))
    
    # eia = \epsilon_a - \epsilon_i
    # p_mo_e = [mo_energy[k][pstart:pend] for k in range(nkpts)]
    # q_mo_e = [mo_energy[k][qstart:qend] for k in range(nkpts)]

    # epq = np.empty((nkpts, plen, qlen), dtype=mo_energy[0].dtype)
    # for k in range(nkpts):
    #     epq[k] = p_mo_e[k][:,None] - q_mo_e[k]

    # dipole in MO basis = -I p(p,q) / (\epsilon_p - \epsison_q)
    dipole = np.empty((3, nkpts, plen, qlen), dtype=ip_mo.dtype)
    for x in range(3):
        #TODO check: should be 1 or -1 * (\epsilon_p - \epsison_q)
        # dipole[x] = -1. * ip_mo[x] / epq

        # switch to pure momentum operator (is it the velocity gauge in dipole approximation?)
        dipole[x] = ip_mo[x]

    return dipole

from scipy.constants import physical_constants
ha_2_ev = 1/physical_constants["electron volt-hartree relationship"][0]
n_states=15
spectral_width=0.1

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

import scipy
def optical_absorption_singlet(eom, scan, eta, kshift=0, tol=1e-5, maxiter=500):
    """Compute BSE spectra.

    Args:
        eom ([type]): [description]
        scan ([type]): [description]
        eta ([type]): [description]
        kshift (int, optional): [description]. Defaults to 0.
        tol ([type], optional): [description]. Defaults to 1e-5.
        maxiter (int, optional): [description]. Defaults to 500.
    """
    
    #OK
    ieta = 1j*eta
    omega_list = scan
    spectrum = np.zeros((3, len(omega_list)), dtype=np.complex)
    
    from pyscf.pbc.ci import kcis_rhf
    counter = kcis_rhf.gmres_counter(rel=True)
    LinearSolver = scipy.sparse.linalg.gcrotmk
    
    
    
    
    dipole = get_dipole_mo(eom, "all", "all")

    # b = <\Phi_{\alpha} | \bar{\dipole} | \Phi_0>
    # b is needed to solve a.x=b linear equations
    b0, b_vector = get_effective_dipole_left(eom, dipole, kshift)
    b_size = b_vector.shape[1]
    # check the \phi_0 component of b vector
    print(f"b0 = {b0}")
    if any(abs(b0) > 1e-6):
        logger.warn(eom, 'Large b0 detected! b0 (x,y,z) = {}). Consider adding b0.conj() * x0 contribution \
                    to spectra'.format(b0))

    # e = <\Phi_0 | (1+\Lambda) \bar{\dipole^{\dagger}} | \Phi_{\alpha}>
    e0, e_vector = get_effective_dipole_right(eom, dipole, kshift, ov_oovv=b_vector)

    # solve linear equations A.x = b
    diag = eom.get_diag(kshift, imds)
    if x0 is None:
        x0 = np.zeros((3, b_size), dtype=b_vector.dtype)

    for i, omega in enumerate(omega_list):
        matvec = lambda vec: eeccsd_matvec_singlet(eom, vec, kshift, imds=imds) * (-1.) + (omega + ieta) * vec
        A = scipy.sparse.linalg.LinearOperator((b_size, b_size), matvec=matvec, dtype=diag.dtype)

        # preconditioner
        # M is the inverse of P, where P should be close to A, but easy to solve.
        # We choose P = H_diags shifted by omega + ieta.
        M = scipy.sparse.diags(np.reciprocal(diag * (-1.) + omega + ieta), format='csc', dtype=diag.dtype)

        for x in range(3):

            sol, info = LinearSolver(A, b_vector[x], x0=x0[x], tol=tol, maxiter=maxiter, M=M, callback=counter,
                                     **kwargs)
            if info == 0:
                print('Frequency', np.round(omega,3), 'converged in', counter.niter, 'iterations')
            else:
                print('Frequency', np.round(omega,3), 'not converged after', counter.niter, 'iterations')
            counter.reset()

            x0[x] = sol
            spectrum[x,i] = np.dot(e_vector[x], sol)

            sol0 = b0[x] + np.dot(sol, amplitudes_to_vector_singlet(imds.Fov, imds.woOvV, kconserv2))
            sol0 /= omega + ieta
            spec0 = e0[x] * sol0
            
            #OK
            logger.debug(eom, 'b0.conj * x0 contribution to spectrum = %.15g', spec0)

            spectrum[x, i] += spec0

    return -1./np.pi*spectrum.imag, x0

def XiaoStyleSpectra(mybse, nexc=n_states):
    """Compute full CCSD spectra.

    Args:
        eom ([type]): [description]
        scan ([type]): [description]
        eta ([type]): [description]
        kshift (int, optional): [description]. Defaults to 0.
        tol ([type], optional): [description]. Defaults to 1e-5.
        maxiter (int, optional): [description]. Defaults to 500.
        imds ([type], optional): [description]. Defaults to None.
    """

    conv, excitations, es, xys = mybse.kernel(nstates=nexc)
    energies_ev = np.asarray(es)[:nexc-5]*ha_2_ev
    
    #xkia dipole matrix elements in MO basis
    dipole = get_dipole_mo(mybse)
    
    # Convolve lineshapes to make spectra
    x_range_bse = np.linspace(energies_ev.min()*0.9, energies_ev.max()*1.1, num=1000)
    intensity_bse = np.zeros((3, len(x_range_bse)))
    
    for x in range(3):
        for e, xy in zip(energies_ev , xys[:nexc-5]):
            f = abs(np.sum(dipole[x]*xy[0]))**2 #0 to take just TDA part
            intensity_bse[x,:] += gaussian(x_range_bse, e, spectral_width) * f

    intensity_bse = (2/3)*np.sum(intensity_bse, axis=0)/x_range_bse#**2

    # # Rough Normalization
    dx = (x_range_bse[-1] - x_range_bse[0])/x_range_bse.size
    area = (intensity_bse*dx).sum()
    intensity_bse /= area

    return x_range_bse, intensity_bse

if __name__ == '__main__':
    from kbse_static_turbomole_frozen import BSE
    from pyscf.pbc import gto, dft
    import numpy as np
    # from pyscf.pbc.tools import pyscf_ase, lattice
    
    ##############################
    # Create a "Cell"
    ##############################
    
    # cell = gto.Cell()
    # # Candidate formula of solid: c, si, sic, bn, bp, aln, alp, mgo, mgs, lih, lif, licl
    # formula = 'c'
    # ase_atom = lattice.get_ase_atom(formula)
    # cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    # cell.a = ase_atom.cell
    # cell.unit = 'B'
    # cell.pseudo = 'gth-lda'
    # cell.basis = 'gth-szv'
    # cell.verbose = 7
    # cell.build()
    
    xc = 'PBE'
    
    cell = gto.Cell()
    cell.build(
            a = np.eye(3),
           mesh = [50]*3,
            atom = """  C      1.0701      0.4341     -0.0336
          C      0.8115     -0.9049     -0.1725
          C     -0.6249     -1.0279     -0.0726
          H      1.9842      1.0231     -0.0364
          H      1.5156     -1.7176     -0.3255
          H     -1.2289     -1.9326     -0.1322
          O     -0.0873      1.1351      0.1422
          N     -1.1414      0.1776      0.1122""",
            dimension = 0,
            basis = 'gth-dzvp')
    
    kdensity = 1
    kmesh = [kdensity, kdensity, kdensity]
    scaled_center=[0.0, 0.0, 0.0]
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    #must have exxdiv=None
    mymf = dft.KRKS(cell, kpts, exxdiv=None).density_fit()
    # from pyscf.pbc.gto import df
    # mymf.with_df = df.DF(cell)
    # formula = 'molecule'
    # mymf.with_df._cderi_to_save = formula+'.h5'
    mymf.xc = xc
    mymf.max_memory=10000
    mymf.kernel()
    
    from pyscf.pbc.gw import krgw_ac
    mygw = krgw_ac.KRGWAC(mymf)
    mygw.linearized = True
    mygw.ac = 'pade'
    # without finite size corrections
    mygw.fc = False
    mygw.kernel()
    print('gw mo energies', mygw.mo_energy)
    mybse = BSE(mygw, TDA=True, singlet=True)
    
    
    x_range_bse, intensity_bse = XiaoStyleSpectra(mybse)
    import matplotlib.pyplot as plt
    ax = plt.figure(figsize=(5, 6), dpi=100).add_subplot()
    # ax.plot(x_range_tddft, intensity_tddft, label='TDDFT@'+xc)
    ax.plot(x_range_bse, intensity_bse, label='BSE@GW@'+xc, color='red')
    plt.ylim([0,1.7])
    plt.xlim([0,11])
    plt.legend(loc='best')
    plt.savefig('periodic_enumerated_spectrum.png')