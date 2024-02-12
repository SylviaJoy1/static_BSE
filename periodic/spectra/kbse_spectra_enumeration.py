#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:05:09 2023

@author: sylviabintrim
"""

from functools import reduce
import scipy

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
    nocc = eom.nocc
    nmo = eom.nmo
    dtype = 'complex128'
    scf = bse.gw._scf

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

import math
def optical_absorption_singlet(bse, scan, nexc=55, eta=0.005):
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

    #xkia dipole matrix elements in MO basis
    dipole = get_dipole_mo(bse)
    
    #x-axis
    omega_list = scan
    
    spectrum = np.zeros((3, len(omega_list)), dtype='complex128')

    conv, excitations, es, xys = bse.kernel(nstates=nexc)
    xys = np.asarray(xys)
    
    for i, omega in enumerate(omega_list):
        for x in range(3):
            for e, xy in zip(es,xys):
                if abs(omega - e) < eta:
                    spec0 = abs(np.sum(dipole[x]*xy))**2
                    # spectrum[x, i] += spec0
                    spectrum[x, :] += [spec0*math.exp(-10000*abs(o - e)**2) for o in omega_list]

    return np.sum(spectrum, axis=0)/omega_list**2

if __name__ == '__main__':
    from kbse_static_turbomole import BSE
    from pyscf.pbc import gto, dft
    import numpy as np
    from pyscf.pbc.tools import pyscf_ase, lattice
    
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
    # cell.pseudo = 'gth-pbe'
    # cell.basis = 'gth-dzvp'
    # cell.verbose = 7
    # cell.build()
    
    cell = gto.Cell()
    cell.build(unit = 'A',
            a = np.eye(3)*100,
            # mesh = [200]*3,
            atom = '''H 0 0 0; H 0 0 1.4''',
            dimension = 0,
            basis = 'gth-szv')

    ##############################
    #  K-point SCF
    ##############################
    kdensity = 1
    kmesh = [kdensity, kdensity, kdensity]
    scaled_center=[0.0, 0.0, 0.0]
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    #must have exxdiv=None
    mymf = dft.KRKS(cell, kpts, exxdiv=None).density_fit()
    mymf.xc = 'lda'
    mymf.max_memory=10000
    mymf.conv_tol = 1e-12
    ekrhf = mymf.kernel()
    print(ekrhf*27.2114)
    
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
    conv, excitations, e, xy = bse.kernel(nstates=4)
    print('bse energies', e*27.2114)
    
    from pyscf.pbc import tdscf
    mytd = mymf.TDDFT()
    # mytd.nstates = 2
    # mytd.singlet = False
    mytd.run()
    print('tddft energies', [e*27.2114 for e in mytd.e])
    
    wmin = 12/27.2
    wmax = 18/27.2
    dw = 0.005
    
    nw = int((wmax - wmin) / dw) + 1
    scan = np.linspace(wmin, wmax, nw)
    
    spectrum = optical_absorption_singlet(bse, scan)
    import matplotlib.pyplot as plt
    plt.plot(scan*27.2114, spectrum)
    plt.xlabel('Energy (eV)', fontsize=14)
    plt.ylabel('$\epsilon_2$', fontsize=14)
    plt.savefig('periodic_C_spectrum', dpi = 200)
    plt.show()