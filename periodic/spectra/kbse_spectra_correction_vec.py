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
    mo_energy = scf.mo_energy
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

def optical_absorption_singlet(bse, scan, eta, kshift=0, tol=1e-5, maxiter=500, eris=None, imds=None, x0=None,
                               partition=None, **kwargs):
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
    dipole = get_dipole_mo(bse, "occ", "vir")
    # b is needed to solve a.x=b linear equations
    b_vector = dipole
    b_size = b_vector.shape[1]
    e_vector = dipole

    # solve linear equations A.x = b
    ieta = 1j*eta
    omega_list = scan
    spectrum = np.zeros((3, len(omega_list)), dtype=np.complex)

    eris = bse.eris
    QP_diff = (bse.mf.mo_energy[:nocc, None]-bse.mf.mo_energy[None, nocc:]).T
    i_mat = 4*np.einsum('Pia,Qia,ai->PQ', eris.Lov, eris.Lov, 1./QP_diff)
    i_tilde = np.linalg.inv(np.eye(np.shape(i_mat)[0])-i_mat)
    
    diag = bse.get_diag(eris, i_tilde)
    if x0 is None:
        x0 = np.zeros((3, b_size), dtype=b_vector.dtype)

    from pyscf.pbc.ci import kcis_rhf
    counter = kcis_rhf.gmres_counter(rel=True)
    LinearSolver = scipy.sparse.linalg.gcrotmk

    for i, omega in enumerate(omega_list):
        #omega - H' + i * eta
        matvec = lambda vec: bse.matvec(vec, eris, i_tilde) * (-1.) + (omega + ieta) * vec
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

            sol0 = b0[x] 
            sol0 /= omega + ieta
            spec0 = e0[x] * sol0

            spectrum[x, i] += spec0

    return -1./np.pi*spectrum.imag, x0


if __name__ == '__main__':
    from kbse_static_turbomole import BSE
    from pyscf.pbc import gto, dft
    import numpy as np
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