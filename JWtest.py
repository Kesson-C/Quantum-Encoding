# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:13:37 2021

@author: mne081069
"""
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType
from qiskit.chemistry import FermionicOperator
from ground import groupedFermionicOperator
from qiskit.aqua.algorithms import NumPyEigensolver
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
def get_qubit_op(dist):
    driver = PySCFDriver(atom="Li 0 0 0; H .0 .0 " + str(dist)#+ ";H .0 .0 " + str(dist*2)
                             # + ";H .0 .0 " + str(dist*3)+";H .0 .0 " + str(dist*4)#+";H .0 .0 " + str(dist*5)
                         , unit=UnitsType.ANGSTROM, hf_method = HFMethodType.UHF
                         , spin=0,charge=0, basis='sto3g')
    molecule = driver.run()
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    num_electrons = molecule.num_alpha + molecule.num_beta
    print(ferOp.modes,num_electrons)
    g = groupedFermionicOperator(ferOp, num_electrons)
    t1=time.time()
    qubitOp = g.to_paulis()
    t2=time.time()
    print(t2-t1)
    qubitOpJW = ferOp.mapping(map_type='jordan_wigner', threshold=1e-8)
    return qubitOp, qubitOpJW

qubitOp, qubitOpJW=get_qubit_op(1)
result_exact = NumPyEigensolver(qubitOp).run()
energy_exact = (np.real(result_exact.eigenvalues))[0]
print(energy_exact)
result_exactJW = NumPyEigensolver(qubitOpJW).run()
energy_exactJW = (np.real(result_exactJW.eigenvalues))[0]
print(energy_exactJW)
