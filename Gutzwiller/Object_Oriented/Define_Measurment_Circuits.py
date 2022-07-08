
import copy 
from Define_Slater_Circuit import slater_circ
import numpy as np
from Define_Paulis import  Mdot

# Applies a square-root of the iSWAP operator.  This is needed for the measurment basis roation.
def apply_riswap(qc_in,j,i):
    qc = copy.deepcopy(qc_in)
    qc.cx(j,i)
    qc.ry(np.pi/2,j)
    qc.rz(np.pi/8,j)
    qc.cx(i,j)
    qc.rz(-np.pi/4,j)
    qc.cx(i,j)
    qc.rz(np.pi/8,j)
    qc.ry(-np.pi/2,j)
    qc.cx(j,i)
    return qc

# Applies the measurment basis rotation
def apply_Um(qc_in,j,i):
    qc = copy.deepcopy(qc_in)
    qc.rz(-np.pi/4,j)
    qc.rz(np.pi/4,i)
    qc = apply_riswap(qc,i,j)
    return qc

# Used to swap the indecies in F
def swap_idx(i,j,Op):
    S = np.identity(len(Op))
    S[i,i] = 0
    S[j,j] = 0
    S[i,j] = 1
    S[j,i] = 1
    return Mdot([S,Op])

# Creates a single circuit which is ready to be measured
def create_circ(Fd,pauli):
    label = ''
    for pi in range(len(pauli)):
        if pauli[pi] == 'X' or pauli[pi] == 'Y':
            label = label + str(pi)
    if label == '':
        qc = slater_circ(Fd)
        psi = copy.deepcopy(qc)
        psi.measure(psi.qubits,psi.clbits)
    else:
        i = int(label[0])
        j = int(label[1])
        Fd_swap = swap_idx(1,j,swap_idx(0,i,Fd))
        qc = slater_circ(Fd_swap)
        psi0 = copy.deepcopy(qc)
        psi = apply_Um(psi0,0,1)
        psi.measure(psi.qubits,psi.clbits)
    return psi

# Collects all the circuits that need to be measured into a list
def create_circs(Fd,paulis):
    circs = []
    for pauli in paulis:
        circ = create_circ(Fd,pauli)
        circs.append(circ)
    return circs
