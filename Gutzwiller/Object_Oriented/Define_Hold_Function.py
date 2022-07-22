
from qiskit.circuit import Qubit, Instruction, QuantumRegister
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Prints out a list of gate names in the order that they are applied
def print_circ(circ):
    import numpy as np
    Lc = len(circ_new.data)
    tst = []
    for i in range(Lc):
        angle = np.round(circ_new.data[i][0].params,3)
        idx = circ_new.data[i][1][0].index
        name = circ_new.data[i][0].name
        tst.append([i,name,idx,angle])
    return tst

# Creates an Ry rotation gate in the language of instructions so that it can be inserted into the instructions
def ry(angle,qbit,circ):
    circ.ry(angle,qbit)
    out = circ.data[-1]
    circ.data.pop(-1)
    return out

# Creates an Cx rotation gate in the language of instructions so that it can be inserted into the instructions
def cx(q1,q2,circ):
    circ.cx(q1,q2,qbit)
    out = circ.data[-1]
    circ.data.pop(-1)
    return out

#Inserts a Ry(phi)Ry(-phi) identity to hold the circuit for a period of time
def insert_hold(angle,qbit,i,circ):
    reg_name = circ.data[0][1][0].register.name
    N = circ.data[0][1][0].register.size
    circ.data.insert(i+1,ry(angle,qbit,circ))
    circ.data.insert(i+2,ry(-angle,qbit,circ))
    return 

#Creates a new circuit with Ry(phi)Ry(-phi) inserted after every Cnot gate
def create_hold_circ(angle,circ_in):
    circ = copy.deepcopy(circ_in)
    Lc = len(circ.data)
    i=0
    while i < Lc:
        idx = circ.data[i][1][0].index
        name = circ.data[i][0].name
        #print(i,name)
        #print(print_circ(circ))
        if name == 'cx':
            #print(i)
            qbit = circ.data[i][1][0].index
            insert_hold(angle,qbit,i,circ)
            Lc += 2
        i += 1
    return circ

# Creates a hold circuit for each circuit in a list
def create_hold_circs(angle,circs_in):
    out = []
    for circ_in in circs_in:
        out.append(create_hold_circ(angle,circ_in))
    return out


    
