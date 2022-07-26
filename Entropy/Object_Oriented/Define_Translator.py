
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from Define_Paulis import I,X,Y,Z,Mdot,bkt

def rho0(N):
    rho = [[0 for i in range(2**N)] for j in range(2**N)]
    rho[0][0] = 1
    return rho

def h(i,N):
    return 1/np.sqrt(2)*(Z(i,N) + X(i,N))

def rx(phi,i,N):
    return np.cos(phi/2)*I(N) - 1j*np.sin(phi/2)*X(i,N)

def ry(phi,i,N):
    return np.cos(phi/2)*I(N) - 1j*np.sin(phi/2)*Y(i,N)

def rz(phi,i,N):
    return np.cos(phi/2)*I(N) - 1j*np.sin(phi/2)*Z(i,N)

def cx(i,j,N):
    return 1/2*( I(N) + Z(i,N) + X(j,N) - Mdot([Z(i,N),X(j,N)]) )
    
def translate(cir):
    N = len(cir.qubits)
    rho = rho0(N)
    for gate in cir:
        if gate[0].name == 'h':
            q = gate[1][0].index
            rho = Mdot([h(q,N),rho,h(q,N)])
        elif gate[0].name == 'ry':
            q = gate[1][0].index
            phi = gate[0].params[0]
            rho = Mdot([ry(phi,q,N),rho,ry(-phi,q,N)])
        elif gate[0].name == 'rx':
            q = gate[1][0].index
            phi = gate[0].params[0]
            rho = Mdot([rx(phi,q,N),rho,rx(-phi,q,N)])
        elif gate[0].name == 'rz':
            q = gate[1][0].index
            phi = gate[0].params[0]
            rho = Mdot([rz(phi,q,N),rho,rz(-phi,q,N)])
        elif gate[0].name == 'cx':
            q1 = gate[1][0].index
            q2 = gate[1][1].index
            rho = Mdot([cx(q1,q2,N),rho,cx(q1,q2,N)])
        elif gate[0].name == 'x':
            q = gate[1][0].index
            rho = Mdot([X(q,N),rho,X(q,N)])
        elif gate[0].name == 'y':
            q = gate[1][0].index
            rho = Mdot([Y(q,N),rho,Y(q,N)])
        elif gate[0].name == 'z':
            q = gate[1][0].index
            rho = Mdot([Z(q,N),rho,Z(q,N)])
        elif gate[0].name == 'barrier':
            q = None
        elif gate[0].name == 'measure':
            q = gate[1][0].index
            rho = 1/4*Mdot([I(N) - Z(q,N),rho,I(N) - Z(q,N)]) + 1/4*Mdot([I(N) + Z(q,N),rho,I(N) + Z(q,N)])
        else:
            print('The gate '+str(gate[0].name)+'was not recognized and was skipped!')
    return rho

# A function to print out the binary number
def bi(num,N):
    bi = bin(num)
    out = ""
    for i in range(2,len(bi)):
        out = out + bi[i]  
    for i in range(len(bi)-2,N):
        out = '0' + out
    return out

def measure_rho(rho):
    L = len(rho)
    N = int(np.log2(L))
    r = {}
    for i in range(L):
        r[bi(i,N)] = rho[i][i]
    return r
    
