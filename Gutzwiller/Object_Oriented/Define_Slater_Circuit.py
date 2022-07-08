
from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, transpile, QuantumRegister,ClassicalRegister, execute
import numpy as np
from Define_Paulis import Mdot

# A single particle rotation used to diagonalize F
def ry(i,j,phi,N):
    M = (1+0*1j)*np.identity(N)
    M[i,i] = np.cos(phi)
    M[j,j] = np.cos(phi)
    M[i,j] = np.sin(phi)
    M[j,i] = -np.sin(phi)
    return M

#Another single particle rotation used to diagonalize F
def rz(j,phi,N):
    M = (1+0*1j)*np.identity(N)
    M[j,j] = np.exp(1j*phi)
    return M

# The fermi-swap operator which is not used in the final contruction
def fswap(i,j,qc):
    qc.swap(i,j)
    qc.ry(np.pi/2,j)
    qc.cx(i,j)
    qc.ry(-np.pi/2,j)
    return qc
    
#A general application of the ry gate to the quantum register.  
#In the end, we only take j=i+1 so the fswaps are not used.
def R_cc(i,j,phi,qc):
    for l in range(i+1,j):
        qc = fswap(l-1,l,qc)
    qc.ry(-np.pi/2,j-1)
    qc.cx(j-1,j)
    qc.ry(-phi,j)
    qc.cx(j-1,j)
    qc.ry(np.pi/2,j-1)
    qc.ry(-np.pi/2,j)
    qc.cx(j,j-1)
    qc.ry(phi,j-1)
    qc.cx(j,j-1)
    qc.ry(np.pi/2,j)
    for l in range(j-1,i+1-1,-1):
        qc = fswap(l-1,l,qc)
    return qc

#Application of the full rotation to the quantum register
def G_cc(i,j,phi,phiz,qc):
    qc = R_cc(i,j,-phi,qc)
    qc.rz(-phiz,j)
    return qc

#Function which applies both ry and rz to F and returns the angles needed to remove an element at 
#collumn c, row rj by rotating it into collumn c row ri.
def givens(ri,rj,c,F):
    N = len(F)
    if F[rj,c] == 0:
        F_new = F
        phiz = 0
        phi = 0
    elif F[ri,c] == 0:
        phiz = 1j*np.log( F[rj,c]/np.abs(F[rj,c]) ,dtype = 'complex')
        Fz =  Mdot([rz(rj,phiz,N) , F])
        phi = np.pi/2
        F_new = Mdot([ry(ri,rj,phi,N) , Fz])
    else:
        phiz = 1j*np.log( F[rj,c]/F[ri,c] * np.abs(F[ri,c])/np.abs(F[rj,c]) ,dtype = 'complex')
        Fz =  Mdot([rz(rj,phiz,N) , F])
        phi = np.arctan(Fz[rj,c]/Fz[ri,c] ,dtype = 'complex')
        F_new = Mdot([ry(ri,rj,phi,N) , Fz])
    return F_new,phiz,phi

# Generates the slatter circuit which applies the roation F0 to the quantum register.
def slater_circ(F0):
    Fl = [F0]
    pzl = [0]
    pl = [0]
    N = len(F0)
    n = 0
    for i in range(N):
        for j in range(N-1-i):
            Fn, pzn, pn = givens(N-2-j,N-1-j,i,Fl[n])
            n+=1
            Fl.append(Fn)
            pzl.append(pzn)
            pl.append(pn)
            #print(N-2-j,N-1-j,i)
    ph0 = [-1j*np.log(Fl[-1][i,i]) for i in range(N)]    
    qr = QuantumRegister(N)
    cr = ClassicalRegister(N)
    qc = QuantumCircuit(qr , cr)
    qc.x(N-2)
    qc.x(N-1)
    print(' ')
    for i in range(len(ph0)):
        qc.rz(np.real(ph0[i]),i)
    n = len(Fl) - 1
    for i in range(N,0,-1):
        for j in range(i,N):
            #print(j-1,j,i-1)
            qc = G_cc(j-1,j,np.real(pl[n]),np.real(pzl[n]),qc)
            n -= 1
    return qc

# The circuit has a global phase offset from a direct application of the eigenmodes defined by the system.  
# This function calculates that phase. 
def phase_offset(F0):
    Fl = [F0]
    pzl = [0]
    pl = [0]
    N = len(F0)
    n = 0
    for i in range(N):
        for j in range(N-1-i):
            Fn, pzn, pn = givens(N-2-j,N-1-j,i,Fl[n])
            n+=1
            Fl.append(Fn)
            pzl.append(pzn)
            pl.append(pn)
    ph0 = [-1j*np.log(Fl[-1][i,i]) for i in range(N)]  
    phase = 1
    for i in range(len(ph0)):
        phase *= np.exp(-1j*ph0[i]/2)
    for i in range(len(pzl)):
        phase = phase*np.exp(1j*pzl[i]/2)
    return phase
