
import qiskit.quantum_info as qi
import numpy as np

i0 = np.array([[1,0],[0,1]])
x0 = np.array([[0,1],[1,0]])
y0 = np.array([[0,-1j],[1j,0]])
z0 = np.array([[1,0],[0,-1]])

def I(N):
    out = 1
    for i in range(N):
        out = np.kron(i0,out)
    return out

def X(i,N):
    out = 1
    for j in range(i):
        out = np.kron(i0,out)
    out = np.kron(x0,out)
    for j in range(i+1,N):
        out = np.kron(i0,out)
    return out

def Y(i,N):
    out = 1
    for j in range(i):
        out = np.kron(i0,out)
    out = np.kron(y0,out)
    for j in range(i+1,N):
        out = np.kron(i0,out)
    return out

def Z(i,N):
    out = 1
    for j in range(i):
        out = np.kron(i0,out)
    out = np.kron(z0,out)
    for j in range(i+1,N):
        out = np.kron(i0,out)
    return out

def c(i,N):
    out = 1
    for j in range(i):
        out = np.kron(i0,out)
    out = 1/2*np.kron(x0 + 1j*y0, out)
    for j in range(i+1,N):
        out = np.kron(z0,out)
    return out

def cd(i,N):
    out = 1
    for j in range(i):
        out = np.kron(i0,out)
    out = 1/2*np.kron(x0 - 1j*y0, out)
    for j in range(i+1,N):
        out = np.kron(z0,out)
    return out


def n(i,N):
    return Mdot([cd(i,N),c(i,N)])


import numpy as np


def Mdot(Ol):
    L = len(Ol)
    out = Ol[L-1]
    for i in range(1,len(Ol)):
        out = np.dot(Ol[L-1-i],out)
    return out

def bkt(y1,O,y2):
    return Mdot([np.conjugate(y1),O,y2])
