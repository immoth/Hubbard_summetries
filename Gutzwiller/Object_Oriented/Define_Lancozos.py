
from Define_System import system
from Define_Paulis import I,X,Y,Z, cd,c,n, Mdot, bkt
import scipy.linalg as ln
import numpy as np
from random import random
from scipy.linalg import eigh_tridiagonal 



def Krolov(H,Lamb):
    N = int(np.log2(len(H)))
    
    #Define containers
    vt_l = []
    v_l = []
    a_l = []
    b_l = []

    #Initialize
    v0t = np.array([random() for i in range(0,len(H))])
    v0 = v0t/np.sqrt(bkt(v0t,I(N),v0t))
    a0 = bkt(v0,H,v0)
    vt_l.append(v0t)
    v_l.append(v0)
    a_l.append(a0)
    b_l.append(0)

    #first step
    v1t = Mdot([H,v0]) - a0*v0
    b1 = np.sqrt(bkt(v1t,I(N),v1t))
    v1 = (1/b1)*v1t
    a1 = bkt(v1,H,v1)
    vt_l.append(v1t)
    v_l.append(v1)
    a_l.append(a1)
    b_l.append(b1)
    
    #Iterate through further steps
    for i in range(2,Lamb):
        vit = Mdot([H,v_l[i-1]]) - a_l[i-1]*v_l[i-1] - b_l[i-1]*v_l[i-2]
        bi = np.sqrt(bkt(vit,I(N),vit))
        vi = (1/bi)*vit
        ai = bkt(vi,H,vi)
        vt_l.append(vit)
        v_l.append(vi)
        a_l.append(ai)
        b_l.append(bi)
    b_l = np.delete(b_l,0)
    return a_l, b_l


def Lancozos(H,Lamb):
    a_l, b_l = Krolov(H,20)
    e,y = eigh_tridiagonal(np.real(a_l),np.real(b_l))
    return np.sort(e)
