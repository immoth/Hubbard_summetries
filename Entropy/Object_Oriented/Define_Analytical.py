
import scipy.linalg as ln
import numpy as np
from Define_Paulis import Mdot

def ketbra(psia,psib):
    out = [[0 for i in range(len(psia))] for j in range(len(psib))]
    for i in range(len(psia)):
        for j in range(len(psib)):
            out[i][j] = psia[i]*np.conjugate( psib[j] )
    return np.array(out)
            
def S(rho):
    e,y = ln.eig(rho)
    s = -sum([np.round( (e[n]+10**-28)*np.log(e[n]+10**(-28)) , 8) for n in range(len(e))])
    return s

def FreeE(T,H,rho):
    #print('H: ',np.trace(Mdot([H,rho])))
    #print('S: ', + T*S(rho))
    return np.trace(Mdot([H,rho])) - T*S(rho)
    

def rho(a,b,c,d):
    rho0 = a*ketbra(psi[arg[0]],psi[arg[0]]) + b*ketbra(psi[arg[1]],psi[arg[1]]) + c*ketbra(psi[arg[2]],psi[arg[2]]) + d*ketbra(psi[arg[3]],psi[arg[3]])
    rho0 = rho0/np.trace(rho0)
    return rho0

def rho_op(H,T):
    e,y = ln.eig(H)
    arg = np.argsort(e)
    psi = np.conjugate(np.transpose(y))
    rho = 0 
    for n in range(len(e)):
        rho += np.exp( -e[arg[n]]/T )*ketbra(psi[arg[n]],psi[arg[n]])
    return rho/np.trace(rho)

def S_op(H,T):
    e,y = ln.eig(H)
    arg = np.argsort(e)
    psi = np.conjugate(np.transpose(y))
    so = 0
    z = sum([np.exp(-e[n]/T) for n in range(len(e))])
    for n in range(len(e)):
        so += e[n]/T * np.exp(-e[n]/T) + np.log(z)* np.exp(-e[n]/T)
    return so/z
    
