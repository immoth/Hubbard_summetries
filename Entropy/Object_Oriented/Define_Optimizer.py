
from Define_Pauli_Measure import measure_E,collect_circuits
import numpy as np
from Define_Ansatz import cir_A
from Define_Quantum_Device import analyze_energy


def enforce_bounds(S,Smin,Smax):
    if S < Smin:
        S = Smin
    if S > Smax:
        S = Smax
    return S

def SPSA_old(T,S0,phi0,k_max,hpauli,method = 'simple'):
    beta = 0.201
    A = 10
    a = 0.05
    gamma = 0.101
    c = 0.4
    np.random.seed(10)
    
    #Initalization
    k = 0
    Nt = len(phi0)
    N = len(phi0[0])
    max_S = 0.999*N*np.log(2)
    phi_k = np.array(phi0)
    S_k = S0
    E_l = []
    k_l = []
    
    #Begin Iterations
    for k in range(k_max):
        #Update c and a
        a_k = a/((A + k + 1)**beta)
        c_k = c/((k + 1)**gamma)

        #Find Delta
        Delta_k = np.array(phi_k)
        for t in range(Nt):
            for q in range(N):
                Delta_k[t][q] = 1 - 2*np.random.binomial(size=None, n=1, p= 0.5)
        phi_k_A = phi_k + c_k*Delta_k
        phi_k_B = phi_k - c_k*Delta_k
        
        #Find Delta S
        Delta_Sk =  1 - 2*np.random.binomial(size=None, n=1, p= 0.5)
        S_k_A = S_k + c_k*Delta_Sk
        S_k_B = S_k - c_k*Delta_Sk
        S_k_A = enforce_bounds(S_k_A, 0, max_S)
        S_k_B = enforce_bounds(S_k_B, 0, max_S)
        
        #Find cir 
        cirA = cir_A(S_k_A,phi_k_A)
        cirB = cir_A(S_k_B,phi_k_B)
            
        #Find energy expectation value
        e_A = measure_E(hpauli, cirA, method = method)
        e_B = measure_E(hpauli, cirB, method = method)
        
        
        #Find Free Energy    
        E_A = e_A - S_k_A*T
        E_B = e_B - S_k_B*T
        
        #Calculate gradiant
        g = np.real((E_A-E_B)/(2*c_k)) 
        
        #Update phi
        g_k = g * Delta_k
        phi_k = phi_k - a_k * g_k
        
        #Update S
        g_Sk = g * Delta_Sk
        S_k = S_k - a_k * g_Sk
        S_k = enforce_bounds(S_k, 0, max_S)
        
        #Find new cir
        cir_n = cir_A(S_k,phi_k)
        
        #Find new energy expectation value
        e_n = measure_E(hpauli, cir_n, method = method)
        
        #Find new Free Energy
        E = e_n - S_k*T
        
        print(k,S_k,E)
        E_l.append(E)
        k_l.append(k)

    return E_l,k_l,S_k,phi_k

def SPSA(T,S0,phi0,k_max,hpauli,qd,method = 'matrix'):
    beta = 0.201
    A = 10
    a = 0.05
    gamma = 0.101
    c = 0.4
    np.random.seed(10)
    
    #Initalization
    k = 0
    Nt = len(phi0)
    N = len(phi0[0])
    max_S = 0.999*N*np.log(2)
    phi_k = np.array(phi0)
    S_k = S0
    E_l = []
    k_l = []
    
    #Begin Iterations
    for k in range(k_max):
        #Update c and a
        a_k = a/((A + k + 1)**beta)
        c_k = c/((k + 1)**gamma)

        #Find Delta
        Delta_k = np.array(phi_k)
        for t in range(Nt):
            for q in range(N):
                Delta_k[t][q] = 1 - 2*np.random.binomial(size=None, n=1, p= 0.5)
        phi_k_A = phi_k + c_k*Delta_k
        phi_k_B = phi_k - c_k*Delta_k
        
        #Find Delta S
        Delta_Sk =  1 - 2*np.random.binomial(size=None, n=1, p= 0.5)
        S_k_A = S_k + c_k*Delta_Sk
        S_k_B = S_k - c_k*Delta_Sk
        S_k_A = enforce_bounds(S_k_A, 0, max_S)
        S_k_B = enforce_bounds(S_k_B, 0, max_S)
        
        #Find circs 
        cirA = cir_A(S_k_A,phi_k_A)
        cirB = cir_A(S_k_B,phi_k_B)
        circs_A,labels_A = collect_circuits(hpauli,cirA)
        circs_B,labels_B = collect_circuits(hpauli,cirB)
        
        #Find Results
        results_A = qd.get_results(circs_A, method = method)
        results_B = qd.get_results(circs_B, method = method)
            
        #Find energy expectation value
        e_A = analyze_energy(labels_A,results_A)
        e_B = analyze_energy(labels_B,results_B)
        
        
        #Find Free Energy    
        E_A = e_A - S_k_A*T
        E_B = e_B - S_k_B*T
        
        #Calculate gradiant
        g = np.real((E_A-E_B)/(2*c_k)) 
        
        #Update phi
        g_k = g * Delta_k
        phi_k = phi_k - a_k * g_k
        
        #Update S
        g_Sk = g * Delta_Sk
        S_k = S_k - a_k * g_Sk
        S_k = enforce_bounds(S_k, 0, max_S)
        
        #Find new circs
        cir_n = cir_A(S_k,phi_k)
        circs_n,labels_n = collect_circuits(hpauli,cir_n)
        
        #Find New Results
        results_n = qd.get_results(circs_n, method = method)
        
        #Find new energy expectation value
        e_n = analyze_energy(labels_n,results_n)
        
        #Find new Free Energy
        E = e_n - S_k*T
        
        print(k,S_k,E)
        E_l.append(E)
        k_l.append(k)

    return E_l,k_l,S_k,phi_k
