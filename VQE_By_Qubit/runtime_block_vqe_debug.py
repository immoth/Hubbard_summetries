
####### Circuits ########
from qiskit import QuantumCircuit, QuantumRegister,ClassicalRegister, execute
from qiskit import transpile

##on-block##

def Ui(cir_i,phi_i):  
    Q = len(phi_i)
    for q in range(0,Q):
        cir_i.ry(phi_i[q],q)
    for q in range(0,Q,4):
        cir_i.cx(q,q+1)
        if q + 3 < Q:
            cir_i.cx(q+3,q+2)
    for q in range(1,Q,4):
        cir_i.cx(q,q+1)
        if q + 3 < Q:
            cir_i.cx(q+3,q+2)
    
    return cir_i

def U(cir,phi_b):
    T = len(phi_b)
    for t in range(T):
        cir = Ui(cir,phi_b[t])
    return cir

##of-block##
def Ui_off(cir_i,phi_a, phi_b):  
    Q = len(phi_a)
    for q in range(0,Q):
        cir_i.ry(phi_a[q],q)
        cir_i.cx(Q,q)
        cir_i.ry(-(phi_b[q]-phi_a[q])/2,q)
        cir_i.cx(Q,q)
        cir_i.ry((phi_b[q]-phi_a[q])/2,q)
    for q in range(0,Q,4):
        cir_i.cx(q,q+1)
        if q + 3 < Q:
            cir_i.cx(q+3,q+2)
    for q in range(1,Q,4):
        cir_i.cx(q,q+1)
        if q + 3 < Q:
            cir_i.cx(q+3,q+2)
    
    return cir_i

def U_off(cir,phi_a,phi_b):
    T = len(phi_b)
    Q = len(phi_b[0])
    cir.h(Q)
    for t in range(T):
        cir = Ui_off(cir,phi_a[t],phi_b[t])
    return cir



####### Calculating Energy ########
from qiskit import Aer
from qiskit.visualization import *
from qiskit import quantum_info as qi


##measure paulis##
def measure_pauli(backend,p_label,psi0,method = 'simple'):
    
    #Simple method of calculation####
    if method == 'simple':
        wave0 = qi.Statevector.from_instruction(psi0)
        Op = qi.Operator.from_label(p_label)
        return np.dot(np.conjugate(wave0),np.dot(Op,wave0))
    #################################
        
    #apply rotations#################
    psi = copy.deepcopy(psi0)
    pauli_qs = []
    Z_label = ''
    Q = len(p_label)
    for q,p in enumerate(p_label):
        if p == 'X':
            psi.ry(-np.pi/2,Q-1-q)
            pauli_qs.append(1)
            Z_label += 'Z'
        if p == 'Y':
            psi.rx(np.pi/2,Q-1-q)
            pauli_qs.append(1)
            Z_label += 'Z'
        if p == 'Z':
            pauli_qs.append(1)
            Z_label += 'Z'
        if p == 'I':
            pauli_qs.append(0)
            Z_label += 'I'
    ################################
    
    #Using matrix multiplication####
    if method == 'matrix':
        # add results
        wave = qi.Statevector.from_instruction(psi)
        r = wave.probabilities_dict()
        z_measure = 0
        for key in list(r.keys()):
            n = 0
            for q in range(len(key)):
                if key[q] == '1' and pauli_qs[q] == 1:
                    n += 1
            z_measure += (-1)**n * r[key] 
        return z_measure
    #################################
    
    #Using the qasm simulator########
    if method == "simulator":
        sim = Aer.get_backend("qasm_simulator")
        psi.measure(psi.qubits,psi.clbits)
        r = execute(psi, backend = sim).result().get_counts()
        z_measure = 0
        total = 0
        for key in list(r.keys()):
            n = 0
            for q in range(len(key)):
                if key[q] == '1' and pauli_qs[q] == 1:
                    n += 1
            z_measure += (-1)**n * r[key] 
            total += r[key]
        return z_measure/total
    ###################################
    
    #Using the quantum backend#########
    if method == "quantum":
        psi.measure(psi.qubits,psi.clbits)
        job = backend.run(transpile(psi, backend), meas_level=2, shots=8192) 
        #print("job id: ",job.job_id())
        #print(job_monitor(job))
        r = job.result().get_counts()
        z_measure = 0
        total = 0
        for key in list(r.keys()):
            n = 0
            for q in range(len(key)):
                if key[q] == '1' and pauli_qs[q] == 1:
                    n += 1
            z_measure += (-1)**n * r[key] 
            total += r[key]

        z_measure/total
        return z_measure/total
    ###################################
    
    raise NameError(method + ' is not a recognized method')
    return method + ' is not a recognized method'

##energy on block##
def E_on_block(backend,phi_b,block,method = 'simple'):
    E = 0
    Q = len(phi_b[0])
    qr = QuantumRegister(Q)
    cr = ClassicalRegister(Q)
    cir = QuantumCircuit(qr , cr)
    psi0 = U(cir,phi_b)
    for p in block:
        psi = copy.deepcopy(psi0)
        w = p.coeff
        p_label = p.primitive.to_label()
        E_p = measure_pauli(backend,p_label,psi,method = method)
        E += w*E_p
    return E

##energy off block##
def E_off_block(backend,phi_a,phi_b,block,method = 'simple'):
    E = 0
    Q = len(phi_b[0])
    qr = QuantumRegister(Q+1)
    cr = ClassicalRegister(Q+1)
    cir = QuantumCircuit(qr , cr)
    psi0 = U_off(cir,phi_a,phi_b)
    for p in block:
        psi = copy.deepcopy(psi0)
        w = p.coeff
        p_label = p.primitive.to_label()
        E_px = measure_pauli(backend,'X' + p_label,psi,method = method)
        E_py = measure_pauli(backend,'Y' + p_label,psi,method = method)
        E += w*(E_px + 1j*E_py)/2
    return E

##total energy##
def find_E(backend,user_messenger,alpha,phi,blocks,method = 'simple'):
    E = 0
    for key in list(blocks.keys()):
        if key[0] == key[2]:
            E += alpha[int(key[0])]*alpha[int(key[2])]*E_on_block(backend,phi[int(key[0])],blocks[key],method = method)
        else:
            E += 2*alpha[int(key[0])]*alpha[int(key[2])]*E_off_block(backend,phi[int(key[0])],phi[int(key[2])],blocks[key],method = method)
        user_messenger.publish({'key':key,'E':E})
    return E



####### Optimization ########
import copy
import numpy as np

beta = 0.201
A = 10
a = 0.05
gamma = 0.101
c = 0.4

beta_a = 0.201
A_a = 10
a_a = 0.05
gamma_a = 0.101
c_a = 0.4



def SPSA(backend, user_messenger, k_max, phi, alpha, blocks, method = 'simple',hold = False):
    #Initalization
    k = 0
    phi_k = np.array(phi)
    alpha_k = np.array(alpha)
    bL = len(phi_k)
    T = len(phi_k[0])
    Q = len(phi_k[0][0])
    E_l = []
    hold_l = []
    
    #Begin Iterations
    for k in range(k_max):
        #Update c and a
        a_k = a/((A + k + 1)**beta)
        c_k = c/((k + 1)**gamma)
        a_ak = a_a/((A_a + k + 1)**beta_a)
        c_ak = c_a/((k + 1)**gamma_a)

        #Find Delta
        Delta_k = np.array(phi_k)
        for b in range(bL):
            for t in range(T):
                for q in range(Q):
                    Delta_k[b][t][q] = 1 - 2*np.random.binomial(size=None, n=1, p=0.5)
        phi_k_A = phi_k + c_k*Delta_k
        phi_k_B = phi_k - c_k*Delta_k
        
        #Find Delta Alpha
        Delta_ak = np.array(alpha_k)
        for n in range(bL):
            Delta_ak[n] = 1 - 2*np.random.binomial(size=None, n=1, p= 0.5)
        alpha_k_A = alpha_k + c_ak*Delta_ak
        alpha_k_B = alpha_k - c_ak*Delta_ak
        norm_A = 1/np.sqrt(np.dot(alpha_k_A,alpha_k_A))
        norm_B = 1/np.sqrt(np.dot(alpha_k_B,alpha_k_B))
        alpha_k_A = norm_A*alpha_k_A
        alpha_k_B = norm_B*alpha_k_B
            
        #Find E    
        user_messenger.publish({'starting iteration':k})
        E_A = find_E(backend,alpha_k_A, phi_k_A, blocks, method = method)
        user_messenger.publish({'k':k,'E_A':E_A})
        E_B = find_E(backend,alpha_k_B, phi_k_B, blocks, method = method)
        user_messenger.publish({'k':k,'E_B':E_B})
        
        #Calculate gradiant
        g = np.real((E_A-E_B)/(2*c_k)) 
        
        #Update phi
        g_k = g * Delta_k
        phi_k = phi_k - a_k * g_k
        
        #Update alpha
        g_ak = g * Delta_ak
        alpha_k = alpha_k - a_ak * g_ak
        norm = 1/np.sqrt(np.dot(alpha_k,alpha_k))
        alpha_k = norm*alpha_k

        
        #Calculate new E
        E_f = np.real(find_E(backend,alpha_k,phi_k,blocks, method = method))
        
        #Print and save E
        #print('k=',k,'c_k=',c_k,'a_k=',a_k,'g=',g,'E_A=',E_A,'E_B=',E_B,'E_f=',E_f)
        user_messenger.publish({'k':k,'E_f':E_f})
        E_l.append(E_f)
        
        if hold == True:
            hold_k = {'E':E_f,'c':c_k,'a':a_k,'phi':phi_k,'Delta':Delta_k,'c_a':c_ak,'a_a':a_ak,'alpha':alpha_k,'Delta_a':Delta_ak}
            hold_l.append(hold_k)
    if hold == True:
        return hold_l
    else:
        return E_l,phi_k



####### Main ########
def main0(backend, user_messenger, **kwargs):
    k_max = kwargs.pop('k_max', 10)
    phi = kwargs.pop('phi')
    alpha = kwargs.pop('alpha')
    blocks = kwargs.pop('blocks')
    block = blocks['0,1']
    p_label = 'X' +block[0].primitive.to_label()
    Q = len(phi[0][0])
    Q = len(phi[0][0])
    qr = QuantumRegister(Q+1)
    cr = ClassicalRegister(Q+1)
    cir = QuantumCircuit(qr , cr)
    psi = U_off(cir,phi[0],phi[1])
    out = measure_pauli(backend,p_label,psi,method = 'quantum')
    #user_messenger.publish({"Starting program with k_max": k_max})
    #out = SPSA(backend, user_messenger, k_max, phi, alpha, blocks, method = 'quantum', hold = True)
    return out


####### Main ########
def main(backend, user_messenger, **kwargs):
    k_max = kwargs.pop('k_max', 10)
    phi = kwargs.pop('phi')
    alpha = kwargs.pop('alpha')
    blocks = kwargs.pop('blocks')
    block = blocks['0,1']
    #out = E_off_block(backend,phi[0],phi[1],block,method = 'quantum')
    out = find_E(backend,user_messenger,alpha,phi,blocks,method = 'quantum')
    #user_messenger.publish({"Starting program with k_max": k_max})
    #out = SPSA(backend, user_messenger, k_max, phi, alpha, blocks, method = 'quantum', hold = True)
    return out

