
from qiskit import IBMQ,transpile
from qiskit.tools.monitor import job_monitor
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-afrl', group='air-force-lab', project='quantum-sim')
from qiskit import Aer
import copy
from qiskit import quantum_info as qi
from qiskit import execute
from Define_Translator import translate, measure_rho
import numpy as np
#print('finished import')

class Quantum_Device:
    
    def __init__(self, backend = 'ibm_lagos', layout = [0,1,3,5]):
        self.backend = provider.get_backend(backend)
        self.layout = layout

    #Returns the results of measuring circs_in
    #The method can be:
    #    'matrix' which computes the results classicaly using matrix multiplication
    #    'simulator' which computes the results using the qasm simulator
    #    'quantum' which computers the results on the phasical backend using the qubits defined in the layout
    def get_results(self,circs_in, method = 'matrix',save_id_file = None):
        circs = copy.deepcopy(circs_in)
        #Exact Calculation
        if method == 'matrix':
            # add results
            r = []
            for l in range(len(circs)):
                rho = translate(circs[l])
                r_l = measure_rho(rho)
                r.append(r_l)
        #Qiskit Simulation        
        if method == "simulator":
            sim = Aer.get_backend("qasm_simulator")
            r = execute(circs, backend = sim).result().get_counts() 
        #Device
        if method == "quantum":
            backend = self.backend
            layout = self.layout
            job = backend.run(transpile(circs, backend,initial_layout = layout), meas_level=2, shots=8192)
            job_id = job.job_id()
            print(job_id)
            np.save(save_id_file,[job_id])
            r = job.result().get_counts()
        return r
    
    #Normalizes the results
    def normalize_results(self,results):
        results_out = []
        for result in results:
            result_out = {}
            states = list(result.keys())
            total = 0
            for state in states:
                total += result[state]
            for state in states:
                result_out[state] = result[state]/total
            results_out.append(result_out)
        return results_out
    
    #Removes states from the results with close to zero wieght
    #This can greatly imporve runtime 
    def chop_results(self,results):
        results_chop = []
        for result in results:
            result_chop = {}
            states = list(result.keys())
            for state in states:
                if result[state] > 10**(-5):
                    result_chop[state] = result[state]
            results_chop.append(result_chop)
        return results_chop

def analyze_pauli(label,result):
    r = result
    pauli_qs = label
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

def analyze_energy(labels,results):
    e = 0
    for p in range(len(labels)):
        e += labels[p][0] * analyze_pauli(labels[p][1],results[p])
    return e
