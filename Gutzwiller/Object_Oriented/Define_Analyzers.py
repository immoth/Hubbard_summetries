
import numpy as np

def analyze_GG(g,result, post_select = True):
    r_keys = list(result.keys())
    gg = 0
    for su in r_keys:
        for sd in r_keys:
            wu = result[su]
            wd = result[sd]
            gg_sr = 1
            nu_total = 0
            nd_total = 0
            for i in range(len(su)):
                nu = int(su[i])
                nd = int(sd[i])
                nu_total += nu
                nd_total += nd
                gg_sr = gg_sr*np.exp(-2*g*nu*nd)
            if post_select == False:    
                gg = gg + wu*wd*gg_sr
            if post_select == True:
                if nu_total == 2 and nd_total == 2:
                    gg = gg + wu*wd*gg_sr
    return gg

def analyze_GDG(g,d,result, post_select = True):
    r_keys = list(result.keys())
    gg = 0
    for su in r_keys:
        for sd in r_keys:
            wu = result[su]
            wd = result[sd]
            nu_total = 0
            nd_total = 0
            gg_sr = 1
            d_sr = 0
            for i in range(len(su)):
                nu = int(su[i])
                nd = int(sd[i])
                nu_total += nu
                nd_total += nd
                gg_sr = gg_sr*np.exp(-2*g*nu*nd)
                d_sr = d_sr + d*nu*nd
            if post_select == False:    
                gg = gg + wu*wd*gg_sr*d_sr
            if post_select == True:
                if nu_total == 2 and nd_total == 2:
                    gg = gg + wu*wd*gg_sr*d_sr
    return gg

def analyze_GMG(g,m,result, post_select = True):
    r_keys = list(result.keys())
    gg = 0
    for su in r_keys:
        for sd in r_keys:
            wu = result[su]
            wd = result[sd]
            nu_total = 0
            nd_total = 0
            gg_sr = 1
            m_sr = 0
            for i in range(len(su)):
                nu = int(su[i])
                nd = int(sd[i])
                nu_total += nu
                nd_total += nd
                gg_sr = gg_sr*np.exp(-2*g*nu*nd)
                m_sr = m_sr + m*nu + m*nd
            if post_select == False:    
                gg = gg + wu*wd*gg_sr*m_sr
            if post_select == True:
                if nu_total == 2 and nd_total == 2:
                    gg = gg + wu*wd*gg_sr*m_sr
    return gg

def find_sset(pauli):
    sset = [i for i in range(len(pauli))] #Swapped set
    fset = []
    hset = []
    N=len(pauli)
    for i in range(2):
        if pauli[i] == 'Z' or pauli[i] == 'I':
            fset.append(i)
    for i in range(2,len(pauli)):
        if pauli[i] == 'X' or pauli[i] == 'Y':
                hset.append(i)
    for i in range(len(fset)):
        sset[hset[i]] = fset[i]
        sset[fset[i]] = hset[i]
    return sset

def analyze_GPG(g,pauli,paulis,results, post_select = True):
    N = len(pauli)
    sset = find_sset(pauli)
    idx = paulis.index(pauli)
    resultd = results[idx]
    resultu = results[0]
    ru_keys = list(resultu.keys())
    rd_keys = list(resultd.keys())
    gg = 0
    for su in ru_keys:
        for sd in rd_keys:
            wu = resultu[su]
            wd = resultd[sd]
            nu_total = 0
            nd_total = 0
            gg_sr = 1
            z0 = 1
            z1 = 1
            if sd[N-1] == '1':
                z0 = -1
            if sd[N-2] == '1':
                z1 = -1
            k_sr = z0 - z1
            for i in range(len(su)):
                nu = int(su[N-1-sset[i] ])
                nd = int(sd[N-1-i])
                nu_total += nu
                nd_total += nd
                if i == 0 or i == 1:
                    gg_sr = gg_sr*np.exp(-g*nu)
                else:
                    gg_sr = gg_sr*np.exp(-2*g*nu*nd) 
            if post_select == False:    
                gg = gg + wu*wd*gg_sr*k_sr
            if post_select == True:
                if nu_total == 2 and nd_total == 2:
                    gg = gg + wu*wd*gg_sr*k_sr
    return gg

def analyze_GKG(g,k,paulis,results, post_select = True):
    out = 0
    for p in range(1,len(paulis),2):
        pauli = paulis[p]
        out += k/2*analyze_GPG(g,pauli,paulis,results, post_select = post_select) #for spin up
        out += k/2*analyze_GPG(g,pauli,paulis,results, post_select = post_select) #for spin down
    return out


def analyze_energy(g,u,k,d,paulis,results,post_select = True):
    num = analyze_GMG(g,u,results[0],post_select) +analyze_GDG(g,d,results[0],post_select) + analyze_GKG(g,k,paulis,results,post_select)
    dom = analyze_GG(g,results[0],post_select)
    return num/dom
