
import numpy as np

#Generates the set of indecies after the Pauli X and Y terms have been taken to sites 0 and 1
#fset holds the indeces 0 or 1 which do not have X or Y in the original Pauli string
#hset holds the indeces > 1 which have X or Y in the original Pauli sting
#The new pauli string will have indces in fset swapped with those in hset
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

#Finds the total number of particles in a state
#This is used for post selection
def n_total(state):
    n_total = 0
    for i in range(len(state)):
        n_total += int(state[i])
    return n_total

# Returns the expectation value of <G2(g)G2(g)> using the result measured in the z-basis 
def analyze_GG2(g,nbrs,result, post_select = True):
    r_keys = list(result.keys())
    gg = 0
    for su in r_keys:
        for sd in r_keys:
            wu = result[su]
            wd = result[sd]
            gg_sr = 1
            N = len(su)
            for pair in nbrs:
                i = N-1-pair[0]
                j = N-1-pair[1]
                nui = int(su[i])
                ndi = int(sd[i])
                ni = nui + ndi
                nuj = int(su[j])
                ndj = int(sd[j])
                nj = nuj + ndj
                gg_sr = gg_sr*np.exp(-2*g*ni*nj)
            if post_select == False:    
                gg = gg + wu*wd*gg_sr
            if post_select == True:
                if n_total(su) == 2 and n_total(sd) == 2:
                    gg = gg + wu*wd*gg_sr
    return gg

# Returns the expectation value of <G2(g)D(d)G2(g)> using the result measured in the z-basis 
def analyze_GDG2(g,d,nbrs,result, post_select = True):
    r_keys = list(result.keys())
    gg = 0
    for su in r_keys:
        for sd in r_keys:
            wu = result[su]
            wd = result[sd]
            gg_sr = 1
            d_sr = 0
            for i in range(len(su)):
                nu = int(su[i])
                nd = int(sd[i])
                d_sr = d_sr + d*nu*nd
            N = len(su)
            for pair in nbrs:
                i = N-1-pair[0]
                j = N-1-pair[1]
                nui = int(su[i])
                ndi = int(sd[i])
                ni = nui + ndi
                nuj = int(su[j])
                ndj = int(sd[j])
                nj = nuj + ndj
                gg_sr = gg_sr*np.exp(-2*g*ni*nj)
            if post_select == False:    
                gg = gg + wu*wd*gg_sr*d_sr
            if post_select == True:
                if n_total(su) == 2 and n_total(sd) == 2:
                    gg = gg + wu*wd*gg_sr*d_sr
    return gg

# Returns the expectation value of G2(g)M(m)G2(g) using the result measured in the z-basis 
def analyze_GMG2(g,m,nbrs,result, post_select = True):
    r_keys = list(result.keys())
    gg = 0
    for su in r_keys:
        for sd in r_keys:
            wu = result[su]
            wd = result[sd]
            gg_sr = 1
            m_sr = 0
            for i in range(len(su)):
                nu = int(su[i])
                nd = int(sd[i])
                m_sr = m_sr + m*nu + m*nd
            N = len(su)
            for pair in nbrs:
                i = N-1-pair[0]
                j = N-1-pair[1]
                nui = int(su[i])
                ndi = int(sd[i])
                ni = nui + ndi
                nuj = int(su[j])
                ndj = int(sd[j])
                nj = nuj + ndj
                gg_sr = gg_sr*np.exp(-2*g*ni*nj)
            if post_select == False:    
                gg = gg + wu*wd*gg_sr*m_sr
            if post_select == True:
                if n_total(su) == 2 and n_total(sd) == 2:
                    gg = gg + wu*wd*gg_sr*m_sr
    return gg

# This returns the results for <Gb> where G2 P G2 = P Gb 
# This is not used in the final construction but can be useful for debugging
def analyze_GG2b(g,nbrs,pauli,paulis,results, post_select = True):
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
            gg_sr = 1
            for pair in nbrs:
                ui = N-1-pair[0]
                di = N-1-sset[pair[0]]
                uj = N-1-pair[1]
                dj = N-1-sset[pair[1]]
                nui = int(su[ui])
                ndi = int(sd[di])
                ni = nui + ndi
                nuj = int(su[uj])
                ndj = int(sd[dj])
                nj = nuj + ndj
                if di == 0 and dj == 1:
                    gg_sr = gg_sr*np.exp(-g*(nui+nuj)**2) 
                elif dj == 0 and di == 1:
                    gg_sr = gg_sr*np.exp(-g*(nui+nuj)**2)
                elif di == 0 or di == 1:
                    gg_sr = gg_sr*np.exp(-g*(1+2*nui)*nj)
                elif dj == 0 or dj == 1:
                    gg_sr = gg_sr*np.exp(-g*ni*(1+2*nuj))
                else:
                    gg_sr = gg_sr*np.exp(-2*g*ni*nj)
            if post_select == False:    
                gg = gg + wu*wd*gg_sr
            if post_select == True:
                if n_total(su) == 2 and n_total(sd) == 2:
                    gg = gg + wu*wd*gg_sr
    return gg

# Returns the expectation value of G2(g)P G2(g) where P is a pauli string using the results for P
def analyze_GPG2(g,nbrs,pauli,paulis,results, post_select = True):
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
            gg_sr = 1
            z0 = 1
            z1 = 1
            if sd[N-1] == '1':
                z0 = -1
            if sd[N-2] == '1':
                z1 = -1
            k_sr = z0 - z1
            N  = len(su)
            for pair in nbrs:
                ui = N-1-pair[0]
                di = N-1-sset[pair[0]]
                uj = N-1-pair[1]
                dj = N-1-sset[pair[1]]
                nui = int(su[ui])
                ndi = int(sd[di])
                ni = nui + ndi
                nuj = int(su[uj])
                ndj = int(sd[dj])
                nj = nuj + ndj
                if di == N-1 and dj == N-2:
                    gg_sr = gg_sr*np.exp(-g*(nui+nuj)**2) 
                elif dj == N-1 and di == N-2:
                    gg_sr = gg_sr*np.exp(-g*(nui+nuj)**2)
                elif di == N-1 or di == N-2:
                    gg_sr = gg_sr*np.exp(-g*(1+2*nui)*nj)
                elif dj == N-1 or dj == N-2:
                    gg_sr = gg_sr*np.exp(-g*ni*(1+2*nuj))
                else:
                    gg_sr = gg_sr*np.exp(-2*g*ni*nj)
            if post_select == False:    
                gg = gg + wu*wd*gg_sr*k_sr
            if post_select == True:
                if n_total(su) == 2 and n_total(sd) == 2:
                    gg = gg + wu*wd*gg_sr*k_sr 
    return gg

# Returns the expection value of <G(g)K(k)G(g)> using the results for each pauli string 
def analyze_GKG2(g,k,nbrs,paulis,results, post_select = True):
    out = 0
    for p in range(1,len(paulis),2):
        pauli = paulis[p]
        out += k/2*analyze_GPG2(g,nbrs,pauli,paulis,results, post_select = post_select) #for spin up
        out += k/2*analyze_GPG2(g,nbrs,pauli,paulis,results, post_select = post_select) #for spin down
    return out
