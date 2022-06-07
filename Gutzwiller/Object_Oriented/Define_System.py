
from Define_Paulis import I,X,Y,Z, cd,c,n, Mdot, bkt
import scipy.linalg as ln
import numpy as np
import matplotlib.pyplot as plt


class system:
    lattice_types = ['square','triangle','line']
    
    def __init__(self,lattice,sites):
        self.lattice = lattice
        self.N = sites
        if self.lattice not in self.lattice_types:
            raise Exception('lattice must be on of: ' + str(self.lattice_types))
        
        if self.lattice == 'square':
            if self.N != 4:
                raise Exception('I have only defined the square lattices for 4 sites')
        
        if self.lattice == 'triangle':
            if self.N != 4:
                raise Exception('I have only defined the triangle lattices for 4 sites')
            
    def K(self,k):
        if self.lattice == 'square':
            N=4
            Kout = 0*I(2*N)
            for i in range(0,N-1):
                Kout = Kout + Mdot([cd(i,2*N),c(i+1,2*N)]) + Mdot([cd(i+1,2*N),c(i,2*N)])
                Kout = Kout + Mdot([cd(i+N,2*N),c(i+1+N,2*N)]) + Mdot([cd(i+1+N,2*N),c(i+N,2*N)])
            Kout = Kout + Mdot([cd(0,2*N),c(N-1,2*N)]) + Mdot([cd(N-1,2*N),c(0,2*N)])
            Kout = Kout + Mdot([cd(0+N,2*N),c(N-1+N,2*N)]) + Mdot([cd(N-1+N,2*N),c(0+N,2*N)])
            return k*Kout
        if self.lattice == 'triangle':
            Kout = 0*I(8)
            Kout = Kout + Mdot([cd(0,8),c(1,8)]) + Mdot([cd(0,8),c(2,8)])+ Mdot([cd(0,8),c(3,8)])
            Kout = Kout + Mdot([cd(1,8),c(0,8)]) + Mdot([cd(1,8),c(2,8)])+ Mdot([cd(1,8),c(3,8)])
            Kout = Kout + Mdot([cd(2,8),c(0,8)]) + Mdot([cd(2,8),c(1,8)])+ Mdot([cd(2,8),c(3,8)])
            Kout = Kout + Mdot([cd(3,8),c(0,8)]) + Mdot([cd(3,8),c(1,8)])+ Mdot([cd(3,8),c(2,8)])
            N=4
            Kout = Kout + Mdot([cd(N+0,8),c(N+1,8)]) + Mdot([cd(N+0,8),c(N+2,8)])+ Mdot([cd(N+0,8),c(N+3,8)])
            Kout = Kout + Mdot([cd(N+1,8),c(N+0,8)]) + Mdot([cd(N+1,8),c(N+2,8)])+ Mdot([cd(N+1,8),c(N+3,8)])
            Kout = Kout + Mdot([cd(N+2,8),c(N+0,8)]) + Mdot([cd(N+2,8),c(N+1,8)])+ Mdot([cd(N+2,8),c(N+3,8)])
            Kout = Kout + Mdot([cd(N+3,8),c(N+0,8)]) + Mdot([cd(N+3,8),c(N+1,8)])+ Mdot([cd(N+3,8),c(N+2,8)])
            return k*Kout
        if self.lattice == 'line':
            N=self.N
            Kout = 0*I(2*N)
            for i in range(0,N-1):
                Kout = Kout + Mdot([cd(i,2*N),c(i+1,2*N)]) + Mdot([cd(i+1,2*N),c(i,2*N)])
                Kout = Kout + Mdot([cd(i+N,2*N),c(i+1+N,2*N)]) + Mdot([cd(i+1+N,2*N),c(i+N,2*N)])
            Kout = Kout + Mdot([cd(0,2*N),c(N-1,2*N)]) + Mdot([cd(N-1,2*N),c(0,2*N)])
            Kout = Kout + Mdot([cd(0+N,2*N),c(N-1+N,2*N)]) + Mdot([cd(N-1+N,2*N),c(0+N,2*N)])
            return k*Kout
        else:
            return None
    
    def D(self,d):
        N = self.N
        Dout = 0*I(2*N)
        for i in range(0,N):
            Dout = Dout + Mdot([n(i,2*N),n(i+N,2*N)])
        return d*Dout

    def M(self,u):
        N = self.N
        Dout = 0*I(2*N)
        for i in range(0,N):
            Dout = Dout + n(i,2*N) 
            Dout = Dout + n(i+N,2*N)
        return u*Dout
    
    def H(self,k,u,d):
        return self.K(k) + self.D(d) + self.M(u)
    
    def G(self,g):
        N = self.N
        out = I(2*N)
        for i in range(N):
            out = Mdot([ out , I(2*N) + (np.exp(-g)-1)*Mdot([n(i,2*N),n(i+N,2*N)]) ])
        return out
    
    def K_single(self, k):
        if self.lattice == 'square':
            N = 4
            h = [[0 for i in range(N)] for ii in range(N)]
            for i in range(0,N-1):
                h[i][i+1] = -k
                h[i+1][i] = -k 
            h[N-1][0] = -k
            h[0][N-1] = -k
            return h
        if self.lattice == 'triangle':
            h = [[0 for i in range(4)] for ii in range(4)]
            h[0][1] = -k; h[0][2] = -k; h[0][3] = -k;
            h[1][0] = -k; h[1][2] = -k; h[1][3] = -k;
            h[2][0] = -k; h[2][1] = -k; h[2][3] = -k;
            h[3][0] = -k; h[3][1] = -k; h[3][2] = -k;
            return h
        if self.lattice == 'line':
            N = self.N
            h = [[0 for i in range(N)] for ii in range(N)]
            for i in range(0,N-1):
                h[i][i+1] = -k
                h[i+1][i] = -k 
            h[N-1][0] = -k
            h[0][N-1] = -k
            return h
        
    def Fl(self):
        e,y = ln.eigh(self.K_single(1))
        return np.transpose(y)

    def Fld(self):
        e,y = ln.eigh(self.K_single(1))
        return np.conjugate(y)

    def ad(self,n):
        F = self.Fl()
        N = len(F)
        Fd = np.conjugate(np.transpose(F))
        out = Fd[0][n]*cd(0,N)
        for i in range(1,N):
            out = out + Fd[i][n]*cd(i,N)
        return out

    def a(self,n):
        F = self.Fl()
        N = len(F)
        Fd = np.conjugate(np.transpose(F))
        out = F[n][0]*c(0,N)
        for i in range(1,N):
            out = out + F[n][i]*c(i,N)
        return out
    
    def psi0(self):
        N = self.N
        y = [0 for i in range(2**N)]
        y[0] = 1
        return y

    def psi1(self,n_list):
        psi1 = self.psi0()
        for n in n_list:
            ad = self.ad(n)
            psi1 = Mdot([ad,psi1])
        return psi1
    
    def psi_spin(self,n_list):
        psi1 = self.psi1(n_list)
        return np.kron(psi1,psi1)
    
    def pauli_strings(self):
        if self.lattice == 'square':
            return ['ZZZZ','XXII','YYII','IXXI','IYYI','IIXX','IIYY','XZZX','YZZY']
        if self.lattice == 'triangle':
            return ['ZZZZ','XXII','YYII','XZXI','YZYI','XZZX','YZZY','IXXI','IYYI','IXZX','IYZY','IIXX','IIYY']
        if self.lattice == 'line':
            N = self.N
            P0 = ''
            for i in range(N):
                P0 = P0 + 'Z'
            P_list = [P0]
            for i in range(N-1):
                PX = ''
                PY = ''
                for j in range(N):
                    if j == i or j == i+1:
                        PX = PX + "X"
                        PY = PY + "Y"
                    else:
                        PX = PX + "I"
                        PY = PY + "I"
                P_list.append(PX)
                P_list.append(PY)
            PX = 'X'
            PY = 'Y'
            for j in range(N-2):
                PX = PX + "Z"
                PY = PY + "Z"
            PX = PX + "X"
            PY = PY + "Y"
            P_list.append(PX)
            P_list.append(PY)
            return P_list
        return None
            
            
    

    def draw(self):
        if self.lattice == 'square':
            circles_y = [0,1,1,0]
            circles_x = [0,0,1,1]
            lines_y = [0,1,1,0,0]
            lines_x = [0,0,1,1,0]
            plt.scatter(circles_x,circles_y,s=500)
            plt.plot(lines_x,lines_y)
            plt.ylim(-1,2)
            plt.xlim(-1,2)
        if self.lattice == 'triangle':
            circles_y = [1,2,1,0]
            circles_x = [0,1,2,1]
            lines_y = [1,2,1,0,1,1,2,0]
            lines_x = [0,1,2,1,0,2,1,1]
            plt.scatter(circles_x,circles_y,s=500)
            plt.plot(lines_x,lines_y)
            plt.ylim(-1,3)
            plt.xlim(-1,3)
        if self.lattice == 'line':
            N = self.N
            circles_y = [0 for i in range(int(np.floor(N/2)))] + [1 for i in range(int(np.ceil(N/2)))]
            circles_x = [i for i in range(int(np.floor(N/2)))] + [int(np.floor(N/2))-1 - i for i in range(int(np.ceil(N/2)))]
            lines_y = [0 for i in range(int(np.floor(N/2)))] + [1 for i in range(int(np.ceil(N/2)))] + [0]
            lines_x = [i for i in range(int(np.floor(N/2)))] + [int(np.floor(N/2))-1 - i for i in range(int(np.ceil(N/2)))] + [0]
            plt.scatter(circles_x,circles_y,s=500)
            plt.plot(lines_x,lines_y)
            plt.ylim(-1,2)
            plt.xlim(-2,int(np.ceil(N/2)))
        return plt.show()

   
            
