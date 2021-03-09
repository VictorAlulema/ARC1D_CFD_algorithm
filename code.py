import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import sparse


class ARC1D:
    def __init__(self, domain, Ncells,
                 BoundaryConditions,
                 InitialConditions,
                 gamma, Courant, tol,
                 k2, k4):
        """
        ARC1D for the subsonic channel flow
        Inputs:
        domain: [x_init,x_end]
        Ncells: number of cells
        Boundadry conditions: This is an input dictionary to construct the BoundConditions  vector

                BC =  {'rho':[rho1,rho2],
                       'u':[u1,u2],
                       'p':[p1,p2],
                       'e':[e1,e2]
                       }
        InitialConditions: This is the initial Q
        gamma: constant
        Courant: Courante number
        tol: stopping criteria based on the residuals

        Output:
            Final Q
            plots
        """
        plt.rc('font', family='serif')
        plt.rc('lines', linewidth=1.5)
        plt.rc('font', size=9)
        plt.rc('legend', **{'fontsize': 9})

        self.x_init = domain[0]
        self.x_end = domain[1]
        self.Ncells = Ncells
        self.tol = tol
        self.k2 = k2
        self.k4 = k4
        self.C = Courant
        self.N = int(self.Ncells - 1)
        self.x = ARC1D.grid(self)
        self.S = ARC1D.S_x(self)
        self.BC = BoundaryConditions  # This is an input dictionary
        SS = np.array([self.S[1:-1],
                       self.S[1:-1],
                       self.S[1:-1]]).flatten()
        self.Q = InitialConditions * SS
        self.gamma = gamma
        self.dS = ARC1D.S_derivative(self)
        self.operator = ARC1D.difference_operator(self)
        self.BCE = ARC1D.BoundConditions(self)
        """
        Execution routine
        """
        norm_residual = 1
        p = ARC1D.GetPressure(self, self.Q[:self.N],
                              self.Q[self.N:2 * self.N],
                              self.Q[2 * self.N:])
        a = ARC1D.GetSoundSpeed(self, p, self.Q[:self.N])
        E = ARC1D.Evector(self)

        self.fig = plt.figure(figsize = (9,3.5))
        self.ax = self.fig.add_subplot(131)
        self.ay = self.fig.add_subplot(132)
        self.az = self.fig.add_subplot(133)
        self.az.set_xlabel('Iterations')
        self.az.set_ylabel('Residual norm')
        self.fig.show()
        self.residuals = []
        self.norm_residuals = [1]
        counter = 1
        while norm_residual > tol:
            # 
            self.ax.cla()
            self.ax.set_xlabel('x [m]')
            self.ax.set_ylabel('Pressure [Pa]')
            self.ay.cla()
            self.ay.set_xlabel('x [m]')
            self.ay.set_ylabel('Mach')
            self.ax.ticklabel_format(style='sci',axis='y',scilimits=(0,1))
            #
            h = self.C * (self.x[1] - self.x[0]) / (abs(self.Q[self.N: 2 * self.N] / self.Q[:self.N]) + a)
            h = np.concatenate((h, h, h), axis=None)
            #h = np.mean(h)
            # Solve
            A = np.identity(3 * self.N) + h * self.operator @ ARC1D.FluxA(self) \
                - h * ARC1D.JacobianG(self) - h * ARC1D.FluxL(self, p, a)
            B = h * (- self.operator @ ARC1D.Evector(self) -
                     self.BCE + ARC1D.Fvector(self) +
                     ARC1D.D_2(self, p, a) + ARC1D.D_4(self, p, a))
            self.Q += np.linalg.solve(A, B)
            # Get values
            p = ARC1D.GetPressure(self, self.Q[:self.N], self.Q[self.N:2 * self.N], self.Q[2 * self.N:])
            a = ARC1D.GetSoundSpeed(self, p, self.Q[:self.N])
            Mach = self.Q[self.N: 2 * self.N] / self.Q[:self.N]/a
            residual = np.sqrt(sum(B ** 2))
            self.residuals.append(residual)
            first_residual =  self.residuals[0]
            norm_residual = residual / first_residual
            self.norm_residuals.append(norm_residual)
            self.ax.plot(self.x, p / self.S[1:-1], 'k-',linewidth = 0.9)
            self.ay.plot(self.x, Mach, 'b-',linewidth = 0.9)
            self.az.semilogy([counter - 1, counter],
                         [self.norm_residuals[counter-1],
                          self.norm_residuals[counter]],
                         'g-', linewidth = 0.9)
            self.fig.tight_layout()
            self.fig.canvas.draw()
            time.sleep(0.001)
            counter = counter + 1
        print('Convergence reached')
        plt.close('all')
        self.counter  =  counter
        
        
    def grid(self):
        """
        Construct the computational grid given the
        domain boundaries: xinit and xend
        """
        dx = (self.x_end - self.x_init) / self.Ncells
        x = np.linspace(self.x_init + dx,
                        self.x_end, self.N,
                        endpoint=False)
        return x

    def S_x(self):
        """
        Variable area as function of x: S(x) for the subsonic case
        """
        x = np.concatenate(([self.x_init], self.x, [self.x_end]))
        S = np.ones_like(x)
        for i, j in enumerate(x):
            if j < 5:
                S[i] = 1 + 1.5 * (1 - (j / 5)) ** 2
            elif j >= 5:
                S[i] = 1 + 0.5 * (1 - (j / 5)) ** 2
        return S

    def difference_operator(self):
        """
        Contruct the difference operator for the spatial derivative
        """
        center_diagonal = np.zeros((self.N, 1)).ravel()
        diagonal_left = -1 * np.ones((self.N, 1)).ravel()
        diagonal_right = 1 * np.ones((self.N, 1)).ravel()
        a = diagonal_left.shape[0]
        diagonals = [center_diagonal, diagonal_left, diagonal_right]
        A = sparse.diags(diagonals, [0, -1, 1], shape=(a, a)).toarray()
        A[0, 0] = 0
        A[0, 1] = 1
        A[-1, -1] = 0
        A[-1, -2] = -1
        A = 1 / (2 * (self.x[1] - self.x[0])) * A
        operator = np.block([[A, np.zeros((self.N, self.N)), np.zeros((self.N, self.N))],
                             [np.zeros((self.N, self.N)), A, np.zeros((self.N, self.N))],
                             [np.zeros((self.N, self.N)), np.zeros((self.N, self.N)), A]])
        return operator

    def BoundConditions(self):
        """
        Create the boundary conditions vector based on the given BoundaryConditions input dictionary
        """
        bcE = np.zeros(3 * self.N)
        bcE[0] = -self.BC['rho'][0] * self.BC['u'][0] / 2 / (self.x[1] - self.x[0]) * self.S[0]
        bcE[self.N - 1] = self.BC['rho'][-1] * self.BC['u'][-1] / 2 / (self.x[1] - self.x[0]) * self.S[-1]
        bcE[self.N] = - ((self.BC['rho'][0] * self.BC['u'][0] ** 2 +
                          self.BC['p'][0]) / 2 / (self.x[1] - self.x[0])) * self.S[0]
        bcE[2 * self.N - 1] = (self.BC['rho'][-1] * self.BC['u'][-1] ** 2 + self.BC['p'][-1]) / 2 / (
                    self.x[1] - self.x[0]) * self.S[-1]
        bcE[2 * self.N] = -(self.BC['u'][0] * (self.BC['e'][0] +
                                               self.BC['p'][0]) / 2 / (self.x[1] - self.x[0])) * self.S[0]
        bcE[3 * self.N - 1] = self.BC['u'][-1] * (self.BC['e'][-1] +
                                                  self.BC['p'][-1]) / 2 / (self.x[1] - self.x[0]) * self.S[-1]
        return bcE

    def S_derivative(self):
        """
        Compute numerically the first derivative of S(x)
        """
        center_diagonal = np.zeros((self.N, 1)).ravel()
        diagonal_left = -1 * np.ones((self.N, 1)).ravel()
        diagonal_right = 1 * np.ones((self.N, 1)).ravel()
        a = diagonal_left.shape[0]
        diagonals = [center_diagonal, diagonal_left, diagonal_right]
        A = sparse.diags(diagonals, [0, -1, 1], shape=(a, a)).toarray()
        A[0, 0] = 0
        A[0, 1] = 1
        A[-1, -1] = 0
        A[-1, -2] = -1
        A = 1 / (2 * (self.x[1] - self.x[0])) * A
        ds = np.zeros(self.N)
        ds[0], ds[-1] = - self.S[0] / 2 / (self.x[1] - self.x[0]), self.S[-1] / 2 / (self.x[1] - self.x[0])
        DS = A @ self.S[1:-1] + ds
        return DS

    def Evector(self):
        """
        Compute E given Q
        """
        P = ARC1D.GetPressure(self, self.Q[:self.N],
                              self.Q[self.N: 2 * self.N],
                              self.Q[2 * self.N:])
        E = np.empty(np.shape(self.Q))
        E[:self.N] = self.Q[self.N:2 * self.N]
        E[self.N:2 * self.N] = self.Q[self.N: 2 * self.N] ** 2 / self.Q[:self.N] + P
        E[2 * self.N:] = self.Q[self.N: 2 * self.N] / self.Q[:self.N] * (self.Q[2 * self.N:] + P)
        return E

    def Fvector(self):
        """
        Compute F given Q
        """
        P = ARC1D.GetPressure(self, self.Q[:self.N],
                              self.Q[self.N: 2 * self.N],
                              self.Q[2 * self.N:]) / self.S[1:-1]
        F = np.zeros(3 * self.N)
        F[self.N: 2 * self.N] = P * self.dS
        return F

    def FluxA(self):
        """
        Compute flux jacobian matrix of E
        """
        B_1 = np.zeros((self.N, self.N))
        B_2 = np.identity(self.N)
        B_3 = np.zeros((self.N, self.N))
        B_4 = 0.5 * (self.gamma - 1) * self.Q[self.N:2 * self.N] ** 2 / self.Q[:self.N] ** 2
        B_5 = (3 - self.gamma) * self.Q[self.N: 2 * self.N] / self.Q[:self.N]
        B_6 = np.identity(self.N) * (self.gamma - 1)
        B_7 = (self.gamma - 1) * self.Q[self.N: 2 * self.N] ** 3 / self.Q[:self.N] ** 3 \
            - self.gamma * self.Q[self.N: 2 * self.N] * self.Q[2 * self.N:] / self.Q[: self.N] ** 2
        B_8 = self.gamma * self.Q[2 * self.N:] / self.Q[:self.N] \
            - 1.5 * (self.gamma - 1) * self.Q[self.N: 2 * self.N] ** 2 / self.Q[: self.N] ** 2
        B_9 = self.gamma * self.Q[self.N: 2 * self.N] / self.Q[: self.N]
        A = np.block([[B_1, B_2, B_3],
                      [np.diag(B_4), np.diag(B_5), B_6],
                      [np.diag(B_7), np.diag(B_8), np.diag(B_9)]])

        return A

    def FluxL(self, P, a):
        """
        Compute flux Jacobian matrix of D
        """
        rho = np.pad(self.Q[:self.N], (1, 1), 'constant',
                     constant_values=(self.BC['rho'][0] * self.S[0],
                                      self.BC['rho'][-1] * self.S[-1])
                     )
        rhou = np.pad(self.Q[self.N: 2 * self.N], (1, 1), 'constant',
                      constant_values=(self.BC['rho'][0] * self.BC['u'][0] * self.S[0],
                                       self.BC['rho'][-1] * self.BC['u'][0] * self.S[-1])
                      )
        e = np.pad(self.Q[2 * self.N:], (1, 1), 'constant',
                   constant_values=(self.BC['e'][0] * self.S[0],
                                    self.BC['e'][-1] * self.S[-1])
                   )
        a = np.pad(a, (1, 1), 'constant',
                   constant_values=(ARC1D.GetSoundSpeed(self, self.BC['p'][0], self.BC['rho'][0]),
                                    ARC1D.GetSoundSpeed(self, self.BC['p'][-1], self.BC['rho'][-1]))
                   )
        dx = self.x[1] - self.x[0]
        C2 = ARC1D.Eps2(self, P) * (abs(rhou / rho) + a)
        C2_1 = 1 / 2 / dx * (C2[1:-1] + C2[2:])
        C2_2 = 1 / 2 / dx * (C2[1:-1] + C2[:-2])
        C2_3 = -1 / 2 / dx * (C2[2:] + 2 * C2[1:-1] + C2[:-2])
        matrix = np.diag(C2_3) + np.diag(C2_2[1:], -1) + np.diag(C2_1[:-1], 1)
        D2 = np.block([[matrix, np.zeros((self.N, self.N)),
                        np.zeros((self.N, self.N))],
                       [np.zeros((self.N, self.N)),
                        matrix, np.zeros((self.N, self.N))],
                       [np.zeros((self.N, self.N)),
                        np.zeros((self.N, self.N)), matrix]]
                      )
        C4 = ARC1D.Eps4(self, P) * (abs(rhou / rho) + a)
        C4_1 = -1 / 2 / dx * (C4[2:-2] + C4[3:-1])
        C4_2 = -1 / 2 / dx * (C4[2:-2] + C4[1:-3])
        C4_3 = 1 / 2 / dx * (4 * C4[2:-2] + C4[1:-3] + 3 * C4[3:-1])
        C4_4 = 1 / 2 / dx * (4 * C4[2:-2] + 3 * C4[1:-3] + C4[3:-1])
        C4_5 = -1 / 2 / dx * (6 * C4[2:-2] + 3 * C4[1:-3] + 3 * C4[3:-1])
        C4_5 = np.pad(C4_5, (1, 1), 'constant',
                      constant_values=(-1 / 2 / dx * (2 * C4[0] + 5 * C4[1] + 3 * C4[2]),
                                       -1 / 2 / dx * (2 * C4[-1] + 5 * C4[-2] + 3 * C4[-3])
                                       )
                      )
        C4_3 = np.insert(C4_3, 0, 1 / 2 / dx * (C4[0] + 4 * C4[1] + 3 * C4[2]))
        C4_1 = np.insert(C4_1, 0, -1 / 2 / dx * (C4[1] + C4[2]))
        C4_4 = np.insert(C4_4, self.N - 2, 1 / 2 / dx * (C4[-1] + 4 * C4[-2] + 3 * C4[-3]))
        C4_2 = np.insert(C4_2, self.N - 2, -1 / 2 / dx * (C4[-1] + C4[-2]))
        matrix = np.diag(C4_5) + np.diag(C4_3, 1) + np.diag(C4_1[:-1], 2) \
            + np.diag(C4_4, -1) + np.diag(C4_2[1:], -2)
        D4 = np.block([[matrix, np.zeros((self.N, self.N)), np.zeros((self.N, self.N))],
                       [np.zeros((self.N, self.N)), matrix, np.zeros((self.N, self.N))],
                       [np.zeros((self.N, self.N)), np.zeros((self.N, self.N)), matrix]])
        return D2 + D4

    def JacobianG(self):
        """
        Compute the jacobian of the source term given Q
        """
        A = (self.gamma - 1) * self.dS * self.Q[self.N: 2 * self.N] ** 2 / self.Q[:self.N] ** 2 / 2 / self.S[1:-1]
        B = -(self.gamma - 1) * self.dS * self.Q[self.N: 2 * self.N] / self.Q[:self.N] ** 2 / self.S[1:-1]
        G = np.block([[np.zeros((self.N, self.N)), np.zeros((self.N, self.N)), np.zeros((self.N, self.N))],
                      [np.diag(A), np.diag(B), np.diag(np.ones(self.N) * (self.gamma - 1) * self.dS)],
                      [np.zeros((self.N, self.N)), np.zeros((self.N, self.N)), np.zeros((self.N, self.N))]])
        return G

    # Below, dissipation terms
    def Eps2(self, P):
        """
        Activate second-dorder artificial dissipation
        """
        P = np.pad(P, (1, 1), 'constant',
                   constant_values=(self.BC['p'][0] * self.S[0],
                                    self.BC['p'][-1] * self.S[-1])
                   )
        Y = np.pad(abs((P[2:] - 2 * P[1:-1] + P[:-2]) / (P[2:] + 2 * P[1:-1] + P[:-2])),
                   (2, 2), 'constant', constant_values=(0, 0)
                   )
        return self.k2 * np.maximum.reduce([Y[2:], Y[1:-1], Y[:-2]])

    def Eps4(self, P):
        """
        Activate fourth-order artificial dissipation
        """
        return np.maximum(np.zeros(self.N + 2), self.k4 - ARC1D.Eps2(self, P))

    def D_2(self, P, a):
        """
        Second order artificial dissipation term.
        """
        rho = np.pad(self.Q[:self.N], (1, 1), 'constant',
                     constant_values=(self.BC['rho'][0] * self.S[0],
                                      self.BC['rho'][-1] * self.S[-1])
                     )
        rhou = np.pad(self.Q[self.N: 2 * self.N], (1, 1), 'constant',
                      constant_values=(self.BC['rho'][0] * self.BC['u'][0] * self.S[0],
                                       self.BC['rho'][-1] * self.BC['u'][-1] * self.S[-1])
                      )
        e = np.pad(self.Q[2 * self.N:], (1, 1), 'constant',
                   constant_values=(self.BC['e'][0] * self.S[0],
                                    self.BC['e'][-1] * self.S[-1])
                   )
        a = np.pad(a, (1, 1), 'constant',
                   constant_values=(ARC1D.GetSoundSpeed(self, self.BC['p'][0], self.BC['rho'][0]),
                                    ARC1D.GetSoundSpeed(self, self.BC['p'][-1], self.BC['rho'][-1]))
                   )
        C = ARC1D.Eps2(self, P) * (abs(rhou / rho) + a)
        dx = self.x[1] - self.x[0]
        C1 = C[1:-1] + C[2:]
        C2 = C[1:-1] + C[:-2]
        C3 = C[2:] + 2 * C[1:-1] + C[:-2]
        drho = 1 / 2 / dx * (C1 * rho[2:] + C2 * rho[:-2] - C3 * rho[1:-1])
        drhou = 1 / 2 / dx * (C1 * rhou[2:] + C2 * rhou[:-2] - C3 * rhou[1:-1])
        de = 1 / 2 / dx * (C1 * e[2:] + C2 * e[:-2] - C3 * e[1:-1])
        return np.array([drho, drhou, de]).flatten()

    def D_4(self, P, a):
        """
        Fourth order artificial dissipation term.
        """
        rho = np.pad(self.Q[:self.N], (1, 1), 'constant',
                     constant_values=(self.BC['rho'][0] * self.S[0],
                                      self.BC['rho'][-1] * self.S[-1])
                     )
        rhou = np.pad(self.Q[self.N: 2 * self.N], (1, 1), 'constant',
                      constant_values=(self.BC['rho'][0] * self.BC['u'][0] * self.S[0],
                                       self.BC['rho'][-1] * self.BC['u'][-1] * self.S[-1])
                      )
        e = np.pad(self.Q[2 * self.N:], (1, 1), 'constant',
                   constant_values=(self.BC['e'][0] * self.S[0],
                                    self.BC['e'][-1] * self.S[-1])
                   )
        a = np.pad(a, (1, 1), 'constant',
                   constant_values=(ARC1D.GetSoundSpeed(self, self.BC['p'][0], self.BC['rho'][0]),
                                    ARC1D.GetSoundSpeed(self, self.BC['p'][-1], self.BC['rho'][-1]))
                   )
        C = ARC1D.Eps4(self, P) * (abs(rhou / rho) + a)
        dx = self.x[1] - self.x[0]
        C1 = C[2:-2] + C[3:-1]
        C2 = C[2:-2] + C[1:-3]
        C3 = 4 * C[2:-2] + C[1:-3] + 3 * C[3:-1]
        C4 = 4 * C[2:-2] + 3 * C[1:-3] + C[3:-1]
        C5 = 6 * C[2:-2] + 3 * C[1:-3] + 3 * C[3:-1]
        drho = np.empty(self.N)
        drhou = np.empty(self.N)
        de = np.empty(self.N)
        drho[1:-1] = 1 / 2 / dx * (-C1 * rho[4:] - C2 * rho[:-4]
                                   + C3 * rho[3:-1] + C4 * rho[1:-3]
                                   - C5 * rho[2:-2])
        drhou[1:-1] = 1 / 2 / dx * (-C1 * rhou[4:] - C2 * rhou[:-4]
                                    + C3 * rhou[3:-1] + C4 * rhou[1:-3]
                                    - C5 * rhou[2:-2])
        de[1:-1] = 1 / 2 / dx * (-C1 * e[4:] - C2 * e[:-4] + C3 * e[3:-1]
                                 + C4 * e[1:-3] - C5 * e[2:-2])
        drho[0] = 1 / 2 / dx * ((C[0] + 2 * C[1] + C[2]) * rho[0]
                                - (2 * C[0] + 5 * C[1] + 3 * C[2]) * rho[1]
                                + (C[0] + 4 * C[1] + 3 * C[2]) * rho[2]
                                - (C[1] + C[2]) * rho[3])
        drhou[0] = 1 / 2 / dx * ((C[0] + 2 * C[1] + C[2]) * rhou[0]
                                 - (2 * C[0] + 5 * C[1] + 3 * C[2]) * rhou[1]
                                 + (C[0] + 4 * C[1] + 3 * C[2]) * rhou[2]
                                 - (C[1] + C[2]) * rhou[3])
        de[0] = 1 / 2 / dx * ((C[0] + 2 * C[1] + C[2]) * e[0]
                              - (2 * C[0] + 5 * C[1] + 3 * C[2]) * e[1]
                              + (C[0] + 4 * C[1] + 3 * C[2]) * e[2]
                              - (C[1] + C[2]) * e[3])
        drho[-1] = 1 / 2 / dx * ((C[-1] + 2 * C[-2] + C[-3]) * rho[-1]
                                 - (2 * C[-1] + 5 * C[-2] + 3 * C[-3]) * rho[-2]
                                 + (C[-1] + 4 * C[-2] + 3 * C[-3]) * rho[-3]
                                 - (C[-2] + C[-3]) * rho[-4])
        drhou[-1] = 1 / 2 / dx * ((C[-1] + 2 * C[-2] + C[-3]) * rhou[-1]
                                  - (2 * C[-1] + 5 * C[-2] + 3 * C[-3]) * rhou[-2]
                                  + (C[-1] + 4 * C[-2] + 3 * C[-3]) * rhou[-3]
                                  - (C[-2] + C[-3]) * rhou[-4])
        de[-1] = 1 / 2 / dx * ((C[-1] + 2 * C[-2] + C[-3]) * e[-1]
                               - (2 * C[-1] + 5 * C[-2] + 3 * C[-3]) * e[-2]
                               + (C[-1] + 4 * C[-2] + 3 * C[-3]) * e[-3]
                               - (C[-2] + C[-3]) * e[-4])
        return np.array([drho, drhou, de]).flatten()

    def GetSoundSpeed(self, P, rho):
        """
        Compute the sound speed
        """
        return np.sqrt(self.gamma * P / rho)

    def GetPressure(self, rho, rhou, e):
        """
        Compute the pressure
        """
        return (self.gamma - 1) * (e - (rhou ** 2 / 2 / rho))


def ARC1Dexceptions():
    if ZeroDivisionError:
        raise Exception('Warning: denominator is equal to zero')
    if ValueError:
        raise Exception('Warning: Invalid value encountered in sqrt')
    if RuntimeWarning:
        raise Exception('Warning: Invalid value encountered in sqrt')
