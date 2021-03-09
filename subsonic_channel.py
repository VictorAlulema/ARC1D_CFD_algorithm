from code import ARC1D
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('lines', linewidth=1.5)
plt.rc('font', size=9)
plt.rc('legend', **{'fontsize': 9})
        
# Load exact solution from previous assigment
exact_solution = np.genfromtxt('solution_exact.txt')
X = exact_solution[:,0]
u_exact = exact_solution[:,1]
M_exact = exact_solution[:,2]
a_exact = exact_solution[:,3]
rho_exact = exact_solution[:,4]
p_exact = exact_solution[:,5]
e_exact = exact_solution[:,6]

# Basic inputs
x_init = 0
x_end = 10
domain = [x_init, x_end]
Ncells = 100
gamma = 1.4
tol = 5e-13

# Boundary conditions
BoundaryConditions =  {'rho':[rho_exact[0],rho_exact[-1]],
                       'u':[u_exact[0],u_exact[-1]],
                       'p':[p_exact[0],p_exact[-1]],
                       'e':[e_exact[0],e_exact[-1]]}

# Initial conditions
dx = (x_end - x_init) / Ncells 
x_0 = np.linspace(x_init + dx,
                x_end, Ncells - 1,
                endpoint=False)
rho_0 = ((rho_exact[-1] - rho_exact[0]) / (x_end - x_init)) * x_0 + rho_exact[0]
u_0 = ((u_exact[-1] - u_exact[0]) / (x_end - x_init)) * x_0 + u_exact[0]
e_0 = ((e_exact[-1] - e_exact[0]) / (x_end - x_init)) * x_0 + e_exact[0]
Q_0 = np.array([rho_0, rho_0 * u_0, e_0]).flatten()

#
k2 = 0
k4 = 0.02
# Effect of Courant number

Courant = 10
Case_1 = ARC1D(domain, Ncells,BoundaryConditions,
               Q_0, gamma,Courant, tol, k2, k4)
iter1 = Case_1.counter
res1 = Case_1.norm_residuals


Courant = 20
Case_2 = ARC1D(domain, Ncells,BoundaryConditions,
               Q_0, gamma,Courant, tol, k2, k4)
iter2 = Case_2.counter
res2 = Case_2.norm_residuals

Courant = 40
Case_3 = ARC1D(domain, Ncells,BoundaryConditions,
               Q_0, gamma,Courant, tol, k2, k4)
iter3 = Case_3.counter
res3 = Case_3.norm_residuals

# Effect of K2 and K4

# k2 = 0, k4 = 0.02
Courant = 40
Case_4 = ARC1D(domain, Ncells,BoundaryConditions,
               Q_0, gamma,Courant, tol, 0, 0.02)
iter4 = Case_4.counter
res4 = Case_4.norm_residuals

# k2 = 0.25, k4 = 0.02
Courant = 40
Case_5 = ARC1D(domain, Ncells,BoundaryConditions,
               Q_0, gamma,Courant, tol, 0.25, 0.02)
iter5 = Case_5.counter
res5 = Case_5.norm_residuals

# k2 = 0.5, k4 = 0.02
Courant = 40
Case_6 = ARC1D(domain, Ncells,BoundaryConditions,
               Q_0, gamma,Courant, tol, 0.5, 0.02)
iter6 = Case_6.counter
res6 = Case_6.norm_residuals


# Effect of mesh size

Ncells = 50
# Initial conditions
dx = (x_end - x_init) / Ncells 
x_0 = np.linspace(x_init + dx,
                x_end, Ncells - 1,
                endpoint=False)
rho_0 = ((rho_exact[-1] - rho_exact[0]) / (x_end - x_init)) * x_0 + rho_exact[0]
u_0 = ((u_exact[-1] - u_exact[0]) / (x_end - x_init)) * x_0 + u_exact[0]
e_0 = ((e_exact[-1] - e_exact[0]) / (x_end - x_init)) * x_0 + e_exact[0]
Q_0 = np.array([rho_0, rho_0 * u_0, e_0]).flatten()
Courant = 40
Case_7 = ARC1D(domain, Ncells,BoundaryConditions,
               Q_0, gamma,Courant, tol, 0, 0.02)
iter7 = Case_7.counter
res7 = Case_7.norm_residuals


Ncells = 100
# Initial conditions
dx = (x_end - x_init) / Ncells 
x_0 = np.linspace(x_init + dx,
                x_end, Ncells - 1,
                endpoint=False)
rho_0 = ((rho_exact[-1] - rho_exact[0]) / (x_end - x_init)) * x_0 + rho_exact[0]
u_0 = ((u_exact[-1] - u_exact[0]) / (x_end - x_init)) * x_0 + u_exact[0]
e_0 = ((e_exact[-1] - e_exact[0]) / (x_end - x_init)) * x_0 + e_exact[0]
Q_0 = np.array([rho_0, rho_0 * u_0, e_0]).flatten()
Courant = 40
Case_8 = ARC1D(domain, Ncells,BoundaryConditions,
               Q_0, gamma,Courant, tol, 0, 0.02)
iter8 = Case_8.counter
res8 = Case_8.norm_residuals


Ncells = 200
# Initial conditions
dx = (x_end - x_init) / Ncells 
x_0 = np.linspace(x_init + dx,
                x_end, Ncells - 1,
                endpoint=False)
rho_0 = ((rho_exact[-1] - rho_exact[0]) / (x_end - x_init)) * x_0 + rho_exact[0]
u_0 = ((u_exact[-1] - u_exact[0]) / (x_end - x_init)) * x_0 + u_exact[0]
e_0 = ((e_exact[-1] - e_exact[0]) / (x_end - x_init)) * x_0 + e_exact[0]
Q_0 = np.array([rho_0, rho_0 * u_0, e_0]).flatten()
Courant = 40
Case_9 = ARC1D(domain, Ncells,BoundaryConditions,
               Q_0, gamma,Courant, tol, 0, 0.02)
iter9 = Case_9.counter
res9 = Case_9.norm_residuals



fig = plt.figure(figsize = (9,3.5))
ax = fig.add_subplot(131)
ay = fig.add_subplot(132)
az = fig.add_subplot(133)
ax.set_xlabel('Iterations')
ax.set_ylabel('Residual norm')
ax.set_title('Effect of $C_n$')

ay.set_xlabel('Iterations')
ay.set_ylabel('Residual norm')
ay.set_title('Effect of k2 and k4')

az.set_xlabel('Iterations')
az.set_ylabel('Residual norm')
az.set_title('Effect of mesh size')
# effect of Courant
ax.semilogy(np.linspace(1,iter1,len(res1)),res1,'k-',lw = 0.9,label='$C_n = 10$')
ax.semilogy(np.linspace(1,iter2,len(res2)),res2,'b-',lw = 0.9,label='$C_n = 20$')
ax.semilogy(np.linspace(1,iter3,len(res3)),res3,'r-',lw = 0.9,label='$C_n = 40$')
# effect of k2 and k4
ay.semilogy(np.linspace(1,iter4,len(res4)),res4,'k-',lw = 0.9,label='$k_2 = 0$')
ay.semilogy(np.linspace(1,iter5,len(res5)),res5,'b-',lw = 0.9,label='$k_2 = 0.25$')
ay.semilogy(np.linspace(1,iter6,len(res6)),res6,'r-',lw = 0.9,label='$k_2 = 0.5$')
# effect of mesh size
az.semilogy(np.linspace(1,iter7,len(res7)),res7,'k-',lw = 0.9,label='N = 50')
az.semilogy(np.linspace(1,iter8,len(res8)),res8,'b-',lw = 0.9,label='N = 100')
az.semilogy(np.linspace(1,iter9,len(res9)),res9,'r-',lw = 0.9,label='N = 200')

ax.legend(numpoints = 1,loc = 'upper right',
          fontsize = 9,frameon = False)
ay.legend(numpoints = 1,loc = 'upper right',
          fontsize = 9,frameon = False)
az.legend(numpoints = 1,loc = 'upper right',
          fontsize = 9,frameon = False)
fig.tight_layout()
plt.show()

