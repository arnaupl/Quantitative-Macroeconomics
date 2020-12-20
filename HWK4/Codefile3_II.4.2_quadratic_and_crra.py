
"""
QUANTITATIVE MACROECONOMICS: HOMEWORK 4

@author: Arnau Pagès López

IN THIS FILE, I PRESENT THE CODE SOLVING ITEM II.4 .2 (partial equilibrium with 
uncertainty) FOR BOTH QUADRATIC AND CRRA UTILITIES.
"""
#Item II.4.2

#Import  libraries that will be used

import numpy as np 
from numpy import vectorize
import matplotlib.pyplot as plt
from itertools import product


#Parameters

rho = 0.06 #discount rate
r = 0.04 #given interest rate
w = 1   #keep  wages normalized to 1
beta = 1/(1+rho) #discount factor
n=100 # number of points in assets grid
maxit=15000 #maximum number of iterations.Set it high enough to allow convergence
#%%  QUADRATIC UTILITY: -the infinitely-lived households economy

gamma = 0      #Change to 0.95 fpr alternative setting
bar_c = 100
var_y = 0.1    #Change to 0.5 for alternative setting 

#two income states
Y = [1-var_y, 1+var_y]
Y = np.array(Y)

#discretize state sapace

A = np.linspace(((-(1+r)/r)*Y[0]), 40, n)

#cartesian product

ay = list(product(Y, A, A))
ay = np.array(ay)

y = ay[:,0]
a_t = ay[:,1]
a_t1 = ay[:,2]

# define transition matrix

trans_matrix = np.array([((1+gamma)/2, (1-gamma)/2), ((1-gamma)/2, (1+gamma)/2)])

cons = y+(1+r)*a_t-a_t1

@vectorize


#return matrix M
  
def M(cons):
    
    return -0.5*(cons-bar_c)**2
     
M = M(cons)
M = np.reshape(M, (1, 2*n*n))
M = np.reshape(M, (2*n, n))

#vector of 0 as initial guess of value function

V_initial = np.zeros(2*n)

#Define matrix omega

def omega_1(A):   
    
    return trans_matrix[0, 0]*(-0.5*(Y[0] + (1+r)*A - A - bar_c)**2)/(1-beta) + trans_matrix[0, 1]*(-0.5*(Y[1] + (1+r)*A - A - bar_c)**2)/(1-beta)

def omega_2(A):
    
    return trans_matrix[1, 0]*(-0.5*(Y[0] + (1+r)*A - A - bar_c)**2)/(1-beta) + trans_matrix[1, 1]*(-0.5*(Y[1] + (1+r)*A - A - bar_c)**2)/(1-beta)

        
omega_1 = omega_1(A)
omega_1 = np.reshape(omega_1, (n,1))
omega_1 = np.tile(omega_1, n)
omega_1 = np.transpose(omega_1)

omega_2 = omega_2(A)
omega_2 = np.reshape(omega_2, (n,1))
omega_2 = np.tile(omega_2, n)
omega_2 = np.transpose(omega_2)

omega = [omega_1, omega_2]
omega = np.reshape(omega, (2*n,n))

#matrix chi:

chi = M + beta*omega

V_1 = np.amax(chi, axis = 1)

V_difference = V_initial - V_1

count = 0

#For differences larger than 1 keep iterating with V_1 as new value function
#Use a maximumm number of iteration high enouh to allow convergence

for V_difference in range(1, maxit):
    
    V_it = V_1
    V_initial = [V_it[0:n], V_it[n:]]
    V_initial = np.array(V_initial)
    
    def omega_1(V_initial):
        
        return trans_matrix[0, 0]*V_initial[0, :] + trans_matrix[0, 1]*V_initial[1, :]
    
    def omega_2(V_initial):
        
        return trans_matrix[1, 0]*V_initial[0, :] + trans_matrix[1, 1]*V_initial[1, :]

    omega_1 = omega_1(V_initial)
    omega_1 = np.reshape(omega_1, (1,n))
    omega_1 = np.tile(omega_1, n)
    omega_1 = np.reshape(omega_1, (n,n))

    omega_2 = omega_2(V_initial)
    omega_2 = np.reshape(omega_2, (1,n))
    omega_2 = np.tile(omega_2, n)
    omega_2 = np.reshape(omega_2, (n,n))
    
    omega = [omega_1, omega_2]
    omega = np.reshape(omega, (2*n, n))
    
    chi = M + beta*omega
    
    V_1 = np.amax(chi, axis = 1)
    
    V_difference = V_it - V_1
    
    count += 1
    

    
chi= M + beta*omega

#Value function for diferent realizations of income

V_y1 = V_1[0:n]
V_y2 = V_1[n:]

#Policy

g = np.argmax(chi, axis = 1)

assetspolicy_y1 = A[g[0:n]]   
assetspolicy_y2 = A[g[n:]]    

conspolicy_y1 = Y[0]*np.ones(n) + (1+r)*A - assetspolicy_y1

conspolicy_y2 = Y[1]*np.ones(n) + (1+r)*A - assetspolicy_y2

# Plot results

plt.figure()
plt.plot(A, conspolicy_y1, color='r', label = 'Consumption policy bad shock')
plt.plot(A, conspolicy_y2, color='b', label = 'Consumption policy good shock')
plt.title('Consumption policies quadratic utility (certainty equivalence) infinite horizon')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()


##############################################################################
                     #######################
##############################################################################
#QUADRATIC UTILITY: -life-cycle economy

#Use terminal condition and solve bacward

omega = np.zeros(2*n*n)
omega = np.reshape(omega, (2*n,n))

count = 0
SV = []
SG = []

cons = y+(1+r)*a_t-a_t1

@vectorize
      
def M(cons):
    
    return -0.5*(cons-bar_c)**2
         
M = M(cons)
M = np.reshape(M,(1, 2*n*n))
M = np.reshape(M,(2*n, n))

for count in range(1, 46):
    
    chi = M + beta*omega
    g = np.argmax(chi, axis = 1)
    omega = np.amax(chi, axis = 1)
    #Store
    SV.append(omega)       
    SG.append(g)
    
    omega = np.reshape(omega, [2*n,1])
    omega = np.tile(omega, n)
    omega = np.transpose(omega)
    omega_1 = omega[:n, :n]
    omega_2 = omega[:n, n:]
    omega = np.concatenate((omega_1, omega_2))
    count = count+1
    
SV = np.array(SV)
SV = np.transpose(SV)
SG = np.array(SG)
SG = np.transpose(SG)

#T=5 T=40

assets_5 = A[SG[0:n, 5]]
assets_40 = A[SG[0:n, 40]]

cons_5 = Y[0]*np.ones(n) + (1+r)*A - assets_5
cons_40 = Y[0]*np.ones(n) + (1+r)*A - assets_40

plt.figure()
plt.plot(A, cons_5,color='green', label = 'Con. T=5')
plt.plot(A, cons_40, color='purple', label = 'Cons. T=40')
plt.title('Policy functions consumption quadratic utility (certainty equivalent) life-cycle economy')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()

#Simulated time paths for consumption drawing randomly realizations of shocks

t = np.linspace(0, 45, 45)

y = np.zeros([1, 45])

for i in range(0, 45):
    
    y[0, i] = np.random.choice((1-var_y, 1+var_y), p = ((1+gamma)/2, (1-gamma)/2))
 
asim = np.zeros([45,1])

for i in range(0, 45):
    
    if y[0, i] < 1:
        
        asim[i] = assetspolicy_y1[i]
    
    if y[0, i] > 1:
        
        asim[i] = assetspolicy_y2[i]
        
cons = np.zeros(45)

for i in range(0, 44):
    
    cons[i] = asim[i]*(1+r)+w*y[0, i]-asim[i+1]
    
    if cons[i] <= 0:
        cons[i] = 0
        
plt.figure()
plt.plot(t, asim, color='purple',label = 'Assets')
plt.title('Assets simulated path 45 periods quadratic utility (certainty equivalent)')
plt.ylabel('a')
plt.xlabel('t')
plt.show()

# Simulation and plot for consumption:

plt.figure()
plt.plot(t, cons,color='purple', label = 'Consumption')
plt.title('Consumption simulated path 45 periods quadratic utility (certainty equivalent)')
plt.ylabel('Consumption')
plt.xlabel('t')
plt.show()

#%% CRRA UTILITY: -the infinitely-lived households economy



sigma = 2       # relative risk aversion. Increase it for alternative settings in which agents are more prudent.
var_y = 0.1    # Change it for alternative settings
gamma = 0       #Change it for alternative settings

#Income process
Y = [1-var_y, 1+var_y]
Y = np.array(Y)

#Discretize state space

A = np.linspace(((-(1+r)/r)*Y[0]), 40, n)

#Cartesian product

ay = list(product(Y, A, A))
ay = np.array(ay)

y = ay[:, 0]
a_t = ay[:, 1]
a_t1 = ay[:, 2]

#Transition matrix

trans_matrix = np.array([((1+gamma)/2, (1-gamma)/2), ((1-gamma)/2, (1+gamma)/2)])

#by b.const
cons = y+(1+r)*a_t-a_t1
 
M = np.zeros(2*n*n)

for i in range(0, 2*n*n):
    
    if cons[i] >= 0:
        
        M[i] = ((cons[i]**(1-sigma))-1)/(1-sigma)
        
    if cons[i] < 0:
        
        M[i] = -100000

M = np.reshape(M, (1, 2*n*n))        
M = np.reshape(M, (2*n, n))

#vector of zeros as inital guess value function

V_initial = np.zeros(2*n)

#Obtain the matrix omega:

def omega_1(A):   
    
    return trans_matrix[0, 0]*(((Y[0] + (1+r)*A - A)**(1-sigma))-1)/((1-sigma)*(1-beta)) + trans_matrix[0, 1]*(((Y[1] + (1+r)*A - A)**(1-sigma))-1)/((1-sigma)*(1-beta))

def omega_2(A):
    
    return trans_matrix[1, 0]*(((Y[0] + (1+r)*A - A)**(1-sigma))-1)/((1-sigma)*(1-beta)) + trans_matrix[1, 1]*(((Y[1] + (1+r)*A - A)**(1-sigma))-1)/((1-sigma)*(1-beta))

        
omega_1 = omega_1(A)
omega_1 = np.reshape(omega_1, (n,1))
omega_1 = np.tile(omega_1, n)
omega_1 = np.transpose(omega_1)

omega_2 = omega_2(A)
omega_2 = np.reshape(omega_2, (n,1))
omega_2 = np.tile(omega_2, n)
omega_2 = np.transpose(omega_2)

omega = [omega_1, omega_2]
omega = np.reshape(omega, (2*n,n))

#Chi matrix

chi = M + beta*omega

V_1 = np.amax(chi, axis = 1)

V_difference = V_initial - V_1

count = 0

#For differences larger than 1 keep iterating with V_1 as new value function
#Use a maximumm number of iteration high enouh to allow convergenc

for V_difference in range(1, maxit):
    
    V_it = V_1
    V_initial = [V_it[0:n], V_it[n:]]
    V_initial = np.array(V_initial)
    
    def omega_1(V_initial):
        
        return trans_matrix[0, 0]*V_initial[0, :] + trans_matrix[0, 1]*V_initial[1, :]
    
    def omega_2(V_initial):
        
        return trans_matrix[1, 0]*V_initial[0, :] + trans_matrix[1, 1]*V_initial[1, :]

    omega_1 = omega_1(V_initial)
    omega_1 = np.reshape(omega_1, (1,n))
    omega_1 = np.tile(omega_1, n)
    omega_1 = np.reshape(omega_1, (n,n))

    omega_2 = omega_2(V_initial)
    omega_2 = np.reshape(omega_2, (1,n))
    omega_2 = np.tile(omega_2, n)
    omega_2 = np.reshape(omega_2, (n,n))
    
    omega = [omega_1, omega_2]
    omega = np.reshape(omega, (2*n, n))
    
    chi = M + beta*omega
    
    V_1 = np.amax(chi, axis = 1)
    
    V_difference = V_it - V_1
    
    count += 1

    
chi = M + beta*omega

# VF taking into account realizations of income

V_y1 = V_1[0:n]
V_y2 = V_1[n:]

#policies

g = np.argmax(chi, axis = 1)

assetspolicy_y1 = A[g[0:n]]   
assetspolicy_y2 = A[g[n:]]    

conspolicy_y1 = Y[0]*np.ones(n) + (1+r)*A - assetspolicy_y1

conspolicy_y2 = Y[1]*np.ones(n) + (1+r)*A - assetspolicy_y2

for i in range(0,n):
    
    if conspolicy_y1[i] < 0:
        conspolicy_y1[i] = 0
    
    if conspolicy_y2[i] < 0:
        conspolicy_y2[i] = 0   

#Plot results

plt.figure()
plt.plot(A, conspolicy_y1, color='r', label = 'Consumption policy bad shock')
plt.plot(A, conspolicy_y2, color='b', label = 'Consumption policy good shock')
plt.title('Consumption policies with CRRA utility (precautionary savings) infinite horizon')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()

#############################################################################
                     #######################
##############################################################################
#CRRA UTILITY: -life-cycle economy

omega = np.zeros(2*n*n)
omega = np.reshape(omega, (2*n,n))

count = 0
SV = []
SG = []

M = np.zeros(2*n*n)

for i in range(0, 2*n*n):
    
    if cons[i] >= 0:
        
        M[i] = ((cons[i]**(1-sigma))-1)/(1-sigma)
        
    if cons[i] < 0:
        
        M[i] = -100000

M = np.reshape(M, (1, 2*n*n))        
M = np.reshape(M, (2*n, n))

for count in range(1, 46):
    
    chi = M + beta*omega
    g = np.argmax(chi, axis = 1)
    omega = np.amax(chi, axis = 1)
    #Store
    SV.append(omega)       
    SG.append(g)
    
    omega = np.reshape(omega, [2*n,1])
    omega = np.tile(omega, n)
    omega = np.transpose(omega)
    omega_1 = omega[:n, :n]
    omega_2 = omega[:n, n:]
    omega = np.concatenate((omega_1, omega_2))
    count = count+1
    
SV = np.array(SV)
SV = np.transpose(SV)
SG = np.array(SG)
SG = np.transpose(SG)



assets_5 = A[SG[0:n, 5]]
assets_40 = A[SG[0:n, 40]]

cons_5 = Y[0]*np.ones(n) + (1+r)*A - assets_5
cons_40 = Y[0]*np.ones(n) + (1+r)*A - assets_40

for i in range(0,n):
    
    if cons_5[i] < 0:
        cons_5[i] = 0
    
    if cons_40[i] < 0:
        cons_40[i] = 0   
        
plt.figure()
plt.plot(A, cons_5, color='purple', label = 'Cons. T=5')
plt.plot(A, cons_40, color='green', label = 'Cons. T=40')
plt.title('Policy functions consumption CRRA utility (precautionary savings) life-cycle economy')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()


#Simulated time paths for consumption drawing randomly realizations of shocks

t = np.linspace(0, 45, 45)

y = np.zeros([1, 45])

for i in range(0, 45):
    
    y[0, i] = np.random.choice((1-var_y, 1+var_y), p = ((1+gamma)/2, (1-gamma)/2))
 
asim = np.zeros([45,1])

for i in range(0, 45):
    
    if y[0, i] < 1:
        
        asim[i] = assetspolicy_y1[i]
    
    if y[0, i] > 1:
        
        asim[i] = assetspolicy_y2[i]
        
cons = np.zeros(45)

for i in range(0, 44):
    
    cons[i] = asim[i]*(1+r)+w*y[0, i]-asim[i+1]
    
    if cons[i] <= 0:
        cons[i] = 0
#plot simulated paths

plt.figure()
plt.plot(t[1:], asim[1:],color='purple', label = 'Assets')
plt.title('Assets simulated path 45 periods CRRA utility (precuationary savings)')
plt.ylabel('a')
plt.xlabel('t')
plt.show()



plt.figure()
plt.plot(t[1:], cons[1:], color='purple', label = 'Consumption')
plt.title('Consumption simulated path 45 periods CRRA utility (precuationary savings)')
plt.ylabel('Consumption')
plt.xlabel('t')
plt.show()
