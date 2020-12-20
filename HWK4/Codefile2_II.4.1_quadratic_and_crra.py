
"""
QUANTITATIVE MACROECONOMICS: HOMEWORK 4


@author: Arnau Pagès López

IN THIS FILE, I PRESENT THE CODE SOLVING ITEM II.4 .1 (partial equilibrium with 
certainty) FOR BOTH QUADRATIC AND CRRA UTILITIES.
"""

#Item II.4.1

#Import  libraries that will be used

import numpy as np 
from numpy import vectorize
import matplotlib.pyplot as plt
from itertools import product

#define parameters:

rho = 0.06 #Disc. rate
r = 0.04   #Exogenously given interest rate r
w = 1      #Normalize w to 1
beta = 1/(1+rho) #disc factor
n=100 #number of points in the grid
maxit=15000  #maximum number of iterations. Must be high enough

#%%  QUADRATIC UTILITY: -the infinitely-lived households economy


gamma = 0  #certainty
var_y = 0  #certainty

#by certainty, income is actually deterministic
Y = [1-var_y, 1+var_y]
Y = np.array(Y)


bar_c=100

#Define a grid for assets tomorrow
A = np.linspace(((-(1+r)/r)*Y[0]), 40, n)   #natural borrowing limit as
                                              #lower point in grid


#Discretize state space
#Cartesian product
ay = list(product(Y, A, A))
ay = np.array(ay)

y = ay[:, 0]
a_t = ay[:, 1]
a_t1 = ay[:, 2]

# Transition matrix, now deterministic

trans_matrix = np.array([((1+gamma)/2, (1-gamma)/2), ((1-gamma)/2, (1+gamma)/2)])

#by budget constraint

cons = y + (1+r)*a_t - a_t1
    
@vectorize
  
def M(cons):
    
    return -0.5*(cons-bar_c)**2
    
#define return matrix M
     
M = M(cons)
M = np.reshape(M, (1, 2*n*n))
M = np.reshape(M, (2*n, n))

#use a vector of zeros as initial guess for value function

V_initial = np.zeros(2*n)

# #Define the matrix omega
    
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

#Define matrix chi

chi= M + beta*omega

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
    

#When convergence is reached redefine chi
    
chi = M + beta*omega



V_y1 = V_1[0:n]
V_y2 = V_1[n:]

#Get policy

g = np.argmax(chi, axis = 1)

#And obtain policies
    
assetspolicy_y1 = A[g[0:n]]   
assetspolicy_y2 = A[g[n:]]    

conspolicy_y1 = np.zeros(n)
conspolicy_y2 = np.zeros(n)

conspolicy_y1 = Y[0]*np.ones(n) + (1+r)*A - assetspolicy_y1

conspolicy_y2 = Y[1]*np.ones(n) + (1+r)*A - assetspolicy_y2

for i in range(0, n):
    
    if conspolicy_y1[i] <= 0:
        
        conspolicy_y1[i] = 0
        
    if conspolicy_y2[i] <= 0:
        
        conspolicy_y2[i] = 0
        
#Plot results

plt.figure()
plt.plot(A, conspolicy_y1, color='r', label = 'Policy function consumption bad shock')
plt.plot(A, conspolicy_y2, color='b', label = 'Policy function consumption good shock')
plt.title('Policy functions  consumption infinite horizon (quadratic utility)')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()

#Now simulate the paths for 45 periods.

y = np.zeros([1, n])

for i in range(0, n):
    #by certainty assumption:
    y[0, i] = 1  

#Assets
        
asim = np.zeros(45)

assetspolicy_y1 = A[g[0:n]]   

g_y1 = g[0:n]

#initial guess
asim[0]  = g_y1[n-1] 

for i in range(1, 45):
    
        asim[i] = g_y1[int(asim[i-1])]
        
for i in range(0, 44):
    
        asim[i] = assetspolicy_y1[int(asim[i])]

          
t = np.linspace(0, 44, 44)

plt.figure()
plt.plot(t, asim[0:44], color='purple', label = 'Assets ')
plt.title('Assets simulated path 45 periods (quadratic utility)')
plt.ylabel('a')
plt.xlabel('t')
plt.show()

#Simulated consumption

cons = np.zeros(44)

for i in range(0, 44):
    
    cons[i] = asim[i]*(1+r)+w*y[0, i]-asim[i+1]
    
    if cons[i] <= 0:
        cons[i] = 0

plt.figure()
plt.plot(t[0:43], cons[0:43], color='purple', label = 'Consumption')
plt.title('Consumption simulated path 45 periods (quadratic utility)')
plt.ylabel('Consumption')
plt.xlabel('t')
plt.show()
#############################################################################
                     #######################
##############################################################################
#QUADRATIC UTILITY: -life-cycle economy

#use termminal consition

# Quadratic utility:

A = np.linspace(((-(1+r)/r)*Y[0]), 10, n)

#cartesian product 

ay = list(product(Y, A, A))
ay = np.array(ay)

y = ay[:,0]
a_t = ay[:,1]
a_t1 = ay[:,2]

#by certainty transition matrix is actually deterministic

trans_matrix = np.array([((1+gamma)/2, (1-gamma)/2), ((1-gamma)/2, (1+gamma)/2)])

cons = y+(1+r)*a_t-a_t1

@vectorize
  
def M(cons):
    
    return -0.5*(cons-bar_c)**2
     
M = M(cons)
M = np.reshape(M,(1, 2*n*n))
M = np.reshape(M,(2*n, n))
omega = np.zeros(2*n*n)
omega = np.reshape(omega, (2*n,n))

count = 0
S_V = []
S_G = []

for count in range(1, 46):
    
    chi = M + beta*omega
    g = np.argmax(chi, axis = 1)
    omega= np.amax(chi, axis = 1)
    #store at eachiteration
    S_V.append(omega)       
    S_G.append(g)
    
    omega= np.reshape(omega, [2*n,1])
    omega = np.tile(omega, n)
    omega = np.transpose(omega)
    omega_1 = omega[:n, :n]
    omega_2 = omega[:n, n:]
    omega= np.concatenate((omega_1, omega_2))
    count = count+1
    
S_V = np.array(S_V)
S_V = np.transpose(S_V)
S_G = np.array(S_G)
S_G = np.transpose(S_G)

#perios T=5 T=40

assets_5 = A[S_G[0:n, 5]]
assets_40 = A[S_G[0:n, 40]]

cons_5 = Y[0]*np.ones(n) + (1+r)*A - assets_5
cons_40 = Y[0]*np.ones(n) + (1+r)*A - assets_40

for i in range(0, n):
    
    if cons_5[i] < 0:
        
        cons_5[i] = 0
    
    if cons_40[i] < 0:
        
        cons_40[i] = 0

plt.figure()
plt.plot(A, cons_5, color='green', label = 'Cons. for T=5')
plt.plot(A, cons_40, color='purple', label = 'Cons. for T=40')
plt.title('Policy dunctions consumption life-cycle economy (quadratic utility)')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()

#%% CRRA UTILITY: -the infinitely-lived households economy

#proceed similarly to with quadratic utility
r = 0.04
sigma = 2
A = np.linspace(((-(1+r)/r)*Y[0]), 40, n)


ay = list(product(Y, A, A))
ay = np.array(ay)

y = ay[:,0]
a_t = ay[:,1]
a_t1 = ay[:,2]


#transition matrix now with certainty is deterministic
trans_matrix = np.array([((1+gamma)/2, (1-gamma)/2), ((1-gamma)/2, (1+gamma)/2)])
#by budget constraint
cons = y + (1+r)*a_t - a_t1
        
M = np.zeros(2*n*n)

for i in range(0, 2*n*n):
    
    if cons[i] >= 0:
        
        M[i] = ((cons[i]**(1-sigma))-1)/(1-sigma)
        
    if cons[i] < 0:
        
        M[i] = -100000

M = np.reshape(M, (1, 2*n*n))        
M = np.reshape(M, (2*n, n))


#initial guess value function is a vector of zeros
V_initial = np.zeros(2*n)

#compute matrix omega
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

#and define chi matrix

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
    
    omega= [omega_1, omega_2]
    omega= np.reshape(omega, (2*n, n))
    
    
 
    
    chi = M + beta*omega
    
    V_1 = np.amax(chi, axis = 1)
    
    V_difference = V_it - V_1
    
    count += 1
    
# redefine chi when convergence is reached
    
chi = M + beta*omega



V_y1 = V_1[0:n]
V_y2 = V_1[n:]

#get policies

g = np.argmax(chi, axis = 1)



#for assets
assetspolicy_y1 = A[g[0:n]]   
assetspolicy_y2 = A[g[n:]]    

#for consumption
conspolicy_y1 = Y[0]*np.ones(n) + (1+r)*A - assetspolicy_y1

conspolicy_y2 = Y[1]*np.ones(n) + (1+r)*A - assetspolicy_y2

for i in range(0, n):
    
    if conspolicy_y1[i] < 0:
        
        conspolicy_y1[i] = 0
    
    if conspolicy_y2[i] < 0:
        
        conspolicy_y2[i] = 0      
    
# Plot policies

plt.figure()
plt.plot(A, conspolicy_y1, color='red', label = 'Consumption policy bad shock')
plt.plot(A, conspolicy_y2, color='blue', label = 'Consumption policy good shock')
plt.title('Policy functions  consumption infinite horizon (CRRA utility)')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()

#Now simulate the paths for 45 periods.

y = np.zeros([1, n])

for i in range(0, n):
    
    y[0, i] = 1  

#Assets
        
asim = np.zeros(45)

assetspolicy_y1 = A[g[0:n]]     # optimal decision of assets 

g_y1 = g[0:n]

#initial guess
asim[0]  = g_y1[n-1] 

for i in range(1, 45):
    
        asim[i] = g_y1[int(asim[i-1])]
        
for i in range(0, 44):
    
        asim[i] = assetspolicy_y1[int(asim[i])]

          
t = np.linspace(0, 44, 44)

plt.figure()
plt.plot(t,asim[0:44], color='purple', label = 'a ')
plt.title('Assets simulated pathe 45 periods (CRRA utility)')
plt.ylabel('Assets')
plt.xlabel('t')
plt.show()

#Consumption

cons = np.zeros(44)

for i in range(0, 44):
    
    cons[i] = asim[i]*(1+r)+w*y[0, i]-asim[i+1]
    
    if cons[i] <= 0:
        cons[i] = 0

plt.figure()
plt.plot(t[0:43], cons[0:43],color='purple', label = 'Cons.')
plt.title('Consumption simulated path 45 periods (CRRA utility)')
plt.ylabel('Consumption')
plt.xlabel('t')
plt.show()

#############################################################################
                     #######################
##############################################################################
#CRRA UTILITY: -life-cycle economy

A = np.linspace(((-(1+r)/r)*Y[0]), 40, n)



ay = list(product(Y, A, A))
ay = np.array(ay)

y = ay[:,0]
a_t = ay[:,1]
a_t1 = ay[:,2]

# Transition matrix now under certainty

trans_matrix = np.array([((1+gamma)/2, (1-gamma)/2), ((1-gamma)/2, (1+gamma)/2)])

cons = y+(1+r)*a_t-a_t1

M = np.zeros(2*n*n)

for i in range(0, 2*n*n):
    
    if cons[i] >= 0:
        
        M[i] = ((cons[i]**(1-sigma))-1)/(1-sigma)
        
    if cons[i] < 0:
        
        M[i] = -100000

M = np.reshape(M, (1, 2*n*n))        
M = np.reshape(M, (2*n, n))

omega = np.zeros(2*n*n)
omega = np.reshape(omega, (2*n,n))

count = 0
S_V = []
S_G = []

for count in range(1, 46):
    
    chi = M + beta*omega
    g = np.argmax(chi, axis = 1)
    omega = np.amax(chi, axis = 1)
    #store at each iteration
    S_V.append(omega)       
    S_G.append(g)
    
    omega = np.reshape(omega, [2*n,1])
    omega = np.tile(omega, n)
    omega = np.transpose(omega)
    omega_1 = omega[:n, :n]
    omega_2 = omega[:n, n:]
    omega = np.concatenate((omega_1, omega_2))
    count = count+1
    
S_V = np.array(S_V)
S_V = np.transpose(S_V)
S_G = np.array(S_G)
S_G = np.transpose(S_G)

#Assets and consuption at T=5 T=40

assets_5 = A[S_G[0:n, 5]]
assets_40 = A[S_G[0:n, 40]]

cons_5 = Y[0]*np.ones(n) + (1+r)*A - assets_5
cons_40 = Y[0]*np.ones(n) + (1+r)*A - assets_40

for i in range(0, n):
    
    if cons_5[i] < 0:
        
        cons_5[i] = 0
    
    if cons_40[i] < 0:
        
        cons_40[i] = 0
        
        
#Plot results.

plt.figure()
plt.plot(A, cons_5, color='green', label = 'Cons. for T=5')
plt.plot(A, cons_40, color='purple', label = 'Cons. for T=40')
plt.title('Policy functions consumption life-cycle economy (CRRA utility)')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('Assets')
plt.show()