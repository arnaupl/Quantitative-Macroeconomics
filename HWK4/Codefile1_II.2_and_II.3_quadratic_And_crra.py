
"""
QUANTITATIVE MACROECONOMICS: HOMEWORK 4

@author: ARNAU PAGÈS LÓPEZ


IN THIS FILE, I PRESENT THE CODE SOLVING ITEMS II.2 (the infinitely lived 
households economy) AND II.3 (the life-cycle economy) FOR BOTH QUADRATIC AND
CRRA UTILITIES.


"""

#Items II.2 and II.3

#Import  libraries that will be used

import numpy as np
from numpy import vectorize
import matplotlib.pyplot as plt
from itertools import product

#Parametrize the model

rho = 0.06        #Discount rate.
beta = 1/(1+rho)  #Discount factor.
r=0.04            #Exogenously given interest rate r
w = 1             #Normalize w to 1
gamma = 0.65       #Correlation (persistence income process)
var_y = 0.28       #Variance income process
n=100             #number of grid points
maxit=15000       #maximum number of iterations. Mus be high enough to allow cnvergence to be reached.

#Given gamma we can define the transition matrix guiding the income markov process
#as in the statement of the exercise

trans_matrix=np.array([((1+gamma)/2, (1-gamma)/2),((1-gamma)/2, (1+gamma)/2)])


##############################################################################

#%%  QUADRATIC UTILITY:    II.2 (the infinitely-lived households economy)

#I will only use discrete methods (brute fore VFI).

#Define the  idiosyncratic two-state income process using 1-income variance for 
#low state, and 1+income variance for high state.
Y=(1-var_y, 1+var_y)

#Define the c-bar parameter of quadratic utility. Must be high enough (100 times 
#max income to avoid saturation of any consumer)
bar_c=100*Y[1]     

#Define a grid for assets tomorrow
A = np.linspace(((-(1+r)/r)*Y[0]), 30, n)    #natural borrowing limit as
                                              #lower point in grid

#Discretize state space
#Cartesian product
ay = list(product(Y, A, A))
ay = np.array(ay)

y = ay[:, 0]
a_t = ay[:, 1]
a_t1 = ay[:, 2]

#By budget constraint:
cons = w*y+(1+r)*a_t-a_t1

@vectorize

#Define return matrix M
def M(cons):
    
    return -0.5*(cons-bar_c)**2
     
M=M(cons)
M=np.reshape(M, (1, n*n*2))
M=np.reshape(M, (n*2, n))

#Vector of 0 as initial guess for Value function.

V_initial=np.zeros(n*2)

#Define the matrix omega, which takes into account the possible realizations for all the points in the value function.

def omega_1(A):   
    
    return trans_matrix[0,0]*(-0.5*(Y[0]+(1+r)*A-A- bar_c)**2)/(1-beta)+trans_matrix[0, 1]*(-0.5*(Y[1]+(1+r)*A-A-bar_c)**2)/(1-beta)

def omega_2(A):
    
    return trans_matrix[1,0]*(-0.5*(Y[0]+(1+r)*A-A-bar_c)**2)/(1-beta)+trans_matrix[1, 1]*(-0.5*(Y[1]+(1+r)*A-A-bar_c)**2)/(1-beta)

        
omega_1=omega_1(A)
omega_1=np.reshape(omega_1, (n,1))
omega_1=np.tile(omega_1, n)
omega_1=np.transpose(omega_1)

omega_2=omega_2(A)
omega_2=np.reshape(omega_2, (n,1))
omega_2=np.tile(omega_2, n)
omega_2=np.transpose(omega_2)

omega=[omega_1, omega_2]
omega=np.reshape(omega, (n*2,n))

#Using M and omega compute the chi matrix:

chi=M+beta*omega

#And define value function updating rule
V_1=np.amax(chi, axis=1)

#Differences to check convergence
V_difference=V_initial-V_1

count=0

#For differences larger than 1 keep iterating with V_1 as new value function
#Use a maximumm number of iteration high enouh to allow convergence

for V_difference in range(1, maxit):
    
    V_it = V_1
    V_initial = [V_it[0:n], V_it[n:]]
    V_initial = np.array(V_initial)
    
    def omega_1(V_initial):
        
        return trans_matrix[0,0]*V_initial[0,:]+trans_matrix[0,1]*V_initial[1,:]
    
    def omega_2(V_initial):
        
        return trans_matrix[1,0]*V_initial[0,:]+trans_matrix[1,1]*V_initial[1,:]

    omega_1= omega_1(V_initial)
    omega_1= np.reshape(omega_1,(1,n))
    omega_1= np.tile(omega_1, n)
    omega_1= np.reshape(omega_1,(n,n))

    omega_2= omega_2(V_initial)
    omega_2= np.reshape(omega_2,(1,n))
    omega_2= np.tile(omega_2,n)
    omega_2= np.reshape(omega_2,(n,n))
    
    omega= [omega_1, omega_2]
    omega= np.reshape(omega, (n*2, n))
    
    chi=M+beta*omega
    
    V_1= np.amax(chi, axis = 1)
    
    V_difference= V_it - V_1
    
    count += 1
    

#Redefine chi when convergence is reached
    
chi= M + beta*omega

#Value functiontaking into account different realizations of y

V_y1= V_1[0:n]
V_y2= V_1[n:]

#Policy

g= np.argmax(chi, axis = 1)


assetspolicy_y1= A[g[0:n]]
assetspolicy_y2= A[g[n:]]    

conspolicy_y1= Y[0]*np.ones(n) + (1+r)*A-assetspolicy_y1

conspolicy_y2= Y[1]*np.ones(n) + (1+r)*A-assetspolicy_y2

for i in range(0, n):
    
    if conspolicy_y1[i] < 0:
        
        conspolicy_y1[i] = 0
        
    if conspolicy_y2[i] < 0:
        
        conspolicy_y2[i] = 0
           
# Plot results:
    
plt.figure()
plt.plot(A, V_y1, color='r', label = 'Value function for bad shock')
plt.plot(A, V_y2, color='b', label = 'Value function for good shock')
plt.title('Value Functions infinite horizon (quadratic utility)')
plt.legend()
plt.ylabel('V')
plt.xlabel('a')
plt.show()
    
plt.figure()
plt.plot(A, assetspolicy_y1, color='r', label = 'Policy function assets for bad shock')
plt.plot(A, assetspolicy_y2, color='b', label = 'Policy function assets for good shock')
plt.title('Policy functions for assets infinite horizon (quadratic utility)')
plt.legend()
plt.ylabel('a´')
plt.xlabel('a')
plt.show()

plt.figure()
plt.plot(A, conspolicy_y1, color='r', label = 'Policy function consumption for bad shock')
plt.plot(A, conspolicy_y2,color='b', label = 'Policy function consumption for good shock')
plt.title('Policy functions for consumption infinite horizon (quadratic utility)')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()


##############################################################################
#%%  QUADRATIC UTILITY:    II.3 (the life-cycle economy)

#Here, using finite horizon terminal condition (At period T+1=45+1 everything is 0)
#as initial condition, I solve backward.

omega=np.zeros((n*2, n))

count=0

while count < 45:
    
    omega= np.amax((M + beta*omega), axis = 1)
    omega= np.reshape(omega,(n*2, 1))
    omega= omega*np.ones((n*2, n))
    
    count += 1

plt.plot(A, omega[0:n, 0],color='r', label = 'Value function for bad shock')
plt.plot(A, omega[n:, 0],color='b', label = 'Value function for good shock')
plt.legend()
plt.title('Value functions  life-cycle economy (quadratic utility)')
plt.ylabel('V')
plt.xlabel('a')
plt.show()

#compute chi matrix
chi=M + beta*omega
#policies
g = np.argmax(chi, axis = 1)


assetspolicy_y1 = A[g[0:n]]     
assetspolicy_y2 = A[g[n:]]

conspolicy_y1 = Y[0]*np.ones(n) + (1+r)*A - assetspolicy_y1

conspolicy_y2 = Y[1]*np.ones(n) + (1+r)*A - assetspolicy_y2

for i in range(0, n):
    
    if conspolicy_y1[i] < 0:
        
        conspolicy_y1[i] = 0
        
    if conspolicy_y2[i] < 0:
        
        conspolicy_y2[i] = 0
        
plt.figure()
plt.plot(A, assetspolicy_y1,color='r', label = 'Policy assets for bad shock')
plt.plot(A, assetspolicy_y2,color='b', label = 'Policy assets for good shock')
plt.legend()
plt.title('Policy functions for assets life-cycle economy (quadratic utility)')
plt.ylabel('a´')
plt.xlabel('a')
plt.show()

plt.figure()
plt.plot(A, conspolicy_y1,color='r', label = 'Policy consumption for bad shock')
plt.plot(A, conspolicy_y2,color='b', label = 'Policy consumption for good shock')
plt.title('Policy functions for consumption life-cycle economy (quadratic utility)')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()

###########################################################################
#%%  CRRA UTILITY:    II.2 (the infinitely-lived households economy)

sigma = 2 #relative risk aversion coefficient

#Proceed similarly as with quadratic utility. Use value function iteration


#Grid assets tomorrow
A = np.linspace(((-(1+r)/r)*Y[0]), 30, n)   

#Cartesian product
ay = list(product(Y, A, A))
ay = np.array(ay)

y = ay[:, 0]
a_t = ay[:, 1]
a_t1 = ay[:, 2]

#By b.c.
cons = y +(1+r)*a_t-a_t1
        
#return matrix M
M = np.zeros(n*n*2)

for i in range(0, n*n*2):
    
    if cons[i] >= 0:
        
        M[i]=((cons[i]**(1-sigma))-1)/(1-sigma)
        
    if cons[i] < 0:
        
        M[i]=-100000

M = np.reshape(M, (1, n*n*2))        
M = np.reshape(M, (n*2, n))

#Value func initial guess

V_initial = np.zeros(n*2)

#Compute the matrix omega taking into account both states.

def omega_1(A):   
    
    return trans_matrix[0, 0]*(((Y[0]+(1+r)*A-A)**(1-sigma))-1)/((1-sigma)*(1-beta))+trans_matrix[0, 1]*(((Y[1]+(1+r)*A-A)**(1-sigma))-1)/((1-sigma)*(1-beta))

def omega_2(A):
    
    return trans_matrix[1, 0]*(((Y[0]+(1+r)*A-A)**(1-sigma))-1)/((1-sigma)*(1-beta))+trans_matrix[1, 1]*(((Y[1]+(1+r)*A-A)**(1-sigma))-1)/((1-sigma)*(1-beta))

        
omega_1 = omega_1(A)
omega_1 = np.reshape(omega_1, (n,1))
omega_1 = np.tile(omega_1, n)
omega_1 = np.transpose(omega_1)

omega_2 = omega_2(A)
omega_2 = np.reshape(omega_2, (n,1))
omega_2 = np.tile(omega_2, n)
omega_2 = np.transpose(omega_2)

omega = [omega_1, omega_2]
omega = np.reshape(omega, (n*2,n))

#compute matrix chi

chi= M +beta*omega

V_1= np.amax(chi, axis = 1)

V_difference =V_initial-V_1

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
    omega = np.reshape(omega, (n*2, n))
    
    chi = M + beta*omega
    
    V_1 = np.amax(chi, axis = 1)
    
    V_difference = V_it - V_1
    
    count += 1
    


    
chi = M + beta*omega

#vf for different states 

V_y1 = V_1[0:n]
V_y2 = V_1[n:]

#policies

g = np.argmax(chi, axis = 1)



assetspolicy_y1 = A[g[0:n]]    
assetspolicy_y2 = A[g[n:]]    

for i in range(0, 2):
    
    assetspolicy_y1[i] = 0
    assetspolicy_y2[i] = 0
    

conspolicy_y1 = Y[0]*np.ones(n) + (1+r)*A - assetspolicy_y1

conspolicy_y2 = Y[1]*np.ones(n) + (1+r)*A - assetspolicy_y2

for i in range(0, n):
    
    if conspolicy_y1[i] < 0:
        
        conspolicy_y1[i] = 0
        
    if conspolicy_y2[i] < 0:
        
        conspolicy_y2[i] = 0
           
# Plot results
    
plt.figure()
plt.plot(A[3:], V_y1[3:], color='r', label = 'Value function for bad shock')
plt.plot(A[3:], V_y2[3:],color='b',  label = 'Value function for good shock')
plt.title('Value functions infinite horizon (CRRA utility)')
plt.legend()
plt.ylabel('V')
plt.xlabel('a')
plt.show()
    
plt.figure()
plt.plot(A[3:], assetspolicy_y1[3:],color='r', label = 'Policy function assets bad shock')
plt.plot(A[3:], assetspolicy_y2[3:],color='b',  label = 'Policy function assets  good shock')
plt.title('Policy functions assets infinite horizon (CRRA utility)')
plt.legend()
plt.ylabel('a´')
plt.xlabel('a')
plt.show()

plt.figure()
plt.plot(A, conspolicy_y1,color='r',  label = 'Policy function consumption bad shock')
plt.plot(A, conspolicy_y2,color='b',  label = 'Policy function consumption good shock')
plt.title('Policy functions consumption infinite horizon (CRRA utility)')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('a')
plt.show()

##############################################################################
#%%  CRRA UTILITY:    II.3 (the life-cycle economy)

#Here, similarly to quadratic utility case, using finite horizon terminal condition (At period T+1=45+1 everything is 0)
#as initial condition, I solve backward.

omega = np.zeros((n*2, n))

count = 0

while count < 45:
    
    omega = np.amax((M + beta*omega), axis = 1)
    omega = np.reshape(omega,(n*2, 1))
    omega= omega*np.ones((n*2, n))
    
    count += 1
    


plt.plot(A[1:], omega[1:n, 1],color='r', label = 'Value function bad shock')
plt.plot(A[1:], omega[n+1:, 1],color='b', label = 'Value function good shock')
plt.title('Value functions life-cycle economy (CRRA utility)')
plt.legend()
plt.ylabel('V')
plt.xlabel('a')
plt.show()



chi = M + beta*omega
g = np.argmax(chi, axis = 1)

assetspolicy_y1 = A[g[0:n]]    
assetspolicy_y2= A[g[n:]]      

conspolicy_y1 = Y[0]*np.ones(n) + (1+r)*A -assetspolicy_y1

conspolicy_y2 = Y[1]*np.ones(n) + (1+r)*A -assetspolicy_y2

for i in range(0, n):
    
    if conspolicy_y1[i] < 0:
        
        conspolicy_y1[i] = 0
        
    if conspolicy_y2[i] < 0:
        
        conspolicy_y2[i] = 0
        
plt.figure()
plt.plot(A, assetspolicy_y1,color='r', label = 'Policy assets bad shock')
plt.plot(A, assetspolicy_y2, color='b',label = 'Policy assets good shock')
plt.title('Policy unctions assets life-cycle economy (CRRA utility)')
plt.legend()
plt.ylabel('a´')
plt.xlabel('a')
plt.show()

plt.figure()
plt.plot(A, conspolicy_y1,color='r', label = 'Policy consumption bad shock')
plt.plot(A, conspolicy_y2,color='b', label = 'Policy consumption good shock')
plt.title('Policy functions consumption life-cycle economy (CRRA utility)')
plt.legend()
plt.ylabel('Consuption')
plt.xlabel('a')
plt.show()