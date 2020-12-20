# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:13:18 2020

@author: Arnau Pagès López

In this code I simulate the neoclassical growth model for 150 periods, under a
labor productivity shock that follows a 5 state Markov chain. This shock represents
the volatility derived from the Covid-19 pandemic, and the lockdown and restriction
policies implmented to stop covid spread.
"""
import math
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe

#Define parameter values.

beta=0.95     #Discount factor
delta=0.4     #Rate of depreciation
alpha=0.675   #share of labor in the production function.
h=0.5         #inelastic labor supply

nu=2.3        #nu and omega are paramters related with the disutility 
omega=5       #of labor.

#Step 1: Create and simulate the Markov chain for gamma paramter.
prob_gamma = [[0.2, 0.8, 0, 0, 0],
              [0.5, 0, 0.5, 0, 0],
              [0, 0.5, 0, 0.5, 0],
              [0, 0, 0.5, 0, 0.5],
              [0, 0, 0, 0.95, 0.05]]  
                 
state_values_gamma=(3,2.5,2,1.5,1)     #5 states
mc = qe.MarkovChain(prob_gamma, state_values=state_values_gamma) 
inestab=mc.simulate(ts_length=80, init=state_values_gamma[0]) #simulate it for 80 periods

#Add periods of stability
stab1=np.ones(20)*state_values_gamma[0]
stab2=np.ones(50)*state_values_gamma[0]
gamma=np.concatenate((stab1,inestab,stab2),axis=0)

#Define some functions to compute the main variables of the model (which depend on k)

def y(kt,gamma):
   return kt**(1-alpha)*(gamma*h)**alpha

def muc (c):
    return 1/c

def inv (kt1,kt):
    return kt1-(1-delta)*kt

def cons (kt1,kt,gamma):
    return y(kt,gamma)-inv(kt1,kt)

def welfaremeasure (kt1,kt,gamma):
    return math.log(cons(kt1,kt,gamma))-(omega)*((h**(1+1/nu))/(1+1/nu))

#STEP 2:Compute steady state capital stock.

kss=(((1-alpha)*(gamma[0]*h)**alpha)/((1/beta)-(1-delta)))**(1/alpha)


n=150   #number of periods I siumlate

#STEP 3: Solve equation 13 in answers pdf for n periods, i.e. simulate the path 
#for capital taking into account teh realizations of the gamma parameter.
def simulation(k, n=n):
    k_0=kss
    k_final=kss
    k[0]=kss #Initial condition
    k[n-1]=kss #Final condition
    k_sim=np.zeros(n)
    for i in range(0,n-2):
        if i==0:
            k_sim[i+1]=beta*muc(y(k[i+1],gamma[i+1])+(1-delta)*k[i+1]-k[i+2])*(1-delta+(1-alpha)*k[i+1]**(-alpha)*(gamma[i+1]*h)**alpha)-muc(y(k_0,gamma[i])+(1-delta)*k_0-k[i+1])            
        elif i==(n-2):
             k_sim[i+1]=beta*muc(y(k[i+1],gamma[i+1])+(1-delta)*k[i+1]-k_final)*(1-delta+(1-alpha)*k[i+1]**(-alpha)*(gamma[i+1]*h)**alpha)-muc(y(k[i],gamma[i])+(1-delta)*k[i]-k[i+1])                            
        else:
             k_sim[i+1]=beta*muc(y(k[i+1],gamma[i+1])+(1-delta)*k[i+1]-k[i+2])*(1-delta+(1-alpha)*k[i+1]**(-alpha)*(gamma[i+1]*h)**alpha)-muc(y(k[i],gamma[i])+(1-delta)*k[i]-k[i+1])    

    return(k_sim)

x0 = np.ones(n)*kss
sim_pathk=fsolve(simulation,x0)
#Obtain the paths for the rest of the relevant variables (investment (equivalently savings),
#output, consumption, etc.)
sim_pathy=y(sim_pathk,gamma)

sim_pathinv=np.zeros(n)
for i in range(0,n-1):
    sim_pathinv[i]=sim_pathk[i+1]-(1-delta)*sim_pathk[i]
sim_pathinv[n-1]=sim_pathinv[n-2]

sim_pathcons=sim_pathy-sim_pathinv

sim_pathwelfaremeasure=np.log(sim_pathcons)-np.ones(n)*((omega)*((h**(1+1/nu))/(1+1/nu)))

#time array
time=np.array(list(range(0,n)))

#Plot the simulated paths.
fig,ax = plt.subplots()    
ax.plot(time, sim_pathk,'-', color='purple', linewidth=2)   
ax.set_title('Simulated path for capital stock')
ax.set_ylabel('k')
ax.set_xlabel('Time')
plt.show()

fig,ax = plt.subplots()    
ax.plot(time, sim_pathy,'-', color='purple', linewidth=2)   
ax.set_title('Simulated path for output')
ax.set_ylabel('Output')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, sim_pathinv,'-', color='purple', linewidth=2)   
ax.set_title('Simulated path for investment (equivalentlly savings)')
ax.set_ylabel('Investment (equivallently savings)')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, sim_pathcons,'-', color='purple', linewidth=2)   
ax.set_title('Simulated path for consumption')
ax.set_ylabel('Consumption')
ax.set_xlabel('Time')
plt.show()



fig,ax = plt.subplots()    
ax.plot(time, sim_pathwelfaremeasure,'-', color='purple', linewidth=2)   
ax.set_title('Simulated path for instantaneous welfare')
ax.set_ylabel('Instantaneous welfare')
ax.set_xlabel('Time')
plt.show()

