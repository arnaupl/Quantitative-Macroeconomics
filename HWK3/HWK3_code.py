# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 20:26:46 2020

@author: ARNAU PAGÈS LÓPEZ
"""
#QUANTITATIVE MACROECONOMICS-HOMEWORK 3

#QUESTION 1: VALUE FUNCTION ITERATION

#1.1 DISCRETE METHODS WITH INELASTIC LABOR SUPPLY h=1

#COMMON PART OF ITEM 1


#Let me first of all define the parameters of the model.
import numpy as np
import time
import matplotlib.pyplot as pl
thetta=0.679
beta=0.988
delta=0.013

#Discretize the state space.
kss=(((1/beta)+delta-1)/(1-thetta))**(-1/thetta) #steady state capital as the maximum element of grid for capital
size_kgrid=200 #size of grid
kgrid=np.linspace(0.05,kss,size_kgrid) #The grid is evenly spaced. If we had some guess about the curvature of value
                                       #function, we could concentrate more points there.
                                       
vguess=np.zeros(size_kgrid) #Initial guess for value function. It will allways be the same.

#This where the common steps for all the methods in 1.1. Now let's focus on specific steps.


#%%1.1.a  BRUTE FORCE ITERATION.
#NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.
start = time.time()

M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j]) 
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
        #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.
epsilon = 0.01 #tolerance
iteratemore = True
X = np.empty((size_kgrid, size_kgrid))
g = np.ones((size_kgrid, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((size_kgrid,1))
count=0
#iteration
while (iteratemore == True):
    count+=1 #to count iterations
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            X[i,j] = M[i,j] + beta*Vi[j]
    for i in range(size_kgrid):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:]) #store decision rules
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj) #store copy for the following iteration
        Vj = np.empty((size_kgrid,1)) #to then update
    else:
        iteratemore = False
        print("SUCCES")

#Policy functions:
gk=np.empty(size_kgrid)
gc=np.empty(size_kgrid)
for i in range(size_kgrid):
    gk[i]=kgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()

#Plot results

fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='green', linewidth=2)   
ax.set_title('Value function brute force method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='green', linewidth=2)   
ax.set_title('Policy function for capital brute force method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='green', linewidth=2)   
ax.set_title('Policy function for consumption brute force method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()

print('Brute force iterations took ' +str(count)+' iterations and '+str(end-start)+' seconds.')
#%%1.1.b.SPEEDING UP TAKING INTO ACCOUNT MONOTONICITY OF DECISION RULE:
 #NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.

start = time.time()

M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000 
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j])
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
        #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.
epsilon = 0.01 #tolerance
iteratemore = True
X = np.empty((size_kgrid, size_kgrid))
g = np.ones((size_kgrid, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((size_kgrid,1))
count=0
while (iteratemore == True):
    count+=1
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            if kgrid[j]>=kgrid[int(g[i])]:  #monotonicity
                X[i,j] = M[i,j] + beta*Vi[j]
            else:
                continue
    for i in range(size_kgrid):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:])
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj)
        Vj = np.empty((size_kgrid,1))
    else:
        iteratemore = False
        print("SUCCES")


   

#Policy functions:
gk=np.empty(size_kgrid)
gc=np.empty(size_kgrid)
for i in range(size_kgrid):
    gk[i]=kgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()



fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='green', linewidth=2)   
ax.set_title('Value function monotonicity method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='green', linewidth=2)   
ax.set_title('Policy function for capital monotonicity method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='green', linewidth=2)   
ax.set_title('Policy function for consumption monotonicity method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()

print('Iterations taking into account monotonicity of decision rule took ' +str(count)+' iterations and '+str(end-start)+' seconds.')
#%%1.1.c  ITERATIONS TAKING INTO ACCOUNT CONCAVITY OF VALUE FUNCTION.
#NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.

start = time.time()



M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j])
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
         #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.
epsilon = 0.01 #tolerance
iteratemore = True
X = np.empty((size_kgrid, size_kgrid))
g = np.ones((size_kgrid, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((size_kgrid,1))
count=0
while (iteratemore == True):
    count+=1
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            X[i,j] = M[i,j] + beta*Vi[j]
            if X[i,j]<X[i,j-1]:#concavity
                break        
    for i in range(size_kgrid):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:])
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj)
        Vj = np.empty((size_kgrid,1))
    else:
        iteratemore = False
        print("SUCCES")


   

#Policy functions:
gk=np.empty(size_kgrid)
gc=np.empty(size_kgrid)
for i in range(size_kgrid):
    gk[i]=kgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()



fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='green', linewidth=2)   
ax.set_title('Value function concavity method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='green', linewidth=2)   
ax.set_title('Policy function for capital concavity method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='green', linewidth=2)   
ax.set_title('Policy function for consumption concavity method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()

print('Iterations taking into account concavity of value function took ' +str(count)+' iterations and '+str(end-start)+' seconds.')

#%%1.1.d  ITERATIONS TAKING INTO ACCOUNT LOCAL SEARCH ON THE DECISION RULE
#NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.

start = time.time()

M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j]) 
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
        #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.
epsilon = 0.01 #tolerance
iteratemore = True
X = np.zeros((size_kgrid, size_kgrid))
g = np.ones((size_kgrid, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((size_kgrid,1))
count=0
while (iteratemore == True):
    count+=1
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            if (j>g[i]) and (j <= g[i] + 1): #small neighborhood
              X[i,j] = M[i,j] + beta*Vi[j]
    for i in range(size_kgrid):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:])
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj)
        Vj = np.empty((size_kgrid,1))
    else:
        iteratemore = False
        print("SUCCES")


   

#Policy functions:
gk=np.empty(size_kgrid)
gc=np.empty(size_kgrid)
for i in range(size_kgrid):
    gk[i]=kgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()



fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='green', linewidth=2)   
ax.set_title('Value function local search method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='green', linewidth=2)   
ax.set_title('Policy function for capital local search method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='green', linewidth=2)   
ax.set_title('Policy function for consumption local search method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()

print('Iterations taking into account local search on the decision rule took ' +str(count)+' iterations and '+str(end-start)+' seconds.')

#%%1.1.e.  ITERATIONS TAKING INTO ACCOUNT BOTH CONCAVITY OF VALUE FUNCTION AND MONOTONICITY OF THE DECISION RULE.
#NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.

start = time.time()



M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j])
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
         #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.
epsilon = 0.01 #tolerance
iteratemore = True
X = np.empty((size_kgrid, size_kgrid))
g = np.ones((size_kgrid, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((size_kgrid,1))
count=0
while (iteratemore == True):
    count+=1
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            if kgrid[j]>=kgrid[int(g[i])]: #monotonicity
                X[i,j] = M[i,j] + beta*Vi[j]
                if X[i,j]<X[i,j-1]: #concavity
                    break
            else:
                continue
    for i in range(size_kgrid):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:])
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj)
        Vj = np.empty((size_kgrid,1))
    else:
        iteratemore = False
        print("SUCCES")


   

#Policy functions:
gk=np.empty(size_kgrid)
gc=np.empty(size_kgrid)
for i in range(size_kgrid):
    gk[i]=kgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()



fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='green', linewidth=2)   
ax.set_title('Value function concavity + monotonicity method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='green', linewidth=2)   
ax.set_title('Policy function for capital concavity + monotonicity method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='green', linewidth=2)   
ax.set_title('Policy function for consumption concavity + monotonicity method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()

print('Iterations taking into account both concavity of value function and monotonicity of decision rule took ' +str(count)+' iterations and '+str(end-start)+' seconds.')

#%%1.1.f)Not able to do it
#%%1.1.g)Not able to do it

###############################################################################
           #####################################################
           #####################################################
#%%#############################################################################
#1.2. DISCRETE METHODS WITH ELASTIC LABOR SUPPLY h

#COMMON PART OF ITEM 2:



#Let me first of all define the parameters of the model.
import numpy as np
import time
import matplotlib.pyplot as pl
thetta=0.679
beta=0.988
delta=0.013
kappa=5.24
nu=2


#Discretize the state space.
kss=((1-thetta)/((1/beta)+delta-1)) #steady state capital as the maximum element of grid for capital
                                                 #Notice that it changes from item 1 as I justify in the PDF.
n=200
size_kgrid=n #size of grid
kgrid=np.linspace(0.05,kss,size_kgrid) #The grid is evenly spaced. If we had some guess about the curvature of value
                                      #function, we could concentrate more points there.
   
#Labor choice is not an state variable but we also discretize it. It is justified in the paper.

size_hgrid=n  
               
hgrid=np.linspace(0.01,1,size_hgrid)
                                   
vguess=np.zeros((n)) #Initial guess for value function. It will allways be the same.

#This where the common steps for all the methods in 1.2. Now let's focus on specific steps.



#%%1.2.a  BRUTE FORCE ITERATION.
#NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.
start = time.time()

M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k ,k' and h in the grids.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]-kgrid[j])-((kappa*(hgrid[j])**(1+(1/nu)))/(1+(1/nu))) 
      
        
    
epsilon = 0.01 #tolerance
iteratemore = True
X = np.empty((n,n))
g = np.ones((n, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((n,1))
count=0
while (iteratemore == True):
    count+=1
    for i in range(n):
        for j in range(n):
            X[i,j] = M[i,j] + beta*Vi[j]
    for i in range(n):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:])
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj)
        Vj = np.empty((n,1))
    else:
        iteratemore = False
        print("SUCCES")


   

#Policy functions:
gk=np.empty(n)
gh=np.empty(n)
gc=np.empty(n)

for i in range(n):
    gk[i]=kgrid[int(g[i])]
    gh[i]=hgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))*(hgrid[i]**(thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()



fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='blue', linewidth=2)   
ax.set_title('Value function brute force method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='blue', linewidth=2)   
ax.set_title('Policy function for capital brute force method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='blue', linewidth=2)   
ax.set_title('Policy function for consumption brute force method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gh, color='blue', linewidth=2)   
ax.set_title('Policy function for labor brute force method')
ax.set_ylabel('g_h(k)')
ax.set_xlabel('k')
pl.show()
print('Brute force iterations took ' +str(count)+' iterations and '+str(end-start)+' seconds.')

#%%1.2.b. SPEEDING UP TAKING INTO ACCOUNT MONOTONICITY OF DECISION RULE:
   #NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.
start = time.time()

M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k ,k' and h in the grids.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]-kgrid[j])-((kappa*(hgrid[j])**(1+(1/nu)))/(1+(1/nu))) 

        
    
epsilon = 0.01 #tolerance
iteratemore = True
X = np.empty((n,n))
g = np.ones((n, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((n,1))
count=0
while (iteratemore == True):
    count+=1
    for i in range(n):
        for j in range(n):
            if kgrid[j]>=kgrid[int(g[i])]:
              X[i,j] = M[i,j] + beta*Vi[j]
            else:
                continue
    for i in range(n):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:])
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj)
        Vj = np.empty((n,1))
    else:
        iteratemore = False
        print("SUCCES")


   

#Policy functions:
gk=np.empty(n)
gh=np.empty(n)
gc=np.empty(n)

for i in range(n):
    gk[i]=kgrid[int(g[i])]
    gh[i]=hgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))*(hgrid[i]**(thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()



fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='blue', linewidth=2)   
ax.set_title('Value function monotonicity method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='blue', linewidth=2)   
ax.set_title('Policy function for capital monotonicity force method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='blue', linewidth=2)   
ax.set_title('Policy function for consumption monotonicity method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gh, color='blue', linewidth=2)   
ax.set_title('Policy function for labor monotonicity method')
ax.set_ylabel('g_h(k)')
ax.set_xlabel('k')
pl.show()
print('Iterations taking into account monotonicity of decision rule took ' +str(count)+' iterations and '+str(end-start)+' seconds.')


#%%1.2.c SPEEDING UP TAKING INTO ACCOUNT CONCAVITY OF THE VALUE FUNCTION
#NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.
start = time.time()

M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k ,k' and h in the grids.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]-kgrid[j])-((kappa*(hgrid[j])**(1+(1/nu)))/(1+(1/nu))) 

        
    
epsilon = 0.01 #tolerance
iteratemore = True
X = np.empty((n,n))
g = np.ones((n, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((n,1))
count=0
while (iteratemore == True):
    count+=1
    for i in range(n):
        for j in range(n):
            X[i,j] = M[i,j] + beta*Vi[j]
            if X[i,j]<X[i,j-1]:
                break
    for i in range(n):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:])
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj)
        Vj = np.empty((n,1))
    else:
        iteratemore = False
        print("SUCCES")


   

#Policy functions:
gk=np.empty(n)
gh=np.empty(n)
gc=np.empty(n)

for i in range(n):
    gk[i]=kgrid[int(g[i])]
    gh[i]=hgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))*(hgrid[i]**(thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()



fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='blue', linewidth=2)   
ax.set_title('Value function concavity method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='blue', linewidth=2)   
ax.set_title('Policy function for capital concavity method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='blue', linewidth=2)   
ax.set_title('Policy function for consumption concavity method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gh, color='blue', linewidth=2)   
ax.set_title('Policy function for labor concavity method')
ax.set_ylabel('g_h(k)')
ax.set_xlabel('k')
pl.show()
print('Iterations taking into account concavity of the value function took ' +str(count)+' iterations and '+str(end-start)+' seconds.')

#%%1.2.d  SPEEDING UP TAKING INTO ACCOUNT LOCAL SEARCH ON THE DECISION RULE:
#NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.
start = time.time()

M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k ,k' and h in the grids.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]-kgrid[j])-((kappa*(hgrid[j])**(1+(1/nu)))/(1+(1/nu))) 

        
    
epsilon = 0.01 #tolerance
iteratemore = True
X = np.empty((n,n))
g = np.ones((n, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((n,1))
count=0
while (iteratemore == True):
    count+=1
    for i in range(n):
        for j in range(n):
            if (j>g[i]) and (j <= g[i] + 1):
              X[i,j] = M[i,j] + beta*Vi[j]
    for i in range(n):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:])
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj)
        Vj = np.empty((n,1))
    else:
        iteratemore = False
        print("SUCCES")


   

#Policy functions:
gk=np.empty(n)
gh=np.empty(n)
gc=np.empty(n)

for i in range(n):
    gk[i]=kgrid[int(g[i])]
    gh[i]=hgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))*(hgrid[i]**(thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()



fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='blue', linewidth=2)   
ax.set_title('Value function local search method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='blue', linewidth=2)   
ax.set_title('Policy function for capital local search method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='blue', linewidth=2)   
ax.set_title('Policy function for consumption local search method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gh, color='blue', linewidth=2)   
ax.set_title('Policy function for labor local search method')
ax.set_ylabel('g_h(k)')
ax.set_xlabel('k')
pl.show()
print('Iterations taking into account local search on the decision rule took ' +str(count)+' iterations and '+str(end-start)+' seconds.')


#%%1.2.e. SPEEDING UP TAKING INTO ACCOUNT  BOTH CONCAVITY OF THE VALUE FUNCTION AND MONOTONICITY OF THE DECISION RULE.
#NOTE: If you see that it solves too fast, please run it again. Sorry for the inconvenience.
start = time.time()

M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k ,k' and h in the grids.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log(((kgrid[i])**(1-thetta))*((hgrid[j])**(thetta))+(1-delta)*kgrid[i]-kgrid[j])-((kappa*(hgrid[j])**(1+(1/nu)))/(1+(1/nu))) 

        
    
epsilon = 0.01 #tolerance
iteratemore = True
X = np.empty((n,n))
g = np.ones((n, 1))
Vi = vguess   # initial value function guess
Vj = np.empty((n,1))
count=0
while (iteratemore == True):
    count+=1
    for i in range(n):
        for j in range(n):
            if kgrid[j]>=kgrid[int(g[i])]:
              X[i,j] = M[i,j] + beta*Vi[j]
              if X[i,j]<X[i,j-1]:
                break
            else:
                continue
    for i in range(n):
        Vj[i] = np.max(X[i,:])
        g[i] = np.argmax(X[i,:])
    if np.max(np.abs(Vj - Vi)) >= epsilon:
        Vi = np.copy(Vj)
        Vj = np.empty((n,1))
    else:
        iteratemore = False
        print("SUCCES")


   

#Policy functions:
gk=np.empty(n)
gh=np.empty(n)
gc=np.empty(n)

for i in range(n):
    gk[i]=kgrid[int(g[i])]
    gh[i]=hgrid[int(g[i])]
    gc[i]=(kgrid[i]**(1-thetta))*(hgrid[i]**(thetta))+(1-delta)*kgrid[i]-gk[i]
    
end = time.time()



fig,ax = pl.subplots()    
ax.plot(kgrid, Vj, color='blue', linewidth=2)   
ax.set_title('Value function concavity + monotonicity method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gk, color='blue', linewidth=2)   
ax.set_title('Policy function for capital concavity + monotonicity method')
ax.set_ylabel('g_k´(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gc, color='blue', linewidth=2)   
ax.set_title('Policy function for consumption concavity + monotonicity method')
ax.set_ylabel('g_c(k)')
ax.set_xlabel('k')
pl.show()


fig,ax = pl.subplots()    
ax.plot(kgrid, gh, color='blue', linewidth=2)   
ax.set_title('Policy function for labor concavity + monotonicity method')
ax.set_ylabel('g_h(k)')
ax.set_xlabel('k')
pl.show()
print('Iterations taking into account both concavity of the value function and monotonicity of the decsion rule took ' +str(count)+' iterations and '+str(end-start)+' seconds.')

#%%1.2.f) Not able to do it
#%%1.2.g) Not able to do it
###############################################################################
               ################################################
###############################################################################
               ################################################
##############################################################################
#%% 1.3.CONTINUOUS METHODS TO APROXIMATE VALUE FUNCTION-CHEBYCHEV ALGORITHM.

#COMMON PART OF ITEM 3


import numpy as np
import matplotlib.pyplot as pl
import time


# Parameters definition

thetta = 0.679 
beta = 0.988 
delta = 0.013 

#With a continuous method the size of the grid can be much smaller, for example, the half than in discrete methods.
size_kgrid = 100 

# Discretize the state space.
kss = (((1/beta)+delta-1)/(1-thetta))**(-1/thetta) # Capital at steady state as upper bound for k grid
kgrid = np.linspace(0.05, kss, size_kgrid) 

#The following steps follow Makoto Nkajima notes:

# Firstly, set the order of polynomials used to  approximate.
n = 8

# Now, set the number of collocation points.
m = size_kgrid
m2 = 2*m

# After that, set a tolerance to error parameter.
epsilon = 0.01 


# Next,compute the collocation points.Chebychev polinomials are defined in the interval [-1,1]. 
#With the following transformation  we convert this in terval in which we want.
#Roots
roots = np.zeros(m)
for i in range(m):
    roots[i] = np.cos((((2*i)-1)/m2)*np.pi)


#Conversion
points = np.zeros(m)
for i in range(m):
    points[i] = (roots[i] + 1)*(kss - 0.05)/2 + 0.05
    


# After that, define an initial guess for the value function y_0. As in previous cases I define a vector of 0.
y_0 = np.zeros((m,1))

#Now use the Chebyshev regression method to get the guess for the coefficients
coefguess = np.polynomial.chebyshev.chebfit(points,y_0,n)

# Next step, value function guess
vguess = np.polynomial.chebyshev.chebval(kgrid,coefguess)
vguess = np.reshape(vguess, (m,1))

#%%1.3.a BRUTE FORCE ITERATIONS
#Compute the matrix M ecactly as in previous methods.

start = time.time()                   
            
M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j]) 
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
        #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.         
            
            
            
#Now, X computation and iterations. I use a function to contain all the steps.

def Chebychev(M,vguess,coefguess, y_0):
    
    X = np.empty((size_kgrid,size_kgrid)) 
    g = np.empty((size_kgrid,1)) 
    y_1 = np.empty((size_kgrid,1)) 
    #Fill X
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            X[i,j] = M[i,j] + beta*vguess[j]
            
    #Updated value func.
    for i in range(size_kgrid):
        g[i] = np.argmax(X[i,:])
        
    for i in range(size_kgrid):     
        y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*vguess[np.int_(g[i])]
        
    
    # Next, update theguess for the coefficients
    
    y_1 = np.reshape(y_1, (m,))
    upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
    V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
    V_1 = np.reshape(V_1, (m,1))
        
    #And compare the consecutive groups of parameters, repeating the algotithm until
    #the difference is always smaller or equal than epsilon, i.e, the tolerance 
    #level that I have imposed is satisfied.
    
    count = 0                                
    while np.any(abs(upd_coef - coefguess) > epsilon): 
        count+=1
        coefguess = upd_coef.copy()                             
        for i in range(size_kgrid):
            for j in range(size_kgrid):         
                X[i,j] = M[i,j] + beta*V_1[j]
    
        for i in range(size_kgrid):
            g[i] = np.argmax(X[i,:])
        
        for i in range(size_kgrid):     
            y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*V_1[np.int_(g[i])]
        
        y_1 = np.reshape(y_1, (m,))
        upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
        V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
        V_1 = np.reshape(V_1, (m,1))
        
    return y_1,count

#I call the function to run the iterations

VF_Chebychev,count = Chebychev(M,vguess,coefguess, y_0)

end = time.time()

#Plot the results

fig,ax=pl.subplots()
ax.plot(kgrid, VF_Chebychev,color='purple', linewidth=2)   
ax.set_title('Value Function Chebyshev algorithm by brute force method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()

print('Brute force iterations with Chebychev regression algorithm took ' +str(count)+' iterations and '+str(end-start)+' seconds.')

#%%1.3.b. MONOTONICITY METHOD


#Compute the matrix M ecactly as in previous methods.

start = time.time()                   
            
M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j]) 
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
        #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.         
            
            
            
#Now, X computation and iterations. I use a function to contain all the steps.

def Chebychev(M,vguess,coefguess, y_0):
    
    X = np.empty((size_kgrid,size_kgrid)) 
    g = np.empty((size_kgrid,1)) 
    y_1 = np.empty((size_kgrid,1)) 
    #Fill X
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            X[i,j] = M[i,j] + beta*vguess[j]
            
    #Updated value func.
    for i in range(size_kgrid):
        g[i] = np.argmax(X[i,:])
        
    for i in range(size_kgrid):     
        y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*vguess[np.int_(g[i])]
        
    
    # Next, update theguess for the coefficients
    
    y_1 = np.reshape(y_1, (m,))
    upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
    V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
    V_1 = np.reshape(V_1, (m,1))
        
    #And compare the consecutive groups of parameters, repeating the algotithm until
    #the difference is always smaller or equal than epsilon, i.e, the tolerance 
    #level that I have imposed is satisfied.
    
    count = 0                                
    while np.any(abs(upd_coef - coefguess) > epsilon): 
        count+=1
        coefguess = upd_coef.copy()                             
        for i in range(size_kgrid):
            for j in range(size_kgrid):  
                if kgrid[j]>=kgrid[int(g[i])]: #monotonicity
                   X[i,j] = M[i,j] + beta*V_1[j]
                else:
                     continue
    
        for i in range(size_kgrid):
            g[i] = np.argmax(X[i,:])
        
        for i in range(size_kgrid):     
            y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*V_1[np.int_(g[i])]
        
        y_1 = np.reshape(y_1, (m,))
        upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
        V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
        V_1 = np.reshape(V_1, (m,1))
        
    return y_1,count

#I call the function to run the iterations

VF_Chebychev,count = Chebychev(M,vguess,coefguess, y_0)

end = time.time()

#Plot the results

fig,ax=pl.subplots()
ax.plot(kgrid, VF_Chebychev,color='purple', linewidth=2)   
ax.set_title('Value Function Chebyshev algorithm by monotonicity method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()

print('Iterations with Chebychev regression algorithm taking into account monotonicity of the optimal decision rule took ' +str(count)+' iterations and '+str(end-start)+' seconds.')
#%%1.3.c CONCAVITY METHOD
#Compute the matrix M ecactly as in previous methods.

start = time.time()                   
            
M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j]) 
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
        #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.         
            
            
            
#Now, X computation and iterations. I use a function to contain all the steps.

def Chebychev(M,vguess,coefguess, y_0):
    
    X = np.empty((size_kgrid,size_kgrid)) 
    g = np.empty((size_kgrid,1)) 
    y_1 = np.empty((size_kgrid,1)) 
    #Fill X
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            X[i,j] = M[i,j] + beta*vguess[j]
            
    #Updated value func.
    for i in range(size_kgrid):
        g[i] = np.argmax(X[i,:])
        
    for i in range(size_kgrid):     
        y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*vguess[np.int_(g[i])]
        
    
    # Next, update theguess for the coefficients
    
    y_1 = np.reshape(y_1, (m,))
    upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
    V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
    V_1 = np.reshape(V_1, (m,1))
        
    #And compare the consecutive groups of parameters, repeating the algotithm until
    #the difference is always smaller or equal than epsilon, i.e, the tolerance 
    #level that I have imposed is satisfied.
    
    count = 0                                
    while np.any(abs(upd_coef - coefguess) > epsilon): 
        count+=1
        coefguess = upd_coef.copy()                             
        for i in range(size_kgrid):
            for j in range(size_kgrid):         
                X[i,j] = M[i,j] + beta*V_1[j]
                if X[i,j]<X[i,j-1]:#concavity
                    break
    
        for i in range(size_kgrid):
            g[i] = np.argmax(X[i,:])
        
        for i in range(size_kgrid):     
            y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*V_1[np.int_(g[i])]
        
        y_1 = np.reshape(y_1, (m,))
        upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
        V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
        V_1 = np.reshape(V_1, (m,1))
        
    return y_1,count

#I call the function to run the iterations

VF_Chebychev,count = Chebychev(M,vguess,coefguess, y_0)

end = time.time()

#Plot the results

fig,ax=pl.subplots()
ax.plot(kgrid, VF_Chebychev,color='purple', linewidth=2)   
ax.set_title('Value Function Chebyshev algorithm by concavity method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()

print('Iterations with Chebychev regression algorithm taking into account concavity of the value function took ' +str(count)+' iterations and '+str(end-start)+' seconds.')



#%%1.3.d LOCAL SEARCH ON THE DECISION RULE METHOD
#Compute the matrix M ecactly as in previous methods.

start = time.time()                   
            
M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j]) 
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
        #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.         
            
            
            
#Now, X computation and iterations. I use a function to contain all the steps.

def Chebychev(M,vguess,coefguess, y_0):
    
    X = np.empty((size_kgrid,size_kgrid)) 
    g = np.empty((size_kgrid,1)) 
    y_1 = np.empty((size_kgrid,1)) 
    #Fill X
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            if (j>g[i]) and (j<=g[i]+4): #small neighborhood
               X[i,j] = M[i,j] + beta*vguess[j]
            
    #Updated value func.
    for i in range(size_kgrid):
        g[i] = np.argmax(X[i,:])
        
    for i in range(size_kgrid):     
        y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*vguess[np.int_(g[i])]
        
    
    # Next, update theguess for the coefficients
    
    y_1 = np.reshape(y_1, (m,))
    upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
    V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
    V_1 = np.reshape(V_1, (m,1))
        
    #And compare the consecutive groups of parameters, repeating the algotithm until
    #the difference is always smaller or equal than epsilon, i.e, the tolerance 
    #level that I have imposed is satisfied.
    
    count = 0                                
    while np.any(abs(upd_coef - coefguess) > epsilon): 
        count+=1
        coefguess = upd_coef.copy()                             
        for i in range(size_kgrid):
            for j in range(size_kgrid):         
                X[i,j] = M[i,j] + beta*V_1[j]
    
        for i in range(size_kgrid):
            g[i] = np.argmax(X[i,:])
        
        for i in range(size_kgrid):     
            y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*V_1[np.int_(g[i])]
        
        y_1 = np.reshape(y_1, (m,))
        upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
        V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
        V_1 = np.reshape(V_1, (m,1))
        
    return y_1,count

#I call the function to run the iterations

VF_Chebychev,count = Chebychev(M,vguess,coefguess, y_0)

end = time.time()

#Plot the results

fig,ax=pl.subplots()
ax.plot(kgrid, VF_Chebychev,color='purple', linewidth=2)   
ax.set_title('Value Function Chebyshev algorithm by local search method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()

print('Iterations with Chebychev regression algorithm taking into account local search on the decision rule took ' +str(count)+' iterations and '+str(end-start)+' seconds.')

#%%1.3.e MONOTONICITY+CONCAVITY METHOD.
#Compute the matrix M ecactly as in previous methods.

start = time.time()                   
            
M=np.zeros((size_kgrid,size_kgrid))
#Fill M with the utility derived from all the possible combinations of k and k' in the kgrid.
for i in range (0,size_kgrid):
    for j in range (0,size_kgrid):
       #Since consumption is restricted to be positive  I set all the combinations that give negative consumption to be
        #a very negative value of utility so that they don't generate confusuion anaymore. In addition we are restricted
        #by the fact that we have to take the ln of consumption, so that actually we need combinations giving a consumption
        #values >= 0 in order to ln be defined.
        
        if kgrid[j]>=((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]): 
            M[i,j]=-500000
        else: #For the rest of cases I compute utility.
            M[i,j]=np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-kgrid[j]) 
        #Notice that as I explain and justify in  the PDF, I ommit the constant disutility of labor in this item 1.
        #Note that some of the elements in M are negative. This is not an issue because utility is  just an ordinal measure
        #of welfare. Cardinality should not matter.         
            
            
            
#Now, X computation and iterations. I use a function to contain all the steps.

def Chebychev(M,vguess,coefguess, y_0):
    
    X = np.empty((size_kgrid,size_kgrid)) 
    g = np.empty((size_kgrid,1)) 
    y_1 = np.empty((size_kgrid,1)) 
    #Fill X
    for i in range(size_kgrid):
        for j in range(size_kgrid):
            X[i,j] = M[i,j] + beta*vguess[j]
            
    #Updated value func.
    for i in range(size_kgrid):
        g[i] = np.argmax(X[i,:])
        
    for i in range(size_kgrid):     
        y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*vguess[np.int_(g[i])]
        
    
    # Next, update theguess for the coefficients
    
    y_1 = np.reshape(y_1, (m,))
    upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
    V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
    V_1 = np.reshape(V_1, (m,1))
        
    #And compare the consecutive groups of parameters, repeating the algotithm until
    #the difference is always smaller or equal than epsilon, i.e, the tolerance 
    #level that I have imposed is satisfied.
    
    count = 0                                
    while np.any(abs(upd_coef - coefguess) > epsilon): 
        count+=1
        coefguess = upd_coef.copy()                             
        for i in range(size_kgrid):
            for j in range(size_kgrid):    
                if kgrid[j]>=kgrid[int(g[i])]:#monotonicity
                    X[i,j] = M[i,j] + beta*V_1[j]
                    if X[i,j]<X[i,j-1]:#concavity
                       break
                else:
                    continue
        for i in range(size_kgrid):
            g[i] = np.argmax(X[i,:])
        
        for i in range(size_kgrid):     
            y_1[i] = np.log((kgrid[i])**(1-thetta)+(1-delta)*kgrid[i]-g[i])+beta*V_1[np.int_(g[i])]
        
        y_1 = np.reshape(y_1, (m,))
        upd_coef = np.polynomial.chebyshev.chebfit(points,y_1,n)
        V_1 = np.polynomial.chebyshev.chebval(kgrid,upd_coef)
        V_1 = np.reshape(V_1, (m,1))
        
    return y_1,count

#I call the function to run the iterations

VF_Chebychev,count = Chebychev(M,vguess,coefguess, y_0)

end = time.time()

#Plot the results

fig,ax=pl.subplots()
ax.plot(kgrid, VF_Chebychev,color='purple', linewidth=2)   
ax.set_title('Value Function Chebyshev algorithm by concavity + monotonicity method')
ax.set_ylabel('V(k)')
ax.set_xlabel('k')
pl.show()

print('Iterations with Chebychev regression algorithm taking into account both concavity of the value function and monotonicity of the decision rule took ' +str(count)+' iterations and '+str(end-start)+' seconds.')
