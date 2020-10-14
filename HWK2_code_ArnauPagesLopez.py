# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 08:18:24 2020

@author: ARNAU PAGÈS LÓPEZ

"""
#QUANTITATIVE MACROECONOMICS-HOMEWORK 2
#ARNAU PAGÈS LÓPEZ

#EXERCISE 1:
    
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

#A)

#First of all let me define the kown parameters of the model

tetta=0.67   #labor share in production function
h=0.31      #For all t, inelastic labor suply for the moment
#And I define the value of z that I found analitically.
z=1.63

#Now, According to the analytical solutions that I obtained, and the information
#that Raül gives us, at steady state the following 5 conditions must hold:
#1=beta*((1-tetta)*(kss**(-tetta))*((z*h)**tetta)+(1-delta))                   EQ(9) FROM ANSWERS SHEET-EULER EQUATION
#css+delta*kss=yss                                                             EQ(2) FROM ANSWERS SHEET-FEASIBILITY CONSTRAINT
#yss=(kss**(1-tetta))*((z*h)**tetta)                                           EQ(3) FROM ANSWERS SHEET-PRODUCTION FUNCTION
#kss/yss=4                                                                     CAPITAL/OUTPUT RATIO
#iss/yss=0.25, which at steady state is equivalent to: ((delta*kss)/yss)=0.25  INVESTMENT/OUTPUT RATIO

#Hence we have a system of 5 non-linear equations and 5 unkowns: kss,css,yss,beta,delta
#I will use fsolve to solve this.
def steadystate(vars):
    #Note that equations must be written, such that Python understands that we're looking for roots.
    #Hence, I must let the zero on the RHS.
    kss,css,yss,beta,delta=vars
    eq1=1-beta*((1-tetta)*(kss**(-tetta))*((z*h)**tetta)+(1-delta))
    eq2=css+delta*kss-yss
    eq3=yss-(kss**(1-tetta))*((z*h)**tetta)
    eq4=(kss/yss)-4
    eq5=((delta*kss)/yss)-0.25
    return [eq1, eq2, eq3, eq4, eq5]

kss,css,yss,beta,delta=fsolve(steadystate,(1,1,1,1,1)) #initial values
#I finally solve for the implicit variable iss
iss=delta*kss    
#And store my results in a dictionary to make observation easier
Sstate1={"kss":kss,"css":css,"yss":yss,"beta":beta,"delta":delta,"iss":iss}
print(Sstate1)
#As we can check values are very closed to those obtained analiticaly, so it worked well.


#B)

#Now we 've the same parameters but economy recieves a positive productivity shock that doubles z:
z2=2*z
#I'm going to use excatly the same procedure as in part A), but with the new z, to solve for the new steady state:
def steadystate2(vars):
    kss2,css2,yss2,beta2,delta2=vars
    eq1=1-beta2*((1-tetta)*(kss2**(-tetta))*((z2*h)**tetta)+(1-delta2))
    eq2=css2+delta2*kss2-yss2
    eq3=yss2-(kss2**(1-tetta))*((z2*h)**tetta)
    eq4=(kss2/yss2)-4
    eq5=((delta2*kss2)/yss2)-0.25
    return [eq1, eq2, eq3, eq4, eq5]    

kss2,css2,yss2,beta2,delta2=fsolve(steadystate2,(1,1,1,1,1))
iss2=delta2*kss2
Sstate2={"kss2":kss2,"css2":css2,"yss2":yss2,"beta2":beta2,"delta2":delta2,"iss2":iss2}
print(Sstate2)


#C)

#First, let me define two simple functions that will make the following steps easier to type.

#The first one is the first derivative of our log-utility function:

def fdu(c):
    return 1/c
#And the second one is the production function:
def y(k,z):
    return k**(1-tetta)*(z*h)**tetta

#And now, what I do is to use the Euler equation rewritten as a function of only capital as I derive in answers sheet. 
#I set up an "initial" Euler equation, a final Euler equation and an Euler equation for all periods between.
#The solution for the following problem of non linear equations is a sequence of capitals such that Euler eq.
#holds every period. With a sufficiently large number of simulations we should see how economy approaches to steady state.

n=149   #number of periods I siumlate


def transition(k, n=n):
    k_0=kss
    k_final=kss2
    k[0]=kss #Initial condition
    k[n-1]=kss2 #Final condition
    k_trans=np.zeros(n)
    for i in range(0,n-2):
        if i==0:
            k_trans[i+1]=fdu(y(k_0,z2)+(1-delta)*k_0-k[i+1])-beta*fdu(y(k[i+1],z2)+(1-delta)*k[i+1]-k[i+2])*(1-delta+(1-tetta)*(1/(k[i+1])**(tetta))*((y(k_0,z2))/(k_0**(1-tetta))))      # In the last term I just rewritted it in a different way bekause I realized that it maked computation easier.                           
        elif i==(n-2):
            k_trans[i+1]=fdu(y(k[i],z2)+(1-delta)*k[i]-k[i+1])-beta*fdu(y(k[i+1],z2)+(1-delta)*k[i+1]-k_final)*(1-delta+(1-tetta)*(1/(k[i+1])**(tetta))*((y(k[i],z2))/(k[i]**(1-tetta))))    # In the last term I just rewritted it in a different way bekause I realized that it maked computation easier.                                                   
        else:
            k_trans[i+1]=fdu(y(k[i],z2)+(1-delta)*k[i]-k[i+1])-beta*fdu(y(k[i+1],z2)+(1-delta)*k[i+1]-k[i+2])*(1-delta+(1-tetta)*(1/(k[i+1])**(tetta))*((y(k[i],z2))/(k[i]**(1-tetta))))     # In the last term I just rewritted it in a different way bekause I realized that it maked computation easier.

    return(k_trans)
x0=np.linspace(4,8,n) #Initial values. I choosed them in a manner such that they are not too far of what I guess that will be the solution.
trans_pathk=fsolve(transition,x0)  #This is the transition path for capital
#Once I get the transition path for capital, it should be more or less straightforward to solve for
#transition paths of the rest of variables:
trans_pathy=y(trans_pathk,z2) #Transition path output

#Transition path for savings:
trans_paths=np.zeros(n)

for i in range(0,n-1):
        trans_paths[i]=trans_pathk[i+1]-(1-delta)*trans_pathk[i]

trans_paths[n-1]=trans_paths[n-2]
#Transition path for consumption.
trans_pathcons=trans_pathy-trans_paths
#As labor supply is inelastic, transition path for labor is straightforward.  
trans_pathlabor=np.ones(n)*h
                        

#Now let me add the steady state observations at the begining of each transition path, just 
#to make my plots more informative.
trans_pathk=np.insert(trans_pathk,0,kss)
trans_pathy=np.insert(trans_pathy,0,yss)
trans_paths=np.insert(trans_paths,0,iss)  #iss because investment equals savings in this models.
trans_pathcons=np.insert(trans_pathcons,0,css)
trans_pathlabor=np.insert(trans_pathlabor,0,h)

#Create time vector
time=np.array(list(range(0,(n+1))))


#And finally plot the figures:
fig,ax = plt.subplots()    
ax.plot(time, trans_pathk,'.', color='green', linewidth=2)   
ax.set_title('Transition path for capital')
ax.set_ylabel('Capital stock')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, trans_paths,'.', color='green', linewidth=2)   
ax.set_title('Transition path for savings')
ax.set_ylabel('Savings')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, trans_pathcons,'.', color='green', linewidth=2)     
ax.set_title('Transition path for consumption')
ax.set_ylabel('Consumption')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, trans_pathlabor, 'g-', linewidth=2)   
ax.set_title('Transition path for labor')
ax.set_ylabel('Labor supply')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, trans_pathy,'.', color='green', linewidth=2)   
ax.set_title('Transition path for output')
ax.set_ylabel('Output')
ax.set_xlabel('Time')
plt.show()




#D)
#Here the procedure is very similar to what I did in previous questionbut with some changes.
#What I do is to repeat the same I did before but now with the capital in period 9  of the transition(which if we count period 0
#is, indeed, the period 10) as initial value, and the first steady state as final value. I compute the sequence of 
#capital that solves for it and then in the first positions of that sequence I add the first 10 periods of the previous 
#transition of part c)


n2=140


def secondtransition(k, n2=n2):
    k_0=trans_pathk[9]
    k_final=kss
    k[0]=trans_pathk[9]
    k[n2-1]=kss
    k_2ndtrans=np.zeros(n2)
    for i in range(0,n2-2):
        if i==0:
            k_2ndtrans[i+1]=fdu(y(k_0,z)+(1-delta)*k_0-k[i+1])-beta*fdu(y(k[i+1],z)+(1-delta)*k[i+1]-k[i+2])*(1-delta+(1-tetta)*(1/(k[i+1])**(tetta))*((y(k_0,z))/(k_0**(1-tetta))))              # In the last term I just rewritted it in a different way bekause I realized that it maked computation easier.            
        elif i==(n2-2):
            k_2ndtrans[i+1]=fdu(y(k[i],z)+(1-delta)*k[i]-k[i+1])-beta*fdu(y(k[i+1],z)+(1-delta)*k[i+1]-k_final)*(1-delta+(1-tetta)*(1/(k[i+1])**(tetta))*((y(k[i],z))/(k[i]**(1-tetta))))       # In the last term I just rewritted it in a different way bekause I realized that it maked computation easier.                                        
        else:
            k_2ndtrans[i+1]=fdu(y(k[i],z)+(1-delta)*k[i]-k[i+1])-beta*fdu(y(k[i+1],z)+(1-delta)*k[i+1]-k[i+2])*(1-delta+(1-tetta)*(1/(k[i+1])**(tetta))*((y(k[i],z))/(k[i]**(1-tetta))))     # In the last term I just rewritted it in a different way bekause I realized that it maked computation easier.
            
    return(k_2ndtrans)
x02=np.linspace(6,4,n2)
trans_pathk2=fsolve(secondtransition,x02)

#Once I have the transition path for capital I solve for the rest of transition paths.
trans_pathy2=y(trans_pathk2,z) #output

trans_paths2=np.zeros(n2)

for i in range(0,n2-1):
        trans_paths2[i]=trans_pathk2[i+1]-(1-delta)*trans_pathk2[i]

trans_paths2[n2-1]=trans_paths2[n2-2] #savings

trans_pathcons2=trans_pathy2-trans_paths2 #consumption

trans_pathlabor2=np.ones(n2)*h #labor


#Finally, add periods 0 to 9 of part c) vectors to get the complete transition dynamics:
trans_pathk2=np.concatenate((trans_pathk[0:10],trans_pathk2))   
trans_pathy2=np.concatenate((trans_pathy[0:10],trans_pathy2)) 
trans_paths2=np.concatenate((trans_paths[0:10],trans_paths2)) 
trans_pathcons2=np.concatenate((trans_pathcons[0:10],trans_pathcons2)) 
trans_pathlabor2=np.concatenate((trans_pathlabor[0:10],trans_pathlabor2))



#And plot results:

fig,ax = plt.subplots()    
ax.plot(time, trans_pathk2,'.', color='green', linewidth=2)   
ax.set_title('New transition path for capital')
ax.set_ylabel('Capital stock')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, trans_paths2,'.', color='green', linewidth=2)   
ax.set_title('New transition path for savings')
ax.set_ylabel('Savings')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, trans_pathcons2,'.', color='green', linewidth=2)     
ax.set_title('New transition path for consumption')
ax.set_ylabel('Consumption')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, trans_pathlabor2, 'g-', linewidth=2)   
ax.set_title('New transition path for labor')
ax.set_ylabel('Labor supply')
ax.set_xlabel('Time')
plt.show()


fig,ax = plt.subplots()    
ax.plot(time, trans_pathy2,'.', color='green', linewidth=2)   
ax.set_title('New transition path for output')
ax.set_ylabel('Output')
ax.set_xlabel('Time')
plt.show()

#E)
#I was not able to do this part.

#%%
##############################################################################
##############################################################################

#EXERCISE 2:
    

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize      
import seaborn as sns

#A)

#The first thing to do is obviously to define all the parameters that the exercise is giving to us: 


Af=1
Anf=1
rho=1.1
kf=0.2
knf=0.2
omega=20
gamma=0.9
i_0=0.2
N=1

# Now I use lambda function of Python to create my objjective, with corresponds to onjective (29)in my answers sheet.
#Notice, I set c(TW) and beta(HC) to be points in a grid that I will define after, because obviously I cannot
#proceed for all beta and c between 0 and 1 since there are infinite values.
    
func =lambda h:-1* (Af*h[0]**((rho-1)/rho)+gr[j]*Anf*h[1]**((rho-1)/rho))**(rho/(rho-1)) - kf*h[0]-knf*h[1] - omega*((1-gamma)*gr[i]*(i_0*h[0]**2/N)) 

# Here I set the constraint:

constr= ({'type':'ineq','fun': lambda h: N - h[0] - h[1] })

#Now I generate the above mentioned grid, where n are the points to evaluate, hence I'm taking 
#discrete points to simulate a continuous evaluation as I also explained above.

n=10                                

gr = np.linspace(0,1,n) 
          
#I create now two arrays of zeros that will be storing the results for Hf and Hnf.
Hf = np.zeros(shape=(n,n))     

Hnf = np.zeros(shape=(n,n))     

#And I proceed now to the optimization. Actually we need many optimizations, so the best is to run loops.
#A duble loop. Fore each time the outer runs, the inner loop completes all its cycle. This is the way
#to ensure that I'm analyzing all the combinations of betas and c in the grid.
for i in range(n):
        for j in range(n):
            h_0 = [0.5,0.5]         # Initial values that minimize requires
            bou = [(0,1),(0,1)]     #Also need to provide bounds  for the optimal values. In this case the bounds are clear from exercise statement. Beta and c must lie in [0,1]
            #Finally just optimize using minimize and store the results.
            solu = minimize(func,h_0,constraints=constr,bounds=bou)      
            Hf[i][j] = solu.x[0]                      
            Hnf[i][j] =solu.x[1]                      
            
# Hence we already have the two key variables optimal values defined for a continuum of betas and c.
#So the remaining job is to compute the optimal values rest of variables that we are required, using 
#previous results.


# H is straightforward

H = Hf+Hnf


## Hf/H ratio is easy as well

Hf_Hratio = Hf/H


# Output may be a little bit trickyer because we need to use the values of c and beta in the grid, but
#a double loop again solves it.

Oup = np.zeros(shape=(n,n))    #To store results for output

for i in range (n):
    for j in range(n):
       Oup[i][j]= (Af*Hf[i][j]**((rho-1)/rho)+gr[j]*Anf*Hnf[i][j]**((rho-1)/rho))**(rho/(rho-1))





# Amount of infections also involves some computation with the c and beta by expressions (20) (21) (22)
#in answers sheet

I = np.zeros(shape=(n,n))    
for i in range(n):
    for j in range(n):
        I[i][j] = Hf[i][j]**2*10*gr[i]

#Deaths

Deaths  = (1-gamma)*I

#Welfare
#Notice that once we have all the other variables defined welfare is straightforward to compute,
#by using welfare function in (27) of answers sheet:
    



# welfare:
    
W = np.zeros(shape=(n,n))    
    
for i in range(n):
    for j in range(n):
        W[i][j] = (Af*Hf[i][j]**((rho-1)/rho)+gr[j]*Anf*Hnf[i][j]**((rho-1)/rho))**(rho/(rho-1)) - kf*Hf[i][j]-knf*Hnf[i][j] - omega*((1-gamma)*gr[i]*(i_0*Hf[i][j]**2/N))




#And finally I plot the results using heatmaps

space = np.around(np.linspace(0,0.9,10),decimals=1)  # Determine the values of the axis
    
size = np.around(np.linspace(0+int(n/20),n-int(n/20),10),decimals=0) 

fig, ax = plt.subplots()
sns.heatmap(H,cbar_kws={"label":"$H$ value"},xticklabels =space,yticklabels=space,vmin=0,vmax=1)
plt.title("$H$",fontsize=20)
ax.set_xticks(size)
ax.set_yticks(size)
plt.xlabel("$c(TW)$")
plt.ylabel("$β(HC)$")

fig, ax = plt.subplots()
sns.heatmap(Hf,cbar_kws={"label":"$H_f$ value"},xticklabels =space,yticklabels=space)
plt.title("$H_f$",fontsize=20)
ax.set_xticks(size)
ax.set_yticks(size)
plt.xlabel("$c(TW)$")
plt.ylabel("$β(HC)$")

fig, ax = plt.subplots()
sns.heatmap(Hnf,cbar_kws={"label":"$H_{nf}$ value"},xticklabels =space,yticklabels=space)
plt.title("$H_{nf}$",fontsize=20)
ax.set_xticks(size)
ax.set_yticks(size)
plt.xlabel("$c(TW)$")
plt.ylabel("$β(HC)$")



fig, ax = plt.subplots()
sns.heatmap(Hf_Hratio,cbar_kws={"label":"$H_f/H$ value"},xticklabels =space,yticklabels=space)
plt.title("$H_f/H$",fontsize=20)
ax.set_xticks(size)
ax.set_yticks(size)
plt.xlabel("$c(TW)$")
plt.ylabel("$β(HC)$")

fig, ax = plt.subplots()
sns.heatmap(Oup,cbar_kws={"label":"Output value"},xticklabels =space,yticklabels=space)
plt.title("Output",fontsize=20)
ax.set_xticks(size)
ax.set_yticks(size)
plt.xlabel("$c(TW)$")
plt.ylabel("$β(HC)$")

fig, ax = plt.subplots()
sns.heatmap(W,cbar_kws={"label":"Welfare value"},xticklabels =space,yticklabels=space)
plt.title("Welfare",fontsize=20)
ax.set_xticks(size)
ax.set_yticks(size)
plt.xlabel("$c(TW)$")
plt.ylabel("$β(HC)$")

fig, ax = plt.subplots()
sns.heatmap(I,cbar_kws={"label":"$I$ value"},xticklabels =space,yticklabels=space)
plt.title("$Infections$",fontsize=20)
ax.set_xticks(size)
ax.set_yticks(size)
plt.xlabel("$c(TW)$")
plt.ylabel("$β(HC)$")

fig, ax = plt.subplots()
sns.heatmap(Deaths,cbar_kws={"label":"Deaths value"},xticklabels =space,yticklabels=space)
plt.title("Deaths",fontsize=20)
ax.set_xticks(size)
ax.set_yticks(size)
plt.xlabel("$c(TW)$")
plt.ylabel("$β(HC)$")




    
plt.show()
    
#B)For the results in part b what I did is to change the values of rho and omega defined at the beginin of part a),
# save the new graphs in the PDF, and then restore the original values.
