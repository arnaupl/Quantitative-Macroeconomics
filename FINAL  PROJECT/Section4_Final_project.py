# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:12:21 2020

@author: Arnau Pagès López

In this code I solve the Ramsey government optimal taxation problem in th economy
of Section 3. I do it in finite horizon (only 6 periods t=0,1,2,3,4,5) and under 
the assumption of commitment.
"""

#Ramsey optimal taxation probem for 10 periods (T=9)
#Repeat the code of the begining of Part 3, to get here the same parametrization and
#the capital at the pre-covid steady state which will be the initial capital in ramsey problem.
import math
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize                                               

#Use the same parametrization as in Section 3.
beta=0.7     #Discount factor
delta=0.4    #Rate of depreciation
alpha=0.675  #Labor share in Cobb-Douglas production function
A_w=1.6      #TFP associated to labor at workplace
A_h=0.7      #TFP associated to labor from home
i1=0         #Infection rate pre-covid.
i2=0.2       #Infection rate covid.
rho=1.1      #Elasticity of substitution
phi=0.7      #Productivity loss associated with teleworking.
omega=5      #Omega and nu are parameters associated with the disutility of labor.
nu=2.3
q=15         #Factor augmenting the disutility of working at workplace if the infection rate is positive
z=(rho-1)/rho #Substitution parameter

#Compute the capital in the pre-covid steady state to use it as k0
def steadystate_PREcovid(vars):
    kss_pre,h_wss_pre=vars
    eq1=beta*(1-delta+(1-alpha)*kss_pre**(-alpha)*(((A_w-i1)*h_wss_pre**(z)+phi*A_h*(1-h_wss_pre)**(z))**(1/z))**(alpha))-1
    eq2=(1/(kss_pre**(1-alpha)*(((A_w-i1)*h_wss_pre**(z)+phi*A_h*(1-h_wss_pre)**(z))**(1/z))**(alpha)-delta*kss_pre))*(((alpha*kss_pre**(1-alpha)*((A_w-i1)*z*h_wss_pre**(z-1)-A_h*phi*z*(1-h_wss_pre)**(z-1)))*((((A_w-i1)*h_wss_pre**(z)+phi*A_h*(1-h_wss_pre)**(z))**(1/z))**(alpha)))/(z*((A_w-i1)*h_wss_pre**(z)+phi*A_h*(1-h_wss_pre)**(z))))-(omega+q*i1)*h_wss_pre**(1/nu)+omega*(1-h_wss_pre)**(1/nu)
    return [eq1, eq2]

kss_pre,h_wss_pre=fsolve(steadystate_PREcovid,(0.5,0.5)) 

#Initial conditions
k0=kss_pre
b0=30
tauk0=0
#Define the stream of public expenditure. Increase in periods 2 and 3 simulates the pandemic.
g=np.array([45,45,90,90,45,45])
#Final conditions. Very imortant in finite horizon.
k6,c6,h_w6,h_h6,b6=0,0,0,0,0

#DEFINE THE FOLLOWING BATCH OF FUNCTIONS TO MAKE THE NOTATION IN THE 
#LAGRANGIAN A BIT EASIER.

#The covid or no covid situation is determinied by the infection rate i
# i1 is wern no covid and i2 when covid.

#Output under no covid (periods 0,1 and 4,5)
def ynocov(k,h_w,h_h):
    return (k**(1-alpha))*(((A_w-i1)*h_w**z+phi*A_h*h_h**z)**(1/z))**alpha
#Output under covid (periods 2,3)
def ycov(k,h_w,h_h):
    return (k**(1-alpha))*(((A_w-i2)*h_w**z+phi*A_h*h_h**z)**(1/z))**alpha


#Disutility of labor at workplace under no covid (periods 0,1 and 4,5)
def disuw_nocov(h_w):
    return ((omega+q*i1)*h_w**(1+1/nu))/(1+1/nu)
#Disutility of labor at workplace under covid (periods 2,3)
def disuw_cov(h_w):
    return ((omega+q*i2)*h_w**(1+1/nu))/(1+1/nu)


#Disutility of labor at from home under no covid (periods 0,1 and 4,5). 
#In this case, it is the same in both situations, but is to make easier the writting of the lagrangian.
def disuh_nocov(h_h):
    return (omega*h_h**(1+1/nu))/(1+1/nu)
#Disutility of labor from home under covid (periods 2,3)
#In this case, it is the same in both situations, but is to make easier the writting of the lagrangian.
def disuh_cov(h_h):
    return (omega*h_h**(1+1/nu))/(1+1/nu)

#Marginal utility of consumption (under logarithmic preferences)
def muc(c):
    return (1/c)


#Marginal disutility of labor at workplace under no covid (periods 0,1 and 4,5)
def margdisuw_nocov(h_w):
    return (omega+q*i1)*h_w**(1/nu)
#Marginal disutility of labor at workplace under covid (periods 2,3)
def margdisuw_cov(h_w):
    return (omega+q*i2)*h_w**(1/nu)


#Marginal disutility of labor at from home under no covid (periods 0,1 and 4,5). 
def margdisuh_nocov(h_h):
    return omega*h_h*(1/nu)
#Marinal disutility of labor from home under covid (periods 2,3)
def margdisuh_cov(h_h):
    return omega*h_h**(1/nu)


#Prices of factors under covid and under no covid. They come from FOC of firms.
def rnocov(k,h_w,h_h):
    return (1-alpha)*k**(-alpha)*(((A_w-i1)*h_w**z+phi*A_h*h_h**z)**(1/z))**alpha
def rcov(k,h_w,h_h):
    return (1-alpha)*k**(-alpha)*(((A_w-i2)*h_w**z+phi*A_h*h_h**z)**(1/z))**alpha

def wnocov(k,h_w,h_h):
    return (alpha*(A_w-i1)*k**(1-alpha)*h_w**(z-1)*(((A_w-i1)*h_w**z+phi*A_h*h_h**z)**(1/z))**alpha)/((A_w-i1)*h_w**z+phi*A_h*h_h**z)
def wcov(k,h_w,h_h):
    return (alpha*(A_w-i2)*k**(1-alpha)*h_w**(z-1)*(((A_w-i2)*h_w**z+phi*A_h*h_h**z)**(1/z))**alpha)/((A_w-i2)*h_w**z+phi*A_h*h_h**z) 

def pnocov(k,h_w,h_h):
    return (alpha*A_h*phi*k**(1-alpha)*h_h**(z-1)*(((A_w-i1)*h_w**z+phi*A_h*h_h**z)**(1/z))**alpha)/((A_w-i1)*h_w**z+phi*A_h*h_h**z)
def pcov(k,h_w,h_h):
    return (alpha*A_h*phi*k**(1-alpha)*h_h**(z-1)*(((A_w-i2)*h_w**z+phi*A_h*h_h**z)**(1/z))**alpha)/((A_w-i2)*h_w**z+phi*A_h*h_h**z)



#Define the objective function, i.e. the lagrangian from expression (49) in 
#the answers pdf. Multiply by -1, because as most of softwares, Python does not
#maximize, it minimizes, hence if we want to maximize  the expression we need to
#minimize the expression multiplyed by -1.

#For the variables whose initial value is given, important to remember to not 
#include them at period 0. Also
    
def objective_function(x):
    c0,c1,c2,c3,c4,c5=x[0],x[1],x[2],x[3],x[4],x[5]
    k1,k2,k3,k4,k5=x[6],x[7],x[8],x[9],x[10]
    h_w0,h_w1,h_w2,h_w3,h_w4,h_w5=x[11],x[12],x[13],x[14],x[15],x[16]
    b1,b2,b3,b4,b5=x[17],x[18],x[19],x[20],x[21]
    tauk1,tauk2,tauk3,tauk4,tauk5=x[22],x[23],x[24],x[25],x[26]
    tauw0,tauw1,tauw2,tauw3,tauw4,tauw5=x[27],x[28],x[29],x[30],x[31],x[32]
    tauh0,tauh1,tauh2,tauh3,tauh4,tauh5=x[33],x[34],x[35],x[36],x[37],x[38]
    lamb0,lamb1,lamb2,lamb3,lamb4,lamb5=x[39],x[40],x[41],x[42],x[43],x[44]
    mu0,mu1,mu2,mu3,mu4,mu5=x[45],x[46],x[47],x[48],x[49],x[50]
    sigma0,sigma1,sigma2,sigma3,sigma4,sigma5=x[51],x[52],x[53],x[54],x[55],x[56]
    iota0,iota1,iota2,iota3,iota4,iota5=x[57],x[58],x[59],x[60],x[61],x[62]
    chi0,chi1,chi2,chi3,chi4,chi5=x[63],x[64],x[65],x[66],x[67],x[68]
    return (-1)*(beta**0*(math.log(c0)-disuw_nocov(h_w0)-disuh_nocov(1-h_w0)+g[0]+lamb0*(tauk0*rnocov(k0,h_w0,1-h_w0)*k0+tauw0*wnocov(k0,h_w0,1-h_w0)*h_w0+tauh0*pnocov(k0,h_w0,1-h_w0)*(1-h_w0)+b0-g[0]-((1)/((1-tauk1)*rnocov(k1,h_w1,1-h_w1)+(1-delta)))*b1)+mu0*(muc(c0)-beta*muc(c1)*((1-tauk1)*rnocov(k1,h_w1,1-h_w1)+1-delta))+sigma0*(margdisuw_nocov(h_w0)-muc(c0)*((1-tauw0)*wnocov(k0,h_w0,1-h_w0)))+iota0*(margdisuh_nocov(1-h_w0)-muc(c0)*((1-tauh0)*pnocov(k0,h_w0,1-h_w0)))+chi0*(ynocov(k0,h_w0,1-h_w0)-c0-g[0]-k1+(1-delta)*k0))                                     
                +beta**1*(math.log(c1)-disuw_nocov(h_w1)-disuh_nocov(1-h_w1)+g[1]+lamb1*(tauk1*rnocov(k1,h_w1,1-h_w1)*k1+tauw1*wnocov(k1,h_w1,1-h_w1)*h_w1+tauh1*pnocov(k1,h_w1,1-h_w1)*(1-h_w1)+b1-g[1]-((1)/((1-tauk2)*rcov(k2,h_w2,1-h_w2)+(1-delta)))*b2)+mu1*(muc(c1)-beta*muc(c2)*((1-tauk2)*rcov(k2,h_w2,1-h_w2)+1-delta))+sigma1*(margdisuw_nocov(h_w1)-muc(c1)*((1-tauw1)*wnocov(k1,h_w1,1-h_w1)))+iota1*(margdisuh_nocov(1-h_w1)-muc(c1)*((1-tauh1)*pnocov(k1,h_w1,1-h_w1)))+chi1*(ynocov(k1,h_w1,1-h_w1)-c1-g[1]-k2+(1-delta)*k1))                                              
                +beta**2*(math.log(c2)-disuw_cov(h_w2)-disuh_cov(1-h_w2)+g[2]+lamb2*(tauk2*rcov(k2,h_w2,1-h_w2)*k2+tauw2*wcov(k2,h_w2,1-h_w2)*h_w2+tauh2*pcov(k2,h_w2,1-h_w2)*(1-h_w2)+b2-g[2]-((1)/((1-tauk3)*rcov(k3,h_w3,1-h_w3)+(1-delta)))*b3)+mu2*(muc(c2)-beta*muc(c3)*((1-tauk3)*rcov(k3,h_w3,1-h_w3)+1-delta))+sigma2*(margdisuw_cov(h_w2)-muc(c2)*((1-tauw2)*wcov(k2,h_w2,1-h_w2)))+iota2*(margdisuh_cov(1-h_w2)-muc(c2)*((1-tauh2)*pcov(k2,h_w2,1-h_w2)))+chi2*(ycov(k2,h_w2,1-h_w2)-c2-g[2]-k3+(1-delta)*k2))           
                +beta**3*(math.log(c3)-disuw_cov(h_w3)-disuh_cov(1-h_w3)+g[3]+lamb3*(tauk3*rcov(k3,h_w3,1-h_w3)*k3+tauw3*wcov(k3,h_w3,1-h_w3)*h_w3+tauh3*pcov(k3,h_w3,1-h_w3)*(1-h_w3)+b3-g[3]-((1)/((1-tauk4)*rnocov(k4,h_w4,1-h_w4)+(1-delta)))*b4)+mu3*(muc(c3)-beta*muc(c4)*((1-tauk4)*rnocov(k4,h_w4,1-h_w4)+1-delta))+sigma3*(margdisuw_cov(h_w3)-muc(c3)*((1-tauw3)*wcov(k3,h_w3,1-h_w3)))+iota3*(margdisuh_cov(1-h_w3)-muc(c3)*((1-tauh3)*pcov(k3,h_w3,1-h_w3)))+chi3*(ycov(k3,h_w3,1-h_w3)-c3-g[3]-k4+(1-delta)*k3))                        
                +beta**4*(math.log(c4)-disuw_nocov(h_w4)-disuh_nocov(1-h_w4)+g[4]+lamb4*(tauk4*rnocov(k4,h_w4,1-h_w4)*k4+tauw4*wnocov(k4,h_w4,1-h_w4)*h_w4+tauh4*pnocov(k4,h_w4,1-h_w4)*(1-h_w4)+b4-g[4]-((1)/((1-tauk5)*rnocov(k5,h_w5,1-h_w5)+(1-delta)))*b5)+mu4*(muc(c4)-beta*muc(c5)*((1-tauk5)*rnocov(k5,h_w5,1-h_w5)+1-delta))+sigma4*(margdisuw_nocov(h_w4)-muc(c4)*((1-tauw4)*wnocov(k4,h_w4,1-h_w4)))+iota4*(margdisuh_nocov(1-h_w4)-muc(c4)*((1-tauh4)*pnocov(k4,h_w4,1-h_w4)))+chi4*(ynocov(k4,h_w4,1-h_w4)-c4-g[4]-k5+(1-delta)*k4))                   
                +beta**5*(math.log(c5)-disuw_nocov(h_w5)-disuh_nocov(1-h_w5)+g[5]+lamb5*(tauk5*rnocov(k5,h_w5,1-h_w5)*k5+tauw5*wnocov(k5,h_w5,1-h_w5)*h_w5+tauh5*pnocov(k5,h_w5,1-h_w5)*(1-h_w5)+b5-g[5]-((1)/((1-delta)))*b6)+mu5*(muc(c5))+sigma5*(margdisuw_nocov(h_w5)-muc(c5)*((1-tauw5)*wnocov(k5,h_w5,1-h_w5)))+iota5*(margdisuh_nocov(1-h_w5)-muc(c5)*((1-tauh5)*pnocov(k5,h_w5,1-h_w5)))+chi5*(ynocov(k5,h_w5,1-h_w5)-c5-g[5]-k6+(1-delta)*k5)))   #This one is shorter because to avoid computational issues I carefully  delated the terms that are 0 by the terminal condition.
    
#Define bounds for the diferent variables. I use them mainly as non negativity constraints.    
bounds_c0,bounds_c1,bounds_c2,bounds_c3,bounds_c4,bounds_c5=(0.01,5000),(0.01,5000),(0.01,5000),(0.01,5000),(0.01,5000),(0.01,5000)
bounds_c=[bounds_c0,bounds_c1,bounds_c2,bounds_c3,bounds_c4,bounds_c5]

bounds_k1,bounds_k2,bounds_k3,bounds_k4,bounds_k5=(0.01,5000),(0.01,5000),(0.01,5000),(0.01,5000),(0.01,5000)
bounds_k=[bounds_k1,bounds_k2,bounds_k3,bounds_k4,bounds_k5]

bounds_h_w0,bounds_h_w1,bounds_h_w2,bounds_h_w3,bounds_h_w4,bounds_h_w5=(0.01,0.99),(0.01,0.99),(0.01,0.99),(0.01,0.99),(0.01,0.99),(0.01,0.99)
bounds_h_w=[bounds_h_w0,bounds_h_w1,bounds_h_w2,bounds_h_w3,bounds_h_w4,bounds_h_w5]

bounds_b1,bounds_b2,bounds_b3,bounds_b4,bounds_b5=(0.01,5000),(0.01,5000),(0.01,5000),(0.01,5000),(0.01,5000)
bounds_b=[bounds_b1,bounds_b2,bounds_b3,bounds_b4,bounds_b5]


bounds_tauk1,bounds_tauk2,bounds_tauk3,bounds_tauk4,bounds_tauk5=(0,1),(0,1),(0,1),(0,1),(0,1)
bounds_tauk=[bounds_tauk1,bounds_tauk2,bounds_tauk3,bounds_tauk4,bounds_tauk5]


bounds_tauw0,bounds_tauw1,bounds_tauw2,bounds_tauw3,bounds_tauw4,bounds_tauw5=(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)
bounds_tauw=[bounds_tauw0,bounds_tauw1,bounds_tauw2,bounds_tauw3,bounds_tauw4,bounds_tauw5]


bounds_tauh0,bounds_tauh1,bounds_tauh2,bounds_tauh3,bounds_tauh4,bounds_tauh5=(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)
bounds_tauh=[bounds_tauh0,bounds_tauh1,bounds_tauh2,bounds_tauh3,bounds_tauh4,bounds_tauh5]

#VERY IMPORTANT THE NON NEGATIVITY CONSTRAIN T OF LAGRANGE MULTIPLIERS.
bounds_lamb0,bounds_lamb1,bounds_lamb2,bounds_lamb3,bounds_lamb4,bounds_lamb5=(0,10000),(0,10000),(0,10000),(0,10000),(0,10000),(0,10000)
bounds_lamb=[bounds_lamb0,bounds_lamb1,bounds_lamb2,bounds_lamb3,bounds_lamb4,bounds_lamb5]

bounds_mu0,bounds_mu1,bounds_mu2,bounds_mu3,bounds_mu4,bounds_mu5=(0,10000),(0,10000),(0,10000),(0,10000),(0,10000),(0,10000)
bounds_mu=[bounds_mu0,bounds_mu1,bounds_mu2,bounds_mu3,bounds_mu4,bounds_mu5]


bounds_sigma0,bounds_sigma1,bounds_sigma2,bounds_sigma3,bounds_sigma4,bounds_sigma5=(0,10000),(0,10000),(0,10000),(0,10000),(0,10000),(0,10000)
bounds_sigma=[bounds_sigma0,bounds_sigma1,bounds_sigma2,bounds_sigma3,bounds_sigma4,bounds_sigma5]


bounds_iota0,bounds_iota1,bounds_iota2,bounds_iota3,bounds_iota4,bounds_iota5=(0,10000),(0,10000),(0,10000),(0,10000),(0,10000),(0,10000)
bounds_iota=[bounds_iota0,bounds_iota1,bounds_iota2,bounds_iota3,bounds_iota4,bounds_iota5]


bounds_chi0,bounds_chi1,bounds_chi2,bounds_chi3,bounds_chi4,bounds_chi5=(0,10000),(0,10000),(0,10000),(0,10000),(0,10000),(0,10000)
bounds_chi=[bounds_chi0,bounds_chi1,bounds_chi2,bounds_chi3,bounds_chi4,bounds_chi5]



bounds=[bounds_c0,bounds_c1,bounds_c2,bounds_c3,bounds_c4,bounds_c5,bounds_k1,bounds_k2,bounds_k3,bounds_k4,bounds_k5,bounds_h_w0,bounds_h_w1,bounds_h_w2,bounds_h_w3,bounds_h_w4,bounds_h_w5,bounds_b1,bounds_b2,bounds_b3,bounds_b4,bounds_b5,bounds_tauk1,bounds_tauk2,bounds_tauk3,bounds_tauk4,bounds_tauk5,bounds_tauw0,bounds_tauw1,bounds_tauw2,bounds_tauw3,bounds_tauw4,bounds_tauw5,bounds_tauh0,bounds_tauh1,bounds_tauh2,bounds_tauh3,bounds_tauh4,bounds_tauh5,bounds_lamb0,bounds_lamb1,bounds_lamb2,bounds_lamb3,bounds_lamb4,bounds_lamb5,bounds_mu0,bounds_mu1,bounds_mu2,bounds_mu3,bounds_mu4,bounds_mu5,bounds_sigma0,bounds_sigma1,bounds_sigma2,bounds_sigma3,bounds_sigma4,bounds_sigma5,bounds_iota0,bounds_iota1,bounds_iota2,bounds_iota3,bounds_iota4,bounds_iota5,bounds_chi0,bounds_chi1,bounds_chi2,bounds_chi3,bounds_chi4,bounds_chi5]

#Initial guess
x0=[700,700,700,700,700,700,400,400,400,400,400,0.5,0.5,0.5,0.5,0.5,0.5,60,60,60,60,60,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]   
#Solve the problem
solution=minimize(objective_function,x0,method='SLSQP',bounds=bounds)

#Obtain the optimal tax sequences.
tauk_sequence=solution.x[22:27]
tauk_sequence_def=np.transpose(np.insert(tauk_sequence,0,tauk0)) #recall that tauk0 was given

tauw_sequence=np.transpose(solution.x[27:33])

tauh_sequence=np.transpose(solution.x[33:39])

#Plot results.
time=np.array(list(range(0,(6))))


fig,ax = plt.subplots()    
ax.plot(time, g,'-', color='purple', linewidth=2)  
ax.set_title('Exogenous sequence of government expenditure (g)')
ax.set_ylabel('g')
ax.set_xlabel('Time')
ax.legend(loc='upper right')
ax.set_xticks(time)
ax.set_yticks(np.linspace(10,100,5))
plt.show()




fig,ax = plt.subplots()    
ax.plot(time, tauk_sequence_def,'-', color='blue', linewidth=2,label='Tax on capital')  
ax.plot(time, tauw_sequence,'-', color='red', linewidth=2,label='Tax on labor at workplace')  
ax.plot(time, tauh_sequence,'-', color='green', linewidth=2,label='Tax on labor from home')  
ax.set_title('Sequences of optimal taxtes')
ax.set_ylabel('tau')
ax.set_xlabel('Time')
ax.legend(loc='upper right')
ax.set_xticks(time)
ax.set_yticks(np.linspace(0,1,10))
plt.show()
