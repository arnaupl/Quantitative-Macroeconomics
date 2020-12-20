# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:44:00 2020

@author: Arnau Pagès López


In this code, I simulate the extended version of the SIR model in which I introduce
deaths and social distancing policies. First I simulate the model without social
distancing (alpha=0) and the I introduce social distancing policies (alpha=0.3) and 
compare.
"""
#SIR MODEL WITH SOCIAL DISTANCING POLICIES AND DEATHS


#FIRST SETTING WITHOUT SOCIAL DISTANCING POLICIES (i.e. alpha=0)
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Define parameter values.
c=0.30              #rate of contact
delta=1/14          #recovery rate. Is expressed as 1/days it takes to recover.
epsilon=0.02        #rate of mortality
alpha=0             #parameter measuring social distance (i.e. locdowns,etc)


time=np.linspace(0,175,175)  #time in days

#Define the differential equations that describe the model (eq 4,5,6,7 in the
#answers pdf.)
def differentials(y,time,N,alpha,c,delta,epsilon):
    S,I,Rec,D=y
    dSdt=-(1-alpha)*c*S*I/N                  #Expresses contact between infectives and susceptibles.Social distance reduces this contact.
    dIdt=(1-alpha)*c*S*I/N-delta*I-epsilon*I #New infected people appears as a result of contact, but also som infected recover or die.
    dRecdt=delta*I                           #Recovery.    
    dDdt=epsilon*I                           #Deaths. 
    return dSdt,dIdt,dRecdt,dDdt

N=100       #Population size. Keeping it to 100 makes the graph more easy to interpret in %


#Define initial values for the groups (in %)
I0=0.01             #Initial percentage of infected. (in %)
Rec0=0              #Initial percentage of recovered (and immune) (in%)
D0=0                #Initial percentage of deaths (in%). 
S0=N-I0-Rec0-D0     #The rest are susceptible.

y0=S0,I0,Rec0,D0 #vector of initial values.

#Integrate the differential equations over time, given the initial values and the parameters.
out=odeint(differentials,y0,time,args=(N,alpha,c,delta,epsilon))

#Transpose for convenience to plot.
Susceptibles,Infected,Recovered,Deaths=np.transpose(out)

#Plot results
fig,ax = plt.subplots()    
ax.plot(time, Susceptibles,'-', color='orange', linewidth=2,label='Susceptible')  
ax.plot(time, Infected,'-', color='red', linewidth=2,label='Infected')  
ax.plot(time, Recovered,'-', color='green', linewidth=2,label='Recovered (immune)')  
ax.plot(time,Deaths,'-', color='black', linewidth=2,label='Deaths')  
ax.set_title('Pandemic dynamics without social distancing policies (alpha=0)')
ax.set_ylabel('%')
ax.set_xlabel('Time (days)')
ax.legend(loc='right')
plt.show()

#SECOND SETTING WITH SOCIAL DISTANCING POLICIES (i.e.social_distance>0)

#Now I redo with positive social distance policies to see how the curve flats
#That is I increase alpha parameter from 0 to 0.3. The rest 
#of things remain identical as in the case above


alpha2=0.3   #parameter measuring social distance (i.e. locdowns,etc)




def differentials2(y2,time,N,alpha2,c,delta,epsilon):
    S2,I2,Rec2,D2=y2
    dSdt2=-(1-alpha2)*c*S2*I2/N                  
    dIdt2=(1-alpha2)*c*S2*I2/N-delta*I2-epsilon*I2 
    dRecdt2=delta*I2                                       
    dDdt2=epsilon*I2                                      
    return dSdt2,dIdt2,dRecdt2,dDdt2


I2_0=0.01            
Rec2_0=0               
D2_0=0               
S2_0=N-I2_0-Rec2_0-D2_0 

y2_0=S2_0,I2_0,Rec2_0,D2_0
out2=odeint(differentials2,y2_0,time,args=(N,alpha2,c,delta,epsilon))

Susceptibles2,Infected2,Recovered2,Deaths2=np.transpose(out2)

#Plot the new results
fig,ax = plt.subplots()    
ax.plot(time, Susceptibles2,'-', color='orange', linewidth=2,label='Susceptible')  
ax.plot(time, Infected2,'-', color='red', linewidth=2,label='Infected')  
ax.plot(time, Recovered2,'-', color='green', linewidth=2,label='Recovered (immune)')  
ax.plot(time,Deaths2,'-', color='black', linewidth=2,label='Deaths')  
ax.set_title('Pandemic dynamics with some social distancing policies (alpha=0.3)')
ax.set_ylabel('%')
ax.set_xlabel('Time (days)')
ax.legend(loc='center left')
plt.show()

#And more importantly,compare this results with the results without 
#social distance m¡poliices
fig,ax = plt.subplots()    
ax.plot(time, Susceptibles,'-', color='orange', linewidth=2,label='Susceptible (alpha=0)')  
ax.plot(time, Susceptibles2,'--', color='orange', linewidth=2,label='Susceptible (alpha=0.3)')  
ax.plot(time, Infected,'-', color='red', linewidth=2,label='Infected(alpha=0)')  
ax.plot(time, Infected2,'--', color='red', linewidth=2,label='Infected(alpha=0.3)')  
ax.plot(time, Recovered,'-', color='green', linewidth=2,label='Recovered (immune) (alpha=0)')  
ax.plot(time, Recovered2,'--', color='green', linewidth=2,label='Recovered (immune) (alpha=0.3)')  
ax.plot(time,Deaths,'-', color='black', linewidth=2,label='Deaths (alpha=0)')  
ax.plot(time,Deaths2,'--', color='black', linewidth=2,label='Deaths (alpha=0.3)') 
ax.set_title('Change on pandemic dynamics depending on social distance (alpha=0 vs alpha=0.3)')
ax.set_ylabel('%)')
ax.set_xlabel('Time (days)')
ax.legend(bbox_to_anchor=(1.1,0.5))
plt.show()



