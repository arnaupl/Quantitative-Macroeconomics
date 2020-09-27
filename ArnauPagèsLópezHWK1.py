# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:42:42 2020

@author: Arnau Pagès López
"""
#QUANTITATIVE MACROOECONOMICS-PS1

#EXECISE 1:

#a)Impact of  Covid-19 on monthly employment rate evolution in the U.S.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
#I import data
cps=pd.read_csv(r'C:\Users\usuario\Documents\SECOND YEAR\First trimester\Quantitative Macroeconomics\Homeworks\Enunciats i meva resposta\PS1\datacps.txt')
#Depending on the researcher, there are different deffinitions of the employment rate.
#Here I compute it using the formula ER=EMPLOYED(10+12 in cps data)/LABOR FORCE(10+12+21+22 in cps data).
#We generate the dummy variable duemp in roder to indicate if they are employed 
#(employment status 10 or 12) or unemployed.This dummy will be useful in a second, 
#when I execute a map() command on our data. Map() requires a function giving providing
#an instruction.The instruction will be given by this dummy which is very simple and easy
#to interpret for my pruposals.
duemp = {10:1,
           12:1,
           21:0,
           22:0
    } 

#Once the dummy is defined  I just must execute the instruction with map on my data,
#and add it's "result" to  data
cps["EMPLOYMENT DISTRIBUTION"]=cps["EMPSTAT"].map(duemp) 

#Now, if I group data by year and month, and I compute each average of 
#"EMPLOYMENT DISTRIBUTION", I will already have the employment rate.
emp_rate= cps.groupby(["YEAR","MONTH"])["EMPLOYMENT DISTRIBUTION"].mean()

#Obviously, this is the REAL evolution of the employment rate. But we are
#asked to do a prediction of employment for 2020.What I will do first is to
# create a new variable which is the employment of 2018-2019. This can be directly 
#substracted from emp_rate.After that I will create a  new data frame from my original dataframe(cps).
# This dataframe will only contain 2018 and 2019 data.
# In that data frame I will compute 2020 predictions. Finally
#I will mix or better said, concattenate this predicted data with the true date of 2018/2019
# obtaing avariable which will be based on real data for 2018-2019, and predictions for 2020.


#Employment rate evolution upon december 2019:
emp_rate_no2020 = emp_rate[emp_rate.index.get_level_values('YEAR') <= 2019]

#New dataframe excluding 2020
except2020 = cps.loc[cps['YEAR'] <= 2019] #dataframe excluding 2020
#Now, I change years to 2020 to run the prediction, which will be nothing more
#than the average of 2018-2019 for the same montth.
except2020.loc[except2020['YEAR']<2020,'YEAR'] = 2020
#And compute the prediction
predictionx2020=except2020.groupby(["YEAR","MONTH"])["EMPLOYMENT DISTRIBUTION"].mean()

#Finally I concatenate temp_rate_no2020 and predictionx2020 to get data ready to plot.
emp_rate_predicted=pd.concat([emp_rate_no2020, predictionx2020])

#And finally results are ready to plot
fig,ax = plt.subplots()

emp_rate_predicted.plot(label="Predicted Employment Rate")
emp_rate.plot(label="Real Employment Rate")


plt.xlabel('Time')
plt.ylabel('Monthly Employment Rate')
plt.legend()
plt.title ('Impact of Covid on Monthly Employment Rate (US)')




#b)Redo by education group

#In order to group people by education level, I will simply create a 
#new column in the cps dataframe that will do so.
#First generate that column
cps["EDUC LEVEL"]="" 
#and now, start filling it
cps.loc[cps["EDUC"] <= 72,"EDUC LEVEL"] = 1 #Lower than High school
cps.loc[(cps["EDUC"] >= 73) & (cps["EDUC"]<111),"EDUC LEVEL"] = 2 #High school
cps.loc[cps["EDUC"] == 111,"EDUC LEVEL"] = 3 #College
cps.loc[cps["EDUC"]>=123 ,"EDUC LEVEL"] = 4 #More than college
#Now, compute averages as before but grouping also by EDUC LEVEL.But procedure is the same.
#Also we take advantatge of our already generated dummy for employment
educ_emprate= cps.groupby(["YEAR","MONTH","EDUC LEVEL"])["EMPLOYMENT DISTRIBUTION"].mean()
#I now create a dataframe from educ_emprate, because it will make me easier to work with
educ_data=pd.DataFrame(educ_emprate)
#This new dataframe is weird and dificult to work with, so we will change the indexes
#to match groups.
indexes=["Lowerthanhighschool", "Highschool", "College", "Morethancollege"]
educ_data["indexes"]=indexes*32

#Finally we plot our results
dates= pd.date_range(start='2018/01/01', periods=32, freq='M')
fig, ax = plt.subplots(facecolor="w")
for index in indexes:
    education_level = educ_data[educ_data["indexes"] == index]
    ax.plot(dates, education_level["EMPLOYMENT DISTRIBUTION"], label=f"{index}")
plt.xticks(rotation=45)
ax.set_ylabel("Employment Rate")
plt.title("Employment Rate by Education Level")
plt.legend()
plt.show()




#c)Redo by industry, creating two groups of industry according their ability to telework.
#This question has a very similar procedure as question a), but with a much longer instruction
#for map(). What I do is to classify industries according the ability to telework. 0 means no ability
#to telework and 1 means ability to telework. Note that I do the assignation according to my own
#criterium.
instructions={170:0,
              180:0,
              190:0,
              270:0,
              280:0,
              290:0,
              370:0,
              390:0,
              470:0,
              490:0,
              770:0,
              1070:0,
              1080:0,
              1090:0,
              1170:0,
              1180:0,
              1190:0,
              1270:0,
              1280:0,
              1290:0,
              1370:0,
              1470:0,
              1480:0,
              1490:0,
              1570:0,
              1590:0,
              1670:0,
              1680:0,
              1690:0,
              1770:0,
              1790:0,
              3770:0,
              3780:0,
              3790:0,
              3875:0,
              1870:0,
              1880:0,
              1890:0,
              1990:0,
              2070:0,
              2090:0,
              2170:0,
              2180:0,
              2190:0,
              2270:0,
              2280:0,
              2290:0,
              2370:0,
              2380:0,
              2390:0,
              2470:0,
              2480:0,
              2490:0,
              2590:0,
              2670:0,
              2680:0,
              2690:0,
              2770:0,
              2780:0,
              2790:0,
              2870:0,
              2880:0,
              2890:0,
              2970:0,
              2980:0,
              2990:0,
              3070:0,
              3080:0,
              3095:0,
              3170:0,
              3180:0,
              3190:0,
              3365:0,
              3370:0,
              3380:0,
              3390:0,
              3470:0,
              3490:0,
              3570:0,
              3580:0,
              3590:0,
              3670:0,
              3680:0,
              3690:0,
              3895:0,
              3960:0,
              3970:0,
              3980:0,
              3990:0,
              4070:0,
              4080:0,
              4090:0,
              4170:0,
              4180:0,
              4195:0,
              4265:0,
              4270:0,
              4280:0,
              4290:0,
              4370:0,
              4380:0,
              4390:0,
              4470:0,
              4480:0,
              4490:0,
              4560:0,
              4570:0,
              4580:0,
              4585:1,
              4590:1,
              4670:0,
              4680:0,
              4690:0,
              4770:0,
              4780:0,
              4795:0,
              4870:0,
              4880:0,
              4890:0,
              4970:0,
              4980:0,
              4990:0,
              5070:0,
              5080:0,
              5090:0,
              5190:0,
              5275:0,
              5280:0,
              5295:1,
              5370:0,
              5380:0,
              5390:1,
              5470:0,
              5480:0,
              5490:0,
              5570:0,
              5580:0,
              5590:1,
              5591:0,
              5592:1,
              5670:0,
              5680:1,
              5690:0,
              5790:0,
              6070:0,
              6080:0,
              6090:0,
              6170:0,
              6180:0,
              6190:0,
              6270:0,
              6280:0,
              6290:0,
              6370:0,
              6380:0,
              6390:0,
              570:0,
              580:0,
              590:0,
              670:0,
              680:0,
              690:0,
              6470:1,
              6480:1,
              6490:1,
              6570:0,
              6590:0,
              6670:1,
              6672:1,
              6680:1,
              6690:1,
              6695:1,
              6770:0,
              6780:1,
              6870:1,
              6880:1,
              6890:1,
              6970:1,
              6990:1,
              7070:1,
              7080:0,
              7170:1,
              7180:1,
              7190:1,
              7270:1,
              7280:1,
              7290:1,
              7370:1,
              7380:1,
              7390:1,
              7460:1,
              7470:1,
              7480:0,
              7490:0,
              7570:1,
              7580:1,
              7590:1,
              7670:1,
              7680:1,
              7770:1,
              7780:1,
              7790:0,
              7860:1,
              7870:1,
              7880:1,
              7890:1,
              7970:0,
              7980:0,
              7990:0,
              8070:0,
              8080:0,
              8090:0,
              8170:0,
              8180:0,
              8190:0,
              8270:0,
              8290:0,
              8370:0,
              8380:0,
              8390:0,
              8470:0,
              8560:0,
              8570:0,
              8580:0,
              8590:0,
              8660:0,
              8670:0,
              8680:0,
              8690:0,
              8770:0,
              8780:0,
              8790:0,
              8870:0,
              8880:0,
              8970:0,
              8990:0,
              9070:0,
              9080:0,
              9090:0,
              9160:0,
              9170:0,
              9180:1,
              9190:1,
              9290:0,
              9370:0,
              9380:0,
              9390:0,
              9470:0,
              9480:1,
              9490:1,
              9570:1,
              9590:0,
              9890:0,
              }
#Notice that actually I'm just generating a dummy variable.
#Taking advantatge of map, we now create a telework variable.
cps["TLWRK"]=cps["IND"].map(instructions)
tlwrk= cps.groupby(["YEAR","MONTH","TLWRK"])["EMPLOYMENT DISTRIBUTION"].mean()
tlwrkwmprate= pd.DataFrame(tlwrk) #dataframe associated to tlwrk
classification=["No_Telework","Telework"]
tlwrkwmprate["classification"]=classification*32

#Plot results
fig, ax = plt.subplots()

for group in classification:
    teleworkers = tlwrkwmprate[tlwrkwmprate["classification"]==group]
    ax.plot(dates,teleworkers["EMPLOYMENT DISTRIBUTION"],label=f"{group}")
plt.xticks(rotation=45)
ax.set_ylabel("Employment Rate")
plt.title("Employment Rate by Industry's ability to Telework")
plt.legend()
plt.show()

#d) Redo by occupation
#Actually the cathegory I used is Class worker (CLASSWRK) instead of OCC because
#makes much more easier to group professions in an interesting way
#I will proceed in a very similar way as for previous items
#Here I'm just grouping classes of workers in more general grups and generating an instruction
#for map(). The groups are:
     #1Self employed
     #2Private sector employed
     #3Public sector employed
     #4Unpaid family worker
workerinstr={13:1,
             14:1,
             21:2,
             22:2,
             23:2,
             24:3,
             25:3,
             26:3,
             27:3,
             28:3,
             29:4}

#I use the key and map operator to generate a new column
cps["TYPE OF WORKER"]=cps["CLASSWKR"].map(workerinstr)

#Group data by type of worker and empployment distribution
ocupation = cps.groupby(["YEAR","MONTH","TYPE OF WORKER"])["EMPLOYMENT DISTRIBUTION"].mean()
ocupation_emprate = pd.DataFrame(ocupation) #define ocupation as a dataframe

#Similary as I did for education, generate a reference for each type.
refs=["Self_Employed","Private_Sector_Employed","Public_Sector_Employed","Family_Worker"]
ocupation_emprate["refs"]=refs*32

#P
fig, ax = plt.subplots(facecolor="w")
for ref in refs:
    ocu_class = ocupation_emprate[ocupation_emprate["refs"] == ref]
    ax.plot(dates, ocu_class["EMPLOYMENT DISTRIBUTION"], label=f"{ref}")
plt.xticks(rotation=45)
ax.set_ylabel("Employment Rate")
plt.title("Employment Rate by Type of Worker")
plt.legend()
plt.show()

#############################################################################################

#EXERCISE 2:REDO FOR AVERAGE WEEKLY HOURS
#a) Avg.weekly hours
weeklyhrs = cps[cps["AHRSWORKT"]<999].groupby(["YEAR","MONTH"])["AHRSWORKT"].mean()
w_hours = pd.DataFrame(weeklyhrs) #dataframe
hoursworked = w_hours["AHRSWORKT"]

fig, ax = plt.subplots()

ax.plot(dates,hoursworked,label="Average Weekly Worked Hours")
plt.xticks(rotation=45)
plt.title("Average Weekly Hours Worked Evolution")
plt.legend
plt.show()

#Could't do 2020 prediction

#b)Redo by Education

wh_education= cps[cps["AHRSWORKT"]<999].groupby(["YEAR","MONTH","EDUC LEVEL"])["AHRSWORKT"].mean() #The same as with employment but by avg weekly hours
educat = pd.DataFrame(wh_education)
indexes=["Lowerthanhighschool", "Highschool", "College", "Morethancollege"]
educat["indexes"]=indexes*32


fig, ax = plt.subplots()
for index in indexes:
    edu_level = educat[educat["indexes"] == index]
    ax.plot(dates, edu_level["AHRSWORKT"], label=f"{index}")
plt.xticks(rotation=45)
ax.set_ylabel("Average Weekly Hours Worked")
plt.title("Average Weekly Hours Worked by Education Level")
plt.legend()
plt.show()


#c)Redo by industry
tlwrk= cps[cps["AHRSWORKT"]<999].groupby(["YEAR","MONTH","TLWRK"])["AHRSWORKT"].mean()
tlwrkempr= pd.DataFrame(tlwrk)
tlwrkempr["classification"]=classification*32

#PLOT MY RESULTS

fig, ax = plt.subplots()

for group in classification:
    teleworkers = tlwrkempr[tlwrkempr["classification"]==group]
    ax.plot(dates,teleworkers["AHRSWORKT"],label=f"{group}")
plt.xticks(rotation=45)
ax.set_ylabel("Average Weekly Worked Hours")
plt.title("Average Weekly Worked Hours by Industry's ability to telework")
plt.legend()
plt.show() 

#d) Redo by occupation
ocupation = cps[cps["AHRSWORKT"]<999].groupby(["YEAR","MONTH","TYPE OF WORKER"])["AHRSWORKT"].mean()
ocupation_emprate = pd.DataFrame(ocupation)

#Ref for each type
ocupation_emprate["refs"]=refs*32

#Plot my results.I firstly delate family workers because their scale is very different
#and data not comparible.

refs=["Self_Employed","Private_Sector_Employed","Public_Sector_Employed"]


fig, ax = plt.subplots()
for ref in refs:
    ocu_class = ocupation_emprate[ocupation_emprate["refs"] == ref]
    ax.plot(dates, ocu_class["AHRSWORKT"], label=f"{ref}")
plt.xticks(rotation=45)
ax.set_ylabel("Average Weekly Worked Hours")
plt.title("Average Weekly Hours Worked by Type of Worker")
plt.legend()
plt.show()

#############################################################################################

#EXERCISE 3:
    
    #I didn't had time to do this one
#############################################################################################
#EXERCISE 4: REDO FOR WAGES OR EARNINGS OR INCOME
#a)wages/earnings
#I do for average weekly earnings

earnw = cps[cps["EARNWEEK"]<9999].groupby(["YEAR","MONTH"])["EARNWEEK"].mean()
earn_week = pd.DataFrame(earnw)
weeklyearnings = earn_week["EARNWEEK"]

#Plot results:

fig, ax = plt.subplots()

ax.plot(dates,weeklyearnings,label="Average Weekly Earnings")
plt.xticks(rotation=45)
plt.ylabel("Average Earnings")
plt.legend()
plt.title("Average Weekly    Earnings Evolution")
plt.show()

#Couldn't do 2020 prediction

#b)education level


earn_education= cps[cps["EARNWEEK"]<9999].groupby(["YEAR","MONTH","EDUC LEVEL"])["EARNWEEK"].mean() 
educat = pd.DataFrame(earn_education)
indexes=["Lowerthanhighschool", "Highschool", "College", "Morethancollege"]
educat["indexes"]=indexes*32

#Plot results
fig, ax = plt.subplots()
for index in indexes:
    edu_level = educat[educat["indexes"] == index]
    ax.plot(dates, edu_level["EARNWEEK"], label=f"{index}")
plt.xticks(rotation=45)
ax.set_ylabel("Average Weekly Earnings")
plt.title("Average Weekly Earnings by Education Level")
plt.legend()
plt.show()


#c) industry

tlwrk= cps[cps["EARNWEEK"]<9999].groupby(["YEAR","MONTH","TLWRK"])["EARNWEEK"].mean()
teleworkear= pd.DataFrame(tlwrk)
teleworkear["classification"]=classification*32

#Plot my results

fig, ax = plt.subplots()

for group in classification:
    teleworkers = teleworkear[teleworkear["classification"]==group]
    ax.plot(dates,teleworkers["EARNWEEK"],label=f"{group}")
plt.xticks(rotation=45)
ax.set_ylabel("Average Weekly Earnings")
plt.title("Average Weekly Earnings by Industry's ability to telework")
plt.legend()
plt.show() 


#d)Ocupation

ocupation = cps[cps["EARNWEEK"]<9999].groupby(["YEAR","MONTH","TYPE OF WORKER"])["EARNWEEK"].mean()
ocup_earn = pd.DataFrame(ocupation)

#I skip self employed and house workers for reasons of data comparability.
#Ref for each type
refs=["Private_Sector_Employed","Public_Sector_Employed"]
ocup_earn["refs"]=["Private_Sector_Employed","Public_Sector_Employed"]*26+["Self_Employed"]+["Private_Sector_Employed","Public_Sector_Employed"]*6

#Plot my results
fig, ax = plt.subplots()
for ref in refs:
    ocu_class = ocup_earn[ocup_earn["refs"] == ref]
    ax.plot(dates, ocu_class["EARNWEEK"], label=f"{ref}")
plt.xticks(rotation=45)
ax.set_ylabel("Average Weekly Earnings")
plt.title("Average Weekly Earnings by Type of Worker")
plt.legend()
plt.show()

#############################################################################################
#EXERCISE 5: REDO FOR YOUR COUNTRY
#I didn't had time to do this.






















