
"""
QUANTITATIVE MACROECONOMICS: HOMEWORK 4

@author: Arnau Pagès López

THIS CODE IS BORROWED FROM SARGENT'S QUANTECON WEBSITE AND SLIGHTLY MODIFIED. CAN BE FOUND IN :
https://python.quantecon.org/aiyagari.html

THE CODE SOLVES ITEM II.5.1 OF THE HOMEWORK :SIMPLE ABHI MODEL.



"""
#Use our parameters from previous exercises


import numpy as np
from numba import jit

class Household:
    """
    "This class takes the parameters that define a household asset accumulation
    problem and computes the corresponding reward and transition matrices R
    and Q required to generate an instance of DiscreteDP, and thereby solve
    for the optimal policy."""
   
    r=0.04
    rho=0.06
    beta=1/(1+rho)
    var_y=0.5
    Y=[1-var_y,1+var_y]
    
    def __init__(self,
                 r=0.04,                      # interest rate
                 w=1.0,                       # wages
                 β=1/(1+rho),                 # discount factor
                 a_min=0,
                 Π=[[0.7, 0.3], [0.3, 0.7]],  # Markov chain
                 z_vals=[1-var_y, 1+var_y],   # exogenous states
                 a_max=18,
                 a_size=200):

        # Store values, set up grids over a and z
        self.r, self.w, self.β = r, w, β
        self.a_min, self.a_max, self.a_size = a_min, a_max, a_size

        self.Π = np.asarray(Π)
        self.z_vals = np.asarray(z_vals)
        self.z_size = len(z_vals)

        self.a_vals = np.linspace(a_min, a_max, a_size)
        self.n = a_size * self.z_size

        # Build the array Q
        self.Q = np.zeros((self.n, a_size, self.n))
        self.build_Q()

        # Build the array R
        self.R = np.empty((self.n, a_size))
        self.build_R()

    def set_prices(self, r, w):
        """
        Use this method to reset prices.  Calling the method will trigger a
        re-build of R.
        """
        self.r, self.w = r, w
        self.build_R()

    def build_Q(self):
        populate_Q(self.Q, self.a_size, self.z_size, self.Π)

    def build_R(self):
        self.R.fill(-np.inf)
        populate_R(self.R, self.a_size, self.z_size, self.a_vals, self.z_vals, self.r, self.w)


# Do the hard work using JIT-ed functions
        
sigma=2 #relative risk aversion coefficient

@jit(nopython=True)
def populate_R(R, a_size, z_size, a_vals, z_vals, r, w):
    n = a_size * z_size
    for s_i in range(n):
        a_i = s_i // z_size
        z_i = s_i % z_size
        a = a_vals[a_i]
        z = z_vals[z_i]
        for new_a_i in range(a_size):
            a_new = a_vals[new_a_i]
            c = w * z + (1 + r) * a - a_new
            if c > 0:
                R[s_i, new_a_i] = (c**(1-sigma)-1)/(1-sigma) #CRRA utility



@jit(nopython=True)
def populate_Q(Q, a_size, z_size, Π):
    n = a_size * z_size
    for s_i in range(n):
        z_i = s_i % z_size
        for a_i in range(a_size):
            for next_z_i in range(z_size):
                Q[s_i, a_i, a_i * z_size + next_z_i] = Π[z_i, next_z_i]


@jit(nopython=True)
def asset_marginal(s_probs, a_size, z_size):
    a_probs = np.zeros(a_size)
    for a_i in range(a_size):
        for z_i in range(z_size):
            a_probs[a_i] += s_probs[a_i * z_size + z_i]
    return a_probs

import quantecon as qe
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
from quantecon.markov import DiscreteDP

#%%

rho=0.03

A = 1.0
N = 1.0
α = 0.33
β = 1/(1+rho)
δ = 0.05


def r_to_w(r):
    """
    Equilibrium wages associated with a given interest rate r.
    """
    return A * (1 - α) * (A * α / (r + δ))**(α / (1 - α))

def rd(K):
    """
    Inverse demand curve for capital.  The interest rate associated with a
    given demand for capital K.
    """
    return A * α * (N / K)**(1 - α) - δ


def prices_to_capital_stock(am, r):
    """
    Map prices to the induced level of capital stock.
    
    Parameters:
    ----------
    
    am : Household
        An instance of an aiyagari_household.Household 
    r : float
        The interest rate
    """
    w = r_to_w(r)
    am.set_prices(r, w)
    aiyagari_ddp = DiscreteDP(am.R, am.Q, β)
    # Compute the optimal policy
    results = aiyagari_ddp.solve(method='policy_iteration')
    # Compute the stationary distribution
    stationary_probs = results.mc.stationary_distributions[0]
    # Extract the marginal distribution for assets
    asset_probs = asset_marginal(stationary_probs, am.a_size, am.z_size)
    # Return K
    return np.sum(asset_probs * am.a_vals)


# Create an instance of Household
am = Household(a_max=20)

# Use the instance to build a discrete dynamic program
am_ddp = DiscreteDP(am.R, am.Q, am.β)

# Create a grid of r values at which to compute demand and supply of capital
num_points = 30
r_vals = np.linspace(0.005, 0.04, num_points)

# Compute supply of capital
k_vals = np.empty(num_points)
for i, r in enumerate(r_vals):
    k_vals[i] = prices_to_capital_stock(am, r)

# Plot against demand for capital by firms
fig, ax = plt.subplots(figsize=(11, 8))
ax.plot(k_vals, r_vals, lw=2, alpha=0.6, color='b',label='supply of capital')
ax.plot(k_vals, rd(k_vals), lw=2, alpha=0.6, color='r', label='demand of capital')
ax.set_xlabel('capital stock')
ax.set_ylabel('r')
ax.legend(loc='upper right')
plt.show()


# Report the endogenous distribution of wealth. 
#STEP 1: Stationary distribution of wealth. 
am_ddp = DiscreteDP(am.R, am.Q, am.β)
results = am_ddp.solve(method='policy_iteration')
# Compute the stationary distribution
stationary_probs = results.mc.stationary_distributions[0]
# Extract the marginal distribution for assets
asset_probs = asset_marginal(stationary_probs, am.a_size, am.z_size)

#PLOT
plt.hist(asset_probs, bins=None, range=None, density=False,weights=None,cumulative=False,bottom=None,histtype='bar', align='mid',orientation='vertical', rwidth=None,log=False,color='purple')
plt.title('Assets stationary distribution')

Amean=np.mean(asset_probs)


#%% 
#Compute Gini and compare with Krueger, Mitman and Perri (2016).

#Function to compute GINI coefficient.
def gini_computation(array):
    array = array.flatten()
    if np.amin(array) < 0:
        #Avoid negative values
        array -= np.amin(array)
    #Avoid 0
    array += 0.0000000000000000001
    #Sort values
    array = np.sort(array)
    #Index.
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    #Compute GINI
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

#Obtain coefficeint:
Gini=gini_computation(asset_probs)