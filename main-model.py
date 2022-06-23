'''
Wonjin Lee
03-18-2020
-------------------------------------------------------------------------------
This Python program replicates the baseline model in the following paper:
Guerrieri, Veronica and Guido Lorenzoni. 2017. "Credit Crises, Precautionary 
Savings, and the Liquidity Trap." Quarterly Journal of Economics.

Note: 
1. The model is solved using the endogenous grid method.
2. This program is JIT-complied whenever possible.
-------------------------------------------------------------------------------
'''

import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy import interpolate
from numba import jit, int64, float64
from numba.experimental import jitclass
import os
path_dir = '/Users/wonjin/GitHub/credit-crunch'
os.chdir(path_dir)

#------------------------------------------------------------------------------
# 1. Define data structure
#------------------------------------------------------------------------------
GL_data = [
    ('γ', float64), ('η', float64), ('ψ', float64), ('r', float64),
    ('β', float64), ('φ', float64), ('ν', float64), ('B', float64),
    ('b_grid', float64[:]), ('θ_grid', float64[:]),
    ('P', float64[:,:]), ('p_star', float64[:])
]

#------------------------------------------------------------------------------
# 2. Set up the model class
#------------------------------------------------------------------------------
@jitclass(GL_data)
class GL_Economy:
    def __init__(
            self, γ, η, ψ, r,
            β, φ, ν, B,
            b_grid, θ_grid,
            P, p_star
        ):

        self.γ, self.η, self.ψ, self.r = γ, η, ψ, r
        self.β, self.φ, self.ν, self.B = β, φ, ν, B
        self.b_grid, self.θ_grid = b_grid, θ_grid
        self.P, self.p_star = P, p_star

    def tax_scheme(self):
        γ, η, ψ, r = self.γ, self.η, self.ψ, self.r
        β, φ, ν, B = self.β, self.φ, self.ν, self.B
        θ_grid = self.θ_grid
        b_grid = self.b_grid
        P, p_star = self.P, self.p_star

        τ = (p_star[1]*ν + r/(1+r)*B) / (1 - p_star[1])
        z  = np.append(ν, -τ*np.ones((1,len(θ_grid)-1)))
        return z

    def c_const(self, c, iθ, b1, b2, r, z):
        γ, η, ψ = self.γ, self.η, self.ψ
        θ_grid = self.θ_grid
        L_parms = (ψ/θ_grid)**(1/η)
        n = np.maximum(0, 1 - L_parms[iθ]*c**(γ/η))
        return b1 - b2/(1+r) - c + θ_grid[iθ]*n + z[iθ]

    def pol_fn_upd(self, iθ, EU_c, z, c_const):
        γ, η, ψ, r = self.γ, self.η, self.ψ, self.r
        β, φ, ν, B = self.β, self.φ, self.ν, self.B
        θ_grid = self.θ_grid
        b_grid = self.b_grid
        L_parms = (ψ/θ_grid)**(1/η)
        nC = 100

        # Unconstrained HHs: The Euler eq'n holds with equality.
        c = ((1+r) * β * EU_c[iθ, :]) ** (-1/γ)
        n = np.maximum(0, 1 - L_parms[iθ]*(c**(γ/η)))
        b = b_grid[b_grid >= -φ] / (1+r) + c - θ_grid[iθ]*n - z[iθ]
        
        # Constrained HHs: The Euler eq'n holds with inequality.
        if b[0] > -φ:
            c_c = np.linspace(c_const[iθ], c[0], nC)
            n_c = np.maximum(0, 1 - L_parms[iθ]*(c_c**(γ/η)))
            b_c = -φ/(1+r) + c_c - θ_grid[iθ]*n_c - z[iθ]
            b   = np.append(b_c[0:nC-1], b)
            c  =  np.append(c_c[0:nC-1], c)
        return c, b
    
    def ss_dist_upd(self, adj_b_pol, wght, mu):
        nθ = len(self.θ_grid)
        nb = len(self.b_grid)
        mu_upd = np.zeros((nθ, nb))
        for iθ in range(nθ):
            for ib in range(nb):
                for iθ_p in range(nθ):
                    mu_upd[ iθ_p, adj_b_pol[iθ,ib] ] = (
                        (1-wght[iθ, ib])*P[iθ, iθ_p]*mu[iθ,ib] 
                        + mu_upd[ iθ_p, adj_b_pol[iθ,ib] ]
                        )
                    mu_upd[ iθ_p, adj_b_pol[iθ,ib] + 1 ] = (
                        wght[iθ,ib]*P[iθ, iθ_p]*mu[iθ,ib] 
                        + mu_upd[ iθ_p, adj_b_pol[iθ,ib] + 1 ]
                        )
        return mu_upd

#------------------------------------------------------------------------------
# 3. Solve the model
#------------------------------------------------------------------------------
def solve(model, tol=1e-10, maxiter=500):
    γ, η, ψ, r = model.γ, model.η, model.ψ, model.r
    β, φ, ν, B, = model.β, model.φ, model.ν, model.B

    # Allocation
    b_grid = np.ascontiguousarray(model.b_grid)
    θ_grid = np.ascontiguousarray(model.θ_grid)
    P, p_star = np.ascontiguousarray(model.P), np.ascontiguousarray(model.p_star)
    nb, nθ = b_grid.size, θ_grid.size

    c_lower_bnd = np.ones(nθ)
    n_pol = np.zeros((nθ, nb))
    y_pol = np.zeros((nθ, nb))
    b_pol = np.zeros((nθ, nb))
    mpcs = np.zeros((nθ, nb))
    c_pol, c_pol_upd = np.zeros((nθ, nb)), np.zeros((nθ, nb))
    c_min = 1e-6  # min(consumption)
    c_pol[:,:] = np.maximum(c_min, r*np.ones((nθ,1)) @ b_grid.reshape(1,-1))
    c_pol_upd[:,:] = c_pol[:,:]
    L_parms = (ψ/θ_grid)**(1/η)
    db = 0.01 # Epsilon change in bond holdings for MPC

    # Taxation and transfer
    z = model.tax_scheme()

    for iθ in range(nθ):
        sol = root_scalar(
            model.c_const, args=(iθ, φ, φ, r, z)
            , method='toms748', bracket=[c_min, 100]
            )
        c_lower_bnd[iθ] = sol.root

    it = 0
    dist = 10.0
    maxiter = 10_000

    # Solve for the consumption policy function.
    while (it < maxiter) and (dist > tol):
        # print(dist)
        EU_c = P@(c_pol**(-γ))
        EU_c = EU_c[:,b_grid>=-φ]

        for iθ in range(nθ):
            c, b = model.pol_fn_upd(iθ, EU_c, z, c_lower_bnd)
            f = interpolate.interp1d(
                b, c, kind = 'linear',  fill_value = 'extrapolate')
            c_pol_upd[iθ, :] = f(b_grid)

        # Update until converge
        c_pol_upd[:,:] = np.maximum(c_pol_upd, c_min)
        dist = np.max(np.abs(c_pol_upd - c_pol))
        c_pol[:,:] = c_pol_upd[:,:]
        it += 1

    # Policy functions
    for iθ in range(nθ):
        n_pol[iθ, :] = np.maximum(0, 1 - L_parms[iθ]*c_pol[iθ, :]**(γ/η))
        y_pol[iθ, :] = θ_grid[iθ] * n_pol[iθ, :]
        b_pol[iθ, :] = np.maximum(
            (1+r) * (b_grid + y_pol[iθ, :] - c_pol[iθ,:] + z[iθ]), -φ
            )
        f = interpolate.interp1d(
            b_grid, c_pol[iθ,:], kind = 'linear',  fill_value = 'extrapolate'
            )
        mpcs[iθ, :]  = (f(b_grid + db) - c_pol[iθ,:] )/ db

    # Compute the stationary distribution
    adj_b_pol = np.digitize(b_pol, b_grid) - 1 # adj_b_pol = a_{j-1}
    wght = (b_pol - b_grid[adj_b_pol]) / (b_grid[adj_b_pol+1] - b_grid[adj_b_pol])

    it = 0
    dist = 10.0
    mu = np.ones((nθ, nb))/(nθ*nb) # Initial guess

    while (it < maxiter) and (dist > tol):
        mu_upd = model.ss_dist_upd(adj_b_pol, wght, mu)
        dist = np.max(np.abs(mu_upd-mu))
        mu[:,:] = mu_upd/np.sum(mu_upd) 
        it += 1

    return c_pol, n_pol, b_pol, mpcs, mu


#------------------------------------------------------------------------------
# 4. Compute the model solutions and plot them.
#------------------------------------------------------------------------------

## Set up the model parameters
#------------------------------------------------------------------------------
## Income process 1: Use the authors' MC process estimates
import scipy.io as sio
data_inc = sio.loadmat('inc_process.mat')
θ_grid_temp = np.exp(data_inc["x"])
P_temp = data_inc["Pr"]
p_star = data_inc["pr"]
θ_grid = np.append(0,θ_grid_temp)

#------------------------------------------------------------------------------
## Income process 2: Use the quantecon package to estimate MC process
'''
I cannot replicate the authors' income process.
'''
# nθ = 12
# mc = qe.markov.tauchen(0.967, 0.017, 0, 2.5133, nθ)
# θ_grid_temp, P_temp = np.exp(mc.state_values), mc.P
# print(θ_grid_temp)
# print(mc.stationary_distributions[0])
# θ_grid = np.append(0,θ_grid_temp)
# p_star = mc.stationary_distributions[0]
#------------------------------------------------------------------------------

# Earning process with unemployment
emp = 0.8820
unemp = 0.0573
nθ = len(θ_grid)
first_row = np.append(1-emp, emp*p_star )
rest_row = np.append(unemp*np.ones((nθ-1,1)), (1-unemp)*P_temp, axis = 1)
P = np.append(first_row[None,:], rest_row, axis = 0)

# Stationary distribution for the earning process
p_star = np.zeros(len(P))
p_star[len(P)-1] = 1
dist = 10
tol = 1e-8
while dist > tol:
    p_star_upd = p_star@P
    dist = np.max( np.abs(p_star - p_star_upd) )
    p_star = p_star_upd

# Parameters
nb = 200
bmin = -2
bmax = 50
b_grid = bmin + (np.arange(1, nb+1)/nb)**2 * (bmax - bmin)
γ = 4
Frisch_ela = 1
r = 2.5/100/4
NE = 0.4
η = 1/Frisch_ela * (1-NE)/NE
β = 0.9774
ν = 0.1670
B = 2.6712
φ = 1.6005
ψ = 15.8819

# Solve the model
qe.tic()
GL = GL_Economy(γ, η, ψ, r, β, φ, ν, B, b_grid, θ_grid, P, p_star)
c_pol, n_pol, b_pol, mpcs, mu = solve(GL)
qe.toc()

# Plot the policy functions and SS distirbution.
# Note: In order to make the same figure as the paper, limit the bond grid.
Y1 = 0.4174
index_b = (b_grid> -ϕ) & (b_grid < 50*Y1)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_title("Consumption")
ax.grid()
ax.plot(b_grid[index_b]/(4*Y1), c_pol[1,index_b], label="$θ^2$", lw=2, alpha=0.7)
ax.plot(b_grid[index_b]/(4*Y1), c_pol[7,index_b], label="$θ^8$", lw=2, alpha=0.7)
ax.set_xlabel("$b$")
ax.legend(loc='best')
plt.savefig('fig/consumption.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_title("Labor supply")
ax.grid()
ax.plot(b_grid[index_b]/(4*Y1), n_pol[1,index_b], label="$θ^2$", lw=2, alpha=0.7)
ax.plot(b_grid[index_b]/(4*Y1), n_pol[7,index_b], label="$θ^8$", lw=2, alpha=0.7)
ax.set_xlabel("$b$")
ax.legend(loc='best')
plt.savefig('fig/labor_supply.pdf')
plt.show()

index_b = (b_grid> -ϕ -0.2) & (b_grid < 50*Y1) # Domain
mu_b = np.sum(mu, axis=0) # add row wise
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_title("Bond distribution")
ax.grid()
ax.plot(b_grid[index_b]/(4*Y1), mu_b[index_b], label="$θ^2$", lw=2, alpha=0.7)
ax.set_xlabel("$b$")
ax.legend(loc='best')
plt.savefig('fig/dist.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_title("Saving policy")
ax.grid()
ax.plot(
    b_grid[index_b]/(4*Y1), (p_star@b_pol[:,index_b]-b_grid[index_b])/(4*Y1), 
    lw=2, alpha=0.7)
ax.set_xlabel("$b$")
ax.legend(loc='best')
plt.savefig('fig/saving.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_title("MPC")
ax.grid()
ax.plot(
    b_grid[index_b]/(4*Y1), (p_star@mpcs[:,index_b])/(4*Y1), 
    lw=2, alpha=0.7)
ax.set_xlabel("$b$")
ax.legend(loc='best')
plt.savefig('fig/mpc.pdf')
plt.show()