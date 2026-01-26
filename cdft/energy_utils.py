import neural_utils as neural
from scipy.integrate import simpson
import numpy as np

def get_betaFex_alongrho(model, rho_array, elec_array, T, dx):
    
    alphas = np.linspace(0, 1, 100)
    integrands = np.empty_like(alphas)
    for j, alpha in enumerate(alphas):
        integrands[j] = simpson(rho_array * neural.get_c1(model, rho_array*alpha, elec_array, params={"T": T}, dx=dx))
    betaFex = -simpson(integrands, x=alphas)
        
    return   betaFex

def get_betaFex_alongphi(rho_array, T, model, elec, dx):
    
    alphas = np.linspace(0, 1, 100)
    integrands = np.empty_like(alphas)
    for j, alpha in enumerate(alphas):
        elecgrad_array = elec*alpha
        integrands[j] = simpson(elec * neural.get_n1(model, rho_array, elecgrad_array, params={"T": T}, dx=dx))
    betaFex = simpson(integrands, x=alphas)
        
    return   betaFex

def get_betaFid(rho, T, dx):
    """
    Calculate the ideal free energy Fid for a given density profile.

    rho: The density profile
    dx: The discretization of the input layer of the model
    """ 
    lambdacubed = T ** (-3 / 2)
    valid = rho > 0
    return simpson(rho[valid] * (np.log(lambdacubed*rho[valid]) - 1), dx=dx) 

def get_free_energy(T, rho, elec, betamu, dx, model_c1, model_n1):
    
    betaFid = get_betaFid(rho, T, dx)
    betaFexrho = get_betaFex_alongrho(model_c1, rho, elec, T, dx)
    betaFexphi = get_betaFex_alongphi(rho, T, model_n1, elec, dx)
    betamuterm = -simpson(rho * betamu, dx=dx)
    
    #Total = Fid + Fex1 + Fex2 + muterm
    return betaFid, betaFexrho, betaFexphi, betamuterm

def get_betamu(T, rho_array, elec_array, model_c1, dx):
    c1 = neural.get_c1(model_c1, rho_array, elec_array, params={"T": T}, dx=dx)
    lambdacubed = T ** (-3 / 2)
    return np.log(lambdacubed*np.mean(rho_array)) - np.mean(c1)