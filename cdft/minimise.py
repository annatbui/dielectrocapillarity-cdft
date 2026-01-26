import numpy as np
import lmft_utils as lmft
import plot_utils as plt
import neural_utils as neural
import scipy.constants as const
from scipy.optimize import curve_fit

alpha_updates_default_onetype = {
    10: 0.001,
    20: 0.01,
    50: 0.01,
    100: 0.05,
    300: 0.05,
    900: 0.08,
    2000: 0.1,
    5000: 0.1,
}

alpha_updates_default = {
    10: 0.001,
    20: 0.01, 
    50: 0.01,
    100: 0.05,
    300: 0.05,
    900: 0.08,
    2000: 0.1,
    5000: 0.1,
}


alpha_restr_updates_default = {
    1100: 0.0015,
    1200: 0.002,
    1500: 0.005,
    1800: 0.008,
    2000: 0.010,
    2200: 0.015,
}


def minimise_SR(model_c1, model_n, zbins, muloc, elecR, temp, dx,
                      initial_guess, plot=False,
                      maxiter=10000, alpha_initial=1e-6, alpha_updates=None, 
                      print_every=1000, plot_every=1000, tolerance=5e-6,
                      output_dict=False, symmetric=False, chargesymmetry=False, zero_charge=False):
    """
    Compute density profile using Picard iteration for short-range interactions.

    Parameters
    ----------
    model_c1 : tf.keras.Model
        Model to compute direct correlation function.
    model_n : tf.keras.Model
        Model to compute charge density.
    zbins : ndarray
        Spatial grid points.
    muloc : ndarray
        Local chemical potential profile.
    elecR : ndarray
        Electrostatic potential used in calculations.
    temp : float
        Temperature (K).
    dx : float
        Grid spacing.
    initial_guess : float
        Initial density value.
    plot : bool
        Enable plotting.
    maxiter : int
        Maximum iteration count.
    alpha_initial : float
        Initial damping factor.
    alpha_updates : dict
        Iteration-based alpha schedule.
    print_every : int
        Frequency of status output.
    plot_every : int
        Frequency of plot updates.
    tolerance : float
        Convergence threshold.
    output_dict : bool
        Toggle for detailed model output.
    symmetric : bool
        Enforce symmetry in profile.
    chargesymmetry : bool
        Enforce symmetry in charge profile.
    zero_charge : bool
        Set all charges to zero.

    Returns
    -------
    tuple of (zbins, rho, n) or (None, None, None)
    """
    
    # setting up grid
    rho_new = np.zeros_like(zbins)
    valid = np.isfinite(muloc)
    rho = initial_guess * np.ones_like(zbins)
    log_rho_new = np.zeros_like(zbins)
    log_rho = np.zeros_like(zbins)
    valid = np.isfinite(muloc)
    log_rho = np.log(initial_guess)
    log_rho[~valid] = -np.inf 
    
    #elec_grad = np.gradient(elec, zbins)
    params = {"T":temp}
    

    # Picard iteration parameter
    alpha = alpha_initial
    if alpha_updates is None:
        alpha_updates = alpha_updates_default

    if plot:
        fig, ax = plt.configure_plot_charge(zbins)
        color_count = 0


    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]

        # correlation from trained SR model
        c1_pred = neural.get_c1(model_c1, rho, elecR, params, dx, output_dict=output_dict) #- 0.8476* elec
        if symmetric:
            c1_pred = 0.5 * (c1_pred + c1_pred[::-1])
        
        # update density
        log_rho_new[valid] = muloc[valid] + c1_pred[valid] + np.log(temp**1.5)
        log_rho_new[~valid] = -np.inf 
        rho_new = np.exp(log_rho_new)
        
       
        log_rho = (1 - alpha) * log_rho + alpha * log_rho_new
        rho = np.exp(log_rho)
        
        
        n = neural.get_n1(model_n, rho, elecR, params, dx)
        n[rho < 1e-5] = 0.0
        
        # check convergence
        delta = np.max(np.abs(rho_new - rho))
         

        if np.isnan(delta):
            print("Not converged: delta is NaN")
            return None, None
        
        

        if plot and i % plot_every == 0:
            plt.plot_interactive_density_charge(fig, ax, zbins, rho, n, muloc, elecR, elecR, color_count)
            color_count += 1

        if plot and i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")

        if delta < tolerance:
            
            print(f"Converged after {i} iterations (delta = {delta})")
            if plot:
                plt.plot_end_density_charge(zbins, rho, n, muloc, elecR, elecR, ax)
                
            return zbins, rho, n

    print(f"Not converged after {i} iterations (delta = {delta})")
    return None, None, None



def minimise_LR(model_c1, model_n, zbins, muloc, elec, temp,
                            kappa_inv, dielectric, mu_correction, dx,
                            initial_guess, plot=False,
                            maxiter=10000, alpha_initial=1e-6, 
                            alpha_updates=None,  alpha_restr_updates=None,
                            print_every=1000, plot_every=1000, tolerance=5e-6,
                            tolerance_restr = 1e-3,
                            output_dict=False, symmetric=False, chargesymmetry=False, zero_charge=False):
    """
    Compute density profile including long-range electrostatics with restructuring.

    Parameters
    ----------
    model_c1 : tf.keras.Model
        Neural network model for direct correlation.
    model_n : tf.keras.Model
        Neural network model for charge density.
    zbins : ndarray
        Spatial grid.
    muloc : ndarray
        Local chemical potential.
    elec : ndarray
        External electrostatic potential.
    temp : float
        Temperature (K).
    kappa_inv : float
        Inverse Debye length.
    dielectric : float
        Dielectric constant.
    mu_correction : ndarray
        Long-range correction term.
    dx : float
        Grid spacing.
    initial_guess : float
        Starting density profile.
    plot : bool
        Enable live plot updates.
    maxiter : int
        Max iteration steps.
    alpha_initial : float
        Initial damping.
    alpha_updates : dict
        Iteration-based alpha update.
    alpha_restr_updates : dict
        Alpha schedule for restructuring field.
    print_every : int
        Output interval.
    plot_every : int
        Plot update interval.
    tolerance : float
        Convergence tolerance (density).
    tolerance_restr : float
        Convergence tolerance (potential).

    Returns
    -------
    tuple of (zbins, rho, n, elecR) or (None, None, None, None)
    """
    
    # setting up grid
    rho_new = np.zeros_like(zbins)
    n = np.zeros_like(zbins)
    valid = np.isfinite(muloc)
    rho = initial_guess * np.ones_like(zbins)
    log_rho_new = np.zeros_like(zbins)
    log_rho = np.zeros_like(zbins)
    valid = np.isfinite(muloc)
    log_rho = np.log(initial_guess)
    log_rho[~valid] = -np.inf 
    kbins = lmft.compute_wave_numbers(len(zbins), zbins[1] - zbins[0])
    elecR = elec
    
    params = {"T":temp}
    prefactor_restructure = lmft.calculate_prefactor(temp, dielectric)
    delta_restr = 1 # initial value for delta
    
    n_k = np.zeros_like(kbins)

    # Picard iteration parameter
    alpha = alpha_initial
    alpha_restr = 0.01
    if alpha_updates is None:
        alpha_updates = alpha_updates_default
    if alpha_restr_updates is None:
        alpha_restr_updates = alpha_restr_updates_default

    if plot:
        fig, ax = plt.configure_plot_charge(zbins)
        color_count = 0


    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]
        if i in alpha_restr_updates:
            alpha_restr = alpha_restr_updates[i]
            
        c1_pred_SR = neural.get_c1(model_c1, rho, elecR, params, dx, output_dict=output_dict) 
        c1_pred_LR = c1_pred_SR - mu_correction
        if symmetric:
            c1_pred_LR = 0.5 * (c1_pred_LR + c1_pred_LR[::-1])
       
       
        # update density
        log_rho_new[valid] = muloc[valid] + c1_pred_LR[valid] + np.log(temp**1.5)
        log_rho_new[~valid] = -np.inf 
        rho_new = np.exp(log_rho_new)
        log_rho = (1 - alpha) * log_rho + alpha * log_rho_new
        rho = np.exp(log_rho)
        
        
      
        # charge density
        n = neural.get_n1(model_n, rho, elecR, params, dx)
        if chargesymmetry:
            n = 0.5 * (n + n[::-1])    
        elif symmetric:
            n = 0.5 * (n + n[::-1])
        if zero_charge:
            n = 0.0 * n
            
        n[rho < 1e-5] = 0.0
            
        kbins, n_k = lmft.fourier_transform(zbins, n, kbins)
            
        lmf_z = lmft.restructure_electrostatic_potential(n_k, kbins, zbins, kappa_inv) * prefactor_restructure
        
        elecR_new = elec + lmf_z
      
        # check convergence
        delta_rho = np.max(np.abs(rho_new - rho))
        delta_restr = np.max(np.abs(elecR_new - elecR))


        if delta_restr > tolerance_restr:
            elecR = (1 - alpha_restr) * elecR + alpha_restr * elecR_new

        if np.isnan(delta_rho):
            print("Not converged: delta is NaN")
            return None, None, None, None
        
        

        if plot and i % plot_every == 0:
            plt.plot_interactive_density_charge(fig, ax, zbins, rho, n, muloc, elec, elecR, color_count)
            color_count += 1

        if plot and i % print_every == 0:
            print(f"Iteration {i}: delta = {delta_rho}, delta_restr = {delta_restr}")

        if delta_rho < tolerance and delta_restr < tolerance_restr:
            print(f"=====================================================")
            print(f"Converged after {i} iterations")
            print(f"Final delta = {delta_rho:.7f} [AA^-3]")
            print(f"Final delta_restr = {delta_restr:.7f} [kT/AAe]")
            if plot:
                plt.plot_end_density_charge(zbins, rho, n, muloc, elecR, elecR, ax)
                
            return zbins, rho, n, elecR

    print(f"Not converged after {maxiter} iterations (delta = {delta_rho:.7f})")
    return None, None, None, None

