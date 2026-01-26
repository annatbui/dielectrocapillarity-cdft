import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv


# Enable or disable Tensor Float 32 Execution
tf.config.experimental.enable_tensor_float_32_execution(False)

@keras.utils.register_keras_serializable()
class GradientLayer(keras.layers.Layer):
    def call(self, inputs):
        # Compute numerical gradient using central difference (approximated)
        grad = 0.5 * (inputs[:, 2:] - inputs[:, :-2])  # Central difference
        grad = tf.pad(grad, [[0, 0], [1, 1]])  # Pad to keep the same shape
        return grad

def generate_windows(array, bins):
    """
    Generate sliding windows for the input array with a given bin size.

    Parameters:
    - array (np.ndarray): Input array.
    - bins (int): Number of bins on each side of the central bin.
    - mode (str): Padding mode for np.pad (default is "wrap").

    Returns:
    - np.ndarray: Array of sliding windows.
    """
    padded_array = np.pad(array, bins, mode="wrap")
    windows = np.empty((len(array), 2 * bins + 1))
    for i in range(len(array)):
        windows[i] = padded_array[i:i + 2 * bins + 1]
    return windows


def get_c1(model, density_profile, elec_profile, params, dx, return_c2=False, output_dict=False):
    """
    Infer the one-body direct correlation profile from a given density profile 
    using a neural correlation functional.

    Parameters:
    - model (tf.keras.Model): The neural correlation functional.
    - density_profile (np.ndarray): The density profile.
    - dx (float): The discretization of the input layer of the model.
    - input_bins (int): Number of input bins for the model.
    - return_c2 (bool or str): If False, only return c1(x). If True, return both 
                               c1 as well as the corresponding two-body direct 
                               correlation function c2(x, x') which is obtained 
                               via autodifferentiation. If 'unstacked', give c2 
                               as a function of x and x-x', i.e., as obtained 
                               naturally from the model.

    Returns:
    - np.ndarray: c1(x) or (c1(x), c2(x, x')) depending on the value of return_c2.
    """
    input_bins = model.input_shape[1][1]
    window_bins = (input_bins - 1) // 2
    rho_windows = generate_windows(density_profile, window_bins).reshape(density_profile.shape[0], input_bins, 1)
    elec_windows = generate_windows(elec_profile, window_bins).reshape(elec_profile.shape[0], input_bins, 1)
    paramsInput = {key: tf.convert_to_tensor(np.full(density_profile.shape[0], value)) for key, value in params.items()}
    input_name = list(model.input.keys())[0]
    if return_c2:
        rho_windows = tf.Variable(rho_windows)
        elec_windows = tf.Variable(elec_windows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rho_windows)
            result = model({input_name: rho_windows, "elec": elec_windows, **paramsInput})
        jacobiWindows = tape.batch_jacobian(result, rho_windows).numpy().squeeze() / dx
        c1_result =  result.numpy().flatten()
        c2_result = np.row_stack([np.roll(np.pad(jacobiWindows[i], (0,density_profile.shape[0]-input_bins)), i-window_bins) for i in range(density_profile.shape[0])])
        return c1_result, c2_result
    
    return model.predict_on_batch({input_name: rho_windows, "elec": elec_windows, **paramsInput}).flatten()



def get_n1(model, density_profile, elec_profile, params, dx, return_c2=False):
    """
    Get charge density from the neural network.

    Parameters:
    - model (tf.keras.Model): The neural functional for charge density.
    - density_profile (np.ndarray): The density profile.
    - elec_profile (np.ndarray): The electric potential profile.
    - input_bins (int): Number of input bins for the model.
    
    
    Returns:
    - np.ndarray: n(x) charge density.
    """
    input_bins = model.input_shape[1][1]
    paramsInput = {key: tf.convert_to_tensor(np.full(density_profile.shape[0], value)) for key, value in params.items()}
    window_bins = (input_bins - 1) // 2
    rho_windows = generate_windows(density_profile, window_bins).reshape(density_profile.shape[0], input_bins, 1)
    elec_windows = generate_windows(elec_profile, window_bins).reshape(elec_profile.shape[0], input_bins, 1)
    input_name = list(model.input.keys())[0]
    if return_c2:
        rho_windows = tf.Variable(rho_windows)
        elec_windows = tf.Variable(elec_windows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rho_windows)
            result = model({input_name: rho_windows, "elec": elec_windows, **paramsInput})
            jacobiWindows = tape.batch_jacobian(result, rho_windows).numpy().squeeze() / dx
        c1_result =  result.numpy().flatten()
        c2_result = np.row_stack([np.roll(np.pad(jacobiWindows[i], (0,density_profile.shape[0]-input_bins)), i-window_bins) for i in range(density_profile.shape[0])])
        return c1_result, c2_result
        
    return model.predict_on_batch({input_name: rho_windows, "elec": elec_windows, **paramsInput}).flatten()  * 1e-3




def pad_pbc(xbins, muloc):
    """
    Pad arrays z and muloc_z for periodic boundary conditions (PBC).
    
    Returns:
    --------
    tuple
        Tuple containing z (padded array of positions), muloc_z (padded local chempot), and L (length scale).
    """
    muloc_z = muloc
    z = xbins - xbins[0] # shift
    dz = z[1] - z[0]
    z = np.append(z, z[-1] + dz)
    muloc_z = np.append(muloc_z, 0.5*(muloc_z[-1] + muloc_z[0]))
    L = z[-1] - z[0]
    
    return z, muloc_z, L
