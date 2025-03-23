import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import json
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import torch.nn as nn
from tqdm import tqdm
import glob
from scipy.interpolate import CubicSpline

import nn_tools as nt
import torch

def evaluate_model(nn_state_path, eos_csv_path, inputs_json_path, number_bins, device, mtype, shape=1, wc_range_steps=1000):
    """
    Evaluates the model by calculating the Mean Squared Error (MSE) for a range of Wilson Coefficients (WC)
    and plots the variation of MSE with respect to WC.

    Parameters
    ----------
    nn_state_path : str
        Path to the saved model state dictionary.
    eos_csv_path : str
        Path to the CSV file containing the theoretical data (bin heights).
    inputs_json_path : str
        Path to the JSON file containing the input parameters (WC).
    number_bins : int
        Number of bins used in the model.
    device : torch.device
        The device to run the model on (e.g., 'cpu' or 'cuda').
    wc_range_steps : int, optional
        Number of steps in the WC range (default is 1000).

    Returns
    -------
    float
        The predicted Wilson Coefficient with the minimum MSE.
    float
        The actual Wilson Coefficient from the input JSON file.
    np.ndarray
        The array of MSE values for each WC in the range.
    """
    # Initialize the model

    if mtype.lower() == 'linear':
        model = nt.base_model(1, number_bins**4).to(device)

    else:
        model = nt.CVAE(input_shape=shape, output_shape=10000, latent_dim=32)

    # Load in Theoretical Data and Inputs (WC)
    eos_df = pd.read_csv(eos_csv_path, index_col=0)
    
    with open(inputs_json_path, 'r') as f:
        inputs = json.load(f)

    actual_wc = inputs['wc_0']
    eos_bin_heights = eos_df['bin_height'].values

    # Create range spanning whole WC limits (0 to 1)
    wc_range = np.linspace(0, 1.0, wc_range_steps)

    pairwise_elements = torch.Tensor(np.array([[value] for value in wc_range])).to(device)

    # Load in model and predict bin height
    model.eval()
    model.load_state_dict(torch.load(nn_state_path, map_location=device, weights_only=False))
    model.to(device)

    with torch.no_grad():
        if mtype.lower() == 'linear':
            outputs = model(pairwise_elements)

        else:
            outputs = model.generate_histogram(pairwise_elements)

        outputs = outputs.to('cpu').detach().numpy()

    # Calculate MSE for each of the outputs
    mse_list = []
    for output in outputs:
        mse = np.mean((output - eos_bin_heights)**2)
        mse_list.append(mse)

    # Find the minimum MSE and its associated WC
    mse_np = np.array(mse_list)

    sorted_indices = np.argsort(mse_np)
    arg_min = sorted_indices[0]

    prediction = wc_range[arg_min]

    return prediction, actual_wc

def wilson_accuracy(nn_state_path, data_path, number_bins, device, mtype, self=True, shape=1, wc_range_steps=1000):
    if self:
        files = glob.glob(os.path.join(data_path, 'model_outputs/*.csv'))
    
    else:
        files = glob.glob(os.path.join(data_path, 'normal_targets/*.csv'))

    filenames = [os.path.splitext(os.path.basename(file))[0] for file in files]

    first_prediction = []
    wcs_obj = []

    for f in tqdm(filenames):
        event_data = data_path + f'/normal_targets/{f}.csv'
        config_data = data_path + f'/normal_inputs/{f}.json'

        prediction, actual_wc = evaluate_model(nn_state_path, event_data, config_data, number_bins, device, mtype, shape)
        
        first_prediction.append(prediction)
        wcs_obj.append(actual_wc)
        
    return first_prediction, wcs_obj

import numpy as np
import matplotlib.pyplot as plt

def QQ_Plot(prediction, objective):
    """
    Generates a QQ plot comparing predicted and actual Wilson coefficients,
    including reflections where necessary and fitting a trend line.

    Parameters
    ----------
    pred : list or np.ndarray
        List or array of predicted Wilson coefficients.
    obj : list or np.ndarray
        List or array of actual Wilson coefficients.
    """
    plt.figure(figsize=(10, 10))

    # Convert lists to NumPy arrays
    objective = np.array(objective)
    prediction = np.array(prediction)

    # Identify conditions for reflection
    mask_1 = (objective < 0.5) & (prediction > 0.5) | (objective > 0.5) & (prediction < 0.5)

    # Create updated prediction lists with reflections where necessary
    pred_updated = np.where(mask_1, 1 - prediction, prediction)

    # Fit the new trend line using updated predictions
    vals = np.polyfit(objective, pred_updated, 1)
    func = np.poly1d(vals)

    # Plot original and reflected points together
    plt.plot(objective, pred_updated, 'x', label='Model Prediction', color='blue')

    # Plot the new trend line
    plt.plot(sorted(objective), func(sorted(objective)), linestyle='dashed', label=f'Gradient: {vals[0]:.2f}, \
             Offset {vals[1]:.2f}', color='black')

    # Labels and styling
    plt.xlabel('Actual Wilson Coefficient')
    plt.ylabel('Predicted Wilson Coefficient')
    plt.title('Model Performance Evaluation')
    plt.legend()
    plt.grid()
    plt.show()

def plot_1d_stk(data_path, x_var, filenumber):

    """
        Generates a 3D surface plot for one dataset and overlays a second dataset for comparison.

        Parameters:
        - data_path (str): Path to the CSV file.
        - x_var (str): Column name for the x-axis variable.
        - y_var (str): Column name for the y-axis variable.
        - integrate_vars (list): List of column names to integrate (sum) over.
        - comparison_data (pd.DataFrame): Second dataset with two columns (x_var, y_var) and one row.
        - bin_height (str): Column name for the bin heights (default: "bin_height").
        - grid_size (int): Resolution of the interpolation grid (default: 50).
        - cmap: Colormap for the surface plot (default: cm.viridis).
        """

    # Both share the same base directory, stored in normal_targets and model_outputs separately
    eos_path = os.path.join(data_path, f'normal_targets/model_WET_mu_{filenumber}.csv')
    nn_path = os.path.join(data_path, f'model_outputs/model_WET_mu_{filenumber}.csv')

    input_data = os.path.join(data_path, f'normal_inputs/model_WET_mu_{filenumber}.json')

    with open(input_data, 'r') as f:
        wilson_info = json.load(f)
    
    wc = wilson_info['wc_0']

    df_eos = pd.read_csv(eos_path)
    df_nn = pd.read_csv(nn_path)

    # Integrate over the remaining two columns to produce 3D plots
    df_eos_integrated = df_eos.groupby([x_var])["bin_height"].sum().reset_index()
    df_nn_integrated = df_nn.groupby([x_var])["bin_height"].sum().reset_index()

    x_unq = np.unique(df_eos_integrated[x_var].values)

    # extract the bin_midpoints - both datasets share the same bin_midpoints
    if x_var != 'q2' and x_var != 'phi':
        x_unq = (x_unq * 2) - 1
    
    elif x_var == 'q2':
        x_unq = (x_unq * 10.48) - 0.02

    else:
        x_unq =(x_unq * 2 * np.pi) 

    # The heights however are different and need to be extracted
    height_eos = df_eos_integrated["bin_height"].values
    height_nn = df_nn_integrated["bin_height"].values

    plt.rcParams["font.family"] = "Serif"
    plt.rcParams['font.size'] = 15

    w = np.diff(x_unq)[-1]
    # Create subplots
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the NN dataset
    ax.bar(x_unq, height_nn, width=w, color='blue', edgecolor='black', alpha=0.35, label='CVAE Output')

    # Plot the EOS dataset
    ax.step(np.append(x_unq, x_unq[-1] + w) - w/2, np.append(height_eos, height_eos[-1]), where='post', color='black', 
            linewidth=2.5, label='EOS Output')
    ax.vlines(x_unq[0] - w/2, 0, height_eos[0], color='black', linewidth=2.5)
    ax.vlines(x_unq[-1] + w/2, 0, height_eos[-1], color='black', linewidth=2.5)

    # cs_nn = CubicSpline(x_unq, height_nn)
    # cs_eos = CubicSpline(x_unq, height_eos)

    # x_span = np.linspace(min(x_unq), max(x_unq), 200, endpoint=True)

    # ax.plot(x_span, cs_nn(x_span), color='blue')
    # ax.plot(x_span, cs_eos(x_span), color='red')

    # Add labels and title
    ax.set_xlabel(f'{x_var}', fontsize=20, loc='right')
    ax.set_ylabel('Integrated Bin Heights', fontsize=20, loc='top')
    ax.set_title(f'Model Performance Comparison (Wilson Coefficient cSR: {wc:.3f})', fontsize=20)

    # Add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)

    # Add legend
    ax.legend(fontsize=20)

    plt.show()


def plot_1d_stk_tau(data_path, x_var, filenumber):

    """
        Generates a 3D surface plot for one dataset and overlays a second dataset for comparison.

        Parameters:
        - data_path (str): Path to the CSV file.
        - x_var (str): Column name for the x-axis variable.
        - y_var (str): Column name for the y-axis variable.
        - integrate_vars (list): List of column names to integrate (sum) over.
        - comparison_data (pd.DataFrame): Second dataset with two columns (x_var, y_var) and one row.
        - bin_height (str): Column name for the bin heights (default: "bin_height").
        - grid_size (int): Resolution of the interpolation grid (default: 50).
        - cmap: Colormap for the surface plot (default: cm.viridis).
        """

    # Both share the same base directory, stored in normal_targets and model_outputs separately
    eos_path = os.path.join(data_path, f'normal_targets/model_WET_tau_{filenumber}.csv')
    nn_path = os.path.join(data_path, f'model_outputs/model_WET_tau_{filenumber}.csv')

    input_data = os.path.join(data_path, f'normal_inputs/model_WET_tau_{filenumber}.json')

    with open(input_data, 'r') as f:
        wilson_info = json.load(f)
    
    wc = wilson_info['wc_0']

    df_eos = pd.read_csv(eos_path)
    df_nn = pd.read_csv(nn_path)

    # Integrate over the remaining two columns to produce 3D plots
    df_eos_integrated = df_eos.groupby([x_var])["bin_height"].sum().reset_index()
    df_nn_integrated = df_nn.groupby([x_var])["bin_height"].sum().reset_index()

    x_unq = np.unique(df_eos_integrated[x_var].values)

    # extract the bin_midpoints - both datasets share the same bin_midpoints
    if x_var != 'q2' and x_var != 'phi':
        x_unq = (x_unq * 2) - 1
    
    elif x_var == 'q2':
        x_unq = (x_unq * 10.48) - 0.02

    else:
        x_unq =(x_unq * 2 * np.pi) 

    # The heights however are different and need to be extracted
    height_eos = df_eos_integrated["bin_height"].values
    height_nn = df_nn_integrated["bin_height"].values

    plt.rcParams["font.family"] = "Serif"
    plt.rcParams['font.size'] = 15

    w = np.diff(x_unq)[-1]
    # Create subplots
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the NN dataset
    ax.bar(x_unq, height_nn, width=w, color='blue', edgecolor='black', alpha=0.35, label='CVAE Output')

    # Plot the EOS dataset
    ax.step(np.append(x_unq, x_unq[-1] + w) - w/2, np.append(height_eos, height_eos[-1]), where='post', color='black', 
            linewidth=2.5, label='EOS Output')
    ax.vlines(x_unq[0] - w/2, 0, height_eos[0], color='black', linewidth=2.5)
    ax.vlines(x_unq[-1] + w/2, 0, height_eos[-1], color='black', linewidth=2.5)

    # cs_nn = CubicSpline(x_unq, height_nn)
    # cs_eos = CubicSpline(x_unq, height_eos)

    # x_span = np.linspace(min(x_unq), max(x_unq), 200, endpoint=True)

    # ax.plot(x_span, cs_nn(x_span), color='blue')
    # ax.plot(x_span, cs_eos(x_span), color='red')

    # Add labels and title
    ax.set_xlabel(f'{x_var}', fontsize=20, loc='right')
    ax.set_ylabel('Integrated Bin Heights', fontsize=20, loc='top')
    ax.set_title(f'Model Performance Comparison (Wilson Coefficient cSR: {wc:.3f})', fontsize=20)

    # Add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)

    # Add legend
    ax.legend(fontsize=20)

    plt.show()

    fig.suptitle(f"Wilson Coefficient cSR: {wc:.3f}")
    
    plt.show()



