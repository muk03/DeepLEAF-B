import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import glob
import os
from tqdm import tqdm

class data_manipulator():
    """
    A class used to manipulate and normalize data.

    Attributes
    ----------
    pth : Path
        The path to the data file.
    data : DataFrame
        The data loaded from the CSV file.

    Methods
    -------
    bin_minmax_normalise(nbins)
        Normalizes the data using min-max normalization and bins it into the specified number of bins.

    bin_minmax_normalise_2d(nbins, col1, col2)
        Normalizes the data using min-max normalization and performs 2D binning on the specified columns.

    config_normalise()
        Normalizes the input variables - the wcs
    """

    def __init__(self, data_path : str, wc_path : str):
        """
        Initializes the data_manipulator with the path to the data file, must be a csv.

        Parameters
        ----------
        data_path : str
            The path to the CSV file containing the data.
        
        wc_path : str
            The path to the json file containing the wilson coefficient etc.
        """

        self.data_pth = data_path
        self.wc_pth = wc_path

    # Ratio of signals added as input here as well
    def bin_minmax_normalise_4d(self, nbins, ratio=0):
        """
        Normalizes the data using min-max normalization and performs 4D binning.

        Parameters
        ----------
        nbins : int
            The number of bins to use for binning the data.

        Returns
        -------
        DataFrame
            A DataFrame containing the binned and normalized data with five columns: bin centers for each dimension and bin height.
        """
        # Read and normalize the data
        data = pd.read_csv(self.data_pth)

        # - Changes really start here -
        hist, edges = np.histogramdd(data.values, bins=nbins)

        bin_centers = [0.5 * (edges[i][1:] + edges[i][:-1]) for i in range(4)]
        xpos, ypos, zpos, wpos = np.meshgrid(*bin_centers, indexing="ij")

        # Flatten the arrays and create the resulting DataFrame
        # q2,cos_theta_l,cos_theta_d,phi

        result_df = pd.DataFrame({
            'q2': xpos.ravel(),
            'cos_theta_l': ypos.ravel(),
            'cos_theta_d': zpos.ravel(),
            'phi': wpos.ravel(),
            'bin_height': hist.ravel()
        })

        return result_df
    

    # Take a look at this later - the input files shape needs to be appropriately changed
    def config_normalise(self, ratio=0, bkg=False):
        """
        Normalizes the configuration values from a JSON file.

        The method reads a JSON configuration file, normalizes the values of the Wilson coefficients
        (or placeholders in the SM case) to a range of [0, 1], and retains the lepton mass without normalization.
        The normalization is performed for all items except the last three, which are assumed to be binary labels
        or other non-normalizable values.

        Returns
        -------
        dict
            A dictionary containing the normalized configuration values, including the lepton mass.
        """
        norm_inputs = {}

        with open(self.wc_pth, 'r') as file:
            config = json.load(file)
        
        # - Think of a way to make this more modular - For different ranges -

        for idx, val in enumerate(config.values()):
            norm_inputs[f'wc_{idx}'] = (val + 10)/20

        # =-=-= The Signal Amount is maximally 0.5 - 50% background and 50% signal =--=-=
        if bkg:
            norm_inputs = norm_inputs | {'signal_amount' : ratio * 0.5}

        return norm_inputs

# Added Ratio of Signals as argument
def file_to_normalise_4d(raw_path, out_path, num_bins, bkg=False):
    """
    Normalizes and bins data from CSV files and normalizes configuration values from JSON files.

    The function processes all CSV files in the specified raw_path directory, normalizes the data using
    min-max normalization, bins it into the specified number of bins, and saves the resulting DataFrame
    to the out_path directory. It also processes all JSON configuration files in the raw_path directory,
    normalizes the configuration values, and saves the resulting dictionary to the out_path directory.

    Parameters
    ----------
    raw_path : str
        The path to the directory containing the raw CSV and JSON files.
    out_path : str
        The path to the directory where the normalized and binned data will be saved.
    num_bins : int
        The number of bins to use for binning the data.

    Returns
    -----------
    None
    """

    os.makedirs(os.path.join(out_path, 'normal_targets'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'normal_inputs'), exist_ok=True)

    files = glob.glob(os.path.join(raw_path, f'*.csv'))
    filenames = [os.path.splitext(os.path.basename(file))[0] for file in files]
    
    for f in (pbar := tqdm(filenames)):

        if bkg:
            ratio = np.random.uniform(0, 1)

        else:
            ratio = 0

        pbar.set_description(f"Processing File No. {f}")
        d_class = data_manipulator(os.path.join(raw_path, f'{f}.csv'), os.path.join(raw_path, f'{f}.json'))

        # include ratio argument here as well
        bin_norm_data = d_class.bin_minmax_normalise_4d(num_bins, ratio)

        # - Config normalise remains the same -
        norm_ins = d_class.config_normalise(ratio, bkg)
        
        # - Store data, this should now work with 4d Binning -
        bin_norm_data.to_csv(out_path + '/normal_targets/' + f'{f}.csv')
        with open(out_path + '/normal_inputs/' + f'{f}.json', 'w') as f:
            json.dump(norm_ins, f,  indent = 4)

def info_extract_4d(data_dir):
    """
    Prepares the data for training and testing by matching input configurations with target data for 2D binning.

    Parameters
    ----------
    data_dir : str
        The directory containing the normalized target CSV files and input JSON files.

    Returns
    -------
    tuple
        A tuple containing all inputs and all targets.
    """
    # - Filewalk over directory to construct list of all file paths - 
    files = glob.glob(os.path.join(data_dir, f'normal_targets/*.csv'))

    # - Extract just the filename - Common b/w inputs and targets -
    filenames = [os.path.splitext(os.path.basename(file))[0] for file in files]

    all_inputs = []
    all_targets = []

    # - Scan over filenames and extract information correctly -
    for f in filenames:
        event_data = data_dir + f'/normal_targets/{f}.csv'
        config_data = data_dir + f'/normal_inputs/{f}.json'

        binned_data = pd.read_csv(event_data, index_col=0)

        with open(config_data, 'r') as f:
            inputs = json.load(f)

        input_params = list(inputs.values())
        targets = binned_data['bin_height'].values

        flat_targets = targets.reshape(-1)

        all_inputs.append(np.array(input_params))
        all_targets.append(np.array(flat_targets))
    
    return np.array(all_inputs), np.array(all_targets), np.array(filenames)

def load_minmax_heights(data_dir, test_idx):
    """
    Loads and normalizes the target data using min-max normalization, then splits the data into training and testing sets.

    Parameters
    ----------
    data_dir : str
        The directory containing the normalized target CSV files and input JSON files.
    test_idx : int
        The index at which to split the data into training and testing sets.

    Returns
    -------
    tuple
        A tuple containing the training inputs, training targets, testing inputs, and testing targets as PyTorch tensors.
    """

    # =-=-= Load in all the data where the 4D binning has been done =-=-=
    norm_inputs, unnorm_targets, filenames = info_extract_4d(data_dir)

    max_height = np.max(unnorm_targets)
    min_height = np.min(unnorm_targets)

    # =-=-= Normalise the Bin-Heights of all data =-=-=
    norm_targets = (unnorm_targets - min_height) / (max_height - min_height)

    train_inputs, train_targets = norm_inputs[:test_idx], norm_targets[:test_idx]
    test_inputs, test_targets = norm_inputs[test_idx:], norm_targets[test_idx:]

    # =-=-= Convert to torch tensor, ready to deploy =-=-=
    tensor_inputs =  torch.Tensor(train_inputs)
    tensor_targets = torch.Tensor(train_targets)

    test_inputs =  torch.Tensor(test_inputs)
    test_targets = torch.Tensor(test_targets)

    files = glob.glob(os.path.join(data_dir, f'normal_targets/*.csv'))

    if max_height == 1.0:
        print('CSV Files: Bin Heights Already Normalised.')
        return tensor_inputs, tensor_targets, test_inputs, test_targets, filenames

    else:
        rescale_dict = {'min_height' : min_height,
                        'max_height' : max_height}
        
        with open(f'{data_dir}/rescale_info.json', 'w') as f:
            json.dump(rescale_dict, f, indent=4)

    # =-=-= Now do the same normalisation, but applied piece by piece to the csvs =-=-=
        for event_data in tqdm(files):
            binned_data = pd.read_csv(event_data, index_col=0)
            binned_data['bin_height'] = (binned_data['bin_height'] - min_height) / (max_height - min_height)
            binned_data.to_csv(event_data)

    return tensor_inputs, tensor_targets, test_inputs, test_targets, filenames