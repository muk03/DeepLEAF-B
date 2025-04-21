import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import os
import pandas as pd
import random
import json

# =====================================================================

class CVAE_Opt(nn.Module):
    """
Convolutional Variational Autoencoder (CVAE) Model that takes in Optimised Hyperparameters.

Args:
    input_shape (int): The dimensionality of the input data.
    output_shape (int): The dimensionality of the output data.
    latent_dim (int): The dimensionality of the latent space.
    encoder_layers (list of int): List of dimensions for each layer in the encoder.
    decoder_layers (list of int): List of dimensions for each layer in the decoder.
    dropout_enc (list of bool): List indicating whether to apply dropout for each encoder layer.
    dropout_dec (list of bool): List indicating whether to apply dropout for each decoder layer.
"""
    def __init__(self, input_shape, output_shape, latent_dim, encoder_layers, decoder_layers, dropout_enc, dropout_dec):
        super(CVAE_Opt, self).__init__()

        self.encoder_base = self.layer_opt(input_shape, encoder_layers, dropout_enc)

        self.encoder_mu = nn.Linear(encoder_layers[-1], latent_dim, bias=True)
        self.encoder_std = nn.Linear(encoder_layers[-1], latent_dim)

        self.decode = self.layer_opt(latent_dim, decoder_layers, dropout_dec, output_shape)


    def layer_opt(self, input_dim, layer_dims, dropout_bool, output_dim=None):
        """
        Constructs a sequential model with specified layers and dropout.

        Args:
            input_dim (int): The dimensionality of the input data.
            layer_dims (list of int): List of dimensions for each layer.
            dropout_bool (list of bool): List indicating whether to apply dropout for each layer.
            output_dim (int, optional): The dimensionality of the output data. Defaults to None.

        Returns:
            nn.Sequential: A sequential model with the specified layers and dropout.
        """
        layers = []
        for dim, drop in zip(layer_dims, dropout_bool):
            layers.append(nn.Linear(input_dim, dim, bias=True))
            layers.append(nn.LeakyReLU())

            if drop:
                layers.append(nn.Dropout(0.2))

            input_dim = dim
        
        if output_dim:
            layers.append(nn.Linear(input_dim, output_dim, bias=True))

        return nn.Sequential(*layers)
            
    def encoder(self, x):
        """
        Encodes the input data into mean and log variance of the latent space.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The mean of the latent space.
            torch.Tensor: The log variance of the latent space.
        """
        base = self.encoder_base(x)
        mean = self.encoder_mu(base)
        logvar = self.encoder_std(base)
        return mean, logvar
    
    def decoder(self, x):
        """
        Decodes the latent space representation back to the original data space.

        Args:
            x (torch.Tensor): The latent space representation.

        Returns:
            torch.Tensor: The reconstructed data.
        """
        reconstruction= self.decode(x)
        return reconstruction

    def reparameterize(self, mean, logvar):
        """
        Reparameterizes the latent space using the mean and log variance.

        Args:
            mean (torch.Tensor): The mean of the latent space.
            logvar (torch.Tensor): The log variance of the latent space.

        Returns:
            torch.Tensor: The reparameterized latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)

        return reconstruction, mean, logvar
    
    def generate_histogram(self, wilson_coefficient):
        self.eval()
        with torch.no_grad():
            mean, logvar = self.encoder(wilson_coefficient)
            z = self.reparameterize(mean, logvar)
            generated_histogram = self.decoder(z)
            return generated_histogram

# =====================================================================

def loss_function(reconstruction, x, mean, logvar, beta=0.001):
    # Reconstruction loss (mean squared error)
    MSE = nn.functional.mse_loss(reconstruction, x, reduction='mean')
    
    return MSE

def loss_function_KLD(reconstruction, x, mean, logvar, beta=0.1, target_std=1.0):
    MSE = torch.nn.functional.mse_loss(reconstruction, x, reduction='mean')

    target_logvar = torch.log(torch.tensor(target_std ** 2)).to(x.device)
    KLD = -0.5 * torch.mean(1 + (logvar - target_logvar) 
                                - (logvar.exp() / target_std ** 2))
    
    loss = MSE + beta * KLD
    return loss

# =====================================================================

def train_epoch_early_CVAE(model, dataloader, dataloader_validation, criterion, optimizer, device):
    """
    Conducts one epoch of training on the provided model.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer for gradient descent.
        device (torch.device): The device to perform training on (CPU or GPU).
    
    Returns:
        float: Average loss over the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    total_MSE = 0.0
    total_KLD = 0.0

    num_batches = len(dataloader)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device
        
        optimizer.zero_grad()  # Zero the gradients

        reconstruction, mean, logvar = model(inputs)  # Forward pass
        loss, MSE, KLD = criterion(reconstruction, targets, mean, logvar)  # Calculate loss

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
        total_MSE += MSE.item()
        total_KLD += KLD.item() 

    with torch.no_grad():
        model.eval()

        total_validation_loss = 0.0
        num_batches_validation = len(dataloader_validation)

        for batch_idx, (inputs, targets) in enumerate(dataloader_validation):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, mu, logstd = model(inputs)
            validation_loss, _, _ = criterion(outputs, targets, mu, logstd)
            
            total_validation_loss += validation_loss.item()

        average_validation_loss = total_validation_loss / num_batches_validation
        
    average_loss = total_loss / num_batches  # Calculate average loss for the epoch
    avg_MSE = total_MSE /num_batches
    avg_KLD = total_KLD / num_batches

    return average_loss, average_validation_loss, avg_MSE, avg_KLD

# =====================================================================

def train_epoch_CVAE(model, dataloader, criterion, optimizer, device):
    """
    Conducts one epoch of training on the provided model.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer for gradient descent.
        device (torch.device): The device to perform training on (CPU or GPU).
    
    Returns:
        float: Average loss over the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device
        
        optimizer.zero_grad()  # Zero the gradients

        reconstruction, mean, logvar = model(inputs)  # Forward pass
        loss, _, _ = criterion(reconstruction, targets, mean, logvar)  # Calculate loss

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        total_loss += loss.item()  # Accumulate the loss

    average_loss = total_loss / num_batches  # Calculate average loss for the epoch
    return average_loss

# =====================================================================

def test_epoch_CVAE(model, dataloader, criterion, device):
    """
    Conducts one epoch of testing on the provided model.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): DataLoader for the testing dataset.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to perform testing on (CPU or GPU).
    
    Returns:
        float: Average loss over the epoch.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device
            
            reconstruction, mean, logvar = model(inputs)  # Forward pass
            loss, _, _ = criterion(reconstruction, targets, mean, logvar)  # Calculate loss
            
            total_loss += loss.item()  # Accumulate the loss

    average_loss = total_loss / num_batches  # Calculate average loss for the epoch
    return average_loss

# =====================================================================

def test_visualise_4d_CVAE(model, dataloader, device, train_index, out_path, nbins, filenames):
    """
    Visualizes the model outputs for a given dataset in 4D and saves the results to CSV files.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device to perform evaluation on (CPU or GPU).
        train_index (int): The starting index for the file pairs.
        out_path (str): The output directory to save the results.
        nbins (int): Number of bins for the histogram.
        file_pairs (list): List of file pairs containing data and configuration paths.
    
    Returns:
        pd.DataFrame: DataFrame containing the model outputs.
    """
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device
            
            reconstruction = model.generate_histogram(inputs)   # Forward pass
            
            outputs = reconstruction.to('cpu')

            fname = os.path.basename(f'{filenames[train_index + batch_idx]}.csv')
            target_file = out_path + f'/normal_targets/{fname}'

            eos_file = pd.read_csv(target_file)

            # Flatten the arrays and create the resulting DataFrame
            result_df = pd.DataFrame({
                'q2': eos_file['q2'].values,
                'cos_theta_l': eos_file['cos_theta_l'].values,
                'cos_theta_d': eos_file['cos_theta_d'].values,
                'phi': eos_file['phi'].values,
                'bin_height': outputs.ravel()
            })

            os.makedirs(os.path.join(out_path, 'model_outputs'), exist_ok=True)
            result_df.to_csv(os.path.join(out_path, 'model_outputs', fname), index=False)

    return result_df

# =====================================================================

def BO_CVAE(path, train_inputs, train_targets):
    """
        Builds a configuration dictionary for the Convolutional Variational Autoencoder (CVAE) 
        using Bayesian Optimization results.

        Args:
            path (str): The path to the JSON file containing the optimized hyperparameters.
            train_inputs (list): The training input data.
            train_targets (list): The training target data.

        Returns:
            dict: A dictionary containing the configuration for the CVAE model.
    """
    cfg_CVAE = {'batch_train' : 64,
                    'batch_test' : 8,
                    'epochs' : 150,
                    'dim': [len(train_inputs[0]), len(train_targets[0])],
                    }
    
    opt_path = path.split('/')[1] + '.json'

    with open(opt_path, 'r') as f:
            opt = json.load(f)

    n_enc = opt['n_encoder_layers']
    n_dec = opt['n_decoder_layers']

    enc = []
    enc_drop = []
    dec = []
    dec_drop = []

    for n in range(n_enc):
        enc.append(opt[f'encoder_layer_{n}'])
        enc_drop.append(opt[f'dropout_enc_{n}'])

    for m in range(n_dec):
        dec.append(opt[f'decoder_layer_{m}'])
        dec_drop.append(opt[f'dropout_dec_{m}'])

    latent_shape = opt['latent_dim']

    cfg_CVAE['encoder'] = enc
    cfg_CVAE['encoder_drop'] = enc_drop

    cfg_CVAE['decoder'] = dec
    cfg_CVAE['decoder_drop'] = dec_drop

    cfg_CVAE['latent_dim'] = latent_shape

    return cfg_CVAE

# =====================================================================