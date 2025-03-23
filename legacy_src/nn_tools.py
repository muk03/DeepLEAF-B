import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import os
import pandas as pd
import random
import json

# =====================================================================

class base_model(nn.Module):
    """
    A base neural network model with multiple linear layers and LeakyReLU activations.

    Attributes:
        in_dim (int): The input dimension of the model.
        out_dim (int): The output dimension of the model.
        model (nn.Sequential): The sequential container of the model layers.

    Methods:
        _initialize_weights(): Initializes the weights of the linear layers using Xavier uniform initialization.
        forward(x): Defines the forward pass of the model.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initializes the base_model with the given input and output dimensions.

        Args:
            input_dim (int): The dimension of the input features.
            output_dim (int): The dimension of the output features.
        """
        super().__init__()
        self.in_dim  = input_dim
        self.out_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(self.in_dim, 4 * self.in_dim, bias=True),
            nn.Tanh(), # Changed to Tanh here
            nn.Linear(4 * self.in_dim, 8 * self.in_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(8 * self.in_dim, 16 * self.in_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(16 * self.in_dim, 32 * self.in_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(32 * self.in_dim, 64 * self.in_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(64 * self.in_dim, 128 * self.in_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128 * self.in_dim),
            nn.Linear(128 * self.in_dim, 256 * self.in_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(256 * self.in_dim, 512 * self.in_dim, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512 * self.in_dim, 1024 * self.in_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(1024 * self.in_dim, self.out_dim, bias=True),
            nn.Softplus(), # Changed from Sigmoid to ReLU here
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the linear layers using Xavier uniform initialization.
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        output = self.model(x)
        return output

# =====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
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
        outputs = model(inputs)  # Forward pass

        loss = criterion(outputs, targets)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        total_loss += loss.item()  # Accumulate the loss

    average_loss = total_loss / num_batches  # Calculate average loss for the epoch
    return average_loss

# =====================================================================

def test_epoch(model, dataloader, criterion, device):
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
            
            outputs = model(inputs)  # Forward pass
            
            loss = criterion(outputs, targets)  # Calculate loss
            
            total_loss += loss.item()  # Accumulate the loss

    average_loss = total_loss / num_batches  # Calculate average loss for the epoch
    return average_loss

# =====================================================================

def test_visualise_4d(model, dataloader, device, train_index, out_path, nbins, filenames):
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

    bin_edges = np.linspace(0, 1, nbins + 1)
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device
            
            outputs = model(inputs)  # Forward pass
            
            outputs = outputs.to('cpu')
            reshaped_outputs = np.array(outputs.reshape(nbins, nbins, nbins, nbins))

            fname = os.path.basename(f'{filenames[train_index + batch_idx]}.csv')

            # Create meshgrid for the bin centers
            xpos, ypos, zpos, wpos = np.meshgrid(bins, bins, bins, bins, indexing="ij")

            # Flatten the arrays and create the resulting DataFrame
            result_df = pd.DataFrame({
                'q2': xpos.ravel(),
                'cos_theta_l': ypos.ravel(),
                'cos_theta_d': zpos.ravel(),
                'phi': wpos.ravel(),
                'bin_height': reshaped_outputs.ravel()
            })

            os.makedirs(os.path.join(out_path, 'model_outputs'), exist_ok=True)
            result_df.to_csv(os.path.join(out_path, 'model_outputs', fname), index=False)

    return result_df

# =-=-=-=-=-=-= CVAE IMPLEMENTATION - INITIALISATION TRAINING AND  =-=-=-=-=-=-=-=
class CVAE(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dim):
        super(CVAE, self).__init__()
        self.encoder_base = nn.Sequential(
            nn.Linear(input_shape, 128, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64, bias=True),
            nn.LeakyReLU(),
        )

        self.encoder_mu = nn.Sequential(
            nn.Linear(64, latent_dim),
            )
        
        self.encoder_std = nn.Sequential(
            nn.Linear(64, latent_dim),
            )

        self.decode = nn.Sequential(
            nn.Linear(latent_dim, 128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(128, 256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(256, output_shape, bias=True),
        )

    def encoder(self, x):
        base = self.encoder_base(x)
        mean = self.encoder_mu(base)
        logvar = self.encoder_std(base)

        return mean, logvar
    
    def decoder(self, x):
        reconstruction= self.decode(x)
        return reconstruction

    def reparameterize(self, mean, logvar):
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
    
def loss_function(reconstruction, x, mean, logvar, beta=0.001):
    # Reconstruction loss (mean squared error)
    MSE = nn.functional.mse_loss(reconstruction, x, reduction='mean')

    return MSE

def loss_function_KLD(reconstruction, x, mean, logvar, beta=1e-2):

    MSE = nn.functional.mse_loss(reconstruction, x, reduction='mean')
    
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    loss = MSE + beta * KLD
    return loss

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
        loss = criterion(reconstruction, targets, mean, logvar)  # Calculate loss

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        total_loss += loss.item()  # Accumulate the loss

    average_loss = total_loss / num_batches  # Calculate average loss for the epoch
    return average_loss

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
            loss = criterion(reconstruction, targets, mean, logvar)  # Calculate loss
            
            total_loss += loss.item()  # Accumulate the loss

    average_loss = total_loss / num_batches  # Calculate average loss for the epoch
    return average_loss

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

    bin_edges = np.linspace(0, 1, nbins + 1)
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device
            
            reconstruction = model.generate_histogram(inputs)   # Forward pass
            
            outputs = reconstruction.to('cpu')
            reshaped_outputs = np.array(outputs.reshape(nbins, nbins, nbins, nbins))

            fname = os.path.basename(f'{filenames[train_index + batch_idx]}.csv')

            # Create meshgrid for the bin centers
            xpos, ypos, zpos, wpos = np.meshgrid(bins, bins, bins, bins, indexing="ij")

            # Flatten the arrays and create the resulting DataFrame
            result_df = pd.DataFrame({
                'q2': xpos.ravel(),
                'cos_theta_l': ypos.ravel(),
                'cos_theta_d': zpos.ravel(),
                'phi': wpos.ravel(),
                'bin_height': reshaped_outputs.ravel()
            })

            os.makedirs(os.path.join(out_path, 'model_outputs'), exist_ok=True)
            result_df.to_csv(os.path.join(out_path, 'model_outputs', fname), index=False)

    return result_df

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
    num_batches = len(dataloader)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device
        
        optimizer.zero_grad()  # Zero the gradients

        reconstruction, mean, logvar = model(inputs)  # Forward pass
        loss = criterion(reconstruction, targets, mean, logvar)  # Calculate loss

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        total_loss += loss.item()  # Accumulate the loss

    with torch.no_grad():
        model.eval()

        total_validation_loss = 0.0
        num_batches_validation = len(dataloader_validation)

        for batch_idx, (inputs, targets) in enumerate(dataloader_validation):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, mu, logstd = model(inputs)
            validation_loss = criterion(outputs, targets, mu, logstd)
            
            total_validation_loss += validation_loss.item()

        average_validation_loss = total_validation_loss / num_batches_validation
        
    average_loss = total_loss / num_batches  # Calculate average loss for the epoch

    return average_loss, average_validation_loss

# =-=-=-=-=-=-= BAYESIAN OPTIMISATION CLASS DEFINITIONS =-=-=-=-=-=-=-=

class CVAE_Opt(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dim, encoder_layers, decoder_layers, dropout_enc, dropout_dec):
        super(CVAE_Opt, self).__init__()

        self.encoder_base = self.layer_opt(input_shape, encoder_layers, dropout_enc)

        self.encoder_mu = nn.Linear(encoder_layers[-1], latent_dim, bias=True)
        self.encoder_std = nn.Linear(encoder_layers[-1], latent_dim)

        self.decode = self.layer_opt(latent_dim, decoder_layers, dropout_dec, output_shape)


    def layer_opt(self, input_dim, layer_dims, dropout_bool, output_dim=None):
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
        base = self.encoder_base(x)
        mean = self.encoder_mu(base)
        logvar = self.encoder_std(base)
        return mean, logvar
    
    def decoder(self, x):
        reconstruction= self.decode(x)
        return reconstruction

    def reparameterize(self, mean, logvar):
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
        
def BO_CVAE(path, train_inputs, train_targets):
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

