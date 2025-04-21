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

def train_epoch_early(model, dataloader, dataloader_validation, criterion, optimizer, device):
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

    with torch.no_grad():
        model.eval()

        total_validation_loss = 0.0
        num_batches_validation = len(dataloader_validation)

        for batch_idx, (inputs, targets) in enumerate(dataloader_validation):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            validation_loss = criterion(outputs, targets)
            
            total_validation_loss += validation_loss.item()

    
    average_validation_loss = total_validation_loss / num_batches_validation
        
    average_loss = total_loss / num_batches  # Calculate average loss for the epoch

    return average_loss, average_validation_loss

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

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device
            
            outputs = model(inputs)  # Forward pass
            
            outputs = outputs.to('cpu')

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

            os.makedirs(os.path.join(out_path, 'linear_model_outputs'), exist_ok=True)
            result_df.to_csv(os.path.join(out_path, 'linear_model_outputs', fname), index=False)

    return result_df

# =====================================================================