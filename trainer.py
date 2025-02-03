import torch
import numpy as np
import VRNNVotingModel
import MVRNNAnomalyQuality


def  create_model(input_dim,latent_dim,type_model):
    if type_model=='GRU_LSTM':
        model = VRNNVotingModel(input_dim=input_dim, latent_dim=latent_dim).to(device)
        return model
    else :
        model = MVRNNAnomalyQuality(input_dim=input_dim, latent_dim=latent_dim,state_type=type_model).to(device)
        return model
    return model



import wandb

def train_and_test(model, optimizer, train_loader, test_loader, 
                   name_project="anomaly-detection-listm_200", 
                   epochs=200, step_eval=5, type_train='Long'):
    """
    Initializes a W&B project and trains the model using the specified training type.
    
    Parameters:
    - model: The deep learning model to be trained.
    - optimizer: The optimizer for training.
    - train_loader: DataLoader for training data.
    - test_loader: DataLoader for testing data.
    - name_project: Name of the W&B project.
    - epochs: Number of training epochs.
    - step_eval: Frequency of evaluation steps.
    - type_train: Training mode ('Long' or 'sciles').

    Returns:
    - Training output results.
    """

    # Initialize Weights & Biases (W&B)
    wandb.init(
        project=name_project,
        name="trajectory-model-listall",
        config={
            "epochs": epochs,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
    )

    # Train using the specified method
    if type_train == 'Long':
        outs = train_model_wandb_long(model, optimizer, train_loader, test_loader, 
                                      epochs=epochs, step_eval=step_eval, 
                                      name_project=name_project)
    else:
        outs = train_model_sciles(model, optimizer, train_loader, test_loader, 
                                  epochs=epochs, step_eval=step_eval, 
                                  name_project=name_project)
    
    return outs



import torch

def train_model(inputs):
    """
    Function to train a model using specified parameters.

    Parameters:
    - inputs: A tuple containing:
      - input_dim: Dimension of input data
      - latent_dim: Dimension of latent space
      - type_model: Type of model (e.g., 'Long', 'sciles')
      - train_loader: DataLoader for training data
      - test_loader: DataLoader for test data

    Returns:
    - outs: The output from the training process
    """
    input_dim, latent_dim, type_model, train_loader, test_loader = inputs
    model = create_model(input_dim, latent_dim, type_model)

    name_project_wandb = 'anomaly-detection-listm_200'
    epochs = 20
    step_eval = 5

    # Optimizer settings
    lr = 0.001
    beta0 = 0.8
    beta1 = 0.999
    eps = 1e-08
    weight_decay = 0.0

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        betas=(beta0, beta1), 
        eps=eps, 
        weight_decay=weight_decay
    )

    type_train = 'sciles'  # Choose between 'Long' and 'sciles'
    
    if type_train == 'Long':
        outs = train_model_wandb_long(model, optimizer, train_loader, test_loader, epochs=epochs, step_eval=step_eval, name_project=name_project_wandb)
    else:
        outs = train_model_sciles(model, optimizer, train_loader, test_loader, epochs=epochs, step_eval=step_eval, name_project=name_project_wandb)

    # Save model
    torch.save(outs[0].state_dict(), f'/content/drive/MyDrive/t/model_{type_model}.pth')

    return outs
from concurrent.futures import ThreadPoolExecutor
import wandb

def train_models_in_parallel(dataset, train_loader, test_loader):
    """
    Function to train models in parallel using ThreadPoolExecutor.

    Parameters:
    - dataset: Dataset containing the total number of bins (total_bins)
    - train_loader: DataLoader for training data
    - test_loader: DataLoader for test data

    Returns:
    - results: List of outputs from the parallel training tasks
    """
    # Define the models and parameters
    typemodels = ['Hybrid']
    input_dim = dataset.total_bins
    latent_dim = 100
    max_workers = 10
    name_project = 'anomaly-detection-listm_200'

    # Initialize the wandb project
    wandb.init(
        project=name_project,
        name="trajectory-model-listall",
        config={
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
    )

    # Prepare inputs for each model
    inputs = []
    for type_model in typemodels:
        inputs.append((input_dim, latent_dim, type_model, train_loader, test_loader))

    # Train models in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(train_model, inputs))

    return results

