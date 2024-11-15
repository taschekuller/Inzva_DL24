import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os

def visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx):
    '''
    Visualizes and saves sample predictions for a given batch of images, masks, and model outputs.

    Args:
    - images (torch.Tensor): Input images (batch of tensors).
    - masks (torch.Tensor): Ground truth segmentation masks (batch of tensors).
    - outputs (torch.Tensor): Model outputs (batch of tensors).
    - save_path (str): Directory path where the visualization will be saved.
    - epoch (int): Current epoch number (for labeling the file).
    - batch_idx (int): Index of the current batch (for labeling the file).
    
    Functionality:
    - Displays and saves the first few samples from the batch, showing the input images, ground truth masks, and predicted masks.
    - Applies a sigmoid function to the outputs and uses a threshold of 0.5 to convert them to binary masks.
    '''
    sample_idx = random.sample(range(images.shape[0]), 3)
    images, masks, outputs = images[sample_idx], masks[sample_idx], outputs[sample_idx]
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    masks = masks.permute(0, 2, 3, 1).cpu().numpy()
    outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, ax in enumerate(axes):
        ax[0].imshow(images[i].squeeze(), cmap='gray')
        ax[0].set_title('Input Image')
        ax[0].axis('off')
        ax[1].imshow(masks[i].squeeze(), cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')
        ax[2].imshow(outputs[i].squeeze(), cmap='gray')
        ax[2].set_title('Predicted Mask')
        ax[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'epoch_{epoch}_batch_{batch_idx}.jpg'))
    plt.close()

def plot_train_val_history(train_loss_history, val_loss_history, plot_dir, args):
    '''
    Plots and saves the training and validation loss curves.

    Args:
    - train_loss_history (list): List of training loss values over epochs.
    - val_loss_history (list): List of validation loss values over epochs.
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    
    Functionality:
    - Plots the train and validation loss curves.
    - Saves the plot as a JPG file in the specified directory.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'loss_plot.jpg'))
    plt.close()
    
def plot_metric(x, label, plot_dir, args, metric):
    '''
    Plots and saves a metric curve over epochs.

    Args:
    - x (list): List of metric values over epochs.
    - label (str): Label for the y-axis (name of the metric).
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    - metric (str): Name of the metric (used for naming the saved file).
    
    Functionality:
    - Plots the given metric curve.
    - Saves the plot as a JPEG file in the specified directory.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(x)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.title(f'{metric} over Epochs')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'{metric}.jpg'))
    plt.close()
    