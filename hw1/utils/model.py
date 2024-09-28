import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes, method='random_normal'):
        '''
        Initializes the neural network by setting up the layers and applying the selected weight initialization method.

        Args:
        - input_size (int): The number of input features.
        - num_classes (int): The number of output classes for classification.
        - method (str): The weight initialization method to be used for the layers. Options include:
            'xavier' for Xavier initialization,
            'kaiming' for Kaiming initialization,
            'random_normal' for random normal initialization.
        
        Layers:
        - fc1 to fc6: Fully connected layers with ReLU activation in between.
        - Consider this as a based architecture we have given you. You can play with it as you want.
        '''
        super(NeuralNet, self).__init__()

        # Define the layers
        self.fc1  = nn.Linear(input_size, 800) 
        self.fc2  = nn.Linear(800, 512)
        self.fc3  = nn.Linear(512, 456)
        self.fc4  = nn.Linear(456, 256)
        self.fc5  = nn.Linear(256, 100)
        self.fc6  = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

        # Apply the chosen initialization method to the weights
    
    def initialize_weights(self, method):
        '''
        Initializes the weights of the linear layers according to the specified method.

        Args:
        - method (str): The initialization method for the weights. Can be one of:
            - 'xavier': Applies Xavier uniform initialization.
            - 'kaiming': Applies Kaiming uniform initialization.
            - 'random_normal': Initializes weights from a normal distribution with mean 0 and standard deviation 0.01.

        '''
        
        
        method = torch.nn.init.xavier_uniform_(self.fc1.weight)
        
        

        
        



    def forward(self, x):
        '''
        Defines the forward pass of the neural network, passing the input through all the layers and ReLU activations.

        Args:
        - x (torch.Tensor): Input tensor containing the batch of data with dimensions matching the input_size.

        Returns:
        - torch.Tensor: The output tensor, typically used for classification.
        
        Layers are processed in the following order:
        - fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> ReLU -> fc4 -> ReLU -> fc5 -> ReLU -> fc6.
        '''
        
        fc1 = self.fc1(x)
        relu1 = self.relu(fc1)
        fc2 = self.fc2(relu1)
        relu2 = self.relu(fc2)
        fc3 = self.fc3(relu2)
        relu3 = self.relu(fc3)
        fc4 = self.fc4(relu3)
        relu4 = self.relu(fc4)
        fc5 = self.fc5(relu4)
        relu5 = self.relu(fc5)
        fc6 = self.fc6(relu5)
        out = self.relu(fc6)

        return out
