# We import the necessary modules from PyTorch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A simple Convolutional Neural Network for binary image classification.
    """
    
    def __init__(self):
        """
        Constructor: Defines all the layers the network will use.
        """
        super(CNN, self).__init__()

        # --- Convolutional Block 1 ---
        # Input shape: (Batch_Size, 3, 150, 150)
        # 3 input channels (RGB), 32 output channels, 3x3 kernel, 1 padding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Output after conv1: (Batch_Size, 32, 150, 150)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 max pooling
        # Output after pool1: (Batch_Size, 32, 75, 75)

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Output after conv2: (Batch_Size, 64, 75, 75)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output after pool2: (Batch_Size, 64, 37, 37)

        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Output after conv3: (Batch_Size, 128, 37, 37)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output after pool3: (Batch_Size, 128, 18, 18) 
        # (37 / 2 = 18.5 -> rounds down to 18)

        # --- Fully-Connected (Classifier) Block ---
        # We must flatten the 128x18x18 feature map
        self.fc1 = nn.Linear(in_features=128 * 18 * 18, out_features=512)
        self.dropout = nn.Dropout(0.5) # Dropout layer to prevent overfitting
        self.fc2 = nn.Linear(in_features=512, out_features=1) # 1 output for binary (Cat vs Dog)

    def forward(self, x):
        """
        Defines the forward pass: how data flows through the layers.
        """
        # Pass through Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Pass through Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Pass through Conv Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten the output for the linear layers
        # The -1 automatically calculates the batch size
        x = x.view(-1, 128 * 18 * 18) 
        
        # Pass through FC Block
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) # Final output (logits)
        
        # We don't apply sigmoid here because we'll use
        # nn.BCEWithLogitsLoss for better numerical stability.
        return x