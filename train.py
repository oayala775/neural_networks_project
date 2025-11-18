import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PetImageDataset
from convolutional_nn import CNN
from tqdm import tqdm

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "datasets/PetImages_Processed/train"
VAL_DIR = "datasets/PetImages_Processed/val"
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10

def train_one_epoch(model, loader, optimizer, criterion):
    """
    Runs a single training epoch.
    """
    model.train() 
    running_loss = 0.0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        # Move data to the selected device (GPU or CPU)
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass: get model predictions
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update the model's weights
        optimizer.step()
        
        # Accumulate the loss
        running_loss += loss.item() * inputs.size(0)
    
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(loader.dataset)
    print(f"Train Loss: {epoch_loss:.4f}")

def validate_one_epoch(model, loader, criterion):
    """
    Runs a single validation epoch.
    """
    model.eval() # Set the model to evaluation mode (disables dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            # Apply sigmoid to get probabilities (0 to 1)
            # Then threshold at 0.5 to get binary predictions (0 or 1)
            preds = torch.sigmoid(outputs) > 0.5
            
            # Count correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    print(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")

def main():
    """
    Main function to run the training process.
    """
    print(f"Using device: {DEVICE}")

    # Load Data
    train_dataset = PetImageDataset(data_dir=TRAIN_DIR)
    val_dataset = PetImageDataset(data_dir=VAL_DIR)
    
    # Create DataLoaders to manage batches
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = CNN().to(DEVICE) # Create model and move it to the device
    
    # Loss function for binary classification
    # This is numerically stabler than Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_one_epoch(model, train_loader, optimizer, criterion)
        validate_one_epoch(model, val_loader, criterion)
    
    print("Finished Training.")
    
    #* Uncomment to save the model
    # torch.save(model.state_dict(), "pet_cnn_model.pth")
    # print("Model saved.")

if __name__ == "__main__":
    main()