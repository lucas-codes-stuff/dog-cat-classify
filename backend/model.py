import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import os
import time

# Paths to training and validation data
train_dir = './data/train'
val_dir = './data/val'

# Preprocessing for training and validation (no augmentation on validation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Resize image to 224x224
    transforms.RandomHorizontalFlip(), # Randomly flip image horizontally
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize image
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Resize image to 224x224
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize image
])

# Load training and validation data
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Load pre-trained ResNet model
model = models.resnet18(weights=None)

# Replace the final later with the two output classes
num_ftrs = model.fc.in_features # Number of input features to the final layer
model.fc = nn.Linear(num_ftrs, 2) # 2 classes: cat and dog

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) # Move data to GPU (if available)

        optimizer.zero_grad() # Zero the parameter gradients to avoid accumulation

        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        running_loss += loss.item() # Accumulate loss


    # Evaluate the model on the validation set
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) # Move data to GPU (if available)

            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            val_loss += loss.item() # Accumulate loss

            _, predicted = torch.max(outputs, 1) # Get the predicted class label
            total += labels.size(0) # Accumulate the number of total labels
            correct += (predicted == labels).sum().item() # Accumulate the number of correct labels

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Save the model
model_path = './model.pth'
torch.save(model.state_dict(), model_path)