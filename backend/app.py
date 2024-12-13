from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
import io

# init FastAPI
app = FastAPI()

# Load the model
model_path = "./model.pth"
# Set cpu if available
device = torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model (resnet18)
from torchvision.models

model = resnet18(pretrained=false)
