from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import torch
from torchvision import transforms
import io
import os
import torch.nn.functional as F

# init FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ONLY WORKS IN DOCKER CONTAINER
# Serve React static files
app.mount("/assets", StaticFiles(directory="backend/build/assets"), name="assets")

# Serve the React index.html
@app.get("/")
async def serve_react_app():
    return FileResponse(os.path.join("backend", "build", "index.html"))

# Catch-all route to serve index.html for React Router
@app.get("/{full_path:path}")
async def serve_static(full_path: str):
    return FileResponse(os.path.join("backend", "build", "index.html"))


# Load the model
model_path = os.path.join(os.path.dirname(__file__), "model.pth")
# Set cpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define model (resnet18)
from torchvision.models import resnet18

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2) # Cats and dogs
model.load_state_dict(torch.load(model_path, map_location=device)) # Load model
model.to(device) # Move model to device
model.eval() # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Class labels
classes = ['Cat', 'Dog']

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Make a prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1) # Convert logits to probabilities
            confidence, predicted = torch.max(probabilities, 1) # Get the highest predicted class label and confidence
        
        # Return the prediction
        return {
            'class': classes[predicted.item()],
            'confidence': f"{confidence.item() * 100:.2f}%" # Convert to percentage
        }
    
    except Exception as e:
        return {
            'error': str(e)
        }
