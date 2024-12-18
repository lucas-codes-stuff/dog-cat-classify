# Trigger rebuild

# Build the backend
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install the dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and the model
COPY backend ./backend

# Copy React build files into the backend folder
# COPY frontend/build ./backend/build

# Expose the port
EXPOSE 7860

# Run the backend
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]