# Base image with PyTorch and CUDA (for GPU support)
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Install necessary libraries
RUN pip install fastapi uvicorn timm pillow torchvision
RUN pip install python-multipart

# Copy the inference script and class file
COPY inference.py /app/inference.py
COPY imagenet_classes.txt /app/imagenet_classes.txt

# Set the working directory
WORKDIR /app

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run FastAPI server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
