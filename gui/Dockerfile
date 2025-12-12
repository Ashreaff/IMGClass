# Base image with Python 3.9
FROM python:3.9-slim

# Install Gradio and Requests
RUN pip install gradio requests
RUN pip install python-multipart
# Copy the GUI script
COPY gui.py /app/gui.py

# Set the working directory
WORKDIR /app

# Expose port 7860 for Gradio
EXPOSE 7860

CMD ["python", "gui.py"]
