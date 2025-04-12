# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container first
# This leverages Docker layer caching - dependencies are only reinstalled if requirements.txt changes
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size and virtual env avoids conflicts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and data into the container
# This includes app.py, the data/ directory, and the PRE-BUILT chroma_db/
COPY . .

# Make port 8501 available (Streamlit default)
EXPOSE 8501

# Define healthcheck (tells Docker if the app is running correctly)
HEALTHCHECK CMD streamlit healthcheck

# Run app.py when the container launches using streamlit
# Use --server.port and --server.address for compatibility with cloud hosting
# DO NOT include the --server.fileWatcherType none flag here - it's not needed inside Docker typically
# Near the end of your Dockerfile
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]