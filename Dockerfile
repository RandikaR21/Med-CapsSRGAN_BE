FROM python:3.7.3-stretch

# Make working directories
RUN  mkdir -p  /med-capssrgan-api
WORKDIR  /med-capssrgan-api

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy every file in the source folder to the created working directory
#COPY  /sr_model /sr_model
COPY fastAPI.py .

# Run the python application
CMD ["python", "fastAPI.py"]