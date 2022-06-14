FROM tensorflow/tensorflow

# Make working directories
RUN  mkdir -p  /med-capssrgan-api
WORKDIR  /med-capssrgan-api

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements.txt .


# Install application dependencies from the requirements file
RUN pip install --no-cache-dir --upgrade -r requirements.txt


RUN apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install python-multipart

# Copy every file in the source folder to the created working directory
COPY  /sr_model /med-capssrgan-api/sr_model
COPY /GeneratorToDeploy /med-capssrgan-api/GeneratorToDeploy
COPY fastAPI.py /med-capssrgan-api

# Run the python application
CMD ["uvicorn", "fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]