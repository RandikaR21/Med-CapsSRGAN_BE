FROM python:3.8

RUN apt-get update \
    && apt-get install -y \
        cmake libsm6 libxext6 libxrender-dev protobuf-compiler \
    && rm -r /var/lib/apt/lists/*

RUN useradd -m randika

COPY --chown=randika:randika . /home/randika/app

USER randika

WORKDIR /home/randika/app


# Copy application requirements file to the created working directory
COPY requirements.txt .

RUN pip install --upgrade pip
# Install application dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir opencv-python
RUN pip install --no-cache-dir python-multipart

# Copy every file in the source folder to the created working directory
COPY  /sr_model /home/randika/app/sr_model
COPY /GeneratorToDeploy /home/randika/app/GeneratorToDeploy
COPY fastAPI.py /home/randika/app

# Run the python application
CMD ["uvicorn", "fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]