# Med-CapSRGAN
**Back-End and Deep Learning Component**

*A Final Year Project by Randika Rodrigo*


## Training the model
1. Execute `generator_trainer.py` to train pre-train the generator network.
2. Execute `gan_trainer.py to train` GAN with the Capsule Network Discriminator.

## API with Pre-Trained Model
### Run Locally
To run the API, execute the command
    ```
    uvicorn fastAPI:app --reload
    ```
### With Docker
1. Build Docker Image
    ```
    docker build -t fyp_api_amd -f .\AMD_Dockerfile .
   ```
2. Run the Docker Image
    ```
   docker run -d -p 8000:8000 --name FYP_API_AMD fyp_api_amd
   ```


