# Auralyse_API

A Flask API to load .wav audio and get back valence arousal charts for each second of it.  

The API has been applied along with Docker Intel Tensorflow image 

With the current setup this application can be easily hosted on to any of the platforms for an API.

This has been successfully tested by loading into DigitalOcean.  
To help with the implementation on DigitalOcean, please follow the links  
https://docs.digitalocean.com/products/app-platform/getting-started/sample-apps/flask/  
https://docs.digitalocean.com/products/app-platform/how-to/deploy-from-container-images/

The deployement is done using container via Dockerfile. Currently, we have used TensorFlow based image here.
Other DockerFiles can be utilised, with emphasis on the python and the related package versions.  

A connection to github codebase would mean automatic refresh of the application when changes are made to the code.  

Also included is the jupyter notebook which was used to create Machine Learning models https://github.com/TusharBabuHub/Auralyse_API/blob/main/auralyse_model_training.ipynb
