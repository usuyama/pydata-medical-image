# Lung nodule detection using 3D Convolutional Neural Network

See the [Jupyter notebooks](./notebooks) for the scripts and explanations.

## Data

Download the dataset (CT scan images) from the LUNA16 page.

1. Sign up https://luna16.grand-challenge.org/
2. Log-in and download images from https://luna16.grand-challenge.org/download/
3. Put the data under `inputs` folder

## Prerequisite

- CNTK and keras (https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras)
- SimpleITK
- numpy
- scipy.ndimage

### CNTK and Keras
If you'd like to use CNTK as Keras backend, don't forget to set the environment variable:

    export KERAS_BACKEND=cntk
    
or update the Keras configuration file:

    {
        "epsilon": 1e-07, 
        "image_data_format": "channels_last", 
        "backend": "cntk", 
        "floatx": "float32" 
    }

Note that you don't need to change the Keras frontend codes for other backends (tensorflow or theano).

Please be aware that CNTK Keras support is in beta and there're known issues. Check the [page](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras) and the issue tracker on GitHub.
