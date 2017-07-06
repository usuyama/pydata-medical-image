
# Diabetic Retinopathy classification using 2D CNN

## Data

Download the dataset (retina images) from the Kaggle competition page.

1. Sign up https://www.kaggle.com/c/diabetic-retinopathy-detection
2. Log-in and download images from https://www.kaggle.com/c/diabetic-retinopathy-detection/data
3. Put the data under `inputs` folder

> A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:
>- 0 - No DR
>- 1 - Mild
>- 2 - Moderate
>- 3 - Severe
>- 4 - Proliferative DR

## Prerequisite

- CNTK and Keras (https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras)
- numpy
- pandas
- opencv

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
