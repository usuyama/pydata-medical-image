## PyData Seattle 2017: Medical Image Analysis

This repository is for storing scripts/notebooks for "Medical image processing using Microsoft Deep Learning framework (CNTK)" (https://pydata.org/seattle2017/schedule/presentation/94/).

The presentation slide is available [here](./PyData_medical_images.pdf).

The talk provides 2 examples:
- [2D-CNN: Diabetic Retinopathy Classification](./diabetic_retinopathy)
- [3D-CNN: Lung Nodule Detection](./lung_nodule)

### Machine Environment

It is recommended to prepare a machine with a GPU if you wish to train the 2D/3D deep learning models in the examples. If you don't have one at hand, [Azure Data Science VM](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-virtual-machine-overview) with [N-series VM](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/) is a great option. As of July 2017, you can choose from Tesla M60 and K80.
