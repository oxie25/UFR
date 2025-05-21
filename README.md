# UFR (Universal Fundamental Relations)
[![DOI](https://zenodo.org/badge/963986439.svg)](https://doi.org/10.5281/zenodo.15380776)
## Overview

UFR is a free energy thermodynamic model for learning pure component parameters which can accurately predict liquid phase properties. The current UFR model is calibrated on infinite dilution activity coefficient (IDAC) data of binary liquid mixtures. With the calibrated model, the activity coefficients across the composition range for any binary mixture of components present in the calibration dataset can be predicted, and correspondingly the full vapor-liquid equilibrium curves (T-x-y and P-x-y) as well as liquid-liquid curves. Through data truncation, the models are guaranteed to not overfit onto the dataset.

**Features**

Three Jupyter notebooks are provided for allowing users to train their own model, analyze their model, and produce phase equilibria predictions. Code for producing 

- ```Train_UFR_Model.ipynb``` is used to calibrate the UFR model on an infinite dilution activity coefficient dataset. A limited dataset containing the open source experimental IDAC data reported in the paper, as well COSMO-RS IDAC data published by Qin and Zavala 2023 is included in this repository in ```data```. Documentation is provided to format data tables so that users can provide their own IDAC training data.
- ```UFR_Model_Performance.ipynb``` is used to analyze the performance of the UFR model on the dataset. The notebook will generate the performance plots reported in the paper. There is a section for adding additional molecules into an existing trained model. Glycolide IDAC data is provided as a demonstration of this capability as outlined in the paper.
- ```UFR_Model_Phase_Diagram.ipynb``` is used to predict full vapor-liquid equilibrium (VLE) and liquid-liquid equilibrium (LLE) curves. The notebook will generate the VLE and LLE plots reported in the paper. An example of predicting non-infinite dilution activity coefficients is provided and benchmarked using the COSMO-RS non-IDAC data published by Qin and Zavala 2023. *Note: calculation of all non-IDAC activity coefficients will take about 10-20 minutes*

**Hardware requirements**

The model was calibrated using a GPU using CUDA 12.1. The model can be trained and evaluated on CPU only, however the training is anticipated to be slow.

**Installation and Setup**

To install the code, the repository can be cloned on a local machine using GIT. Anaconda package manager is used for managing the Python packages. A YAML environment file is provided for easy replication of the required environment. Please note that this environment file installs Pytorch CUDA 12.1. If your machine does not have a GPU or it does not run on CUDA 12.1, then modification of the Pytorch install version is required. Please refer to Pytorch documentation to determine which version to install on your computer.

The code has only been tested on a Windows operating system using Python 3.10.14

```
git clone https://github.com/oxie25/UFR.git
cd UFR

conda env create -f environment.yml
conda activate ufr
```

If you prefer to manually install packages, below is a list of the major required packages.
```
torch==2.2.2
numpy
pandas==2.2.2
scipy==1.15.2
matplotlib
seaborn
```

**Files in Repo**

- ```data``` Contains the open source IDAC data, data from Qin and Zavala 2023, small molecule properties data, select vapor pressure correlation data from NIST, glycolide data, and an Excel file containing the cleaning rules used for removing outliers from the main dataset.
- ```data\vle_data``` Contains the predicted VLE curves from a few of the models described in the main text
- ```trained_models\models_from_paper``` Contains trained models with various model architectures
- ```ufr``` Contains the model and analysis functions
