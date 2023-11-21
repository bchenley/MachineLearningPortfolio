## Overview

The BiomedicalImaging project performs semantic segmentation on MRI data from brain cancer patients. The data is obtained from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/). Specifically, we obtain the T1, T1c, T2, and FLAIR from a subset of patients in the "Task01_BrainTumor.tar" file.
We employ Pytorch to develop various models for the segmentation task and evaluate the results. 

## Prerequisites

Before you begin, make sure you have the following prerequisites installed:

- Git (for cloning the repository)
- Python 3.x
- pip (Python package manager)

## Setting upyour virtual environment:

1. Clone the BiomedicalImaging repository to your local machine:

   ```bash
   git clone https://github.com/bchenley/MachineLearningPortfolio.git

2. Navigate to the "BiomedicalImaging" source code directory:

   ```bash
   cd MachineLearningPortfolio/BiomedicalImaging/src

3. Create a virtual environment (or use an existing one)

   ```bash
   python -m venv bmienv       # Create a virtual environment named "bmienv" (if you're using your own, skip this line)
   source bmienv/bin/activate  # Activate the virtual environment (or the environment of your own)

5. Install the required dependencies using the provided requirements.txt file

   ```bash
   pip install -r requirements.text    

## Generating Train-Test Sets

1. With the environment set up and activated, make sure you are in the source code directory and run the "create_train_test_images.py" script to generate the train-test sets:

   ```bash
   python create_train_test_images.py    

  The script will prompt you for the following information: 

   - Path to the DICOM dataset
   - Path to save the generated images
   - Sample size (default is 100)
   - Training size as a percentage (e.g., 0.7 for 70% training, 30% testing)
   - Image registration (y/n): Register images to T1 (if desired)       

2. This script will create train and tests sets and save them the specified location ask pickle files. Please confirm they are located in the desired location. 
                  
