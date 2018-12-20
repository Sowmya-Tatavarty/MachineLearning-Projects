# Machine Learning Engineer Nanodegree
## Distracted Driver Detection

As part of Machine Learning Nano Degree, I am working on completing a Capstone Project, to leverage what I have learned thought the Nanodegree program to author a proposal and solve a problem of my choice by applying machine learning algorithms and techniques.

In this project, we will use dashboard cameras images to detect drivers engaging in distracted behaviors. Given a dataset of 2D dashboard camera images by State Farm in the Kaggle challenge we will classify each driver's behavior. Are they driving attentively, wearing their seatbelt, or taking a selfie with their friends in the back seat?


### Data

The dataset used for this project can be downloaded from Kaggle  [Kaggle - State Farm Distracted Drive Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

**File Descriptions**

	img.zip - zipped folder of all (train/text) images
	sample_submissions.csv - a sample submission file in the correct format
	driver_imgs_list.csv - a list of training images 


Read the Project report (Project Report.pdf) and open "Project summary" IPython notebook to review models trained and their accuracies.
Download the files and extract imgs.zip to 'Distracted_Data' folder in the current working directory.
After extracting files from imgs.zip, make sure that the path for training and test file is /Distracted_Data/imgs/train/ and 
/Distracted_Data/imgs/train/ repsectively. 
Project summary has the option to load saved models to regenerate submission files. Submit the results in the Kaggle competition and view the score
Submissions folder has submissions made to get the score mentioned in the project report document and project summary IPython notebook.

Description for documents in the project repo.
1. Project Report: High level overview of the project. 
2. Project Summary.ipynb: IPython Notebook where I loaded and compared benchmark model with other models and final model built and trained on the Disracted Driver dataset.
3. VGGNet.ipynb:  Benchmark model creation and training. 
4. Distracted_Driver-Inception.ipynb: Fine tuned Inception model 
5. Distracted_Driver-ResNet.ipynb: Fine tuned Resnet model
6. VGG19.ipynb: Final fine tuned VGG19 model with best score on the training and test dataset. 
6. submission_function.py: Util file with function to generate submission files in format required by Kaggle
