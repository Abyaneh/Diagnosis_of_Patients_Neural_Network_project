# Neural Network Group Project


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project 1: Diagnosis of Patients with Fat](#project-1-diagnosis-of-patients-with-fat)
- [Project 2: Clothing Classification](#project-2-clothing-classification)
- [Project 3: House Price Prediction](#project-3-house-price-prediction)
- [Technologies & Tools Used](#technologies--tools-used)
- [How to Run the Project](#how-to-run-the-project)
- [Team Members](#team-members)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This repository features three group projects using neural networks. The work combines efforts on medical diagnosis, clothing classification, and house price prediction. Mohammad Maleki Abyaneh and Mohammad Mohtashami collaborated on these projects to demonstrate deep learning techniques and machine learning applications.

[Back to Top](#table-of-contents)
## Features
- Neural network-based diagnosis of patients with fat.
- Clothing image classification using convolutional neural networks.
- Predictive analysis of house prices using regression models.

## Project 1: Diagnosis of Patients with Fat

**Objective**: Diagnose fatty liver using patient data and medical features.

- **Data Preprocessing**: Removed inconsistencies, such as zero blood pressure, and used standardization.
- **Model Architecture**: Designed using TensorFlow and Keras with early stopping, achieving 89.25% accuracy.
- **Model Evaluation**: Confusion matrix analysis, accuracy, and error graphs were used to assess the model.

## Project 2: Clothing Classification

**Objective**: Classify clothing images using CNN models.

- **Data**: 70,000 images split into training and testing sets.
- **Model Architecture**: CNN with dropout layers to reduce overfitting. Test accuracy reached 92.17%.
- **Models Comparison**: Explored multiple architectures, selecting the best based on accuracy and loss.

## Project 3: House Price Prediction

**Objective**: Predict house prices using machine learning.

- **Contribution by Mohammad Mohtashami**: Implemented machine learning models to predict house prices with key metrics such as MAE and RMSE.
  
## Technologies & Tools Used
- Python (TensorFlow, Keras, Pandas, Scikit-learn)
- CNN for image processing
- Data preprocessing techniques
- Early stopping and checkpointing

## How to Run the Project

```bash
# Clone repository
git clone https://github.com/yourusername/yourproject.git

# Install dependencies
pip install -r requirements.txt

# Run specific projects
python diagnosis_of_patients_with_fat.py
python clothing_classification.py
python house_price_prediction.py
