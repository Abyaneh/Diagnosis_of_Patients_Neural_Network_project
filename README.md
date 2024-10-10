# Neural Network Group Projects


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project 1: Diagnosis of Patients with Fat](#project-1-diagnosis-of-patients-with-fat)
- [Project 2: House Price Prediction](#project-2-house-price-prediction)
- [Project 3: Clothing Classification](#project-3-clothing-classification)
- [Technologies & Tools Used](#technologies--tools-used)
- [How to Run the Project](#how-to-run-the-project)
- [Team Members](#team-members)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

This repository presents three neural network-based projects developed by **Mohammad Maleki Abyaneh** and **Mohammad Mohtashami**. Each project is designed to address a different challenge: diagnosing patients with fatty liver, classifying clothing images, and predicting house prices. These projects apply advanced deep learning techniques and machine learning algorithms, focusing on creating efficient and scalable solutions for real-world problems.

[Back to Top](#table-of-contents)
## Features

- **Diagnosis of Patients with Fat**: A binary classification neural network model designed to diagnose fatty liver using patient medical records.
- **Clothing Classification**: A convolutional neural network (CNN) model used to classify images of clothing into various categories.
- **House Price Prediction**: A machine learning regression model that predicts house prices based on features such as location, size, and age of the house.

[Back to Top](#table-of-contents)
## Project 1: Diagnosis of Patients with Fat

**Objective**: This project aimed to create a neural network model to predict fatty liver disease in patients based on medical attributes like blood sugar, blood pressure, and age.

### Data Characteristics:
- **Dataset**: The dataset includes patient medical records, with features like blood pressure, blood sugar levels, age, and other clinical indicators.
- **Challenges**:
  - **Data Preprocessing**: Cleaned and corrected missing or erroneous data (e.g., zero blood pressure). Used methods such as mean replacement or removal of invalid samples.
  - **Segmentation & Standardization**: The data was split into training (80%) and testing (20%) sets, and standardized using `StandardScaler`.

### Model Architecture:
- The neural network was designed using **TensorFlow** and **Keras**, with two hidden layers. The first hidden layer has 64 neurons, and the second has 32 neurons, both using **ReLU** activation.
- The output layer utilizes the **Sigmoid** function for binary classification.
- **EarlyStopping** and **ModelCheckpoint** were implemented to prevent overfitting and to save the best model.

### Model Performance:
- **Accuracy**: Achieved a final accuracy of **89.25%**.
- **Error Analysis**: Focused on reducing false positives and false negatives, with detailed confusion matrix evaluation.


[Back to Top](#table-of-contents)
## Project 2: House Price Prediction

**Objective**: Predict house prices using various attributes such as size, location, number of rooms, etc.

### Data and Model:
- The dataset includes house-related features like **location**, **square footage**, **number of bedrooms**, and **age** of the property.
- A machine learning regression model was implemented by **Mohammad Mohtashami** to predict house prices.
- **Key Metrics**: Evaluated using **Mean Absolute Error (MAE)** and **Root Mean Square Error (RMSE)** to ensure accurate prediction.

[Back to Top](#table-of-contents)
## Project 3: Clothing Classification

**Objective**: Classify clothing images into categories such as shirts, pants, and dresses using CNN models.

### Data Characteristics:
- **Dataset**: 70,000 images of clothing, divided into 60,000 images for training and validation, and 10,000 for testing.
- **Challenges**:
  - Designed three different CNN models with varying architectures.
  - Experimented with different hyperparameters (epochs, batch size, dropout layers) for model optimization.

### Model 1 Architecture:
- 4 convolutional layers with filter sizes of 32, 64, 128, and 256, each followed by **MaxPooling** layers.
- The final layers include **Flatten**, **Dense (128 neurons)** with ReLU, and an output layer with **Softmax** for multi-class classification.

### Results:
- **Model 1 Test Accuracy**: 91.35%
- **Model 2 Test Accuracy**: 87.83%
- **Model 3 Test Accuracy**: 92.17%
- **Best Model**: Model 3, which included a **Dropout Layer (0.5)** to prevent overfitting, achieved the best performance with a test accuracy of **92.17%**.

### Analysis of Performance:
- Models were evaluated using accuracy and loss graphs. Model 3, with its dropout layer, showed the best balance between training and validation accuracy, reducing overfitting compared to the other models.


[Back to Top](#table-of-contents)

## Technologies & Tools Used

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, Pandas, Scikit-learn, Matplotlib
- **Machine Learning Techniques**: Neural Networks, CNNs, Regression
- **Optimization Techniques**: EarlyStopping, ModelCheckpoint, Dropout

[Back to Top](#table-of-contents)

## How to Run the Project

```bash
# Clone repository
git clone https://github.com/Abyaneh/Neural_Network_projects

# Install dependencies
pip install -r requirements.txt

# Run specific projects
python diagnosis_of_patients_with_fat.py
python clothing_classification.py
python house_price_prediction.py
```
[Back to Top](#table-of-contents)
## Team Members
- **Mohammad Maleki Abyaneh** (me)
- **Mohammad Mohtashami**

[Back to Top](#table-of-contents)

## Contributing
Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Add a new feature'`).
4. Push to your branch (`git push origin feature-branch`).
5. Open a pull request for review.

[Back to Top](#table-of-contents)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Abyaneh/rotten_and_fresh/blob/main/LICENSE) file for details.

[Back to Top](#table-of-contents)

