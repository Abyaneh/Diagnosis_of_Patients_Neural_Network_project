# Diagnosis of Patients

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Diagnosis of Patients with Fat](#diagnosis-of-patients-with-fat)
- [Technologies & Tools Used](#technologies--tools-used)
- [How to Run the Project](#how-to-run-the-project)
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
## Diagnosis of Patients with Fat

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

#### Model 1
![Epoch-Accuracy_and_Epoch-Loss_Scores _for_Model_1](https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/blob/main/Photos/Epoch-Accuracy_and_Epoch-Loss_Scores%20_for_Model_1.png)

#### Model 6 (With RandomizedSearchCV)
![Epoch-Accuracy_and_Epoch-Loss_Scores _for_Model_6_RandomizedSearchCV](https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/blob/main/Photos/Epoch-Accuracy_and_Epoch-Loss_Scores%20_for_Model_6_RandomizedSearchCV.png)

[Back to Top](#table-of-contents)

## Technologies & Tools Used

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, Pandas, Scikit-learn, Matplotlib
- **Machine Learning Techniques**: Neural Networks, Binary Classification
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

