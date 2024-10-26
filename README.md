# Fatty Liver Disease Diagnosis

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Technologies Used](#technologies-used)
- [How to Run the Project](#how-to-run-the-project)
- [Contributing](#contributing)
- [License](#license)
---

## Introduction

This project is a neural network-based solution to diagnose fatty liver disease based on medical data such as blood sugar, blood pressure, and age. The goal is to provide a fast and reliable way to assist in the diagnosis of this condition using machine learning.

[Back to Top](#table-of-contents)

---

## Dataset

- The dataset consists of anonymized patient medical records with features like blood pressure, blood sugar levels, age, and other clinical indicators.
- **Preprocessing**:
  - Removed invalid samples (e.g., zero values for blood pressure).
  - Missing values handled through mean replacement.
  - Data was split into **80% for training** and **20% for testing**.

##### You can download the dataset from [here](https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/blob/main/Diagnosis%20of%20Patients%20with%20Fat/Code/data.csv)

[Back to Top](#table-of-contents)

## Model Architecture

- **Framework**: TensorFlow, Keras
I trained 6 different models that you can see the architectures in [this file](https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/tree/main/Diagnosis%20of%20Patients%20with%20Fat/Model%20shape%20picture/model%20shape) and model 6 was the best one. 

#### Model 6 Architecture:
![Model 6 Architecture](https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/blob/main/Diagnosis%20of%20Patients%20with%20Fat/Model%20shape%20picture/model%20shape/model_first_pro_shuffle_Dropout_correction2_.png)

[Back to Top](#table-of-contents)

## Performance
- **Accuracy**: 88.00% ( for model 6)
- **Error Analysis**: Focused on reducing false positives and false negatives, with detailed confusion matrix evaluation.

#### Model 6 (With RandomizedSearchCV) :
![Epoch-Accuracy_and_Epoch-Loss_Scores _for_Model_6_RandomizedSearchCV](https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/blob/main/Photos/Epoch-Accuracy_and_Epoch-Loss_Scores%20_for_Model_6_RandomizedSearchCV.png)

#### Model 6 (With RandomizedSearchCV) Confusion Matrix:
![Epoch-Accuracy_and_Epoch-Loss_Scores _for_Model_6_RandomizedSearchCV](https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/blob/main/Photos/Confusion%20matrix_model6.png)

[Back to Top](#table-of-contents)

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, Pandas, Scikit-learn
- **Optimization Techniques**: EarlyStopping, ModelCheckpoint

[Back to Top](#table-of-contents)

---

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/tree/main
   ```
2. **Run the model**: Open the first_pro_(....).ipynb notebook and follow the instructions to train and test the model. ( You can replace (....) with all the names of the codes that are in [this file](https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/tree/main/Diagnosis%20of%20Patients%20with%20Fat/Code)

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

