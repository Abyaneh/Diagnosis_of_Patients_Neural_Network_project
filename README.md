# Fatty Liver Disease Diagnosis

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Technologies Used](#technologies-used)
- [How to Run the Project](#how-to-run-the-project)
- [Contact](#contact)

---

## Introduction

This project is a neural network-based solution to diagnose fatty liver disease based on medical data such as blood sugar, blood pressure, and age. The goal is to provide a fast and reliable way to assist in the diagnosis of this condition using machine learning.

---

## Dataset

- **Source**: The dataset consists of anonymized patient medical records with features like blood pressure, blood sugar levels, age, and other clinical indicators.
- **Preprocessing**:
  - Removed invalid samples (e.g., zero values for blood pressure).
  - Missing values handled through mean replacement.
  - Data was split into **80% for training** and **20% for testing**.

---

## Model Architecture

- **Framework**: TensorFlow, Keras
- **Architecture**:
  - Two hidden layers: 64 and 32 neurons with **ReLU** activation.
  - **Sigmoid** function in the output layer for binary classification.
  - Optimized with **EarlyStopping** and **ModelCheckpoint**.
  
![Model Architecture](path/to/fatty_liver_model_architecture.png)

---

## Performance

- **Accuracy**: 89.25%
- **Confusion Matrix Analysis**: Focused on reducing false positives and false negatives.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, Pandas, Scikit-learn
- **Optimization Techniques**: EarlyStopping, ModelCheckpoint

---

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abyaneh/Diagnosis_of_Patients_Neural_Network_project/tree/main
   ```
2. **Run the model**: Open the first_pro_.....ipynb notebook and follow the instructions to train and test the model.


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

