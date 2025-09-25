# 🇬🇧 dslr_42

**If you enjoy this project, feel free to give it a star ⭐️!**

## Introduction

The goal of this project is to build a logistic regression model from scratch that will solve a classification problem.
Throught this project, we will recreate a magic Sorting Hat to sort the new Hogwarts' students to the houses. 🧙🏻‍♂️
Students' data wil be used to train the model.

---

## Logistic Regression
Logistic Regression is a statistical and machine learning technique used to model the probability of a binary outcome (such as yes/no, success/failure, or class A/class B) based on one or more input features. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability that an input belongs to a particular class.

It works by applying the sigmoid function to a linear combination of the input features. The sigmoid function squashes the output to a value between 0 and 1, which can be interpreted as a probability. The model is trained to find the best weights that separate the classes by minimizing a cost function (usually the cross-entropy loss).

Logistic Regression is widely used for classification tasks such as:

    Email spam detection (spam vs. not spam)
    Medical diagnosis (disease vs. no disease)
    Image recognition (cat vs. dog)
  
<img width="563" height="452" alt="Screenshot from 2025-09-25 16-25-47" src="https://github.com/user-attachments/assets/51ff8757-ce94-4003-8b4a-db04202acc26" />

## Usage

1. Clone the repository
```bash
git clone https://github.com/MatLBS/multilayer_perceptron_42.git
cd multilayer_perceptron_42
```
2. Create a virtual environment and install dependencies
```python
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Dataset preview

```bash
python describe.py datasets/dataset_train.csv
```

<img width="1606" height="197" alt="Screenshot from 2025-09-25 16-27-34" src="https://github.com/user-attachments/assets/2a1a4527-a1b0-4fff-bb24-edfb59750437" />

```bash
python histogram.py datasets/dataset_train.csv
```

<img width="1420" height="1097" alt="Screenshot from 2025-09-25 16-28-04" src="https://github.com/user-attachments/assets/828dc78b-f87b-42da-852b-b45830e26aab" />

# Training & Prediction

## Training of an Logistic Regression model

```bash
python logreg_train.py datasets/dataset_train.csv
```
It will create a file 'weights.json' which will be used for predictions

```bash
python logreg_predict.py datasets/dataset_test.csv weights.json
```
Creates a file 'houses.csv' with a predicted house for every future student




