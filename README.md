# 🇬🇧 dslr_42

**If you enjoy this project, feel free to give it a star ⭐️!**

## Introduction

The goal of this project is to build a logistic regression model from scratch to solve a classification problem.
Throughout this project, we will recreate a magic Sorting Hat to sort new Hogwarts students into their houses. 🧙🏻‍♂️
Students' data will be used to train the model.

---

## Logistic Regression

Logistic Regression is a statistical and machine learning technique used to model the probability of a binary outcome (such as yes/no, success/failure, or class A/class B) based on one or more input features. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability that an input belongs to a particular class.

It works by applying the sigmoid function to a linear combination of the input features. The sigmoid function squashes the output to a value between 0 and 1, which can be interpreted as a probability. The model is trained to find the best weights that separate the classes by minimizing a cost function (usually the cross-entropy loss).

Logistic Regression is widely used for classification tasks such as:

- Email spam detection (spam vs. not spam)
- Medical diagnosis (disease vs. no disease)
- Image recognition (cat vs. dog)
  
<img width="563" height="452" alt="Screenshot from 2025-09-25 16-25-47" src="https://github.com/user-attachments/assets/51ff8757-ce94-4003-8b4a-db04202acc26" />

## Usage

1. Clone the repository
```bash
git clone https://github.com/MatLBS/dslr_42.git
cd dslr_42
```
2. Create a virtual environment and install dependencies
```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Dataset preview

```bash
python srcs/describe.py datasets/dataset_train.csv
```

<img width="1606" height="197" alt="Screenshot from 2025-09-25 16-27-34" src="https://github.com/user-attachments/assets/2a1a4527-a1b0-4fff-bb24-edfb59750437" />

```bash
python srcs/histogram.py datasets/dataset_train.csv
```

<img width="1420" height="1097" alt="Screenshot from 2025-09-25 16-28-04" src="https://github.com/user-attachments/assets/828dc78b-f87b-42da-852b-b45830e26aab" />

# Training & Prediction

## Training a Logistic Regression model

```bash
python srcs/logreg_train.py datasets/dataset_train.csv
```
This will create a file `weights.json` which will be used for predictions.

```bash
python srcs/logreg_predict.py datasets/dataset_test.csv weights.json
```
This creates a file `houses.csv` with a predicted house for every future student.

# 🇫🇷 dslr_42

**Si vous appréciez ce projet, n’hésitez pas à lui attribuer une étoile ⭐️ !**

## Introduction

Le but de ce projet est de construire un modèle de régression logistique à partir de zéro pour résoudre un problème de classification.
Tout au long de ce projet, nous allons recréer le célèbre Choixpeau magique pour répartir les nouveaux élèves de Poudlard dans leurs maisons. 🧙🏻‍♂️
Les données des étudiants seront utilisées pour entraîner le modèle.

---

## Régression Logistique

La régression logistique est une technique statistique et d’apprentissage automatique utilisée pour modéliser la probabilité d’un résultat binaire (comme oui/non, succès/échec, ou classe A/classe B) en fonction d’une ou plusieurs variables d’entrée. Contrairement à la régression linéaire, qui prédit des valeurs continues, la régression logistique prédit la probabilité qu’une entrée appartienne à une classe particulière.

Elle fonctionne en appliquant la fonction sigmoïde à une combinaison linéaire des variables d’entrée. La fonction sigmoïde comprime la sortie entre 0 et 1, ce qui peut être interprété comme une probabilité. Le modèle est entraîné afin de trouver les meilleurs poids permettant de séparer les classes en minimisant une fonction de coût (généralement l'entropie croisée).

La régression logistique est largement utilisée pour des tâches de classification telles que :

- Détection de spam dans les emails (spam vs. non-spam)
- Diagnostic médical (malade vs. non-malade)
- Reconnaissance d’images (chat vs. chien)
  
<img width="563" height="452" alt="Screenshot from 2025-09-25 16-25-47" src="https://github.com/user-attachments/assets/51ff8757-ce94-4003-8b4a-db04202acc26" />

## Utilisation

1. Cloner le dépôt
```bash
git clone https://github.com/MatLBS/dslr_42.git
cd dslr_42
```
2. Créer un environnement virtuel et installer les dépendances
```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Aperçu du jeu de données

```bash
python srcs/describe.py datasets/dataset_train.csv
```

<img width="1606" height="197" alt="Screenshot from 2025-09-25 16-27-34" src="https://github.com/user-attachments/assets/2a1a4527-a1b0-4fff-bb24-edfb59750437" />

```bash
python srcs/histogram.py datasets/dataset_train.csv
```

<img width="1420" height="1097" alt="Screenshot from 2025-09-25 16-28-04" src="https://github.com/user-attachments/assets/828dc78b-f87b-42da-852b-b45830e26aab" />

# Entraînement & Prédiction

## Entraînement d’un modèle de régression logistique

```bash
python srcs/logreg_train.py datasets/dataset_train.csv
```
Cela créera un fichier `weights.json` qui sera utilisé pour les prédictions.

```bash
python srcs/logreg_predict.py datasets/dataset_test.csv weights.json
```
Cela créera un fichier `houses.csv` avec une maison prédite pour chaque futur étudiant.

