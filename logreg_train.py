import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from toolkit_dslr.logistic_regression import LogisticRegressionScratch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
from sklearn.model_selection import train_test_split


def write_json(new_data, filename="weights.json"):
    """
    Ajoute ou met à jour les poids/biais d'une maison dans le fichier JSON.
    new_data doit être de la forme {house: {...}} où house est le nom de la maison.
    """
    # Vérifie si le fichier existe, sinon crée un dict vide
    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            try:
                file_data = json.load(file)
            except json.JSONDecodeError:
                file_data = {}
    else:
        file_data = {}

    # Ajoute ou met à jour la maison (clé du dict)
    file_data.update(new_data)

    # Écrit le dict complet dans le fichier
    with open(filename, 'w') as file:
        json.dump(file_data, file, indent=4)


def LogisticRegression(file: str):
    df = pd.read_csv(file)
    imputer = SimpleImputer(strategy="mean")

    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    y_houses = {house: np.array([1 if i == house else 0 for i in df["Hogwarts House"]]) for house in houses}

    X = df.drop(['Index', 'Hogwarts House',
                  'First Name', 'Last Name',
                  'Birthday', "Best Hand"], axis=1)

    X = imputer.fit_transform(X)

    for house in houses:
        y_house = y_houses[house]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        model = LogisticRegressionScratch(learning_rate=0.1, iterations=1000)
        weights, bias = model.fit(X, y_house)

        print("type bias:", type(bias))
        print("bias:", bias)

        data = {
            house: {
                "weights": weights.tolist(),
                "bias": bias
            }
        }
        write_json(data)


# # LogisticRegression with X_train, X_test, y_train, y_test

# def LogisticRegression(file: str):
#     df = pd.read_csv(file)
#     imputer = SimpleImputer(strategy="mean")

#     houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
#     y_houses = {house: np.array([1 if i == house else 0 for i in df["Hogwarts House"]]) for house in houses}

#     X = df.drop(['Index', 'Hogwarts House',
#                   'First Name', 'Last Name',
#                   'Birthday', "Best Hand"], axis=1)

#     X = imputer.fit_transform(X)

#     models = {}
#     for house in houses:
#         y_house = y_houses[house]
#         X_train, X_test, y_train, y_test = train_test_split(X, y_house, test_size=0.2, random_state=42)
        
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
        
#         model = LogisticRegressionScratch(learning_rate=0.1, iterations=1000)
#         model.fit(X_train, y_train)
#         models[house] = (model, scaler)

#     house_preds = []
#     for idx in range(X_test.shape[0]):
#         probas = []
#         for house in houses:
#             model, scaler = models[house]
#             x = X_test[idx].reshape(1, -1)
#             proba = model.predict(x)[0]
#             print(proba)
#             probas.append(proba)
#         best_house_idx = np.argmax(probas)
#         house_preds.append(houses[best_house_idx])


def main():
    try:
        assert len(sys.argv) > 1, "File path is missing"
        assert os.path.exists(sys.argv[1]), "The file does not exists"
        LogisticRegression(sys.argv[1])
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()