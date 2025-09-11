import numpy as np
import pandas as pd
import sys
import os
from toolkit_dslr.logistic_regression import LogisticRegressionScratch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json


def write_json(new_data, filename="weights.json"):
    """
    Ajoute ou met à jour les poids/biais d'une maison dans le fichier JSON.
    new_data doit être de la forme {house: {...}}
    où house est le nom de la maison.
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
    scaler = StandardScaler()

    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    y_houses = {house: np.array([1 if i == house else 0 for i in
                                 df["Hogwarts House"]]) for house in houses}

    X = df.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name',
                 'Birthday', 'Best Hand', 'Arithmancy',
                 'Care of Magical Creatures'], axis=1)

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    for house in houses:
        y_house = y_houses[house]

        model = LogisticRegressionScratch(learning_rate=0.1, iterations=1000)
        weights, bias = model.fit(X, y_house)

        data = {
            house: {
                "weights": weights.tolist(),
                "bias": bias
            }
        }
        write_json(data)


def main():
    try:
        assert len(sys.argv) > 1, "File path is missing"
        assert os.path.exists(sys.argv[1]), "The file does not exists"
        LogisticRegression(sys.argv[1])
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
