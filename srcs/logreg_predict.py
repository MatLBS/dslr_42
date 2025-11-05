import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from toolkit_dslr.logistic_regression import LogisticRegressionScratch
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import json


def predict_houses(data_file: str, weights_file: str):

    df = pd.read_csv(data_file)
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]

    X = df.drop(['Index', 'Hogwarts House', 'First Name',
                 'Last Name', 'Birthday', 'Best Hand',
                 'Arithmancy', 'Care of Magical Creatures'], axis=1)

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)
    model = LogisticRegressionScratch(learning_rate=0.1, iterations=1000)

    with open(weights_file, 'r') as f:
        data = json.load(f)

    house_preds = []
    for idx in range(X.shape[0]):
        probas = []
        for house in houses:
            weights = data[house]["weights"]
            bias = data[house]["bias"]
            x = X[idx].reshape(1, -1)
            proba = model.predict_arguments(x, weights, bias)
            probas.append(proba)
        best_house_idx = np.argmax(probas)
        house_preds.append(houses[best_house_idx])

    with open("houses.csv", "w") as f:
        f.write("Index,Hogwarts House\n")
        for idx, house in enumerate(house_preds):
            f.write(f"{idx},{house}\n")


def main():
    try:
        assert len(sys.argv) == 3, "Provide the dataset and weights file paths"
        assert os.path.exists(sys.argv[1]), "The file does not exists"
        assert os.path.exists(sys.argv[2]), "The file does not exists"
        predict_houses(sys.argv[1], sys.argv[2])
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
