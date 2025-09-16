import numpy as np
import pandas as pd
import sys
import os
from toolkit_dslr.logistic_regression import LogisticRegressionScratch
from toolkit_dslr.lr_utils import write_json, preprocess_data


def LogisticRegression(file: str):
    df = pd.read_csv(file)

    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    y_houses = {house: np.array([1 if i == house else 0 for i in
                                 df["Hogwarts House"]]) for house in houses}
    X = preprocess_data(df)

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
        assert len(sys.argv) == 2, "You must provide the dataset file path"
        assert os.path.exists(sys.argv[1]), "The file does not exists"
        LogisticRegression(sys.argv[1])
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
