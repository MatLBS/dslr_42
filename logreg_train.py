import matplotlib.pyplot as plt
import pandas as pd


def LinearRegression():
    gryffindor = df[df["Hogwarts House"] == "Gryffindor"]
    hufflepuff = df[df["Hogwarts House"] == "Hufflepuff"]
    ravenclaw = df[df["Hogwarts House"] == "Ravenclaw"]
    slytherin = df[df["Hogwarts House"] == "Slytherin"]

def main():
    try:
        data = pd.read_csv('data.csv')
        LinearRegression(data)
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
