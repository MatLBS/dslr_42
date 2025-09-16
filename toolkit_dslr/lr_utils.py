import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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


def preprocess_data(df):

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X = df.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name',
                 'Birthday', 'Best Hand', 'Arithmancy',
                 'Care of Magical Creatures'], axis=1)

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    return X
