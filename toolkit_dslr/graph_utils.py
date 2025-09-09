import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def draw_histogram(file: str) -> None:
    df = pd.read_csv(file)

    gryffindor = df[df["Hogwarts House"] == "Gryffindor"]
    hufflepuff = df[df["Hogwarts House"] == "Hufflepuff"]
    ravenclaw = df[df["Hogwarts House"] == "Ravenclaw"]
    slytherin = df[df["Hogwarts House"] == "Slytherin"]

    columns = df.columns[6:]

    fig, ax = plt.subplots(4, 4, figsize=(14, 10))
    k = 0

    for i in range(4):
        for j in range(4):
            if k < len(columns):
                ax[i][j].hist(gryffindor[columns[k]],
                              bins=20, color='red', alpha=0.7)
                ax[i][j].hist(hufflepuff[columns[k]],
                              bins=20, color='yellow', alpha=0.7)
                ax[i][j].hist(ravenclaw[columns[k]],
                              bins=20, color='blue', alpha=0.7)
                ax[i][j].hist(slytherin[columns[k]],
                              bins=20, color='green', alpha=0.7)
                ax[i][j].set_title(columns[k])
                ax[i][j].legend(
                    ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'],
                    loc="upper right",
                    fontsize=8,
                    frameon=False
                )
                ax[i][j].set_xlabel('Score')
                ax[i][j].set_ylabel('Number of Students')
            else:
                ax[i][j].axis('off')
            k += 1
    plt.tight_layout()
    plt.show()


def draw_scatter_plot(file: str) -> None:
    df = pd.read_csv(file)

    gryffindor = df[df["Hogwarts House"] == "Gryffindor"]
    hufflepuff = df[df["Hogwarts House"] == "Hufflepuff"]
    ravenclaw = df[df["Hogwarts House"] == "Ravenclaw"]
    slytherin = df[df["Hogwarts House"] == "Slytherin"]

    plt.scatter(gryffindor["Astronomy"],
                gryffindor["Defense Against the Dark Arts"],
                color='red', label='Gryffindor', alpha=0.8)
    plt.scatter(hufflepuff["Astronomy"],
                hufflepuff["Defense Against the Dark Arts"],
                color='yellow', label='Hufflepuff', alpha=0.8)
    plt.scatter(ravenclaw["Astronomy"],
                ravenclaw["Defense Against the Dark Arts"],
                color='blue', label='Ravenclaw', alpha=0.8)
    plt.scatter(slytherin["Astronomy"],
                slytherin["Defense Against the Dark Arts"],
                color='green', label='Slytherin', alpha=0.8)
    plt.legend()
    plt.xlabel("Astronomy")
    plt.ylabel("Defense Against the Dark Arts")
    plt.show()


def draw_pair_plot(file: str) -> None:
    df = pd.read_csv(file)
    df = df.drop(['Index',
                  'First Name', 'Last Name',
                  'Birthday', "Best Hand"], axis=1)

    sns.pairplot(df, diag_kind='auto', hue='Hogwarts House', palette={
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }, plot_kws={'alpha': 0.6})

    plt.show()
