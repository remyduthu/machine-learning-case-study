#!/usr/bin/env python3

from matplotlib import pyplot
from os import getcwd, listdir
from pandas import concat, read_csv, to_datetime


def load_df(station_id: int):
    # Read the data from the CSV files
    files = []
    for file in listdir(f"{getcwd()}/data/"):
        if file.endswith(".csv"):
            files.append(
                read_csv(
                    f"{getcwd()}/data/{file}",
                    na_values="mq",
                    sep=";",
                )
            )

    df = concat(files)

    # Filter based on the station ID
    df.query(f"numer_sta == {station_id}", inplace=True)

    # Convert the date into datetime Python objects
    df["date"] = to_datetime(df["date"], format="%Y%m%d%H%M%S")

    return df


def load_dataset(df, param: str):
    # Keep only the date and the parameter columns
    df = df[["date", param]]

    # Convert into a Numpy array
    return df.to_numpy()


def plot_dataset(position, dataset: list, param: str):
    # Plot the dataset values
    position.plot(dataset[:, 0], dataset[:, 1], "o")

    # position.set_xlabel("Date")
    position.set_ylabel(param)


def main():
    print("⚙️  Load the data...")
    df = load_df(station_id=7630)

    _, axis = pyplot.subplots(5, figsize=(18, 28))

    plot_dataset(
        axis[0],
        load_dataset(df, "t"),
        "Température (K)",
    )

    plot_dataset(
        axis[1],
        load_dataset(df, "u"),
        "Humidité (%)",
    )

    plot_dataset(
        axis[2],
        load_dataset(df, "dd"),
        "Direction du vent moyen 10 mn (degré)",
    )

    plot_dataset(
        axis[3],
        load_dataset(df, "ff"),
        "Vitesse du vent moyen 10 mn (m/s)",
    )

    plot_dataset(
        axis[4],
        load_dataset(df, "pres"),
        "Pression station (Pa)",
    )

    pyplot.savefig(f"data/dataset.png")


if __name__ == "__main__":
    main()
