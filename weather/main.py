#!/usr/bin/env python3

from datetime import datetime
from matplotlib import pyplot
from os import getcwd, listdir
from pandas import concat, read_csv, to_datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Utilities


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

    # Convert the dates into Pandas Timestamps
    df["date"] = to_datetime(df["date"], format="%Y%m%d%H%M%S").astype(int) / 10 ** 9

    return df


def load_ds(df, param: str):
    # Keep only the date and the parameter columns
    df = df[["date", param]]

    # Drop rows with NaN values
    df = df.dropna()

    # Convert into a Numpy array
    ds = df.to_numpy()

    # Reshape X as we work on a single feature
    X = ds[:, 0].reshape(-1, 1)
    y = ds[:, 1]

    return train_test_split(X, y)


def plot_ds(position, param, dataset, y_pred):
    X_train, X_test, y_train, y_test = dataset

    # Add the training data
    position.scatter(
        ts_to_date(X_train),
        y_train,
        alpha=0.1,
        label="Training",
    )

    # Add the test data
    position.scatter(ts_to_date(X_test), y_test, alpha=0.4, label="Test")

    # Add the predictions
    position.scatter(ts_to_date(X_test), y_pred, alpha=0.4, label="Predictions")

    position.set(
        xlabel="Date",
        ylabel=param,
    )

    position.legend()


def ts_to_date(X: list):
    ts_list = X[:, 0]

    return [datetime.fromtimestamp(ts) for ts in ts_list]


# Models


def make_temperature_model(df, position):
    # Load the temperature dataset
    t = load_ds(df, "t")
    X_train, X_test, y_train, y_test = t

    # Build & fit the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    plot_ds(
        position,
        "Température (K)",
        dataset=t,
        y_pred=model.predict(X_test),
    )

    return model.score(X_test, y_test) * 100


def make_humidity_model(df, position):
    h = load_ds(df, "u")
    X_train, X_test, y_train, y_test = h

    # Build & fit the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    plot_ds(
        position,
        "Humidité (%)",
        dataset=h,
        y_pred=model.predict(X_test),
    )

    return model.score(X_test, y_test) * 100


# TODO: Try to use another type of Regressor
def make_wind_direction_model(df, position):
    wd = load_ds(df, "dd")
    X_train, X_test, y_train, y_test = wd

    # Build & fit the model
    model = make_pipeline(
        PolynomialFeatures(degree=4),
        StandardScaler(),
        LinearRegression(),
    )
    model.fit(X_train, y_train)

    plot_ds(
        position,
        "Direction du vent moyen 10 mn (degré)",
        dataset=wd,
        y_pred=model.predict(X_test),
    )

    return model.score(X_test, y_test) * 100


def make_wind_speed_model(df, position):
    ws = load_ds(df, "ff")
    X_train, X_test, y_train, y_test = ws

    # Build & fit the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    plot_ds(
        position,
        "Vitesse du vent moyen 10 mn (m/s)",
        dataset=ws,
        y_pred=model.predict(X_test),
    )

    return model.score(X_test, y_test) * 100


def make_atmospheric_pressure_model(df, position):
    p = load_ds(df, "pres")
    X_train, X_test, y_train, y_test = p

    # Build & fit the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    plot_ds(
        position,
        "Pression station (Pa)",
        dataset=p,
        y_pred=model.predict(X_test),
    )

    return model.score(X_test, y_test) * 100


def main():
    # Load the whole data from the CSV files
    print("⚙️  Load the data...")
    # df = load_df(station_id=7630)
    df = load_df(station_id=7460)

    # Create the main figure
    _, axis = pyplot.subplots(5, figsize=(18, 28))

    print(f"Temperature: {make_temperature_model(df, axis[0])}%")
    print(f"Humidity: {make_humidity_model(df, axis[1])}%")
    print(f"Wind direction: {make_wind_direction_model(df, axis[2])}%")
    print(f"Wind speed: {make_wind_speed_model(df, axis[3])}%")
    print(f"Atmospheric pressure: {make_atmospheric_pressure_model(df, axis[4])}%")

    pyplot.tight_layout()
    pyplot.savefig(f"data/dataset.png")


if __name__ == "__main__":
    main()
