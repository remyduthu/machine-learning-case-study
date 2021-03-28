#!/usr/bin/env python3

from datetime import datetime, timedelta
from matplotlib import pyplot
from numpy import array
from os import getcwd, listdir
from pandas import concat, read_csv, to_datetime
from scipy.constants import convert_temperature
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Utilities


def get_day_from_date(date: datetime):
    return array(date.timetuple().tm_yday).reshape(-1, 1)


def load_df(stations: list):
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
    df = df[df["numer_sta"].isin(stations)]

    # Convert the dates into Pandas Timestamps
    df.date = to_datetime(df.date, format="%Y%m%d%H%M%S")

    # Add a column to compare values from year to year
    df["days"] = df.date.dt.day_of_year

    return df


def load_ds(df, param: str):
    # Keep only required columns
    df = df[["date", "days", param]]

    # Drop rows with NaN values
    df = df.dropna()

    # Reshape X as we work on a single feature
    X = df.days.to_numpy().reshape(-1, 1)
    y = df[param].to_numpy()

    return train_test_split(X, y)


def measure_performance(model, ds, y_pred):
    _, X_test, _, y_test = ds

    print(f"• R²: {model.score(X_test, y_test)}")
    print(f"• RMSE: +/-{mean_squared_error(y_test, y_pred, squared=False)}")
    print()

    predict_by_date(model, datetime.now())


def predict_by_date(model, date):
    today = get_day_from_date(date)
    in_five_days = get_day_from_date(date + timedelta(days=5))
    in_a_week = get_day_from_date(date + timedelta(weeks=1))
    in_a_month = get_day_from_date(date + timedelta(weeks=4))

    print(f"• Today: {model.predict(today)[0]}")
    print(f"• In Five Days: {model.predict(in_five_days)[0]}")
    print(f"• In a Week: {model.predict(in_a_week)[0]}")
    print(f"• In a Month: {model.predict(in_a_month)[0]}")
    print()


def plot_ds(position, param, dataset, y_pred):
    X_train, X_test, y_train, y_test = dataset

    # Add the training data
    position.scatter(
        X_train,
        y_train,
        alpha=0.1,
        label="Training",
    )

    # Add the test data
    position.scatter(X_test, y_test, alpha=0.4, label="Test")

    # Add the predictions
    position.scatter(X_test, y_pred, alpha=0.4, label="Predictions")

    # Add dots for today
    today = get_day_from_date(datetime.now())
    position.scatter(today, y_test[today], alpha=0.8, s=128, label="Today (y_test)")
    position.scatter(today, y_pred[today], alpha=0.8, s=128, label="Today (y_pred)")

    position.set(
        xlabel="Days in the Year",
        ylabel=param,
    )

    position.legend()


# Models


def make_temperature_model(df, position):
    # Load the temperature dataset
    ds = load_ds(df, "t")
    X_train, X_test, y_train, y_test = ds

    # Convert temperatures into Celsius degrees
    y_train[:] = [convert_temperature(y, "Kelvin", "Celsius") for y in y_train]
    y_test[:] = [convert_temperature(y, "Kelvin", "Celsius") for y in y_test]

    # Build & fit the model
    model = make_pipeline(
        PolynomialFeatures(6),
        StandardScaler(),
        LinearRegression(),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plot_ds(
        position,
        "Temperature (K)",
        dataset=(X_train, X_test, y_train, y_test),
        y_pred=y_pred,
    )

    return model, ds, y_pred


def make_humidity_model(df, position):
    ds = load_ds(df, "u")
    X_train, X_test, y_train, _ = ds

    # Build & fit the model
    model = make_pipeline(
        PCA(),
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=80),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plot_ds(
        position,
        "Humidity (%)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


# TODO: Finish the wind predictions
def make_wind_direction_model(df, fig):
    ds = load_ds(df, "dd")
    X_train, X_test, y_train, y_test = ds

    # Build & fit the model
    model = RandomForestRegressor(max_depth=10)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    pyplot.polar(
        y_train,
        X_train,
        "o",
        alpha=0.1,
        label="Training",
    )
    pyplot.polar(y_test, X_test, "o", alpha=0.4, label="Test")
    pyplot.polar(y_pred, X_test, "o", alpha=0.8, label="Predictions")

    return model, ds, y_pred


def make_wind_speed_model(df, position):
    ds = load_ds(df, "ff")
    X_train, X_test, y_train, _ = ds

    # Build & fit the model
    model = make_pipeline(
        PCA(whiten=True),
        StandardScaler(),
        RadiusNeighborsRegressor(radius=0.02),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plot_ds(
        position,
        "Average wind speed 10 min (m/s)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


def make_precipitations_model(df, position):
    ds = load_ds(df, "rr24")
    X_train, X_test, y_train, _ = ds

    # Build & fit the model
    model = make_pipeline(
        PCA(whiten=True),
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=128),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plot_ds(
        position,
        "Precipitation in the last 24 hours (mm)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


def make_atmospheric_pressure_model(df, position):
    ds = load_ds(df, "pres")
    X_train, X_test, y_train, _ = ds

    # Build & fit the model
    model = make_pipeline(
        PCA(whiten=True),
        StandardScaler(),
        RadiusNeighborsRegressor(radius=0.014),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plot_ds(
        position,
        "Atmospheric pressure (Pa)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


def main():
    # Load the whole data from the CSV files
    print("⚙️  Load the data...")
    df = load_df(stations=[7558])

    # Create the main figure
    _, axis = pyplot.subplots(4, figsize=(18, 24))

    # Build & fit the models
    print(f"🚧 Building the models (this may take some time)...")
    t, t_ds, t_pred = make_temperature_model(df, axis[0])
    h, h_ds, h_pred = make_humidity_model(df, axis[1])
    p, p_ds, p_pred = make_precipitations_model(df, axis[2])
    a, a_ds, a_pred = make_atmospheric_pressure_model(df, axis[3])

    print("Temperature (K)")
    measure_performance(t, t_ds, t_pred)

    print("Humidity (%)")
    measure_performance(h, h_ds, h_pred)

    print("Precipitations (mm)")
    measure_performance(p, p_ds, p_pred)

    print("Atmospheric Pressure (Pa)")
    measure_performance(a, a_ds, a_pred)

    pyplot.tight_layout()
    pyplot.savefig(f"dist/dataset.png")


if __name__ == "__main__":
    main()
