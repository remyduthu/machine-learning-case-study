#!/usr/bin/env python3

from matplotlib import pyplot
from os import getcwd, listdir
from pandas import concat, read_csv, to_datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

    position.set(
        xlabel="Days in the Year",
        ylabel=param,
    )

    position.legend()


# Models


def make_temperature_model(df, position):
    # Load the temperature dataset
    ds = load_ds(df, "t")
    X_train, X_test, y_train, _ = ds

    # Build & fit the model
    model = make_pipeline(
        PolynomialFeatures(degree=6),
        StandardScaler(),
        LinearRegression(),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plot_ds(
        position,
        "Temperature (K)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


def make_humidity_model(df, position):
    ds = load_ds(df, "u")
    X_train, X_test, y_train, _ = ds

    # Build & fit the model
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            max_depth=6,
            max_features="auto",
            min_samples_split=128,
            n_estimators=128,
            random_state=0,
        ),
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


def make_wind_direction_model(df, position):
    ds = load_ds(df, "dd")
    X_train, X_test, y_train, _ = ds

    # Build & fit the model
    model = make_pipeline(
        PolynomialFeatures(degree=4),
        StandardScaler(),
        LinearRegression(),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plot_ds(
        position,
        "Average wind direction 10 min (degré)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


def make_wind_speed_model(df, position):
    ds = load_ds(df, "ff")
    X_train, X_test, y_train, _ = ds

    # Build & fit the model
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            max_features="auto",
            max_depth=10,
            min_samples_split=32,
            n_estimators=512,
            random_state=0,
        ),
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
    model = RandomForestRegressor()
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
    model = RandomForestRegressor()
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
    df = load_df(station_id=7630)

    # Create the main figure
    _, axis = pyplot.subplots(6, figsize=(18, 32))

    # Build & fit the models
    print(f"🚧 Building the models (this may take some time)...")
    t, t_ds, t_pred = make_temperature_model(df, axis[0])
    h, h_ds, h_pred = make_humidity_model(df, axis[1])
    wd, wd_ds, wd_pred = make_wind_direction_model(df, axis[2])
    ws, ws_ds, ws_pred = make_wind_speed_model(df, axis[3])
    p, p_ds, p_pred = make_precipitations_model(df, axis[4])
    a, a_ds, a_pred = make_atmospheric_pressure_model(df, axis[5])

    print("1. Temperature (K)")
    measure_performance(t, t_ds, t_pred)

    print("2. Humidity (%)")
    measure_performance(h, h_ds, h_pred)

    print("3. Wind Direction (degree)")
    measure_performance(wd, wd_ds, wd_pred)

    print("4. Wind Speed (m/s)")
    measure_performance(ws, ws_ds, ws_pred)

    print("5. Precipitations (mm)")
    measure_performance(p, p_ds, p_pred)

    print("6. Atmospheric Pressure (Pa)")
    measure_performance(a, a_ds, a_pred)

    pyplot.tight_layout()
    pyplot.savefig(f"dist/dataset.png")


if __name__ == "__main__":
    main()
