#!/usr/bin/env python3

from datetime import datetime, timedelta
from matplotlib import pyplot
from numpy import arange, array, ndindex, zeros
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
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVR

# Utilities


def create_wind_direction_matrix(X, y, ax):
    # 4 seasons and 8 directions
    M = zeros([8, 4])

    for i, x in enumerate(X[:, 0]):
        M[x, y[i]] += 1

    ax.matshow(M, cmap=pyplot.cm.Blues)

    for ix, iy in ndindex(M.shape):
        m = M[ix, iy]
        ax.text(iy, ix, str(m), ha="center", va="center")

    # https://stackoverflow.com/a/65151948
    ax.set_xticks(arange(4))
    ax.set_xticklabels(["Winter", "Spring", "Summer", "Autumn"])

    ax.set_yticks(arange(8))
    ax.set_yticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])


def degree_to_direction(angle: int):
    direction = int((angle / 45) + 0.5)

    directions = [0, 1, 2, 3, 4, 5, 6, 7]

    return directions[(direction % 8)]


def day_to_season(day: int, n_days: int):
    if day == 0:
        day += 1

    if day == n_days:
        day -= 1

    return int((day / n_days) * 4)


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


def plot_regression(fig_name, param, dataset, y_pred):
    X_train, X_test, y_train, y_test = dataset

    # Make sure everything is clear
    pyplot.clf()
    pyplot.figure(figsize=(18, 8))

    # Add the training data
    pyplot.scatter(
        X_train,
        y_train,
        alpha=0.1,
        label="Training",
    )

    # Add the test data
    pyplot.scatter(X_test, y_test, alpha=0.4, label="Test")

    # Add the predictions
    pyplot.scatter(X_test, y_pred, alpha=0.4, label="Predictions")

    # Add dots for today
    today = get_day_from_date(datetime.now())
    pyplot.scatter(today, y_test[today], alpha=0.8, s=128, label="Today (y_test)")
    pyplot.scatter(today, y_pred[today], alpha=0.8, s=128, label="Today (y_pred)")

    pyplot.xlabel.set_text = "Day of the Year"
    pyplot.ylabel.set_text = param

    pyplot.legend()

    pyplot.savefig(f"dist/{fig_name}.png")


# Models


def make_temperature_model(df):
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

    plot_regression(
        "temperature",
        "Temperature (Celsius)",
        dataset=(X_train, X_test, y_train, y_test),
        y_pred=y_pred,
    )

    return model, ds, y_pred


def make_humidity_model(df):
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

    plot_regression(
        "humidity",
        "Humidity (%)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


def make_wind_direction_model(df):
    ds = load_ds(df, "dd")

    y_train, y_test, X_train, X_test = ds

    X_train = X_train.astype(int)
    X_test = X_test.astype(int)

    # Convert angles into directions
    X_train[:] = [degree_to_direction(x) for x in X_train]
    X_test[:] = [degree_to_direction(x) for x in X_test]

    # Rehape X
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Flatten y
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Convert days into seasons
    y_train_max = max(y_train)
    y_test_max = max(y_test)

    y_train[:] = [day_to_season(y, y_train_max) for y in y_train[:]]
    y_test[:] = [day_to_season(y, y_test_max) for y in y_test[:]]

    # Build & fit the model
    model = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(
            max_depth=64,
        ),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Plot the results
    _, (ax1, ax2, ax3) = pyplot.subplots(1, 3, sharey=True, figsize=(16, 8))

    ax1.title.set_text("Training")
    create_wind_direction_matrix(X_train, y_train, ax1)

    ax2.title.set_text("Test")
    create_wind_direction_matrix(X_test, y_test, ax2)

    ax3.title.set_text("Predictions")
    create_wind_direction_matrix(X_test, y_pred, ax3)

    pyplot.savefig(f"dist/wind_direction.png")

    return model, ds, y_pred


def make_wind_speed_model(df):
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

    plot_regression(
        "wind_speed",
        "Average wind speed 10 min (m/s)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


def make_precipitation_model(df):
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

    plot_regression(
        "precipitation",
        "Precipitation in the last 24 hours (mm)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


def make_atmospheric_pressure_model(df):
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

    plot_regression(
        "atmospheric_pressure",
        "Atmospheric pressure (Pa)",
        dataset=ds,
        y_pred=y_pred,
    )

    return model, ds, y_pred


def main():
    # Load the whole data from the CSV files
    print("⚙️  Load the data...")
    df = load_df(stations=[7558])

    # Build & fit the models
    print(f"🚧 Building the models (this may take some time)...")
    t, t_ds, t_pred = make_temperature_model(df)
    h, h_ds, h_pred = make_humidity_model(df)
    wd, wd_ds, wd_pred = make_wind_direction_model(df)
    ws, ws_ds, ws_pred = make_wind_speed_model(df)
    p, p_ds, p_pred = make_precipitation_model(df)
    a, a_ds, a_pred = make_atmospheric_pressure_model(df)

    print("Temperature (Celsius)")
    measure_performance(t, t_ds, t_pred)

    print("Humidity (%)")
    measure_performance(h, h_ds, h_pred)

    print("Wind Direction")
    measure_performance(wd, wd_ds, wd_pred)

    print("Wind Speed (m/s)")
    measure_performance(ws, ws_ds, ws_pred)

    print("Precipitations (mm)")
    measure_performance(p, p_ds, p_pred)

    print("Atmospheric Pressure (Pa)")
    measure_performance(a, a_ds, a_pred)


if __name__ == "__main__":
    main()
