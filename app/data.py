from os import getenv
import requests
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime, timedelta

# Alphavnatage API Constants
API_KEY = getenv("ALPHAVANTAGE_API_KEY")
FUNCTION = "TIME_SERIES_INTRADAY_EXTENDED"
INTERVAL = "60min"

# Data Formatting Constants
TARGET = "close"
LOOK_BACK = 11*24
LOOK_AHEAD = 4*24

# Model Loading Constants
MODEL_FOLDER = "models"
MODEL_NAME = "tuned_model1"


def query(ticker, window=24):
    if window <= 12:
        y = 1
        m = window
    elif window <= 24:
        y = 2
        m = window-12
    csv_url = f"http://alphavantage.co/query?function={FUNCTION}&symbol={ticker}"
    csv_url += f"&interval={INTERVAL}&slice=year{y}month{m}&apikey={API_KEY}"
    with requests.Session() as s:
        try:
            download = s.get(csv_url)
            decoded_content = download.content.decode('utf-8')
            data = StringIO(decoded_content)
            df = pd.read_csv(data, delimiter=",", parse_dates=[
                             "time"], index_col="time")
        except:
            print(f"Failed to Load year{y}month{m}")
            df = pd.DataFrame()
    return df


def get_data(ticker):
    num_months_in_past = 5
    for i in range(num_months_in_past):
        if i == 0:
            df = query(ticker, i+1)
        else:
            df = pd.concat([df, query(ticker, i+1)])
    df = df.sort_index(ascending=True)
    return df


def scale_data(df):
    scaled_data = {}
    scaler_dict = {}
    for col in df.columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[col] = scaler.fit_transform(
            df[col].values.reshape(-1, 1)).flatten()
        scaler_dict[col] = scaler
    return pd.DataFrame.from_dict(scaled_data), scaler_dict


def invert_scaling(data, feat_name, scaler_dict):
    return scaler_dict[feat_name].inverse_transform(data)


def create_sequences(scaled_dataframe):
    n_samples = scaled_dataframe.shape[0]
    n_cols = scaled_dataframe.shape[1]

    # Sequence for Future Predictions
    starting_index = n_samples-1-LOOK_BACK
    stopping_index = starting_index+LOOK_BACK
    X = scaled_dataframe.to_numpy()[starting_index:stopping_index]
    X = X.reshape(1, LOOK_BACK, n_cols)

    # Sequences for Past Error
    X_prior = None
    while True:
        stopping_index = starting_index
        starting_index = starting_index-LOOK_BACK
        if starting_index < 0:
            break

        # Divide Data Into Training and Validation Sequences
        x_data = scaled_dataframe.to_numpy()[starting_index:stopping_index]
        x_data = x_data.reshape(1, LOOK_BACK, n_cols)
        y_data = scaled_dataframe[TARGET].to_numpy(
        )[stopping_index:stopping_index+LOOK_AHEAD]
        y_data = y_data.reshape(1, LOOK_AHEAD)

        # Combine Into Tensors
        if X_prior is None:
            X_prior = x_data
            Y_prior = y_data
        else:
            X_prior = np.vstack((X_prior, x_data))
            Y_prior = np.vstack((Y_prior, y_data))

    return X, X_prior, Y_prior


def get_predictions(sequence_array, scaler_dict):
    pretrained_model = tensorflow.keras.models.load_model(
        f"{MODEL_FOLDER}/{MODEL_NAME}")

    scaled_predictions = pretrained_model.predict(sequence_array)
    predictions = invert_scaling(scaled_predictions, TARGET, scaler_dict)
    return predictions


def get_pred_errors_by_hour_ahead(past_X_seq, past_Y_seq, scaler_dict):
    pretrained_model = tensorflow.keras.models.load_model(
        f"{MODEL_FOLDER}/{MODEL_NAME}")
    scaled_predictions = pretrained_model.predict(past_X_seq)
    predictions = invert_scaling(
        scaled_predictions, TARGET, scaler_dict)
    Y_observed = invert_scaling(past_Y_seq, TARGET, scaler_dict)

    n_prediciton_windows = predictions.shape[0]
    absolute_errors_by_hour = np.repeat([0.0], LOOK_AHEAD)
    for i in range(n_prediciton_windows):
        last_error = None
        for j in range(LOOK_AHEAD):
            error = np.absolute(
                predictions[i, j] - Y_observed[i, j])
            if last_error is not None:
                error = np.max([error, last_error])
            absolute_errors_by_hour[j] += error
            last_error = error
    mean_absolute_errors_by_hour = (
        absolute_errors_by_hour) / n_prediciton_windows

    return mean_absolute_errors_by_hour


def create_figure(ticker, raw_data, predictions, errors):

    figure = Figure(figsize=(20, 5))
    axis = figure.add_subplot(1, 1, 1)
    axis.set_title(f"{ticker} - 4 Day Forecast")
    axis.set_xlabel("Time")
    axis.set_ylabel("Closing Price")
    axis.grid()
    axis.plot(
        raw_data.index,
        raw_data[TARGET],
        label="Observed Closing Price",
        c="c"
    )
    s = raw_data.index[-1]
    # axis.set_ylim(bottom=0)
    axis.set_xlim(left=s-timedelta(hours=11*24*3),
                  right=s+timedelta(hours=4*24))
    prediction_date_range = pd.date_range(start=s,
                                          freq="H",
                                          periods=LOOK_AHEAD)
    axis.plot(
        prediction_date_range,
        predictions+1.0*errors,
        label="Predicted Upper Bound",
        linestyle="dashed",
        c="m"
    )
    axis.plot(
        prediction_date_range,
        predictions-1.0*errors,
        label="Predicted Lower Bound",
        linestyle="dashed",
        c="r"
    )
    axis.legend()

    return figure
