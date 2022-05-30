from curses import raw
import mimetypes
from os import getenv
from flask_pymongo import PyMongo
from flask import Flask, render_template, request, url_for, redirect
from requests import Response
from .data import *
import numpy as np
import pandas as pd

import io
import base64


def create_app():

    # Create a Flask app object
    app = Flask(__name__)

    # Connect our application to the MongoDB Instance
    app.config["MONGO_URI"] = getenv('DATABASE_URI')
    MONGO_DB_CLIENT = PyMongo()
    MONGO_DB_CLIENT.init_app(app)
    DB = MONGO_DB_CLIENT.db

    # HOME ROUTE ************************************************************************

    @ app.route("/", methods=['GET', 'POST'])
    def home_page(image=None,
                  message=""):

        # TESTING SPACE =====================================================================

        # ===================================================================================

        # Form Requests
        if request.method == 'POST':

            # Search Form
            if 'ticker' in request.form:
                ticker = request.values['ticker']

                # Get Data
                df = get_data(ticker)

                if df.shape[0] >= 11*24*3:  # At Least 3 Months of Data for Error Bounds

                    # print(f"TICKER: {ticker}\nSHAPE: {df.shape}")
                    # print(f"STARTING DATE: {df.index[0]}")
                    # print(f"ENDING DATE: {df.index[-1]}")

                    message = ""

                    # Scale Data
                    scaled_df, scaler_dict = scale_data(df)

                    # Format Data into Sequences
                    last_X_seq, past_X_seq, past_Y_seq = create_sequences(
                        scaled_df)
                    # print("PAST X SEQ SHAPE: ", past_X_seq.shape)
                    # print("PAST Y SEQ SHAPE: ", past_Y_seq.shape)

                    # Load Model and Get Predictions
                    future_predictions = get_predictions(
                        last_X_seq, scaler_dict).reshape(-1)

                    # Approximate Model Error
                    past_errors = get_pred_errors_by_hour_ahead(
                        past_X_seq, past_Y_seq, scaler_dict)

                    # Create Figure
                    figure = create_figure(ticker=ticker,
                                           raw_data=df,
                                           predictions=future_predictions,
                                           errors=past_errors)

                    # Convert Figure to PNG Image
                    pngImage = io.BytesIO()
                    FigureCanvas(figure).print_png(pngImage)

                    # Encode PNG image to base64 string
                    pngImageB64String = "data:image/png;base64,"
                    pngImageB64String += base64.b64encode(
                        pngImage.getvalue()).decode('utf8')
                    image = pngImageB64String
                else:
                    message = f"No Data Found For {ticker} - Try a Different Ticker or Wait 30s for API limit to Refresh"

                # Requires the following into the jinja2 template
                # <img src="{{ image }}"/>

        return render_template('base.html', image=image, message=message)
    # ************************************************************************************

    return app
