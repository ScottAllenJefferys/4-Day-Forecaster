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

                print(f"TICKER: {ticker}\nSHAPE: {df.shape}")
                print(f"STARTING DATE: {df.index[0]}")
                print(f"ENDING DATE: {df.index[-1]}")

                if df.shape[0] >= 11*24:
                    message = ""

                    # Scale Data
                    scaled_df, scaler_dict = scale_data(df)

                    # Format as a Sequence for Model
                    seq_array = create_sequence(scaled_df)

                    # Load Model and Get Predictions
                    predictions = get_predictions(seq_array, scaler_dict)

                    # Create Figure
                    figure = create_figure(ticker=ticker,
                                           raw_data=df,
                                           predictions=predictions)

                    # Convert Figure to PNG Image
                    pngImage = io.BytesIO()
                    FigureCanvas(figure).print_png(pngImage)

                    # Encode PNG image to base64 string
                    pngImageB64String = "data:image/png;base64,"
                    pngImageB64String += base64.b64encode(
                        pngImage.getvalue()).decode('utf8')
                    image = pngImageB64String
                else:
                    message = f"No Data Found For {ticker}"

                # Requires the following into the jinja2 template
                # <img src="{{ image }}"/>

        return render_template('base.html', image=image, message=message)
    # ************************************************************************************

    return app
