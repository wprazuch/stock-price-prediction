import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import os
from stock_pred import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = dash.Dash()
server = app.server

with open(r"scaler.pickle", "rb") as input_file:
    scaler = pickle.load(input_file)

dataset = pd.read_hdf(r"datasets\tata_dataset.h5")


data = dataset.values
scaled_data = scaler.transform(data)

model = load_model("saved_model.h5")


X_test, _ = utils.generate_sequence_data(scaled_data, history_len=60)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

predicted = dataset.copy()
predicted['Predictions'] = np.hstack([np.zeros((60,)), closing_price.ravel()])


app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=dataset.index,
                                y=dataset["Close"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=predicted.index,
                                y=predicted["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                )
            ])


        ]),

    ])
])


if __name__ == '__main__':
    app.run_server(debug=True)
