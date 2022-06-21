##################################################################################
"""



"""
##################################################################################

### Importation
import pandas as pd
from reflect_antd import (
    Space,
    Typography,
    Select,
    InputNumber,
    DatePicker,
    Card,
    Button,
)
from reflect_html import *
from reflect_plotly import Graph
import plotly.express as px
from datetime import datetime
from TSF_Project.database.Forecasting import MODELS
from TSF_Project.database.datacreation import database_explanation, data_columns
import os
import numpy as np

Title, Paragraph, Text, Link = (
    Typography.Title,
    Typography.Paragraph,
    Typography.Text,
    Typography.Link,
)
Option = Select.Option

################### APP ###################
DATA_FOLDER = "TSF_Project/database"
link = "https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/"


class Application:
    def __init__(self):
        samples = [
            name[: -len(".csv")]
            for name in os.listdir(DATA_FOLDER)
            if name.endswith(".csv")
        ]
        self.database = Select(
            children=[Option(key, key=key) for key in (samples)],
            allowClear=True,
            style=dict(width="100%"),
            placeholder="Please select your dataset",
        )

        self.model = Select(
            children=[
                Option(method.__doc__, key=key) for key, method in MODELS.items()
            ],
            allowClear=True,
            style=dict(width="100%"),
            placeholder="Please select your model",
        )
        self.date = DatePicker(picker="month")
        self.input_number = InputNumber(min=1)

        self.col_option = Select(
            children=[Option(key, key=key) for key in data_columns["data_specific"]],
            allowClear=True,
            style=dict(width="100%"),
            placeholder="Please select your product",
        )

    def explanation(self):
        explanation_princip = [
            Title("Method Explanation"),
            Paragraph(
                """
                The goal of this app is to study the result of several methods of time series Forecasting on different databases."
                
                We can define two major interests in this study.
                
                1. A Supply Chain Problem :
                Knowing the next purchase of a product and his quantity is needed for the optimization of the supply chain.
                Indeed, knowing this information, you can choose how and with which product you want to a fill a truck.
                This is the major asset for Amazon. 
                
                2. A Commercial Problem :
                The second information you can extract from this study is the future behavior.
                If you see a reduction of the purchase of one customer, this can mean this customer is going to leave 
                your company for another.
                So, you need to act in order to avoid that
                
                In a more complex way, the forecast is a way of the future can append. So, if the reality is very different
                from the forecast, something must have arrived.
                """
            ),
        ]
        explanation_model = [
            Title("Model Explanation"),
            Paragraph(
                """
                You can find several models in the dropdown menu : 
                """
            ),
            Paragraph(
                """
            * AR : Autoregression,
            """
            ),
            Paragraph(
                """
            * MA : Moving Average,
            """
            ),
            Paragraph(
                """
            * ARMA : Autoregressive Moving Average,
            """
            ),
            Paragraph(
                """
            * ARIMA : Autoregressive Integrated Moving Average,
            """
            ),
            Paragraph(
                """
            * SARIMA : Seasonal Autoregressive Integrated Moving - Average,
            """
            ),
            Paragraph(
                """
            * VAR : Vector Autoregression,
            """
            ),
            Paragraph(
                """
            * VARMA : Vector Autoregression Moving-Average,
            """
            ),
            Paragraph(
                """
            * SES : Simple Exponential Smoothing,
            """
            ),
            Paragraph(
                """
            * HWES : Holt Winter’s Exponential Smoothing,
            """
            ),
        ]
        if not self.database():
            return div(
                [
                    Typography(explanation_princip),
                    Typography(explanation_model),
                    Space([Button("Link to the article", type="link")]),
                ]
            )
        else:
            data = pd.read_csv(os.path.join(DATA_FOLDER, self.database() + ".csv"))
            data.set_index(pd.to_datetime(data["Unnamed: 0"]), inplace=True)
            data.drop(columns="Unnamed: 0", inplace=True)
            explanation_database = [
                Title("Database Explanation"),
                Paragraph(
                    [
                        "The dataset translates the sales of ten different products, between january 2010 and december 2019. ",
                        f"The monthly purchase has been created by {database_explanation[self.database()]}",
                    ]
                ),
            ]
            return div(
                [
                    Typography(explanation_princip),
                    Typography(explanation_model),
                    Space([Button("Link to the article", type="link")]),
                    Typography(explanation_database),
                ]
            )

    def generic_graph(self):
        if not all((self.model(), self.database(), self.date(), self.input_number())):
            return
        else:
            data = pd.read_csv(os.path.join(DATA_FOLDER, self.database() + ".csv"))
            data.set_index(pd.to_datetime(data["Unnamed: 0"]), inplace=True)
            data.drop(columns="Unnamed: 0", inplace=True)

            date = pd.to_datetime(self.date())
            date = datetime(date.year, date.month, date.day)

            N = int(self.input_number())
            data, dff, dffm = MODELS[self.model()](data, date, int(self.input_number()))
            y_true = data.sum(axis=1)
            y_pred = dff.sum(axis=1)

            y_true.index.name = None

            df = pd.concat([y_true, y_pred], axis=1)
            df.columns = ["Total Purchase", "Total Predicted"]
            fig = px.line(df, y=["Total Purchase", "Total Predicted"])
            fig.update_layout(
                title="Evolution of the total purchase",
                xaxis_title="Date",
                yaxis_title="Purchase",
            )

            test_index = pd.date_range(date, periods=N, freq="MS")
            test_index = test_index[test_index <= y_true.index.max()]
            test_index.freq = None
            test_index = pd.DatetimeIndex(test_index)
            diff = (
                    abs(y_pred.loc[test_index] - y_true.loc[test_index])
                    / y_true.loc[test_index]
            )
            fig_error = px.bar(diff)
            fig_error.update_layout(
                title="Evolution of the absolute percent error",
                xaxis_title="Date",
                yaxis_title="Error",
            )
            return div(
                [
                    Space(Graph(fig)),
                    Space(Graph(fig_error)),
                    Space(
                        [
                            Card(
                                np.round(y_true.loc[test_index].sum()),
                                title="Total Purchase on the overlap time values",
                                extra=a(href=True),
                            ),
                            Card(
                                np.round(y_pred.loc[test_index].sum()),
                                title="Total Prediction on the overlap time values",
                                extra=a(href=True),
                            ),
                        ]
                    ),
                ]
            )

    def specific_graph(self):
        if not all(
                (
                        self.model(),
                        self.database(),
                        self.date(),
                        self.input_number(),
                        self.col_option(),
                )
        ):
            return
        else:
            data = pd.read_csv(os.path.join(DATA_FOLDER, self.database() + ".csv"))
            data.set_index(pd.to_datetime(data["Unnamed: 0"]), inplace=True)
            data.drop(columns="Unnamed: 0", inplace=True)

            date = pd.to_datetime(self.date())
            date = datetime(date.year, date.month, date.day)

            N = int(self.input_number())
            data, dff, dffm = MODELS[self.model()](data, date, int(self.input_number()))
            data = data[self.col_option()]
            dff = dff[self.col_option() + "_" + self.model()]
            dffm = dffm[self.col_option() + "_error_" + self.model()]

            y_true = data
            y_pred = dff

            y_true.index.name = None

            df = pd.concat([y_true, y_pred], axis=1)
            df.columns = ["Purchase", "Predicted"]
            df = df.astype(float)
            fig = px.line(df, y=["Purchase", "Predicted"])
            fig.update_layout(
                title="Evolution of the purchase",
                xaxis_title="Date",
                yaxis_title="Volume",
            )

            test_index = pd.date_range(date, periods=N, freq="MS")
            test_index = test_index[test_index <= y_true.index.max()]
            test_index.freq = None
            test_index = pd.DatetimeIndex(test_index)
            diff = (
                    abs(y_pred.loc[test_index] - y_true.loc[test_index])
                    / y_true.loc[test_index]
            )
            fig_error = px.bar(diff)
            fig_error.update_layout(
                title="Evolution of the absolute percent error",
                xaxis_title="Date",
                yaxis_title="Error",
            )
            return div(
                [
                    Space(Graph(fig)),
                    Space(Graph(fig_error)),
                    Space(
                        [
                            Card(
                                int(np.round(y_true.loc[test_index].sum())),
                                title="Total Purchase on the overlap time values",
                                extra=a(href=True),
                            ),
                            Card(
                                int(np.round(y_pred.loc[test_index].sum())),
                                title="Total Prediction on the overlap time values",
                                extra=a(href=True),
                            ),
                        ]
                    ),
                ]
            )


def app():
    application = Application()
    return div(
        [
            Space([application.database]),
            Space([application.model]),
            Space([application.date, application.input_number]),
            Space([application.generic_graph]),
            Space([application.col_option]),
            Space([application.specific_graph]),
        ]
    )
