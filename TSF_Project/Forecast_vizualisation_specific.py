import pandas as pd
from reflect import autorun
from reflect_antd import Input, Space, Typography
from reflect_html import *
from reflect_antd import Select
from reflect import js
import plotly.express as px
from reflect import autorun
from reflect_plotly import Graph
import plotly.express as px
from reflect_antd import InputNumber
from reflect_antd import DatePicker

Option = Select.Option
import itertools
import os
from datetime import datetime
from TSF_Project.database.Forecasting import *
from TSF_Project.database.datacreation import list_database

################### APP


class Graph_test:
    def __init__(self):
        self.model = Select(
            children=[Option(key, key=key) for key in (support_type)],
            allowClear=True,
            style=dict(width="100%"),
            placeholder="Please select your model",
        )
        self.database = Select(
            children=[Option(key, key=key) for key in (list_database)],
            allowClear=True,
            style=dict(width="100%"),
            placeholder="Please select your dataset",
        )
        self.date = DatePicker(picker="month")
        self.input_number = InputNumber(min=1)

    def graph(self):
        if (
            self.model() is None
            or self.database() is None
            or self.date() is None
            or self.input_number() is None
        ):
            pass
        else:
            model = predictor(self.model())
            data = pd.read_csv(self.database())
            data.set_index(pd.to_datetime(data["Unnamed: 0"]), inplace=True)
            data.drop(columns="Unnamed: 0", inplace=True)
            date = pd.to_datetime(self.date())
            N = int(self.input_number())
            data, dff, dffm = model.predict(data, date, N)
            df = data.join(dff)
            df.fillna(0, inplace=True)
            fig = px.line(dff)
            fig.update_layout(
                title="Evolution of the purchase by region",
                xaxis_title="Date",
                yaxis_title="Purchase",
            )
            fig_error = px.bar(dffm)
            return div(
                [
                    Graph(fig),
                    Graph(fig_error),
                ]
            )


def app():
    page = Graph_test()
    return div(
        [
            Space([page.model, page.database]),
            Space([page.date, page.input_number]),
            page.graph,
        ]
    )
