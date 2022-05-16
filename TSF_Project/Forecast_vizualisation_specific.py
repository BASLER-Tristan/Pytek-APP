import os

import pandas as pd
import plotly.express as px
from reflect_antd import DatePicker, InputNumber, Select, Space
from reflect_html import div
from reflect_plotly import Graph

from database.Forecasting import MODELS

Option = Select.Option

DATA_FOLDER = "database"


class Application:
    def __init__(self):
        samples = [
            name[:-len(".csv")] for name in os.listdir(DATA_FOLDER) if name.endswith(".csv")
        ]
        self.model = Select(
            children=[Option(method.__doc__, key=key) for key, method in MODELS.items()],
            allowClear=True,
            style=dict(width="100%"),
            placeholder="Please select your model",
        )
        self.database = Select(
            children=[Option(key, key=key) for key in (samples)],
            allowClear=True,
            style=dict(width="100%"),
            placeholder="Please select your dataset",
        )
        self.date = DatePicker(picker="month")
        self.input_number = InputNumber(min=1)

    def graph(self):
        if not all((self.model(), self.database(), self.date(), self.input_number())):
            return
        data = pd.read_csv(os.path.join(DATA_FOLDER, self.database() + ".csv"))
        data.set_index(pd.to_datetime(data["Unnamed: 0"]), inplace=True)
        data.drop(columns="Unnamed: 0", inplace=True)
        date = pd.to_datetime(self.date())
        data, dff, dffm = MODELS[self.model()](data, date, int(self.input_number()))
        df = data.join(dff)
        df.fillna(0, inplace=True)
        fig = px.line(dff)
        fig.update_layout(
            title="Evolution of the purchase by region",
            xaxis_title="Date",
            yaxis_title="Purchase",
        )
        fig_error = px.bar(dffm)
        return div([Graph(fig), Graph(fig_error)])


def app():
    application = Application()
    return div(
        [
            Space([application.model, application.database]),
            Space([application.date, application.input_number]),
            application.graph,
        ]
    )
