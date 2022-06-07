from reflect_rcdock import DockLayout
from reflect_utils.md_parsing import parse_md_doc
from .Forecast_vizualisation_specific import app as content


def app():
    defaultLayout = {
        "dockbox": {
            "mode": "horizontal",
            "children": [
                {
                    "mode": "vertical",
                    "children": [
                        {
                            "tabs": [
                                {
                                    "title": "Description",
                                    "content": parse_md_doc(
                                        open("TSF_Project/description.md", "r").read()
                                    ),
                                },
                            ],
                        },
                        {"tabs": [{"title": "Forecast", "content": content()}]},
                    ],
                },
            ],
        }
    }
    return DockLayout(
        defaultLayout=defaultLayout,
        style={
            "width": "100%",
            "height": "100%",
        },
    )
