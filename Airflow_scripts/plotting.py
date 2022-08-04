import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import json


def get_boxplot(args, filename="Boxplot.pdf", title=""):

    data = []

    for arg in args:

        trace = go.Box(y=arg["y"], name=arg["name"])

        data.append(trace)

    layout = go.Layout(title=title)

    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, filename)


def get_barplot(args):

    data = []

    for arg in args:

        trace = go.Bar(y=arg["y"], name=arg["name"])

        data.append(trace)

    layout = go.Layout(title="Barchart")

    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, "fig1.pdf")


def get_lineplot(
    args, filename="Linechart.pdf", title="", xaxis_title="", yaxis_title=""
):

    data = []

    for arg in args:

        trace = go.Scatter(
            x=arg.get("x", np.arange(0, len(arg["y"]))),
            y=arg["y"],
            name=arg.get("name", None),
            line={"color": arg.get("color", None)},
        )
        data.append(trace)

    layout = go.Layout(
        title=title, xaxis={"title": xaxis_title}, yaxis={"title": yaxis_title}
    )

    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, filename)


def get_table(args):
    table_list = args["list"]
    trace = go.Table(
        header=dict(values=["Accuracy", "Precision", "Recall", "F1-Score"]),
        cells=dict(values=table_list),
    )

    data = [trace]


def read_json_data(filename):

    f = open(filename, "r")
    data = f.read()
    data_cleaned = data.replace("\n", "").split("}")
    json_list = []
    for elem in data_cleaned:
        if elem != "":
            json_list.append(json.loads(elem + "}"))

    f.close()
    return json_list


def read_roc_json(filename):

    f = open(filename, "r")
    lj = json.load(f)
    f.close()

    roc = [(key, val["TPR"], val["FPR"]) for key, val in lj.items()]
    return roc


def get_values(column_list, filename):

    json_list = read_json_data(filename)
    metric_dict = {}
    for column in column_list:
        metric = [elem[column] for elem in json_list]
        metric_dict[column] = metric

    return metric_dict
