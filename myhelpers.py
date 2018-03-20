import base64
import datetime
import time as t
import io
import codecs
import pickle
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import savefig
import urllib
import os

from rq import Queue
from worker import conn
import uuid

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation


# file upload function
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return None

    return df


# Generate table function
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


# Convert matplotlib to html image function.
def convert_fig_to_html(fig):
    # Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding.
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    # html = '<img src = "%s"/>' % uri

    return uri


# Encode as portable pickled string
def encode(results):
    pickled_results = codecs.encode(pickle.dumps(results), "base64").decode()
    return pickled_results

# Decode from portable pickled string
def decode(pickled_results):
    unpickled = pickle.loads(codecs.decode(pickled_results.encode(), "base64"))
    return unpickled