
# coding: utf-8

# In[9]:


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

import myfunctions
import myhelpers


####

# Instantiate app.
app = dash.Dash()
server = app.server


# App layout.
app.layout = html.Div([
    # Sidebar
    html.Div([
        html.H4('Pythian Forecasts', id='title', style={'color':'rgb(23,107,143)'}),
        html.P('This application wraps a Python implementation of Facebook\'s Prophet forecasting package. \
               Upload a .csv file with columns date, value, and groups, in that order. \
               Dates must be in yyyy-mm-dd format. Values are the quantity to be predicted. \
               Groups will be used to split the data into separate forecasts by group.', id='helptext'),
        dcc.Upload(
            id='upload_data',
            children=html.Button('Upload File'),
            multiple=False),
        html.P(id='upload_confirmation'),
        # Forecast slider div.
        html.Div([
            html.H6('Set Forecast Horizon:', style={'color':'rgb(23,107,143)'}),
            dcc.Slider(
                id='forecast_periods_slider',
                marks={15: '15',
                   30: '30',
                   45: '45',
                   60: '60',
                   75: '75'},
                min=0,
                max=90,
                step=1,
                value=7)],
            style={'height':100, 'width':'100%', 'display':'inline-block'}
        ), # Close forecast slider div.
        html.Button(
            id='go_button',
            n_clicks=0,
            children='Go!'),
        html.P(id='go_button_confirmation'),
        html.Div(id='intermediate_value', style={'display': 'none'}), # my original invisible div for results...
        html.Div(id='status'), # Div to show status message
        html.Div(id='job_id', style={'display': 'none'}), # invisible div to store current job ID.
        dcc.Interval(
            id='update_interval',
            interval=60*60*5000, # in milliseconds
            n_intervals=0
        ), # This (invisible?) element controls report refresh.

        # Groups dropdown div.
        html.Div([dcc.Dropdown(
                     id='groups_dropdown',
                     placeholder='Select a group.')],
                 style={'height':30, 'width':'100%', 'display':'inline-block'}
                ), # Close groups dropdown div.
        html.H6(id='mape', style={'color':'rgb(23,107,143)'}),
        html.H6(id='rmse', style={'color':'rgb(23,107,143)'})],
        className = 'left',
        style = {
                 'position':'fixed',
                 'width': 360,
                 }
    ), # Close sidebar

    # Main panel
    html.Div([
        html.Img(id='plot'),
        html.Div(id='results_table'),
        html.A(
            html.Button('Download Results'),
            id='download_link',
            download="rawdata.csv",
            href="",
            target="_blank",
            style = {'margin':10}
        )],
        className = 'main',
        style = {
                 'marginLeft':360,
                 'padding':20
                 #'height':'100%',
                 #'backgroundColor': 'blue'
        }
    ) # Close main panel.
]) # Close app layout.


# Upload confirmation callback
@app.callback(
    Output('upload_confirmation', 'children'),
    [Input('upload_data', 'contents')]
)
def confirm_upload(contents):
    if contents is not None:
        children = html.Div('Looking good...')
        return children

# Go button confirmation callback (placeholder)
@app.callback(
    Output(component_id='go_button_confirmation', component_property='children'),
    [Input(component_id='go_button', component_property='n_clicks')]
)
def update_output_div(n_clicks):
    if n_clicks > 0:
        return 'We\'re on our way.'


# this callback checks submits the query as a new job, returning job_id to the invisible div
@app.callback(
    Output('job_id', 'children'),
    [Input('go_button', 'n_clicks'),
     Input('forecast_periods_slider', 'value'),
     Input('upload_data', 'contents'),
     Input('upload_data', 'filename')])
def query_submitted(click, forecast_periods, contents, filename):
    if click == 0 or click is None:
        return ''
    else:
        # a query was submitted, so queue it up and return job_id
        data = myhelpers.parse_contents(contents, filename)
        duration = 20           # pretend the process takes 20 seconds to complete
        q = Queue(connection=conn)
        job_id = str(uuid.uuid4())
        job = q.enqueue_call(func=myfunctions.calculation_script,
                                args=(data, forecast_periods),
                                timeout='3m',
                                job_id=job_id)
        return job_id


#
# To encode results table:
# codecs.encode(pickle.dumps(calculated_data), "base64").decode()
#
# To decode string in hidden div:
# unpickled = pickle.loads(codecs.decode(OBJECT.encode(), "base64"))
#


# this callback checks if the job result is ready.  If it's ready
# the results return to the table.  If it's not ready, it pauses
# for a short moment, then empty results are returned.  If there is
# no job, then empty results are returned.
@app.callback(
    Output('intermediate_value', 'children'),
    [Input('update_interval', 'n_intervals')],
    [State('job_id', 'children')])
def update_results_tables(n_intervals, job_id):
    q = Queue(connection=conn)
    job = q.fetch_job(job_id)
    if job is not None:
        # job exists - try to get result
        result = job.result
        if result is None:
            # results aren't ready, pause then return empty results
            # You will need to fine tune this interval depending on
            # your environment
            t.sleep(5)
            return ''
        if result is not None:
            # results are ready
            return result
    else:
        # no job exists with this id
        return ''


# this callback orders the table to be regularly refreshed if
# the user is waiting for results, or to be static (refreshed once
# per hour) if they are not.
@app.callback(
    dash.dependencies.Output('update_interval', 'interval'),
    [dash.dependencies.Input('job_id', 'children'),
    dash.dependencies.Input('update_interval', 'n_intervals')])
def stop_or_start_table_update(job_id, n_intervals):
    q = Queue(connection=conn)
    job = q.fetch_job(job_id)
    if job is not None:
        # the job exists - try to get results
        result = job.result
        if result is None:
            # a job is in progress but we're waiting for results
            # therefore regular refreshing is required.  You will
            # need to fine tune this interval depending on your
            # environment.
            return 1000
        else:
            # the results are ready, therefore stop regular refreshing
            return 60*60*1000
    else:
        # the job does not exist, therefore stop regular refreshing
        return 60*60*1000


# this callback displays a please wait message in the status div if
# the user is waiting for results, or nothing if they are not.
@app.callback(
    dash.dependencies.Output('status', 'children'),
    [dash.dependencies.Input('job_id', 'children'),
    dash.dependencies.Input('update_interval', 'n_intervals')])
def stop_or_start_table_update(job_id, n_intervals):
    q = Queue(connection=conn)
    job = q.fetch_job(job_id)
    if job is not None:
        # the job exists - try to get results
        result = job.result
        if result is None:
            # a job is in progress and we're waiting for results
            return 'Running query.  This might take a moment - don\'t close your browser!'
        else:
            # the results are ready, therefore no message
            return ''
    else:
        # the job does not exist, therefore no message
        return ''


# Groups dropdown callback.
@app.callback(
    Output('groups_dropdown', 'options'),
    [Input('intermediate_value', 'children')]
)
def update_dropdown(pickled_results):
    if pickled_results is not None:
        # Decode the results dataframe.
        dataframe = pickle.loads(codecs.decode(pickled_results.encode(), "base64"))
        # Make list of dicts with index as both label and value.
        options = [{'label': k, 'value': k} for k in dataframe.index]
        return options
    else:
        return [{}]


# MAPE error callback
@app.callback(
    Output('mape', 'children'),
    [Input('groups_dropdown', 'value'),
     Input('intermediate_value', 'children')]
)
def update_error_table(value, pickled_results):
    if value is not None:
        # Decode the results dataframe.
        dataframe = pickle.loads(codecs.decode(pickled_results.encode(), "base64"))
        data = dataframe['error_mape'][value]
        mape = '{:.2%}'.format(data)
        return 'MAPE: ' + mape


# RMSE error callback
@app.callback(
    Output('rmse', 'children'),
    [Input('groups_dropdown', 'value'),
     Input('intermediate_value', 'children')]
)
def update_error_table(value, pickled_results):
    if value is not None:
        # Decode the results dataframe.
        dataframe = pickle.loads(codecs.decode(pickled_results.encode(), "base64"))
        data = dataframe['error_rmse'][value]
        rmse = '{:.2f}'.format(data)
        return 'RMSE: ' + rmse


# Results table callback
@app.callback(
    Output('results_table', 'children'),
    [Input('groups_dropdown', 'value'),
     Input('intermediate_value', 'children'),
     Input('forecast_periods_slider', 'value')]
)
def update_table(value, pickled_results, forecast_period):
    if value is not None:
        # Decode the results dataframe.
        dataframe = pickle.loads(codecs.decode(pickled_results.encode(), "base64"))
        data = (
            dataframe['forecasts'][value]
            .tail(forecast_period)
            [['ds', 'yhat','yhat_lower','yhat_upper']]
            .rename({'ds':'Date',
                     'yhat':'Forecast',
                     'yhat_lower':'Lower Bound',
                     'yhat_upper':'Upper Bound'}
                     ,axis='columns')
        )
        return myhelpers.generate_table(data)

# Plot forecast callback
@app.callback(
    Output('plot', 'src'),
    [Input('groups_dropdown', 'value'),
     Input('intermediate_value', 'children')]
)
def update_plot(value, pickled_results):
    if value is not None:
        # Decode the results dataframe.
        dataframe = pickle.loads(codecs.decode(pickled_results.encode(), "base64"))
        fig = dataframe['models'][value].plot(dataframe['forecasts'][value])
        return myhelpers.convert_fig_to_html(fig)

# Download link content callback
@app.callback(
    Output('download_link', 'href'),
    [Input('groups_dropdown', 'value'),
     Input('intermediate_value', 'children')])
def update_download_link(value, pickled_results):
    if value is not None:
        # Decode the results dataframe.
        dataframe = pickle.loads(codecs.decode(pickled_results.encode(), "base64"))
        data = dataframe['forecasts'][value]
        csv_string = data.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string

# Download link filename callback
@app.callback(
    Output('download_link', 'download'),
    [Input('groups_dropdown', 'value')])
def update_download_link(value):
    if value is not None:
        # Set the filename to include selected value.
        filename = value + '_data.csv'
        return filename
    else:
        return 'rawdata.csv'

# Dynamic download button callback
@app.callback(
    Output('download_link', 'children'),
    [Input('groups_dropdown', 'value')])
def make_download_button(value):
    if value is not None:
        # Make the button.
        return html.Button('Download Results')
    else:
        return ''


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")

