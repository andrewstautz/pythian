
# coding: utf-8

# In[9]:


import base64
import datetime
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

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation


#### FUNCTIONS

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



# Main calculation script.
def main_script(data, forecast_periods):
    forecast_periods = forecast_periods
    cv_horizon = str(forecast_periods) + ' days'
    if data.shape[1] == 2:
        data['groups'] = 'None'
    data.columns = ['ds', 'y', 'groups']
    data_grouped = data.groupby('groups')
    groups = [group_name for group_name, group_data in data_grouped]
    models = [Prophet(holidays=holidays, holidays_prior_scale=4).fit(group_data) for group_name, group_data in data_grouped]
    futures = [model.make_future_dataframe(periods=forecast_periods) for model in models]
    forecasts = [model.predict(future) for model, future in zip(models, futures)]
    #plots = [model.plot(forecast) for model, forecast in zip(models, forecasts)]
    initials = [ str(group_data.shape[0] - ((forecast_periods * 2) + 2)) + ' days' for group_name, group_data in data_grouped]
    cv_results = [cross_validation(model, horizon=cv_horizon, period=cv_horizon, initial=initial) for model, initial in zip(models, initials)]
    for dataframe in cv_results:
        dataframe['error'] = dataframe['yhat'] - dataframe['y']
        dataframe['abs_error'] = dataframe['error'].transform(np.abs)
        dataframe['squared_error'] = dataframe['error'].transform(np.square)
        dataframe['pct_error'] = np.divide(dataframe['abs_error'], dataframe['y'])
    RMSE = [np.sqrt(np.mean(cv_result['squared_error'])) for cv_result in cv_results]
    MAPE = [np.mean(cv_result['pct_error']) for cv_result in cv_results]
    complete = pd.DataFrame(index = groups,
                        data = {'error_rmse': RMSE,
                                'error_mape': MAPE,
                                'models': models,
                                #'plots': plots,
                                'forecasts': forecasts})
    return complete #The output table with forecast, plots, and errors by group.


####

# Instantiate app.
app = dash.Dash()
server = app.server

# Holidays dataframe (static global variable) 
holidays = pd.DataFrame({'holiday': ['New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day', 
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day', 
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day', 
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day', 
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day', 
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day', 
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day', 
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day', 
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day', 
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day', 
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day', 
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day', 
                                     'New Year Day', 'Martin Luther King Jr. Day', 'Presidents Day', 'Memorial Day', 'Independence Day', 
                                     'Labor Day', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Black Friday', 'Christmas Day', 
                                     'Christmas Eve', 'Christmas Eve', 'Christmas Eve', 'Christmas Eve', 'Christmas Eve', 'Christmas Eve', 
                                     'Christmas Eve', 'New Years Eve', 'New Years Eve', 'New Years Eve', 'New Years Eve', 'New Years Eve', 
                                     'New Years Eve', 'New Years Eve', 'Valentines Day', 'Valentines Day', 'Valentines Day', 'Valentines Day', 
                                     'Valentines Day', 'Valentines Day', 'Valentines Day', 'Halloween', 'Halloween', 'Halloween', 'Halloween', 
                                     'Halloween', 'Halloween', 'Halloween', 'iPhone Launch', 'iPhone Launch', 'iPhone Launch', 'iPhone Launch', 
                                     'iPhone Launch', 'iPhone Launch', 'Pixel Launch', 'Pixel Launch', 'Easter', 'Easter', 'Easter', 'Easter', 
                                     'Easter', 'Easter', 'Easter', 'Samsung Launch', 'Samsung Launch', 'Samsung Launch', 'Samsung Launch', 
                                     'Samsung Launch'],
                         'ds': pd.to_datetime(['2014-01-01', '2014-01-20', '2014-02-17', '2014-05-26', '2014-07-04', '2014-09-01', 
                                               '2014-10-13', '2014-11-11', '2014-11-27', '2014-11-28', '2014-12-25', '2015-01-01', 
                                               '2015-01-19', '2015-02-16', '2015-05-25', '2015-07-03', '2015-09-07', '2015-10-12', 
                                               '2015-11-11', '2015-11-26', '2015-11-27', '2015-12-25', '2016-01-01', '2016-01-18', 
                                               '2016-02-15', '2016-05-30', '2016-07-04', '2016-09-05', '2016-10-10', '2016-11-11', 
                                               '2016-11-24', '2016-11-25', '2016-12-25', '2017-01-02', '2017-01-16', '2017-02-20', 
                                               '2017-05-29', '2017-07-04', '2017-09-04', '2017-10-09', '2017-11-10', '2017-11-23', 
                                               '2017-11-24', '2017-12-25', '2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28', 
                                               '2018-07-04', '2018-09-03', '2018-10-08', '2018-11-12', '2018-11-22', '2018-11-23', 
                                               '2018-12-25', '2019-01-01', '2019-01-21', '2019-02-18', '2019-05-27', '2019-07-04', 
                                               '2019-09-02', '2019-10-14', '2019-11-11', '2019-11-28', '2019-11-29', '2019-12-25', 
                                               '2020-01-01', '2020-01-20', '2020-02-17', '2020-05-25', '2020-07-03', '2020-09-07', 
                                               '2020-10-12', '2020-11-11', '2020-11-26', '2020-11-27', '2020-12-25', '2014-12-24', 
                                               '2015-12-24', '2016-12-24', '2017-12-24', '2018-12-24', '2019-12-24', '2020-12-24', 
                                               '2014-12-31', '2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31', 
                                               '2020-12-31', '2014-02-14', '2015-02-14', '2016-02-14', '2017-02-14', '2018-02-14', 
                                               '2019-02-14', '2020-02-14', '2014-10-31', '2015-10-31', '2016-10-31', '2017-10-31', 
                                               '2018-10-31', '2019-10-31', '2020-10-31', '2014-09-19', '2015-09-25', '2016-03-31', 
                                               '2016-09-16', '2017-09-22', '2017-11-03', '2016-10-20', '2017-10-19', '2014-04-20', 
                                               '2015-04-05', '2016-03-27', '2017-04-16', '2018-04-01', '2019-04-21', '2020-04-12', 
                                               '2014-04-11', '2015-04-10', '2016-03-11', '2017-04-21', '2018-03-16'])})

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
        html.Div(id='intermediate_value', style={'display': 'none'}),
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

# Computation step callback (main script here)
@app.callback(
    Output('intermediate_value', 'children'), 
    [Input('go_button', 'n_clicks'),
     Input('forecast_periods_slider', 'value'),
     Input('upload_data', 'contents'),
     Input('upload_data', 'filename')]
)
def run_calculation(n_clicks, forecast_periods, contents, filename):
     if n_clicks > 0:
        data = parse_contents(contents, filename)
        q = Queue(connection=conn)
        job = q.enqueue_call(func=main_script, args=(data, forecast_periods))
        if job is not None:
            if job.result is not None:
                calculated_data = job.result
                return codecs.encode(pickle.dumps(calculated_data), "base64").decode()

#
# To decode string in hidden div:
# unpickled = pickle.loads(codecs.decode(OBJECT.encode(), "base64"))
#

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
        return generate_table(data)

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
        return convert_fig_to_html(fig)

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
    app.run_server(debug=True)

