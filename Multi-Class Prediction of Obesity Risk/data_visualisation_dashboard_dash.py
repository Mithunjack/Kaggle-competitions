from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# initialise the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# train dataset
df_train = pd.read_csv('~/Schreibtisch/Data/Kaggle/playground-series-s4e2/train.csv').iloc[::3]
# test dataset
df_test = pd.read_csv('~/Schreibtisch/Data/Kaggle/playground-series-s4e2/test.csv').iloc[::3]

# This is the main layout of the dashboard
app.layout = dbc.Container([
    dbc.Col([
        dcc.Markdown('Obesity Dataset Explorer', style={'textAlign': 'center'}),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='1d_plot_columns',
                #options=[{'label': col, 'value': col} for col in df_train.columns],
                options=df_test.columns.to_list(),
                value='age'
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            graph_1d := dcc.Graph(id='1d_plot')
        ])
    ]),
    # Now  plot 2 columns against each other
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='2d_plot_x',
                options=df_train.columns.to_list(),
                value='FAVC'
            ),
            dcc.Dropdown(
                id='2d_plot_y',
                options=df_train.columns.to_list(),
                value='FAVC'
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            graph_2d := dcc.Graph(id='2d_plot')
        ])
    ])
])

@app.callback(
    Output(graph_1d, component_property='figure'),
    Input('1d_plot_data_select', 'value'),
    Input('1d_plot_columns', 'value'),
)
def update_1d_plot(data_select, column):
    fig = px.bar(df_train, x='NObeyesdad', y=column)
    return fig

@app.callback(
    Output(graph_2d, component_property='figure'),
    Input('2d_plot_x', 'value'),
    Input('2d_plot_y', 'value'),
)
def update_2d_plot(x, y):
    fig = px.scatter(df_train, x=x, y=y, color='NObeyesdad')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)