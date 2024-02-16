from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
import plotly.express as px
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
# initialise the app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# train dataset, just change your directory path here
toms_path = '~/Schreibtisch/Data/Kaggle/playground-series-s4e2/train.csv'
hastamithuns_path = '...'
df_train = pd.read_csv('~/Schreibtisch/Data/Kaggle/playground-series-s4e2/train.csv').iloc[::1]

# categorical columns
categorical_columns = df_train.select_dtypes(include=['object']).columns.to_list()
print(categorical_columns)

# numerical columns
numerical_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.to_list()
print(numerical_columns)

def df_numerical_to_categorical_columns(df, numerical_columns):
    df_categorical = deepcopy(df)
    for col in numerical_columns:
        df_categorical[col] = pd.cut(df[col], bins=5, labels=[f'{col}_{i}' for i in range(1, 6)])
    return df_categorical

df_train_categorical = df_numerical_to_categorical_columns(df_train, numerical_columns)
# This is the main layout of the dashboard
app.layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col([
            html.H1('Data Visualisation Dashboard')
        ])
    ]),
    # explain the variables with html
    dbc.Row([

        dcc.Markdown('Description of the variables can be found [here](https://www.kaggle.com/code/chetanaggarwal01/obesity-or-cvd-risk-classify-regressor-cluster) '),
    ]),
    # Now  plot 2 columns against each other
    dbc.Row([
        dcc.Markdown('### Scatterplot of two columns for numerical values'),
        dcc.Markdown('Select two columns to plot against each other'),
        dbc.Col([
            dcc.Dropdown(
                id='2d_plot_x',
                options=numerical_columns,
                value= 'Age'
            ),
            dcc.Dropdown(
                id='2d_plot_y',
                options=numerical_columns,
                value='Height'
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='2d_plot', style={'height': '800px'})
        ])
    ]),
    # parallel_coordinates plot for the train dataset categorical columns
    dbc.Row([
        dcc.Markdown('### Parallel coordinates plot for the train dataset for categorical columns'),
    ]),
    dbc.Row([
        # directly insert the parallel coordinates plot
        dcc.Graph(figure=px.parallel_categories(df_train), style={'height': '800px'})
    ])

])

@app.callback(
    Output('2d_plot', 'figure'),
    Input('2d_plot_x', 'value'),
    Input('2d_plot_y', 'value')
)
def update_2d_plot(x, y):
    fig = px.scatter(df_train, x=x, y=y, color='NObeyesdad', marginal_y='box', marginal_x='box')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

