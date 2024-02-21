from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
from copy import deepcopy
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder


# Initialise the app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Load the dataset
df_train = pd.read_csv('~/Schreibtisch/Data/Kaggle/playground-series-s4e2/train.csv')
# i also want to find out how important a column for the prediction is, so i want to get the correlation of all column, But the categorial need to be one hot encoded first


# Define categorical columns
categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()

# Define numerical columns
numerical_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

target_variable = 'NObeyesdad'

#---------------------------- Data preparation for correlatin ------------------------------
# Splitting the dataset into features and target variable
X = df_train.drop(target_variable, axis=1)
y = df_train[target_variable]

# Convert categorical features using one-hot encoding
labelencoder = LabelEncoder()

def mutual_information(df, target_variable):
    df_deepcopy = deepcopy(df)
    # make categorial columns from numerical columns with pandas cut
    for col in df_deepcopy.columns:
        if df_deepcopy[col].dtype == 'float64' or df_deepcopy[col].dtype == 'int64':
            number_of_unique_values = len(df_deepcopy[col].unique())
            column = pd.cut(df_deepcopy[col], 20)
            df_deepcopy[col] = labelencoder.fit_transform(column)
    # now label encode all categorial columns
    for col in df_deepcopy.columns:
        if df_deepcopy[col].dtype == 'object':
            df_deepcopy[col] = labelencoder.fit_transform(df_deepcopy[col])
    # now we can calculate the mutual information
    mi = mutual_info_classif(df_deepcopy.drop(columns = target_variable), df_deepcopy[target_variable])
    columns = df_deepcopy.drop(columns =target_variable).columns
    # sort the columns by the mutual information
    mi, columns = zip(*sorted(zip(mi, columns)))
    return columns, mi, df_deepcopy

columns, mi, df_deepcopy = mutual_information(df_train, target_variable)
# -----------------------------End-------------------------------------

# Define the layout of the dashboard
app.layout = dbc.Container([
    # Title
    dbc.Row([
        dbc.Col([
            html.H1('Data Visualisation Dashboard')
        ])
    ]),
    # Description of the variables
    dbc.Row([
        dcc.Markdown('Description of the variables can be found [here](https://www.kaggle.com/code/chetanaggarwal01/obesity-or-cvd-risk-classify-regressor-cluster)')
    ]),
    # plot missing values
    dbc.Row([
        dcc.Markdown('### Missing values: Plot the missing values in the dataset with a bar chart'),
        dcc.Graph(figure=px.bar(df_train.isna().sum()), style={'height': '500px'})
    ]),
    # target variable distribution with a pie chart
    dbc.Row([
        dcc.Markdown('### Pie chart: Distribution of the categorical variables'),
        # select the target variable
        dcc.Dropdown(id = 'pie_chart_target_variable', options = [{'label': col, 'value': col} for col in categorical_columns], value = categorical_columns[0]),
        dcc.Graph(id = 'pie_chart', style = {'height': '500'})
    ]),
    # Histogram for numerical columns
    dbc.Row([
        dcc.Markdown('### Histogram: Distribution of the numerical and categorial variables'),
        # create 2 dropdowns aside each other, 1 for the column, and 1 for the color
    dbc.Col([
        dcc.Dropdown(
            id='histogram_column',
            options=[{'label': col, 'value': col} for col in df_train.columns],
            value=numerical_columns[0]
        )], width=6),
        # now the color
        dbc.Col([
            dcc.Dropdown(
                id='histogram_color',
                options=categorical_columns,
                value=target_variable
            )], width=6),

        dcc.Graph(id='histogram', style={'height': '800px'})
    ]),
    # Scatterplot of two columns for numerical values
    dbc.Row([
        dcc.Markdown('### 2D Scatterplot: Plot two numerical columns against each other'),
        dcc.Markdown('Select two columns to plot against each other'),
        dbc.Col([
            dcc.Dropdown(
                id='2d_plot_x',
                options=[{'label': col, 'value': col} for col in numerical_columns],
                value=numerical_columns[0]
            ),
            dcc.Dropdown(
                id='2d_plot_y',
                options=[{'label': col, 'value': col} for col in numerical_columns],
                value=numerical_columns[1]
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='2d_plot', style={'height': '800px'})
        ])
    ]),
    dbc.Row([
        dcc.Markdown('### Mutual Information'),
        dcc.Markdown('The mutual information of a column with the target variable is a measure of how much information the column provides about the target variable. The higher the mutual information, the more important the column is for predicting the target variable. The mutual information is calculated using the mutual_info_classif function from scikit-learn. The numerical variables were converted to caterogorial variables with a histogram with 30 bins. Changing the number of bins also changes the entropys a little bit. But the overall trend remains.')
    ]),
    dbc.Row([
        dcc.Graph(figure=px.bar(
            x=columns,
            y=mi,
        ), style={'height': '800px'})

    
    ])
])

# Callback to update the pie chart
@app.callback(
    Output('pie_chart', 'figure'),
    Input('pie_chart_target_variable', 'value')
)
def update_pie_chart(target_variable):
    fig = px.pie(df_train, names=target_variable, title=f'Distribution of {target_variable}', color=target_variable)
    return fig

# Callback to update the histogram
@app.callback(
    Output('histogram', 'figure'),
    Input('histogram_column', 'value'),
    Input('histogram_color', 'value')
)
def update_histogram(y, color):
    fig = px.histogram(df_train, x=y, color=color, marginal='violin')
    return fig

# Callback to update the scatterplot
@app.callback(
    Output('2d_plot', 'figure'),
    Input('2d_plot_x', 'value'),
    Input('2d_plot_y', 'value')
)
def update_2d_plot(x, y):
    fig = px.scatter(df_train, x=x, y=y, color=target_variable, marginal_y='violin', marginal_x='violin')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
