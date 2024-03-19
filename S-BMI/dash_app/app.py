import csv
import pandas as pd 
import numpy as np
import seaborn as sns
import re
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output


#IMPORT CHILD REFERENCES 
boy_below_5 =  pd.read_csv("https://raw.githubusercontent.com/molne1/WHO_z-score_children_0-19/main/WHO_reference_data/WHO_boy_reference_below_5_years.csv",index_col=0) 
girl_below_5 = pd.read_csv("https://raw.githubusercontent.com/molne1/WHO_z-score_children_0-19/main/WHO_reference_data/WHO_girl_reference_below_5_years.csv",index_col=0)
boy_above_5 = pd.read_csv("https://raw.githubusercontent.com/molne1/WHO_z-score_children_0-19/main/WHO_reference_data/WHO_boy_reference_above_5_years.csv",index_col=0)
girl_above_5 = pd.read_csv("https://raw.githubusercontent.com/molne1/WHO_z-score_children_0-19/main/WHO_reference_data/WHO_girl_reference_above_5_years.csv",index_col=0)

# remove overlap
boy_below_5 = boy_below_5.loc[:MONTHS_AT_AGE_5]
girl_below_5 = girl_below_5.loc[:MONTHS_AT_AGE_5]
#limit age to 18 
boy_above_5 = boy_above_5.loc[:155]
girl_above_5 = girl_above_5.loc[:155]

#DEFINE FUNCTIONS
def create_flexible_SD(df, SD_fixed):
    distance_for_SBMI = []
    SD_lower= SD_fixed[0]
    SD_upper= SD_fixed[1]
    sd_columns =  [col for col in df if col.startswith('SD')]
    x_range = range(0,len(sd_columns))
    x_data = [*x_range]
    for index, row in df.iterrows():
        y_data = df[sd_columns].values.tolist()
        y_f = interp1d(x_data, y_data, 'linear')
        interpolated_values_lower = (y_f(SD_lower))
        interpolated_values_upper = (y_f(SD_upper))
        distance_for_SBMI = interpolated_values_upper - interpolated_values_lower
    df['distance_SBMI']= distance_for_SBMI
    months_column = df.pop('Months')
    df.insert(2, 'Months', months_column)
    return df
def expand_SD_table(df, start_adding_after, end):

    difference = df['distance_SBMI']
    start = start_adding_after+1
    stop = end
    start_adding_column = 'SD'+ str(start_adding_after)
    
    target_column_index = df.columns.get_loc(start_adding_column)
    new_df = df.iloc[:, :target_column_index+1].copy()
    for i in range(start,stop+1):
        column_name = f'SD{i}'
        new_column_value = (df[start_adding_column]+ df['distance_SBMI']*(i-(start-1)))
        new_df.loc[:, column_name] = new_column_value
    

    new_df['distance_SBMI'] = df['distance_SBMI']

    return new_df
def find_sbmi_cut_offs(expanded_df, column_month, desired_sbmi):
    df_18 = expanded_df[expanded_df[column_month]== MONTHS_AT_AGE_18]
    
    sd_columns =  [col for col in expanded_df if col.startswith('SD')]
    list_sd = []
    for name in sd_columns:
        cleaned_name = re.sub("\D", "", name)
        if "neg" in name:
            cleaned_name = "-" + cleaned_name
        list_sd.append(cleaned_name)

    dict_sd_values_for_sbmi = {}

    x_data = df_18[sd_columns].values.tolist()
    x_data = x_data[0]
    y_data = [list_sd]
    y_f = interp1d(x_data, y_data, 'linear')
 
    for bmi in desired_sbmi:
        sd_value = y_f(bmi)
        dict_sd_values_for_sbmi[bmi] = sd_value.item()
        
    return(dict_sd_values_for_sbmi, list_sd)
def SBMI_caluclator(expanded_df, dict_sbmi_cutoffs, list_sd):
    SBMI_dict = {}
    list_sd_number = []
    list_sd = [float(x) for x in list_sd]
    
    sd_columns =  [col for col in expanded_df if col.startswith('SD')]
#     sd_columns = list_sd
    for name in sd_columns:
            list_sd_number.append(float(re.sub("\D","",name)))

    for index,row in expanded_df.iterrows():
        row_dict = {}
    
        for key, value in dict_sbmi_cutoffs.items():

            x_data = row[sd_columns].values.tolist()
            y_data = list_sd
            y_f = interp1d(y_data, x_data,'linear')
            interp_sbmi_value = (y_f(value))
            row_dict[key] =interp_sbmi_value

        SBMI_dict[index] = row_dict
    df_sbmi = pd.DataFrame.from_dict(SBMI_dict, orient='index')
    df_sbmi['Months'] = range(0,217)
    return df_sbmi
def check_decrease(sbmi_df,above_this_age_in_months, change_greater_or_equal_to):
    subset_decreasing_values = pd.DataFrame()
    subset_decreasing_values['Months'] = sbmi_df['Months']
    for column in sbmi_df.columns:
        column_name = f'decrease {column}'
        sbmi_df[column_name] = sbmi_df[column].pct_change()

        
        condition = (sbmi_df[column_name] < 0) & (sbmi_df['Months'] > above_this_age_in_months)
        subset_decreasing_values[[f'decrease_value_{column}',column_name]] = sbmi_df.loc[condition, [column, column_name]]

    return subset_decreasing_values

# Assume your functions and dataframes are defined elsewhere

# Dash app
app = Dash(__name__)

# Define variables and functions
datasets = {'Girl': girl_df, 'Boy': boy_df}
desired_bmi_values = [30, 40, 50, 60]
sd_fixed_values = list(np.arange(0, 4, 0.1))
start_adding_values = list(np.arange(0, 4, 0.1))

# functions 
def create_figure(df, desired_bmi, sd_fixed, start_adding):
    # functions 
    df_flexible = create_flexible_SD(df, sd_fixed)
    df_expand = expand_SD_table(df_flexible, start_adding, 13)
    dict_sbmi_cutoffs, list_sd_girl = find_sbmi_cut_offs(df_expand, 'Months', desired_bmi)
    SBMI_result = SBMI_caluclator(df_expand, dict_sbmi_cutoffs, list_sd_girl)

    # Create traces for SBMI curves
    traces = []
    for column in SBMI_result.columns[:-1]:
        trace = go.Scatter(
            x=SBMI_result['Months'],
            y=SBMI_result[column],
            mode='lines',
            name=column
        )
        traces.append(trace)

    # Create trace for decrease values
    age_in_months = 61
    change_ge = 0
    girl_cheching_decrease = check_decrease(SBMI_result, age_in_months, change_ge)
    for column in girl_cheching_decrease.columns:
        if column.startswith('decrease_value'):
            trace = go.Scatter(
                x=girl_cheching_decrease['Months'],
                y=girl_cheching_decrease[column],
                mode='lines',
                name=column,
                line=dict(color='red')
            )
            traces.append(trace)

    # Create Plotly layout
    layout = go.Layout(
        xaxis=dict(title='Months'),
        yaxis=dict(title='BMI'),
        title='SBMI Curves',
        legend=dict(orientation='h')
    )

    # Create Plotly figure
    fig = go.Figure(data=traces, layout=layout)
    return fig

# Define layout
app.layout = html.Div([
    html.H1('SBMI Curves'),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': dataset, 'value': dataset} for dataset in datasets.keys()],
        value='Girl'
    ),
    dcc.Checklist(
        id='desired-bmi-checklist',
        options=[{'label': str(bmi), 'value': bmi} for bmi in desired_bmi_values],
        value=[30, 40, 50, 60],
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Slider(
        id='sd-fixed-slider',
        min=0,
        max=3.0,
        step=0.1,
        value=[2, 3],
        marks={0: '0', 3: '3.0'}
    ),
    dcc.Slider(
        id='start-adding-slider',
        min=0,
        max=3.0,
        step=0.1,
        value=2,
        marks={0: '0', 3: '3.0'}
    ),
    dcc.Graph(id='sbmi-plot')
])

# Define callback to update the plot
@app.callback(
    Output('sbmi-plot', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('desired-bmi-checklist', 'value'),
     Input('sd-fixed-slider', 'value'),
     Input('start-adding-slider', 'value')]
)
def update_plot(dataset, desired_bmi, sd_fixed, start_adding):
    df = datasets[dataset]
    fig = create_figure(df, desired_bmi, sd_fixed, start_adding)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
