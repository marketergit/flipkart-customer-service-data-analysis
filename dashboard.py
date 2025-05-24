"""
Interactive dashboard for the Flipkart Customer Support Analysis project.
This script creates a Dash web application to visualize the results of the analysis.
"""
import os
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create the Dash app
app = dash.Dash(__name__, 
                title='Flipkart Customer Support Analysis',
                meta_tags=[{'name': 'viewport', 
                           'content': 'width=device-width, initial-scale=1.0'}])
server = app.server

# Load data and models if available
def load_project_data():
    """
    Load data and models for the dashboard
    
    Returns:
    --------
    data : dict
        Dictionary containing loaded data and models
    """
    data = {}
    
    # Load processed data if available
    if os.path.exists('results/processed_data.csv'):
        data['df'] = pd.read_csv('results/processed_data.csv')
        print(f"Loaded processed data with shape: {data['df'].shape}")
    else:
        # Try to load original data
        if os.path.exists('Customer_support_data.csv'):
            data['df'] = pd.read_csv('Customer_support_data.csv')
            print(f"Loaded original data with shape: {data['df'].shape}")
        else:
            print("No data found")
            data['df'] = None
    
    # Load model comparison results if available
    if os.path.exists('results/model_comparison.csv'):
        data['model_comparison'] = pd.read_csv('results/model_comparison.csv')
        print("Loaded model comparison results")
    else:
        data['model_comparison'] = None
    
    # Load project metadata if available
    if os.path.exists('results/project_metadata.json'):
        with open('results/project_metadata.json', 'r') as f:
            data['metadata'] = json.load(f)
        print("Loaded project metadata")
    else:
        data['metadata'] = None
    
    # Load models if available
    models_dir = 'models'
    if os.path.exists(models_dir):
        data['models'] = {}
        for model_file in os.listdir(models_dir):
            if model_file.endswith('.joblib'):
                model_name = os.path.splitext(model_file)[0]
                try:
                    data['models'][model_name] = joblib.load(os.path.join(models_dir, model_file))
                    print(f"Loaded {model_name} model")
                except Exception as e:
                    print(f"Error loading {model_name} model: {e}")
    else:
        data['models'] = None
    
    return data

# Load data
project_data = load_project_data()
df = project_data['df']

# Define color scheme
colors = {
    'background': '#f9f9f9',
    'text': '#333333',
    'primary': '#3366cc',
    'secondary': '#ff9900',
    'success': '#00cc66',
    'warning': '#ffcc00',
    'danger': '#ff3333',
    'light': '#f0f0f0',
    'dark': '#343a40'
}

# Create dashboard layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'}, children=[
    # Header
    html.Div([
        html.Img(src='https://static-assets-web.flixcart.com/fk-p-linchpin-web/fk-cp-zion/img/flipkart-plus_8d85f4.png',
                style={'height': '40px', 'marginRight': '15px'}),
        html.H1('Customer Support Analysis Dashboard',
               style={'color': colors['primary'], 'display': 'inline-block', 'verticalAlign': 'middle'})
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # Project metadata
    html.Div([
        html.H2('Project Overview', style={'color': colors['text']}),
        html.Div([
            html.Div([
                html.H4('Project Details', style={'color': colors['secondary']}),
                html.P(f"Analysis timestamp: {project_data['metadata']['execution_timestamp'] if project_data['metadata'] else 'N/A'}"),
                html.P(f"Data shape: {project_data['metadata']['data_shape'] if project_data['metadata'] else 'N/A'}"),
                html.P(f"Features used: {project_data['metadata']['total_features'] if project_data['metadata'] else 'N/A'}")
            ], className='card', style={'padding': '15px', 'backgroundColor': colors['light'], 'borderRadius': '10px', 'flex': '1'}),
            
            html.Div([
                html.H4('Model Performance', style={'color': colors['success']}),
                html.P(f"Best model: {project_data['metadata']['best_model'] if project_data['metadata'] else 'N/A'}"),
                html.P(f"Accuracy: {project_data['metadata']['best_model_accuracy'] if project_data['metadata'] else 'N/A':.4f}")
            ], className='card', style={'padding': '15px', 'backgroundColor': colors['light'], 'borderRadius': '10px', 'flex': '1', 'marginLeft': '20px'})
        ], style={'display': 'flex', 'marginBottom': '20px'})
    ]),
    
    # Tabs for different visualizations
    dcc.Tabs([
        # Tab 1: Ticket Overview
        dcc.Tab(label='Ticket Overview', children=[
            html.Div([
                html.H3('Ticket Distribution', style={'color': colors['text'], 'marginTop': '20px'}),
                
                # Row 1: Resolution Status and Categories
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='resolution-status-pie',
                            figure=px.pie(
                                df['is_delayed'].map({1: 'Delayed', 0: 'On-time'}).value_counts().reset_index(),
                                values='count',
                                names='is_delayed',
                                title='Ticket Resolution Status',
                                color_discrete_sequence=px.colors.qualitative.Set3
                            ) if df is not None and 'is_delayed' in df.columns else {}
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(
                            id='categories-bar',
                            figure=px.bar(
                                df['category'].value_counts().nlargest(10).reset_index(),
                                x='category',
                                y='count',
                                title='Top 10 Ticket Categories',
                                color='count',
                                color_continuous_scale='Viridis'
                            ) if df is not None and 'category' in df.columns else {}
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ]),
                
                # Row 2: Ticket Volume by Time
                html.Div([
                    dcc.Graph(
                        id='ticket-volume-time',
                        figure=px.line(
                            df.groupby(pd.to_datetime(df['Issue_reported at']).dt.date).size().reset_index(),
                            x='Issue_reported at',
                            y=0,
                            title='Daily Ticket Volume',
                            labels={'Issue_reported at': 'Date', '0': 'Number of Tickets'}
                        ) if df is not None and 'Issue_reported at' in df.columns else {}
                    )
                ], style={'marginTop': '20px'})
            ])
        ]),
        
        # Tab 2: Delay Analysis
        dcc.Tab(label='Delay Analysis', children=[
            html.Div([
                html.H3('Delay Patterns', style={'color': colors['text'], 'marginTop': '20px'}),
                
                # Row 1: Delay by Category and Time
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='delay-by-category',
                            figure=px.bar(
                                df.groupby('category')['is_delayed'].mean().sort_values(ascending=False).nlargest(10).reset_index(),
                                x='category',
                                y='is_delayed',
                                title='Top 10 Categories by Delay Rate',
                                labels={'is_delayed': 'Delay Rate', 'category': 'Category'},
                                color='is_delayed',
                                color_continuous_scale='RdYlGn_r'
                            ) if df is not None and 'category' in df.columns and 'is_delayed' in df.columns else {}
                        )
                    ], style={'width': '100%', 'marginBottom': '20px'}),
                    
                    html.Div([
                        dcc.Graph(
                            id='delay-by-time',
                            figure=px.line(
                                df.groupby(pd.to_datetime(df['Issue_reported at']).dt.hour)['is_delayed'].mean().reset_index(),
                                x='Issue_reported at',
                                y='is_delayed',
                                title='Delay Rate by Hour of Day',
                                labels={'Issue_reported at': 'Hour of Day', 'is_delayed': 'Delay Rate'}
                            ) if df is not None and 'Issue_reported at' in df.columns and 'is_delayed' in df.columns else {}
                        )
                    ], style={'width': '100%'})
                ])
            ])
        ]),
        
        # Tab 3: Agent Performance
        dcc.Tab(label='Agent Performance', children=[
            html.Div([
                html.H3('Agent Analysis', style={'color': colors['text'], 'marginTop': '20px'}),
                
                # Row 1: Agent Workload and Performance
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='agent-workload',
                            figure=px.bar(
                                df['Agent Shift'].value_counts().nlargest(10).reset_index(),
                                x='Agent Shift',
                                y='count',
                                title='Top 10 Agents by Workload',
                                color='count',
                                color_continuous_scale='Blues'
                            ) if df is not None and 'Agent Shift' in df.columns else {}
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(
                            id='agent-performance',
                            figure=px.scatter(
                                df.groupby('Agent Shift').agg({'is_delayed': 'mean', 'Unique id': 'count'}).reset_index(),
                                x='Unique id',
                                y='is_delayed',
                                hover_name='Agent Shift',
                                title='Agent Performance: Workload vs. Delay Rate',
                                labels={'Unique id': 'Number of Tickets (Workload)', 'is_delayed': 'Delay Rate', 'Agent Shift': 'Agent'},
                                color='is_delayed',
                                color_continuous_scale='RdYlGn_r'
                            ) if df is not None and 'Agent Shift' in df.columns and 'is_delayed' in df.columns else {}
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ])
            ])
        ]),
        
        # Tab 4: Model Results
        dcc.Tab(label='Model Results', children=[
            html.Div([
                html.H3('Model Performance Comparison', style={'color': colors['text'], 'marginTop': '20px'}),
                
                # Model comparison table
                html.Div([
                    dash_table.DataTable(
                        id='model-comparison-table',
                        columns=[{'name': col, 'id': col} for col in project_data['model_comparison'].columns] if project_data['model_comparison'] is not None else [],
                        data=project_data['model_comparison'].to_dict('records') if project_data['model_comparison'] is not None else [],
                        style_table={'overflowX': 'auto'},
                        style_header={
                            'backgroundColor': colors['primary'],
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_cell={
                            'padding': '10px',
                            'textAlign': 'center'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': colors['light']
                            }
                        ]
                    )
                ], style={'marginBottom': '30px'}),
                
                # Feature importance plots
                html.Div([
                    html.H4('Feature Importance', style={'color': colors['secondary']}),
                    dcc.Dropdown(
                        id='model-selection',
                        options=[
                            {'label': 'Logistic Regression', 'value': 'logistic_regression'},
                            {'label': 'Random Forest', 'value': 'random_forest'},
                            {'label': 'XGBoost', 'value': 'xgboost'}
                        ],
                        value='random_forest',
                        style={'width': '50%', 'marginBottom': '10px'}
                    ),
                    html.Img(id='feature-importance-img', style={'width': '100%'})
                ])
            ])
        ]),
        
        # Tab 5: Recommendations
        dcc.Tab(label='Recommendations', children=[
            html.Div([
                html.H3('Key Findings and Recommendations', style={'color': colors['text'], 'marginTop': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H4('Key Findings', style={'color': colors['secondary']}),
                        html.Ul([
                            html.Li("Technical issues have the highest delay rates, suggesting the need for specialized training or more resources in this area."),
                            html.Li("Tickets submitted outside business hours experience longer resolution times."),
                            html.Li("Agent workload is unevenly distributed, with some agents handling significantly more tickets than others."),
                            html.Li("Complex issues with multiple product categories have higher delay rates than single-category issues.")
                        ])
                    ], style={'marginBottom': '30px'}),
                    
                    html.Div([
                        html.H4('Recommendations', style={'color': colors['success']}),
                        html.Ul([
                            html.Li("Implement specialized training for technical support agents to reduce delays in that category."),
                            html.Li("Consider adjusting staffing levels during peak hours to better handle ticket volume."),
                            html.Li("Redistribute workload more evenly across agents to prevent burnout and maintain quality."),
                            html.Li("Develop a fast-track system for simple, common issues to improve overall resolution times."),
                            html.Li("Create knowledge base articles for frequently reported issues to empower customer self-service.")
                        ])
                    ], style={'marginBottom': '30px'}),
                    
                    html.Div([
                        html.H4('Implementation Plan', style={'color': colors['primary']}),
                        html.Ol([
                            html.Li("Short-term (1-3 months): Redistribute agent workload and develop knowledge base articles."),
                            html.Li("Medium-term (3-6 months): Implement specialized training programs and adjust staffing schedules."),
                            html.Li("Long-term (6-12 months): Develop and implement the fast-track system for common issues.")
                        ])
                    ])
                ])
            ])
        ])
    ], style={'marginTop': '20px'})
])

# Callback for feature importance image
@app.callback(
    Output('feature-importance-img', 'src'),
    Input('model-selection', 'value')
)
def update_feature_importance(model_name):
    """
    Update feature importance image based on selected model
    
    Parameters:
    -----------
    model_name : str
        Name of the selected model
    
    Returns:
    --------
    src : str
        Path to the feature importance image
    """
    image_path = f'plots/{model_name}_feature_importance.png'
    
    if os.path.exists(image_path):
        return app.get_asset_url(image_path)
    else:
        return ''

# Run the app
if __name__ == '__main__':
    print("Starting Flipkart Customer Support Analysis Dashboard...")
    app.run(debug=True, port=8050)
