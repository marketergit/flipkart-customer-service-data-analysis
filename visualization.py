"""
Visualization module for the Flipkart Customer Support Analysis project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from utils import print_section_header

# Set style for static plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def ensure_plot_dir():
    """Create plots directory if it doesn't exist"""
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("Created 'plots' directory")

def plot_data_distribution(df, categorical_cols=None, numerical_cols=None):
    """
    Plot distributions of categorical and numerical features
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    categorical_cols : list
        List of categorical columns to plot
    numerical_cols : list
        List of numerical columns to plot
    """
    print_section_header("Data Distribution Visualization")
    ensure_plot_dir()
    
    # If columns not specified, use default
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:5]
    
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['number']).columns[:5]
    
    # Plot categorical variables
    for i, col in enumerate(categorical_cols):
        if col in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Get top 10 categories to avoid overcrowding
            value_counts = df[col].value_counts().nlargest(10)
            
            # Create bar plot
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Distribution of {col} (Top 10)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'plots/distribution_{col}.png')
            plt.close()
            print(f"Saved distribution plot for '{col}'")
    
    # Plot numerical variables
    for i, col in enumerate(numerical_cols):
        if col in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Create histogram with KDE
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(f'plots/distribution_{col}.png')
            plt.close()
            print(f"Saved distribution plot for '{col}'")

def plot_ticket_resolution_by_category(df, category_col='category', target_col='is_delayed'):
    """
    Plot ticket resolution by category
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    category_col : str
        The column containing categories
    target_col : str
        The column containing resolution status (1 for delayed, 0 for on-time)
    """
    ensure_plot_dir()
    
    if category_col in df.columns and target_col in df.columns:
        # Get counts by category and resolution status
        resolution_by_category = pd.crosstab(df[category_col], df[target_col])
        
        # Calculate percentage of delayed tickets by category
        resolution_by_category['delayed_pct'] = resolution_by_category[1] / (resolution_by_category[0] + resolution_by_category[1]) * 100
        
        # Sort by percentage of delayed tickets
        resolution_by_category = resolution_by_category.sort_values('delayed_pct', ascending=False)
        
        # Take top 10 categories
        top_categories = resolution_by_category.head(10)
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar plot
        bar_width = 0.35
        index = np.arange(len(top_categories))
        
        plt.bar(index, top_categories[0], bar_width, label='On-time')
        plt.bar(index + bar_width, top_categories[1], bar_width, label='Delayed')
        
        plt.xlabel('Category')
        plt.ylabel('Number of Tickets')
        plt.title('Ticket Resolution by Category (Top 10)')
        plt.xticks(index + bar_width / 2, top_categories.index, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/resolution_by_{category_col}.png')
        plt.close()
        
        # Plot percentage of delayed tickets
        plt.figure(figsize=(14, 8))
        bars = plt.bar(top_categories.index, top_categories['delayed_pct'])
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', rotation=0)
        
        plt.xlabel('Category')
        plt.ylabel('Percentage of Delayed Tickets')
        plt.title(f'Percentage of Delayed Tickets by {category_col} (Top 10)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(top_categories['delayed_pct']) * 1.1)  # Add some space for labels
        plt.tight_layout()
        plt.savefig(f'plots/delayed_pct_by_{category_col}.png')
        plt.close()
        
        print(f"Saved resolution plots by {category_col}")

def plot_time_trends(df, datetime_col='Issue_reported at', target_col='is_delayed'):
    """
    Plot time trends for ticket volume and resolution
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    datetime_col : str
        The column containing the datetime information
    target_col : str
        The column containing resolution status (1 for delayed, 0 for on-time)
    """
    ensure_plot_dir()
    
    if datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        # Create copy of dataframe
        temp_df = df.copy()
        
        # Extract date components
        temp_df['date'] = temp_df[datetime_col].dt.date
        temp_df['hour'] = temp_df[datetime_col].dt.hour
        temp_df['day_of_week'] = temp_df[datetime_col].dt.dayofweek
        temp_df['month'] = temp_df[datetime_col].dt.month
        
        # Daily ticket volume
        daily_volume = temp_df.groupby('date').size().reset_index(name='ticket_count')
        daily_volume['date'] = pd.to_datetime(daily_volume['date'])
        daily_volume = daily_volume.sort_values('date')
        
        plt.figure(figsize=(14, 6))
        plt.plot(daily_volume['date'], daily_volume['ticket_count'], marker='o', linestyle='-')
        plt.title('Daily Ticket Volume')
        plt.xlabel('Date')
        plt.ylabel('Number of Tickets')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/daily_ticket_volume.png')
        plt.close()
        
        # Hourly distribution
        hourly_volume = temp_df.groupby('hour').size().reset_index(name='ticket_count')
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(hourly_volume['hour'], hourly_volume['ticket_count'])
        plt.title('Hourly Ticket Distribution')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Tickets')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/hourly_ticket_distribution.png')
        plt.close()
        
        # Day of week distribution
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_volume = temp_df.groupby('day_of_week').size().reset_index(name='ticket_count')
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(day_volume['day_of_week'], day_volume['ticket_count'])
        plt.title('Ticket Distribution by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Tickets')
        plt.xticks(range(0, 7), day_names)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/day_of_week_distribution.png')
        plt.close()
        
        # Monthly distribution
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_volume = temp_df.groupby('month').size().reset_index(name='ticket_count')
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(month_volume['month'], month_volume['ticket_count'])
        plt.title('Ticket Distribution by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Tickets')
        plt.xticks(range(1, 13), month_names)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/monthly_distribution.png')
        plt.close()
        
        # If target column exists, plot resolution status by time
        if target_col in df.columns:
            # Daily resolution rate
            daily_resolution = temp_df.groupby('date')[target_col].mean().reset_index(name='delayed_rate')
            daily_resolution['date'] = pd.to_datetime(daily_resolution['date'])
            daily_resolution = daily_resolution.sort_values('date')
            
            plt.figure(figsize=(14, 6))
            plt.plot(daily_resolution['date'], daily_resolution['delayed_rate'], marker='o', linestyle='-')
            plt.title('Daily Ticket Delay Rate')
            plt.xlabel('Date')
            plt.ylabel('Percentage of Delayed Tickets')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig('plots/daily_delay_rate.png')
            plt.close()
            
            # Hourly resolution rate
            hourly_resolution = temp_df.groupby('hour')[target_col].mean().reset_index(name='delayed_rate')
            
            plt.figure(figsize=(14, 6))
            bars = plt.bar(hourly_resolution['hour'], hourly_resolution['delayed_rate'])
            plt.title('Hourly Ticket Delay Rate')
            plt.xlabel('Hour of Day')
            plt.ylabel('Percentage of Delayed Tickets')
            plt.xticks(range(0, 24))
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig('plots/hourly_delay_rate.png')
            plt.close()
            
            # Day of week resolution rate
            day_resolution = temp_df.groupby('day_of_week')[target_col].mean().reset_index(name='delayed_rate')
            
            plt.figure(figsize=(14, 6))
            bars = plt.bar(day_resolution['day_of_week'], day_resolution['delayed_rate'])
            plt.title('Ticket Delay Rate by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Percentage of Delayed Tickets')
            plt.xticks(range(0, 7), day_names)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig('plots/day_of_week_delay_rate.png')
            plt.close()
        
        print("Saved time trend plots")

def plot_agent_performance(df, agent_col='Agent Shift', target_col='is_delayed'):
    """
    Plot agent performance metrics
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    agent_col : str
        The column containing agent information
    target_col : str
        The column containing resolution status (1 for delayed, 0 for on-time)
    """
    ensure_plot_dir()
    
    if agent_col in df.columns:
        # Agent workload
        agent_workload = df[agent_col].value_counts().reset_index()
        agent_workload.columns = [agent_col, 'ticket_count']
        
        # Sort by ticket count
        agent_workload = agent_workload.sort_values('ticket_count', ascending=False)
        
        # Take top 15 agents by workload
        top_agents = agent_workload.head(15)
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(top_agents[agent_col], top_agents['ticket_count'])
        plt.title(f'Ticket Volume by {agent_col} (Top 15)')
        plt.xlabel(agent_col)
        plt.ylabel('Number of Tickets')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'plots/workload_by_{agent_col}.png')
        plt.close()
        
        # If target column exists, plot performance metrics
        if target_col in df.columns:
            # Get delayed rate by agent
            agent_performance = df.groupby(agent_col)[target_col].agg(['mean', 'count']).reset_index()
            agent_performance.columns = [agent_col, 'delayed_rate', 'ticket_count']
            
            # Filter agents with at least 10 tickets
            agent_performance = agent_performance[agent_performance['ticket_count'] >= 10]
            
            # Sort by delayed rate
            agent_performance = agent_performance.sort_values('delayed_rate')
            
            # Take top and bottom 10 agents by performance
            best_agents = agent_performance.head(10)
            worst_agents = agent_performance.tail(10)
            
            # Plot best performing agents
            plt.figure(figsize=(14, 8))
            bars = plt.bar(best_agents[agent_col], best_agents['delayed_rate'])
            plt.title(f'Best Performing {agent_col}s (Lowest Delay Rate)')
            plt.xlabel(agent_col)
            plt.ylabel('Delay Rate')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f'plots/best_{agent_col}_performance.png')
            plt.close()
            
            # Plot worst performing agents
            plt.figure(figsize=(14, 8))
            bars = plt.bar(worst_agents[agent_col], worst_agents['delayed_rate'])
            plt.title(f'Worst Performing {agent_col}s (Highest Delay Rate)')
            plt.xlabel(agent_col)
            plt.ylabel('Delay Rate')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f'plots/worst_{agent_col}_performance.png')
            plt.close()
            
            # Scatter plot of workload vs performance
            plt.figure(figsize=(12, 8))
            plt.scatter(agent_performance['ticket_count'], agent_performance['delayed_rate'], alpha=0.7)
            plt.title(f'Workload vs. Performance by {agent_col}')
            plt.xlabel('Number of Tickets (Workload)')
            plt.ylabel('Delay Rate')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'plots/workload_vs_performance_{agent_col}.png')
            plt.close()
        
        print(f"Saved agent performance plots for {agent_col}")

def plot_correlation_matrix(df, features=None):
    """
    Plot correlation matrix of numerical features
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    features : list, optional
        List of features to include in the correlation matrix
    """
    ensure_plot_dir()
    
    # If features not specified, use all numerical columns
    if features is None:
        features = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter to only include specified features
    corr_df = df[features].copy()
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    print("Saved correlation matrix plot")

def create_interactive_dashboard(df, target_col='is_delayed'):
    """
    Create interactive visualizations using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    target_col : str
        The column containing resolution status
        
    Returns:
    --------
    figures : dict
        Dictionary containing plotly figures
    """
    ensure_plot_dir()
    figures = {}
    
    # Filter out unwanted columns for the dashboard
    exclude_cols = ['Unique id', 'Order_id', 'Customer Remarks']
    dashboard_df = df[[col for col in df.columns if col not in exclude_cols]].copy()
    
    # 1. Ticket Volume Over Time
    if 'Issue_reported at' in dashboard_df.columns and pd.api.types.is_datetime64_any_dtype(dashboard_df['Issue_reported at']):
        # Create date column
        dashboard_df['date'] = dashboard_df['Issue_reported at'].dt.date
        
        # Daily ticket volume
        daily_volume = dashboard_df.groupby('date').size().reset_index(name='ticket_count')
        daily_volume['date'] = pd.to_datetime(daily_volume['date'])
        
        # Create figure
        fig = px.line(daily_volume, x='date', y='ticket_count',
                     title='Daily Ticket Volume',
                     labels={'date': 'Date', 'ticket_count': 'Number of Tickets'})
        
        figures['ticket_volume'] = fig
    
    # 2. Delayed vs On-time Tickets
    if target_col in dashboard_df.columns:
        # Create labels
        dashboard_df['status'] = dashboard_df[target_col].map({1: 'Delayed', 0: 'On-time'})
        
        # Count by status
        status_counts = dashboard_df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        # Create figure
        fig = px.pie(status_counts, values='Count', names='Status',
                    title='Ticket Resolution Status',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        
        figures['resolution_status'] = fig
    
    # 3. Ticket Categories
    if 'category' in dashboard_df.columns:
        # Count by category
        category_counts = dashboard_df['category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        # Take top 10 categories
        top_categories = category_counts.head(10)
        
        # Create figure
        fig = px.bar(top_categories, x='Category', y='Count',
                    title='Top 10 Ticket Categories',
                    color='Count',
                    color_continuous_scale='Viridis')
        
        figures['categories'] = fig
        
        # If target column exists, create stacked bar by category
        if target_col in dashboard_df.columns:
            # Count by category and status
            category_status = pd.crosstab(dashboard_df['category'], dashboard_df['status'])
            category_status = category_status.reset_index()
            
            # Take top 10 categories by total
            top_categories = dashboard_df['category'].value_counts().nlargest(10).index
            category_status = category_status[category_status['category'].isin(top_categories)]
            
            # Create figure
            fig = px.bar(category_status, x='category', y=['Delayed', 'On-time'],
                        title='Resolution Status by Category',
                        labels={'category': 'Category', 'value': 'Number of Tickets', 'variable': 'Status'},
                        barmode='stack')
            
            figures['category_status'] = fig
    
    # 4. Agent Performance
    if 'Agent Shift' in dashboard_df.columns and target_col in dashboard_df.columns:
        # Calculate performance metrics by agent
        agent_perf = dashboard_df.groupby('Agent Shift').agg({
            target_col: ['mean', 'count']
        }).reset_index()
        
        agent_perf.columns = ['Agent', 'Delay_Rate', 'Ticket_Count']
        
        # Filter to agents with at least 10 tickets
        agent_perf = agent_perf[agent_perf['Ticket_Count'] >= 10]
        
        # Sort by delay rate
        agent_perf = agent_perf.sort_values('Delay_Rate')
        
        # Create figure
        fig = px.scatter(agent_perf, x='Ticket_Count', y='Delay_Rate', color='Delay_Rate',
                        hover_name='Agent', color_continuous_scale='RdYlGn_r',
                        title='Agent Performance: Workload vs. Delay Rate',
                        labels={'Ticket_Count': 'Number of Tickets (Workload)', 
                                'Delay_Rate': 'Percentage of Delayed Tickets'})
        
        figures['agent_performance'] = fig
    
    # 5. Resolution Time Distribution
    if 'resolution_time_hours' in dashboard_df.columns:
        # Create figure
        fig = px.histogram(dashboard_df, x='resolution_time_hours',
                          title='Distribution of Resolution Time',
                          labels={'resolution_time_hours': 'Resolution Time (hours)'},
                          color_discrete_sequence=['#3366CC'])
        
        figures['resolution_time'] = fig
    
    # Save figures as HTML files
    for name, fig in figures.items():
        fig.write_html(f'plots/{name}.html')
    
    print("Created interactive dashboard visualizations")
    
    return figures

def visualize_data(df, feature_info=None, target_col='is_delayed'):
    """
    Main function to create visualizations
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    feature_info : dict, optional
        Dictionary with information about the features
    target_col : str
        The column containing resolution status
    
    Returns:
    --------
    dashboard_figures : dict
        Dictionary containing plotly figures for the dashboard
    """
    print_section_header("Data Visualization")
    
    # Create output directory
    ensure_plot_dir()
    
    # If feature_info is provided, use it to get column types
    if feature_info:
        categorical_cols = feature_info.get('categorical_columns', None)
        numerical_cols = feature_info.get('numerical_columns', None)
    else:
        categorical_cols = None
        numerical_cols = None
    
    # Plot data distributions
    plot_data_distribution(df, categorical_cols, numerical_cols)
    
    # Plot ticket resolution by category
    if 'category' in df.columns:
        plot_ticket_resolution_by_category(df, 'category', target_col)
    
    # Plot time trends
    if 'Issue_reported at' in df.columns:
        plot_time_trends(df, 'Issue_reported at', target_col)
    
    # Plot agent performance
    if 'Agent Shift' in df.columns:
        plot_agent_performance(df, 'Agent Shift', target_col)
    
    # Plot correlation matrix
    if feature_info and 'numerical_columns' in feature_info:
        important_features = feature_info['numerical_columns'] + [f"{col}_encoded" for col in feature_info['categorical_columns']]
        plot_correlation_matrix(df, important_features)
    else:
        plot_correlation_matrix(df)
    
    # Create interactive dashboard
    dashboard_figures = create_interactive_dashboard(df, target_col)
    
    return dashboard_figures
