"""
Demand Forecasting EDA
Kaggle Competition: demand-forecasting-kernels-only

This script performs Exploratory Data Analysis on the demand forecasting dataset.
Run this after setting up your environment with the required packages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Optional imports
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some statistical tests will be skipped.")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some advanced analysis will be skipped.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directory for graphs
OUTPUT_DIR = 'output_graphs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

def save_plot(filename, dpi=300, bbox_inches='tight'):
    """Helper function to save plots"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved plot: {filepath}")
    plt.show()

def load_data():
    """Load the dataset files"""
    print("Loading datasets...")
    
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        sample_submission = pd.read_csv('data/sample_submission.csv')
        
        print(f"Train dataset shape: {train_df.shape}")
        print(f"Test dataset shape: {test_df.shape}")
        print(f"Sample submission shape: {sample_submission.shape}")
        
        return train_df, test_df, sample_submission
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure to download the dataset first using:")
        print("kaggle competitions download -c demand-forecasting-kernels-only")
        return None, None, None

def explore_basic_info(df, name):
    """Basic information about the dataset"""
    print(f"\n{'='*50}")
    print(f"BASIC INFO: {name.upper()}")
    print(f"{'='*50}")
    
    print(f"Shape: {df.shape}")
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nUnique values per column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")

def plot_sales_distribution(train_df):
    """Enhanced sales distribution analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original distribution
    axes[0, 0].hist(train_df['sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Sales Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sales')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log transformation
    log_sales = np.log1p(train_df['sales'])
    axes[0, 1].hist(log_sales, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Log(Sales + 1) Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Log(Sales + 1)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot with outliers highlighted
    box_plot = axes[0, 2].boxplot(train_df['sales'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightgreen')
    axes[0, 2].set_title('Sales Boxplot', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Sales')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Q-Q plot for normality check
    stats.probplot(train_df['sales'], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Sales vs Normal)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot for log-transformed data
    stats.probplot(log_sales, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Log Sales vs Normal)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistical summary text
    sales_stats = train_df['sales'].describe()
    skewness = stats.skew(train_df['sales'])
    kurtosis = stats.kurtosis(train_df['sales'])
    
    stats_text = f"""Statistical Summary:
Mean: {sales_stats['mean']:.2f}
Median: {sales_stats['50%']:.2f}
Std: {sales_stats['std']:.2f}
Skewness: {skewness:.3f}
Kurtosis: {kurtosis:.3f}
Min: {sales_stats['min']:.2f}
Max: {sales_stats['max']:.2f}"""
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].set_title('Statistical Summary', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    save_plot('01_sales_distribution_analysis.png')
    
    # Perform Shapiro-Wilk test for normality (on sample due to size limits)
    sample_size = min(5000, len(train_df))
    sample_sales = train_df['sales'].sample(sample_size, random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample_sales)
    
    print(f"\nNormality Tests (sample of {sample_size}):")
    print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.2e}")
    print(f"Interpretation: {'Normal' if shapiro_p > 0.05 else 'Not normal'} distribution")

def analyze_temporal_patterns(train_df):
    """Enhanced temporal pattern analysis"""
    if 'date' in train_df.columns:
        # Create a copy to avoid modifying original
        df_temp = train_df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['year'] = df_temp['date'].dt.year
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['day'] = df_temp['date'].dt.day
        df_temp['weekday'] = df_temp['date'].dt.day_name()
        df_temp['quarter'] = df_temp['date'].dt.quarter
        df_temp['week'] = df_temp['date'].dt.isocalendar().week
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        # 1. Time series plot with trend
        daily_sales = df_temp.groupby('date')['sales'].sum().reset_index()
        axes[0, 0].plot(daily_sales['date'], daily_sales['sales'], alpha=0.7, linewidth=1)
        
        # Add rolling average
        rolling_avg = daily_sales['sales'].rolling(window=30).mean()
        axes[0, 0].plot(daily_sales['date'], rolling_avg, color='red', linewidth=2, label='30-day MA')
        
        axes[0, 0].set_title('Daily Sales with Trend', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Total Daily Sales')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Monthly patterns with confidence intervals
        monthly_stats = df_temp.groupby('month')['sales'].agg(['mean', 'std', 'count']).reset_index()
        monthly_stats['ci'] = 1.96 * monthly_stats['std'] / np.sqrt(monthly_stats['count'])
        
        axes[0, 1].bar(monthly_stats['month'], monthly_stats['mean'], 
                      yerr=monthly_stats['ci'], capsize=5, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Average Sales by Month (with 95% CI)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Sales')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Weekday patterns
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_stats = df_temp.groupby('weekday')['sales'].agg(['mean', 'std', 'count'])
        weekday_stats = weekday_stats.reindex(weekday_order)
        weekday_stats['ci'] = 1.96 * weekday_stats['std'] / np.sqrt(weekday_stats['count'])
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
        bars = axes[1, 0].bar(range(len(weekday_stats)), weekday_stats['mean'], 
                             yerr=weekday_stats['ci'], capsize=5, color=colors, alpha=0.8)
        axes[1, 0].set_title('Average Sales by Weekday (with 95% CI)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Weekday')
        axes[1, 0].set_ylabel('Average Sales')
        axes[1, 0].set_xticks(range(len(weekday_stats)))
        axes[1, 0].set_xticklabels(weekday_stats.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Yearly trends
        yearly_stats = df_temp.groupby('year')['sales'].agg(['mean', 'sum', 'count'])
        
        ax_twin = axes[1, 1].twinx()
        bars1 = axes[1, 1].bar(yearly_stats.index - 0.2, yearly_stats['sum']/1000, 
                              width=0.4, label='Total Sales (K)', color='steelblue', alpha=0.7)
        bars2 = ax_twin.bar(yearly_stats.index + 0.2, yearly_stats['mean'], 
                           width=0.4, label='Average Sales', color='orange', alpha=0.7)
        
        axes[1, 1].set_title('Yearly Sales Trends', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Total Sales (Thousands)', color='steelblue')
        ax_twin.set_ylabel('Average Sales', color='orange')
        axes[1, 1].legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Heatmap: Month vs Year
        pivot_month_year = df_temp.pivot_table(values='sales', index='month', columns='year', aggfunc='mean')
        im = axes[2, 0].imshow(pivot_month_year.values, cmap='YlOrRd', aspect='auto')
        axes[2, 0].set_title('Sales Heatmap: Month vs Year', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Year')
        axes[2, 0].set_ylabel('Month')
        axes[2, 0].set_xticks(range(len(pivot_month_year.columns)))
        axes[2, 0].set_xticklabels(pivot_month_year.columns)
        axes[2, 0].set_yticks(range(len(pivot_month_year.index)))
        axes[2, 0].set_yticklabels(pivot_month_year.index)
        plt.colorbar(im, ax=axes[2, 0], label='Average Sales')
        
        # 6. Quarterly analysis
        quarterly_stats = df_temp.groupby(['year', 'quarter'])['sales'].sum().reset_index()
        quarterly_stats['period'] = quarterly_stats['year'].astype(str) + '-Q' + quarterly_stats['quarter'].astype(str)
        
        axes[2, 1].plot(range(len(quarterly_stats)), quarterly_stats['sales'], 
                       marker='o', linewidth=2, markersize=6, color='purple')
        axes[2, 1].set_title('Quarterly Sales Trend', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Quarter')
        axes[2, 1].set_ylabel('Total Sales')
        axes[2, 1].set_xticks(range(0, len(quarterly_stats), 4))
        axes[2, 1].set_xticklabels([quarterly_stats.iloc[i]['period'] for i in range(0, len(quarterly_stats), 4)], rotation=45)
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot('02_temporal_patterns_analysis.png')
        
        # Statistical tests for seasonality
        print("\nSeasonality Analysis:")
        # ANOVA test for monthly differences
        monthly_groups = [group['sales'].values for name, group in df_temp.groupby('month')]
        f_stat, p_val = stats.f_oneway(*monthly_groups)
        print(f"Monthly seasonality ANOVA: F-statistic={f_stat:.2f}, p-value={p_val:.2e}")
        
        # ANOVA test for weekday differences
        weekday_groups = [group['sales'].values for name, group in df_temp.groupby('weekday')]
        f_stat_wd, p_val_wd = stats.f_oneway(*weekday_groups)
        print(f"Weekday seasonality ANOVA: F-statistic={f_stat_wd:.2f}, p-value={p_val_wd:.2e}")

def analyze_store_item_patterns(train_df):
    """Enhanced store and item pattern analysis"""
    if 'store' in train_df.columns and 'item' in train_df.columns:
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        # Store analysis
        store_sales = train_df.groupby('store')['sales'].agg(['mean', 'sum', 'std', 'count', 'median']).reset_index()
        store_sales['cv'] = store_sales['std'] / store_sales['mean']
        store_sales['se'] = store_sales['std'] / np.sqrt(store_sales['count'])
        
        # Total sales by store with error bars
        axes[0, 0].bar(store_sales['store'], store_sales['sum']/1000, 
                      color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Total Sales by Store', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Store ID')
        axes[0, 0].set_ylabel('Total Sales (Thousands)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average sales by store with confidence intervals
        axes[0, 1].bar(store_sales['store'], store_sales['mean'], 
                      yerr=1.96*store_sales['se'], capsize=5,
                      color='lightcoral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Average Sales by Store (with 95% CI)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Store ID')
        axes[0, 1].set_ylabel('Average Sales')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Item analysis - top performers
        item_stats = train_df.groupby('item')['sales'].agg(['mean', 'sum', 'count', 'std']).reset_index()
        item_stats['cv'] = item_stats['std'] / item_stats['mean']
        top_items = item_stats.nlargest(15, 'sum')
        
        colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(top_items)))
        bars = axes[1, 0].bar(range(len(top_items)), top_items['sum']/1000, 
                             color=colors_gradient, alpha=0.8, edgecolor='black')
        axes[1, 0].set_title('Top 15 Items by Total Sales', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Item Rank')
        axes[1, 0].set_ylabel('Total Sales (Thousands)')
        axes[1, 0].set_xticks(range(len(top_items)))
        axes[1, 0].set_xticklabels([f'Item {x}' for x in top_items['item']], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sales variability analysis
        axes[1, 1].scatter(store_sales['mean'], store_sales['cv'], 
                          s=store_sales['sum']/1000, alpha=0.7, c='orange', edgecolors='black')
        axes[1, 1].set_title('Sales Variability vs Average (Store Level)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Average Sales')
        axes[1, 1].set_ylabel('Coefficient of Variation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add store labels
        for i, row in store_sales.iterrows():
            axes[1, 1].annotate(f"S{int(row['store'])}", 
                              (row['mean'], row['cv']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Store performance distribution
        axes[2, 0].boxplot([train_df[train_df['store'] == store]['sales'].values 
                           for store in sorted(train_df['store'].unique())],
                          labels=[f'S{i}' for i in sorted(train_df['store'].unique())])
        axes[2, 0].set_title('Sales Distribution by Store', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Store ID')
        axes[2, 0].set_ylabel('Sales')
        axes[2, 0].tick_params(axis='x', rotation=45)
        axes[2, 0].grid(True, alpha=0.3)
        
        # Item performance variability
        item_cv_top = item_stats.nlargest(15, 'cv')
        axes[2, 1].bar(range(len(item_cv_top)), item_cv_top['cv'], 
                      color='purple', alpha=0.7, edgecolor='black')
        axes[2, 1].set_title('Most Variable Items (Top 15 by CV)', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Item')
        axes[2, 1].set_ylabel('Coefficient of Variation')
        axes[2, 1].set_xticks(range(len(item_cv_top)))
        axes[2, 1].set_xticklabels([f'Item {x}' for x in item_cv_top['item']], rotation=45)
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot('03_store_item_patterns_analysis.png')
        
        # Enhanced heatmap with clustering
        print("\nGenerating Store-Item Sales Heatmap...")
        # Sample items for visualization if too many
        max_items_display = 25
        if len(train_df['item'].unique()) > max_items_display:
            top_items_list = item_stats.nlargest(max_items_display, 'sum')['item'].tolist()
            pivot_data = train_df[train_df['item'].isin(top_items_list)]
        else:
            pivot_data = train_df
            
        pivot_table = pivot_data.pivot_table(values='sales', index='store', columns='item', aggfunc='mean')
        
        plt.figure(figsize=(16, 10))
        mask = pivot_table.isnull()
        sns.heatmap(pivot_table, cmap='RdYlBu_r', cbar_kws={'label': 'Average Sales'},
                   mask=mask, linewidths=0.5, square=False, 
                   cbar=True, annot=False)
        plt.title(f'Average Sales Heatmap: Store vs Item (Top {min(max_items_display, len(pivot_table.columns))} Items)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Item ID', fontsize=12)
        plt.ylabel('Store ID', fontsize=12)
        plt.tight_layout()
        save_plot('04_store_item_heatmap.png')
        
        # Statistical analysis
        print("\nStore-Item Statistical Analysis:")
        # ANOVA for store differences
        store_groups = [group['sales'].values for name, group in train_df.groupby('store')]
        f_stat_store, p_val_store = stats.f_oneway(*store_groups)
        print(f"Store differences ANOVA: F-statistic={f_stat_store:.2f}, p-value={p_val_store:.2e}")
        
        # ANOVA for item differences  
        item_groups = [group['sales'].values for name, group in train_df.groupby('item')]
        f_stat_item, p_val_item = stats.f_oneway(*item_groups)
        print(f"Item differences ANOVA: F-statistic={f_stat_item:.2f}, p-value={p_val_item:.2e}")
        
        # Top and bottom performers
        print(f"\nTop 5 performing stores (by average sales):")
        top_stores = store_sales.nlargest(5, 'mean')[['store', 'mean', 'sum']]
        for _, row in top_stores.iterrows():
            print(f"  Store {int(row['store'])}: Avg={row['mean']:.2f}, Total={row['sum']:.0f}")
            
        print(f"\nTop 5 performing items (by total sales):")
        top_items_display = item_stats.nlargest(5, 'sum')[['item', 'mean', 'sum']]
        for _, row in top_items_display.iterrows():
            print(f"  Item {int(row['item'])}: Avg={row['mean']:.2f}, Total={row['sum']:.0f}")

def correlation_analysis(train_df):
    """Analyze correlations between numerical features"""
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = train_df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        save_plot('05_correlation_matrix.png')

def seasonal_decomposition_analysis(train_df):
    """Enhanced seasonal decomposition with multiple approaches"""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller
        
        if 'date' in train_df.columns:
            df_temp = train_df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'])
            
            # Aggregate daily sales
            daily_sales = df_temp.groupby('date')['sales'].sum().reset_index()
            daily_sales = daily_sales.set_index('date')
            daily_sales = daily_sales.asfreq('D', fill_value=0)
            
            if len(daily_sales) > 2 * 365:
                print("\nPerforming Seasonal Decomposition Analysis...")
                
                # Additive decomposition
                decomp_add = seasonal_decompose(daily_sales['sales'], model='additive', period=365)
                
                # Multiplicative decomposition
                # Add small constant to avoid zero values
                sales_positive = daily_sales['sales'] + 1
                decomp_mult = seasonal_decompose(sales_positive, model='multiplicative', period=365)
                
                fig, axes = plt.subplots(4, 2, figsize=(20, 16))
                
                # Additive decomposition
                decomp_add.observed.plot(ax=axes[0, 0], title='Original Sales (Additive)', color='blue')
                axes[0, 0].grid(True, alpha=0.3)
                
                decomp_add.trend.plot(ax=axes[1, 0], title='Trend Component (Additive)', color='green')
                axes[1, 0].grid(True, alpha=0.3)
                
                decomp_add.seasonal.plot(ax=axes[2, 0], title='Seasonal Component (Additive)', color='orange')
                axes[2, 0].grid(True, alpha=0.3)
                
                decomp_add.resid.plot(ax=axes[3, 0], title='Residual Component (Additive)', color='red')
                axes[3, 0].grid(True, alpha=0.3)
                
                # Multiplicative decomposition
                decomp_mult.observed.plot(ax=axes[0, 1], title='Original Sales (Multiplicative)', color='blue')
                axes[0, 1].grid(True, alpha=0.3)
                
                decomp_mult.trend.plot(ax=axes[1, 1], title='Trend Component (Multiplicative)', color='green')
                axes[1, 1].grid(True, alpha=0.3)
                
                decomp_mult.seasonal.plot(ax=axes[2, 1], title='Seasonal Component (Multiplicative)', color='orange')
                axes[2, 1].grid(True, alpha=0.3)
                
                decomp_mult.resid.plot(ax=axes[3, 1], title='Residual Component (Multiplicative)', color='red')
                axes[3, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                save_plot('06_seasonal_decomposition.png')
                
                # Stationarity test
                print("\nStationarity Analysis:")
                adf_result = adfuller(daily_sales['sales'].dropna())
                print(f"ADF Statistic: {adf_result[0]:.6f}")
                print(f"p-value: {adf_result[1]:.6f}")
                print(f"Critical Values:")
                for key, value in adf_result[4].items():
                    print(f"\t{key}: {value:.3f}")
                
                is_stationary = adf_result[1] < 0.05
                print(f"Series is {'stationary' if is_stationary else 'non-stationary'} (p<0.05)")
                
                # Seasonal strength analysis
                seasonal_strength = 1 - (np.var(decomp_add.resid.dropna()) / np.var(decomp_add.seasonal.dropna() + decomp_add.resid.dropna()))
                trend_strength = 1 - (np.var(decomp_add.resid.dropna()) / np.var(decomp_add.trend.dropna() + decomp_add.resid.dropna()))
                
                print(f"\nDecomposition Strength:")
                print(f"Seasonal Strength: {seasonal_strength:.4f}")
                print(f"Trend Strength: {trend_strength:.4f}")
                
            else:
                print("Not enough data for reliable seasonal decomposition (need > 2 years)")
                
    except ImportError:
        print("statsmodels not available for seasonal decomposition")
    except Exception as e:
        print(f"Error in seasonal decomposition: {e}")

def advanced_statistical_analysis(train_df):
    """Advanced statistical analysis and feature engineering insights"""
    print("Performing advanced statistical tests...")
    
    # Feature engineering for analysis
    df_temp = train_df.copy()
    if 'date' in df_temp.columns:
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['weekday'] = df_temp['date'].dt.weekday
        df_temp['quarter'] = df_temp['date'].dt.quarter
    
    # Interaction effects analysis
    if 'store' in df_temp.columns and 'item' in df_temp.columns:
        # Create store-item interaction
        interaction_stats = df_temp.groupby(['store', 'item'])['sales'].agg(['mean', 'std', 'count']).reset_index()
        
        print(f"\\nStore-Item Interactions:")
        print(f"Total unique store-item combinations: {len(interaction_stats)}")
        print(f"Average sales per combination: {interaction_stats['mean'].mean():.2f}")
        print(f"Most profitable combination: Store {interaction_stats.loc[interaction_stats['mean'].idxmax(), 'store']}, Item {interaction_stats.loc[interaction_stats['mean'].idxmax(), 'item']} (Avg: {interaction_stats['mean'].max():.2f})")
        
        # Two-way ANOVA
        try:
            from scipy.stats import f_oneway
            import itertools
            
            # Sample for computation efficiency
            sample_df = df_temp.sample(min(10000, len(df_temp)), random_state=42)
            
            # Kruskal-Wallis test (non-parametric alternative to ANOVA)
            from scipy.stats import kruskal
            
            # Test for store effects
            store_groups = [group['sales'].values for name, group in sample_df.groupby('store')]
            h_stat_store, p_val_kruskal_store = kruskal(*store_groups)
            print(f"\\nKruskal-Wallis test (Store effect): H-statistic={h_stat_store:.2f}, p-value={p_val_kruskal_store:.2e}")
            
            # Test for item effects
            item_groups = [group['sales'].values for name, group in sample_df.groupby('item')]
            h_stat_item, p_val_kruskal_item = kruskal(*item_groups)
            print(f"Kruskal-Wallis test (Item effect): H-statistic={h_stat_item:.2f}, p-value={p_val_kruskal_item:.2e}")
            
        except ImportError:
            print("Advanced statistical tests require scipy")
    
    # Sales distribution analysis by segments
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sales distribution by store quartiles
    if 'store' in df_temp.columns:
        store_means = df_temp.groupby('store')['sales'].mean()
        q1, q2, q3 = store_means.quantile([0.25, 0.5, 0.75])
        
        low_stores = store_means[store_means <= q1].index
        mid_stores = store_means[(store_means > q1) & (store_means <= q3)].index  
        high_stores = store_means[store_means > q3].index
        
        axes[0, 0].hist(df_temp[df_temp['store'].isin(low_stores)]['sales'], 
                       alpha=0.7, label=f'Low Perf. Stores (n={len(low_stores)})', bins=50)
        axes[0, 0].hist(df_temp[df_temp['store'].isin(mid_stores)]['sales'], 
                       alpha=0.7, label=f'Mid Perf. Stores (n={len(mid_stores)})', bins=50)
        axes[0, 0].hist(df_temp[df_temp['store'].isin(high_stores)]['sales'], 
                       alpha=0.7, label=f'High Perf. Stores (n={len(high_stores)})', bins=50)
        axes[0, 0].set_title('Sales Distribution by Store Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Coefficient of variation analysis
    if 'store' in df_temp.columns:
        store_cv = df_temp.groupby('store')['sales'].agg(['mean', 'std'])
        store_cv['cv'] = store_cv['std'] / store_cv['mean']
        
        axes[0, 1].scatter(store_cv['mean'], store_cv['cv'], s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Mean Sales')
        axes[0, 1].set_ylabel('Coefficient of Variation')
        axes[0, 1].set_title('Sales Variability vs Mean (Store Level)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add correlation
        correlation = np.corrcoef(store_cv['mean'], store_cv['cv'])[0, 1]
        axes[0, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 1].transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Time-based variance analysis
    if 'date' in df_temp.columns:
        monthly_var = df_temp.groupby('month')['sales'].var()
        axes[1, 0].bar(monthly_var.index, monthly_var.values, color='lightblue', alpha=0.7)
        axes[1, 0].set_title('Sales Variance by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Sales growth analysis
    if 'date' in df_temp.columns:
        monthly_sales = df_temp.groupby(df_temp['date'].dt.to_period('M'))['sales'].sum()
        growth_rates = monthly_sales.pct_change().dropna()
        
        axes[1, 1].plot(range(len(growth_rates)), growth_rates.values, marker='o', alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Monthly Sales Growth Rate')
        axes[1, 1].set_xlabel('Month Index')
        axes[1, 1].set_ylabel('Growth Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_growth = growth_rates.mean()
        std_growth = growth_rates.std()
        axes[1, 1].text(0.05, 0.95, f'Mean: {mean_growth:.3f}\\nStd: {std_growth:.3f}', 
                       transform=axes[1, 1].transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    plt.tight_layout()
    save_plot('07_advanced_statistical_analysis.png')

def outlier_analysis(train_df):
    """Comprehensive outlier detection and analysis"""
    print("Performing outlier detection...")
    
    sales_data = train_df['sales'].values
    
    # Multiple outlier detection methods
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Z-score method
    z_scores = np.abs(stats.zscore(sales_data))
    z_outliers = sales_data[z_scores > 3]
    
    axes[0, 0].hist(sales_data, bins=50, alpha=0.7, label='Normal', color='blue')
    axes[0, 0].hist(z_outliers, bins=20, alpha=0.7, label=f'Z-score Outliers (n={len(z_outliers)})', color='red')
    axes[0, 0].set_title('Z-Score Outlier Detection')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. IQR method
    Q1 = np.percentile(sales_data, 25)
    Q3 = np.percentile(sales_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = sales_data[(sales_data < lower_bound) | (sales_data > upper_bound)]
    
    axes[0, 1].boxplot(sales_data)
    axes[0, 1].set_title(f'IQR Method (n={len(iqr_outliers)} outliers)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Modified Z-score (using median)
    median = np.median(sales_data)
    mad = np.median(np.abs(sales_data - median))
    modified_z_scores = 0.6745 * (sales_data - median) / mad
    modified_z_outliers = sales_data[np.abs(modified_z_scores) > 3.5]
    
    axes[0, 2].hist(sales_data, bins=50, alpha=0.7, label='Normal', color='blue')
    axes[0, 2].hist(modified_z_outliers, bins=20, alpha=0.7, 
                   label=f'Modified Z-score Outliers (n={len(modified_z_outliers)})', color='orange')
    axes[0, 2].set_title('Modified Z-Score Method')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Isolation Forest
    try:
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_pred = iso_forest.fit_predict(sales_data.reshape(-1, 1))
        iso_outliers = sales_data[outlier_pred == -1]
        
        axes[1, 0].hist(sales_data, bins=50, alpha=0.7, label='Normal', color='blue')
        axes[1, 0].hist(iso_outliers, bins=20, alpha=0.7, 
                       label=f'Isolation Forest Outliers (n={len(iso_outliers)})', color='green')
        axes[1, 0].set_title('Isolation Forest Method')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
    except ImportError:
        axes[1, 0].text(0.5, 0.5, 'Isolation Forest\\nrequires scikit-learn', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Isolation Forest (Not Available)')
    
    # 5. Percentile method
    p99 = np.percentile(sales_data, 99)
    p1 = np.percentile(sales_data, 1)
    percentile_outliers = sales_data[(sales_data > p99) | (sales_data < p1)]
    
    axes[1, 1].hist(sales_data, bins=50, alpha=0.7, label='Normal', color='blue')
    axes[1, 1].hist(percentile_outliers, bins=20, alpha=0.7, 
                   label=f'99th Percentile Outliers (n={len(percentile_outliers)})', color='purple')
    axes[1, 1].set_title('Percentile Method (1st & 99th)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary comparison
    outlier_methods = {
        'Z-Score (3σ)': len(z_outliers),
        'IQR (1.5×IQR)': len(iqr_outliers),
        'Modified Z-Score': len(modified_z_outliers),
        'Percentile (1-99%)': len(percentile_outliers)
    }
    
    try:
        outlier_methods['Isolation Forest'] = len(iso_outliers)
    except:
        pass
    
    methods = list(outlier_methods.keys())
    counts = list(outlier_methods.values())
    
    bars = axes[1, 2].bar(range(len(methods)), counts, color=['red', 'blue', 'orange', 'purple', 'green'][:len(methods)])
    axes[1, 2].set_title('Outlier Count by Method')
    axes[1, 2].set_xlabel('Detection Method')
    axes[1, 2].set_ylabel('Number of Outliers')
    axes[1, 2].set_xticks(range(len(methods)))
    axes[1, 2].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                       f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot('08_outlier_detection_analysis.png')
    
    # Outlier analysis by categories
    if 'store' in train_df.columns:
        print("\\nOutlier Analysis by Store:")
        for store in sorted(train_df['store'].unique()):
            store_sales = train_df[train_df['store'] == store]['sales']
            store_z_scores = np.abs(stats.zscore(store_sales))
            store_outliers = len(store_sales[store_z_scores > 3])
            outlier_percentage = (store_outliers / len(store_sales)) * 100
            print(f"  Store {store}: {store_outliers} outliers ({outlier_percentage:.2f}%)")
    
    # Impact analysis
    total_sales_with_outliers = sales_data.sum()
    total_sales_without_z_outliers = sales_data[z_scores <= 3].sum()
    outlier_impact = ((total_sales_with_outliers - total_sales_without_z_outliers) / total_sales_with_outliers) * 100
    
    print(f"\\nOutlier Impact Analysis:")
    print(f"Total sales: {total_sales_with_outliers:,.0f}")
    print(f"Sales from Z-score outliers: {total_sales_with_outliers - total_sales_without_z_outliers:,.0f}")
    print(f"Outlier contribution to total sales: {outlier_impact:.2f}%")

def generate_summary_report(train_df, test_df, sample_submission):
    """Generate a summary report of findings"""
    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}")
    
    print(f"\nDataset Overview:")
    print(f"- Training samples: {len(train_df):,}")
    print(f"- Test samples: {len(test_df):,}")
    print(f"- Features in train: {len(train_df.columns)}")
    print(f"- Features in test: {len(test_df.columns)}")
    
    if 'sales' in train_df.columns:
        print(f"\nSales Statistics:")
        print(f"- Mean sales: {train_df['sales'].mean():.2f}")
        print(f"- Median sales: {train_df['sales'].median():.2f}")
        print(f"- Min sales: {train_df['sales'].min():.2f}")
        print(f"- Max sales: {train_df['sales'].max():.2f}")
        print(f"- Sales std: {train_df['sales'].std():.2f}")
    
    if 'store' in train_df.columns:
        print(f"\n- Number of stores: {train_df['store'].nunique()}")
    
    if 'item' in train_df.columns:
        print(f"- Number of items: {train_df['item'].nunique()}")
    
    if 'date' in train_df.columns:
        print(f"- Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    
    print(f"\nData Quality:")
    missing_train = train_df.isnull().sum().sum()
    missing_test = test_df.isnull().sum().sum()
    print(f"- Missing values in train: {missing_train}")
    print(f"- Missing values in test: {missing_test}")

def main():
    """Main EDA function"""
    print("Starting Demand Forecasting EDA...")
    print("="*60)
    
    # Load data
    train_df, test_df, sample_submission = load_data()
    
    if train_df is None:
        print("Cannot proceed without data. Please download the dataset first.")
        return
    
    # Basic exploration
    explore_basic_info(train_df, "Training Data")
    explore_basic_info(test_df, "Test Data")
    explore_basic_info(sample_submission, "Sample Submission")
    
    # Sales distribution analysis
    if 'sales' in train_df.columns:
        print("\nAnalyzing sales distribution...")
        plot_sales_distribution(train_df)
    
    # Temporal analysis
    print("\nAnalyzing temporal patterns...")
    analyze_temporal_patterns(train_df)
    
    # Store and item analysis
    print("\nAnalyzing store and item patterns...")
    analyze_store_item_patterns(train_df)
    
    # Correlation analysis
    print("\nPerforming correlation analysis...")
    correlation_analysis(train_df)
    
    # Seasonal decomposition
    print("\nPerforming seasonal decomposition...")
    seasonal_decomposition_analysis(train_df)
    
    # Advanced analysis
    print("\nPerforming advanced statistical analysis...")
    advanced_statistical_analysis(train_df)
    
    print("\nPerforming outlier detection...")
    outlier_analysis(train_df)
    
    # Generate summary
    generate_summary_report(train_df, test_df, sample_submission)
    
    print("\nEDA completed!")

if __name__ == "__main__":
    main()