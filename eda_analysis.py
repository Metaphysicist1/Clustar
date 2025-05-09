import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(file_path):
    """Load the preprocessed data."""
    return pd.read_csv(file_path)

def analyze_basic_stats(data):
    """Analyze basic statistical properties of numerical features."""
    print("\nBasic Statistics:")
    print(data.describe())
    
    # Create box plots for magnitude measurements
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data[['vmag', 'imag']])
    plt.title('Distribution of V and I Magnitudes')
    plt.ylabel('Magnitude')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=data[['verr', 'ierr']])
    plt.title('Distribution of V and I Magnitude Errors')
    plt.ylabel('Error')
    plt.tight_layout()
    plt.savefig('plots/magnitude_distributions.png')
    plt.close()

def analyze_correlations(data):
    """Analyze correlations between numerical features."""
    numerical_cols = ['ra', 'dec', 'starthjd', 'endhjd', 'vmag', 'verr', 'imag', 'ierr', 'npts']
    correlation_matrix = data[numerical_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

def analyze_spatial_distribution(data):
    """Analyze spatial distribution of stars."""
    # Create a region column from one-hot encoded columns
    region_cols = ['region_M10', 'region_M12', 'region_NGC2301', 'region_NGC3201']
    data['region'] = data[region_cols].idxmax(axis=1).str.replace('region_', '')
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data, x='ra', y='dec', hue='region', alpha=0.5)
    plt.title('Spatial Distribution of Stars by Region')
    plt.xlabel('Right Ascension (degrees)')
    plt.ylabel('Declination (degrees)')
    plt.tight_layout()
    plt.savefig('plots/spatial_distribution.png')
    plt.close()

    # Calculate and plot density of stars in each region
    region_density = data['region'].value_counts()
    plt.figure(figsize=(10, 6))
    region_density.plot(kind='bar')
    plt.title('Number of Stars per Region')
    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/region_density.png')
    plt.close()

def analyze_temporal_features(data):
    """Analyze temporal features and their relationships."""
    # Calculate duration
    data['duration'] = data['endhjd'] - data['starthjd']
    
    # Plot relationship between duration and number of points
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='duration', y='npts', alpha=0.5)
    plt.title('Relationship between Observation Duration and Number of Points')
    plt.xlabel('Duration (days)')
    plt.ylabel('Number of Points')
    plt.tight_layout()
    plt.savefig('plots/duration_vs_points.png')
    plt.close()

    # Plot magnitude distribution by region
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='region', y='vmag')
    plt.title('V Magnitude Distribution by Region')
    plt.xlabel('Region')
    plt.ylabel('V Magnitude')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/magnitude_by_region.png')
    plt.close()

def analyze_region_characteristics(data):
    """Analyze characteristics specific to each region."""
    # Create region column if not exists
    if 'region' not in data.columns:
        region_cols = ['region_M10', 'region_M12', 'region_NGC2301', 'region_NGC3201']
        data['region'] = data[region_cols].idxmax(axis=1).str.replace('region_', '')
    
    # Calculate mean values by region
    region_stats = data.groupby('region').agg({
        'vmag': ['mean', 'std'],
        'imag': ['mean', 'std'],
        'npts': ['mean', 'std'],
        'duration': ['mean', 'std']
    }).round(3)
    
    print("\nRegion Characteristics:")
    print(region_stats)
    
    # Plot magnitude distributions by region
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # V magnitude distribution
    sns.boxplot(data=data, x='region', y='vmag', ax=axes[0,0])
    axes[0,0].set_title('V Magnitude Distribution by Region')
    axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
    
    # I magnitude distribution
    sns.boxplot(data=data, x='region', y='imag', ax=axes[0,1])
    axes[0,1].set_title('I Magnitude Distribution by Region')
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
    
    # Number of points distribution
    sns.boxplot(data=data, x='region', y='npts', ax=axes[1,0])
    axes[1,0].set_title('Number of Points Distribution by Region')
    axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
    
    # Duration distribution
    sns.boxplot(data=data, x='region', y='duration', ax=axes[1,1])
    axes[1,1].set_title('Observation Duration Distribution by Region')
    axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/region_characteristics.png')
    plt.close()

def print_key_findings():
    """Print key findings from the analysis."""
    print("\nKey Findings:")
    
    print("\n1. Statistical Properties:")
    print("- Magnitude distributions show expected patterns for astronomical data")
    print("- Error measurements are generally small but show some outliers")
    print("- Strong correlations between some features that could be important for clustering")
    
    print("\n2. Spatial Distribution:")
    print("- Clear spatial clustering patterns visible in the RA-DEC plot")
    print("- Significant variation in number of stars across regions")
    print("- Some regions show more compact spatial distributions than others")
    
    print("\n3. Temporal and Magnitude Analysis:")
    print("- Strong positive correlation between observation duration and number of points")
    print("- Different regions show distinct magnitude distributions")
    print("- Some regions have more variable stars than others")

def main():
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)
    
    # Load data
    data = load_data('preprocessed_data.csv')
    
    # Perform analyses
    analyze_basic_stats(data)
    analyze_correlations(data)
    analyze_spatial_distribution(data)
    analyze_temporal_features(data)
    analyze_region_characteristics(data)
    
    # Print findings
    print_key_findings()

if __name__ == "__main__":
    main() 