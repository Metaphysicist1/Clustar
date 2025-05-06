import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path, sample_size=None):
    """Load and preprocess the astronomical data."""
    # Load the data
    df = pd.read_csv(file_path)
    
    # Remove duplicates if any
    df = df.drop_duplicates()
    
    # If sample_size is provided, take a random sample
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    # Select features for clustering
    features = ['ra', 'dec', 'starthjd', 'endhjd', 'vmag', 'verr', 'imag', 'ierr', 'npts']
    X = df[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, features

def find_optimal_clusters(X_scaled, max_clusters=5, batch_size=1000):
    """Find the optimal number of clusters using the elbow method and silhouette score."""
    # Calculate inertia (within-cluster sum of squares) for different k values
    inertias = []
    silhouette_scores = []
    K = range(1, max_clusters + 1)
    
    print("\nCalculating cluster metrics...")
    print("Using MiniBatchKMeans for faster computation...")
    
    # First calculate inertias for all k values using MiniBatchKMeans
    for k in K:
        print(f"Processing k={k} (inertia calculation)...")
        kmeans = MiniBatchKMeans(n_clusters=k, 
                                batch_size=batch_size,
                                random_state=42,
                                n_init=3)  # Reduced n_init for speed
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Then calculate silhouette scores (only for k > 1)
    print("\nCalculating silhouette scores...")
    # Use a subset of data for silhouette score calculation
    sample_size = min(10000, len(X_scaled))
    X_sample = X_scaled[np.random.choice(len(X_scaled), sample_size, replace=False)]
    
    for k in range(2, max_clusters + 1):
        print(f"Processing k={k} (silhouette calculation)...")
        kmeans = MiniBatchKMeans(n_clusters=k, 
                                batch_size=batch_size,
                                random_state=42,
                                n_init=3)
        labels = kmeans.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        silhouette_scores.append(score)
    
    # Plot elbow curve and silhouette scores
    print("\nGenerating plots...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png')
    plt.close()
    
    print("Cluster analysis plots saved to 'cluster_analysis.png'")
    
    return inertias, silhouette_scores

def perform_clustering(X_scaled, n_clusters, batch_size=1000):
    """Perform K-means clustering with the specified number of clusters."""
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                            batch_size=batch_size,
                            random_state=42,
                            n_init=3)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans

def visualize_clusters(df, clusters, features):
    """Visualize the clusters using PCA for dimensionality reduction."""
    # Apply PCA to reduce dimensions to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df[features])
    
    # Create a DataFrame with PCA results and cluster assignments
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters
    
    # Plot the clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis')
    plt.title('Cluster Visualization (PCA)')
    plt.savefig('cluster_visualization.png')
    plt.close()

def analyze_clusters(df, clusters, features):
    """Analyze and print cluster characteristics."""
    # Create a copy of the dataframe to avoid modifying the original
    df_analyzed = df.copy()
    
    # Add cluster assignments to the dataframe
    df_analyzed['Cluster'] = clusters
    
    # Calculate cluster statistics
    cluster_stats = df_analyzed.groupby('Cluster')[features].agg(['mean', 'std']).round(3)
    
    # Save cluster statistics to CSV
    cluster_stats.to_csv('cluster_statistics.csv')
    
    # Print cluster sizes
    print("\nCluster Sizes:")
    print(df_analyzed['Cluster'].value_counts().sort_index())
    
    return cluster_stats

def main():
    # File path
    file_path = 'data/clean_numeric_dataset.csv'
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    # Use a sample of 50000 rows for faster processing
    df, X_scaled, features = load_and_preprocess_data(file_path, sample_size=50000)
    
    # Step 2: Find optimal number of clusters
    print("\nStep 2: Finding optimal number of clusters...")
    inertias, silhouette_scores = find_optimal_clusters(X_scaled, max_clusters=5)
    
    # Step 3: Perform clustering
    print("\nStep 3: Performing clustering...")
    n_clusters = 4  # This can be adjusted based on the elbow method results
    clusters, kmeans = perform_clustering(X_scaled, n_clusters)
    
    # Step 4: Visualize clusters
    print("\nStep 4: Visualizing clusters...")
    visualize_clusters(df, clusters, features)
    
    # Step 5: Analyze clusters
    print("\nStep 5: Analyzing clusters...")
    cluster_stats = analyze_clusters(df, clusters, features)
    
    print("\nClustering analysis complete! Check the generated files:")
    print("- cluster_analysis.png: Elbow method and silhouette analysis")
    print("- cluster_visualization.png: PCA visualization of clusters")
    print("- cluster_statistics.csv: Detailed cluster statistics")

if __name__ == "__main__":
    main() 