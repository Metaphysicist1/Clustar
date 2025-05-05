![image](https://github.com/user-attachments/assets/51b3cdb2-df77-4ff8-9d8d-909f22b247ea)

# Clustar: Automated Star Clustering Analysis

Clustar is an advanced astronomical data analysis project that leverages machine learning and clustering techniques to automate the detection and analysis of variable stars and transients from large-scale astronomical surveys.

## Project Overview

Modern astronomical surveys like TESS and LSST generate terabytes of data daily, creating an unprecedented challenge for traditional manual analysis methods. Clustar addresses this challenge by:

1. Automating detection of anomalous variability, scaling to large datasets
2. Prioritizing high-variability clusters, optimizing telescope use
3. Providing robust noise handling, capturing diverse transients (e.g., supernovae, pulsators)

## Key Challenges Addressed

- **Data Volume**: Millions of stars observed in modern surveys generate terabytes of data daily
- **Limited Telescope Time**: Expensive telescope time (e.g., JWST at $100K/hour) requires optimal utilization
- **Manual Analysis Limitations**: Traditional methods are slow and miss subtle patterns
- **Missed Opportunities**: Delayed detection of transients impacts our understanding of stellar evolution and cosmology

## Project Impact

### Scientific Impact

- Accelerate discovery of variable stars, supernovae, and other transients
- Enhance understanding of stellar evolution and galactic dynamics
- Transform transient astronomy through automated analysis

### Operational Impact

- Optimize telescope scheduling, potentially saving $1M-$5M annually
- Prioritize follow-up observations for high-impact targets
- Streamline telescope use through intelligent clustering

## Dataset Description

Our dataset is sourced from the NASA Exoplanet Science Institute, providing rich photometric and light curve data. Key features include:

Dataset is also provided on kaggle: **https://www.kaggle.com/datasets/edgarabasov/star-observations-dataset**

- `star_id`: Unique identifier for each star
- `region`: Part of the sky surveyed
- `ra`: Right Ascension in degrees
- `dec`: Declination in degrees
- `starthjd`: Start time of observation (Heliocentric Julian Date)
- `endhjd`: End time of observation (Heliocentric Julian Date)
- `vmag`: V-band magnitude
- `verr`: V-band magnitude uncertainty
- `imag`: I-band magnitude
- `ierr`: I-band magnitude uncertainty
- `npts`: Number of points in the light curve

## Implementation Details

The project implements a comprehensive analysis pipeline:

1. Data Extraction and Loading

   - Direct integration with NASA database
   - Efficient data loading and preprocessing

2. Feature Analysis and Processing

   - Raw data transformation
   - Standardization using scalers
   - Dimension reduction techniques

3. Clustering Analysis

   - Multiple algorithm implementation (KMeans, PCA, etc.)
   - Parameter optimization
   - Parallel processing capabilities

4. Visualization and Interpretation
   - Interactive result visualization
   - Comprehensive reporting
   - Feature importance analysis

## Project Structure

- `explore.ipynb`: Jupyter notebook containing the data exploration and analysis
- `Cluster_2025.04.16_04.44.28.csv`: Raw dataset from NASA Exoplanet Archive

## Requirements

The project requires the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Getting Started

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open and run the Jupyter notebook `explore.ipynb`

## Data Source

The data was obtained from the NASA Exoplanet Archive (http://exoplanetarchive.ipac.caltech.edu) on April 16, 2025.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
