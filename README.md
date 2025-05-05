# Star Clustering Analysis Project

This project analyzes astronomical data from the NASA Exoplanet Archive to perform clustering analysis on star observations. The dataset contains various measurements of stars including their positions, magnitudes, and observation periods.

## Dataset Description

The dataset contains the following key features:

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

## Project Structure

- `explore.ipynb`: Jupyter notebook containing the data exploration and analysis
- `Cluster_2025.04.16_04.44.28.csv`: Raw dataset from NASA Exoplanet Archive

## Data Processing

The project includes the following data processing steps:

1. Loading and initial exploration of the dataset
2. Handling missing values
3. Feature selection and preprocessing
4. Data normalization
5. Clustering analysis

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
