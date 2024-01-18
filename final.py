import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.impute import SimpleImputer


def process_data(file_path):
    """
    Read, clean, and transpose the data.

    Parameters:
        - file_path: Path to the CSV file containing the data.

    Returns:
        - original_data: Original data read from the CSV file.
        - cleaned_data: Cleaned data with non-numeric characters removed and missing values imputed.
        - transposed_data: Transposed version of the cleaned data.
    """
    # Read the original data
    original_data = pd.read_csv(file_path)

    # Identify and clean non-numeric characters
    columns_to_clean = ['Forest area (% of land area) [AG.LND.FRST.ZS]',
                        'Total greenhouse gas emissions (% change from 1990) [EN.ATM.GHGT.ZG]',
                        'Renewable energy consumption (% of total final energy consumption) [EG.FEC.RNEW.ZS]',
                        'Total reserves (% of total external debt) [FI.RES.TOTL.DT.ZS]']

    cleaned_data = original_data.copy()
    for column in columns_to_clean:
        cleaned_data[column] = pd.to_numeric(cleaned_data[column].astype(str).str.replace(r'[^0-9.]', ''), errors='coerce')

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    cleaned_data[columns_to_clean] = imputer.fit_transform(cleaned_data[columns_to_clean])

    # Transpose the cleaned data
    transposed_data = cleaned_data.transpose()

    return original_data, cleaned_data, transposed_data


def calculate_confidence_interval(prediction, std_dev, z_score=1.96):
    """
    Calculate confidence interval for a prediction.

    Parameters:
        - prediction: Predicted values (array)
        - std_dev: Standard deviations of the predictions (array)
        - z_score: Z-score for the desired confidence level (default is 1.96 for 95% confidence)

    Returns:
        Arrays containing lower and upper bounds of the confidence intervals.
    """
    lower_bound = prediction - z_score * np.repeat(std_dev, len(prediction) // len(std_dev))
    upper_bound = prediction + z_score * np.repeat(std_dev, len(prediction) // len(std_dev))
    return lower_bound, upper_bound


# Define a simple model function with initial parameters
def model_function(x, a, b):
    """
        Compute the values of a simple exponential model.

        The exponential model is defined as: y = a * exp(b * x)

        Parameters:
            - x (array-like): Independent variable values.
            - a (float): Amplitude or scale factor of the exponential function.
            - b (float): Exponential growth or decay rate.

        Returns:
            Array of computed values based on the exponential model for the given x values.
        """
    return a * np.exp(b * x)



# Use the process_data function to obtain the data
original_data, cleaned_data, transposed_data = process_data("6b7a8d1c-190c-45c0-88c6-d330425c1e6f_Data.csv")

columns_to_clean = ['Forest area (% of land area) [AG.LND.FRST.ZS]',
                    'Total greenhouse gas emissions (% change from 1990) [EN.ATM.GHGT.ZG]',
                    'Renewable energy consumption (% of total final energy consumption) [EG.FEC.RNEW.ZS]',
                    'Total reserves (% of total external debt) [FI.RES.TOTL.DT.ZS]']

# Normalize the imputed data
normalized_data = (cleaned_data[columns_to_clean] - cleaned_data[columns_to_clean].mean()) / cleaned_data[columns_to_clean].std()

# Perform clustering (example with k-means)
kmeans = KMeans(n_clusters=4)
cleaned_data['Cluster'] = kmeans.fit_predict(normalized_data)

# Add cluster centers to the dataframe
cleaned_data['ClusterCenter'] = kmeans.predict(normalized_data)

# Calculate silhouette scores
silhouette_avg = silhouette_score(normalized_data, cleaned_data['Cluster'])
print(silhouette_avg)
cleaned_data['SilhouetteScore'] = silhouette_samples(normalized_data, cleaned_data['Cluster'])

# Visualize clusters
plt.scatter(cleaned_data['Forest area (% of land area) [AG.LND.FRST.ZS]'],
            cleaned_data['Total greenhouse gas emissions (% change from 1990) [EN.ATM.GHGT.ZG]'], c=cleaned_data['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Cluster Centers')
plt.xlabel('Forest Area %')
plt.ylabel('Total Greenhouse Gas Emissions')
plt.title('Clustering of Countries with Cluster Centers')
plt.legend()
plt.show()

# Provide initial parameter guesses
initial_params = [1.0, 0.02]

# Fit the model to the data with initial parameters
try:
    params, covariance = curve_fit(model_function, cleaned_data['Time'], cleaned_data['Total greenhouse gas emissions (% change from 1990) [EN.ATM.GHGT.ZG]'], p0=initial_params)

    # Predict future values
    future_years = np.arange(2000, 2042, 1)
    print(future_years)
    predicted_values = model_function(future_years, *params)
    print(predicted_values)

    # Estimate confidence intervals
    std_dev = np.sqrt(np.diag(covariance))  # Diagonal elements for standard deviations
    lower, upper = calculate_confidence_interval(predicted_values, std_dev)

    # Check for NaN or inf in confidence intervals
    if np.any(np.isnan(lower)) or np.any(np.isinf(lower)) or np.any(np.isnan(upper)) or np.any(np.isinf(upper)):
        print("Error in calculating confidence intervals. Check your data or fitting process.")
    else:
        # Visualize the fitted model and confidence range
        plt.scatter(cleaned_data['Time'], cleaned_data['Total greenhouse gas emissions (% change from 1990) [EN.ATM.GHGT.ZG]'], label='Actual Data')
        plt.plot(future_years, predicted_values, label='Fitted Model', color='red')
        plt.fill_between(future_years, lower, upper, color='black', alpha=0.2, linewidth=10, label='Confidence Interval')
        plt.xlabel('Year')
        plt.ylabel('Total Greenhouse Gas Emissions')
        plt.legend()
        plt.title('Fitted Model with Confidence Interval')
        plt.show()

except Exception as e:
    print(f"Error in curve fitting: {e}")

