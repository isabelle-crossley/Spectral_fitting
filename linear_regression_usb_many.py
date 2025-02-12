# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:57:36 2025

@author: wlmd95
"""

import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import re

# Load pure spectra and combined datasets
def import_arrays(filepath):
    # Read the text from file
    with open(filepath, 'r') as file:
        text = file.read()

    # Pattern to extract arrays
    pattern = r"\[.*?\]"

    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    # Convert matched strings to arrays
    arrays = []
    for match in matches:
        array = re.findall(r"\d+\.*\d*", match)
        arrays.append([float(num) for num in array])
    arrays = np.array(arrays)

    return arrays[:3]

def fit_linear_regression(fluorescein, nile_red, combined_spectrum, wavelengths, normalise=True):
    """
    Linear regression to fit the weighting factors of fluorescein and Nile red spectra
    of a given combined spectrum.
    """
    # Normalize
    if normalise:
        fluorescein = (fluorescein - np.min(fluorescein)) / (np.max(fluorescein) - np.min(fluorescein))
        nile_red = (nile_red - np.min(nile_red)) / (np.max(nile_red) - np.min(nile_red))
        combined_spectrum = (combined_spectrum - np.min(combined_spectrum)) / (np.max(combined_spectrum) - np.min(combined_spectrum))
    
    # Combine spectra into a feature matrix
    X = np.vstack((fluorescein, nile_red)).T  # Shape: (n_samples, 2)
    y = combined_spectrum  # Target combined spectrum

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Retrieve the optimized weights and intercept
    weights = model.coef_
    intercept = model.intercept_

    # Predict the combined spectrum
    y_pred = model.predict(X)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(wavelengths, y, label='Observed Combined Spectrum', color='blue')
    plt.plot(wavelengths, y_pred, label='Predicted Combined Spectrum', color='red', linestyle='--')
    plt.legend()
    plt.xlabel("Wavelength")
    plt.ylabel("Normalized Intensity")
    plt.title("Linear Regression Fit")
    plt.show()

    return weights, intercept

# Paths to pure spectra
fluorescein = import_arrays(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240122\fluorescein_test.txt")
nile_red = import_arrays(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240122\pink_fluoro_txt")
fluorescein_x, fluorescein_y = fluorescein[0], fluorescein[1]
nile_red_x, nile_red_y = nile_red[0], nile_red[1]

# Folder containing combined datasets
combined_data_folder = r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240311"
#combined_data_folder = r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240307"
#combined_data_folder = r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240125"

# Process all combined datasets in the folder
results = []
for filename in os.listdir(combined_data_folder):
    print('.....\n\n')
    if filename.endswith(".txt"):  # Ensure only text files are processed
        print(filename)
        file_path = os.path.join(combined_data_folder, filename)
        combined_dataset = import_arrays(file_path)
        combined_x, combined_y = combined_dataset[0], combined_dataset[1]

        # Fit the regression model
        weights, intercept = fit_linear_regression(
            fluorescein_y, nile_red_y, combined_y, combined_x, normalise=True
        )
        
        # Store results
        results.append({
            "filename": filename,
            "weights": weights,
            "intercept": intercept
        })

        # Print results for each file
        print(f"File: {filename}")
        print(f"  Optimized weights: Fluorescein = {weights[0]:.4f}, Nile Red = {weights[1]:.4f}")
        print(f"  Sum of weights: {weights[0] + weights[1]:.4f}")
        print(f"  Intercept: {intercept:.4f}\n")

# Optionally, save results to a file
'''results_output_path = r"C:\Path\to\output_results.txt"
with open(results_output_path, "w") as file:
    for result in results:
        file.write(f"File: {result['filename']}\n")
        file.write(f"  Optimized weights: Fluorescein = {result['weights'][0]:.4f}, Nile Red = {result['weights'][1]:.4f}\n")
        file.write(f"  Intercept: {result['intercept']:.4f}\n\n")
'''