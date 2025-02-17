# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:58:50 2024

@author: wlmd95
"""

import os 
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import re
import csv
import pandas as pd


def chi_squared(observed, expected, error):
    return np.sum((observed - expected) ** 2 / error**2)

def chi_squared_reduced(observed, expected, error):
    return chi_squared(observed, expected, error)/np.size(observed)

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
    arrays=np.array(arrays)

    return arrays[:2]

def read_csv(filename):
    # Read CSV using pandas
    df = pd.read_csv(filename)

    # Ensure the correct columns exist
    if 'Wavelength' not in df.columns or 'Intensity' not in df.columns:
        raise ValueError(f"File {filename} does not have the expected columns: 'Wavelength' and 'Intensity'")

    # Convert to numpy array (transpose to match previous format)
    return df[['Wavelength', 'Intensity']].to_numpy().T

# test data 
fluorescein = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra files\fluorescein_mkid_250212.csv")  
nile_red = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra files\nile_red_mkid_250212.csv")  
combined_dataset_1 = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra files\mixed_spectrum_100_fluorescein_mkid_250212.csv")  
combined_dataset_2 = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra files\mixed_spectrum_90_fluorescein_mkid_250212.csv")


#fitting process
def fit_spectra_strength(variables, coef):
    spectrum1_strength, spectrum2_strength, background_strength = variables  # Added background_strength

    fluorescein, nile_red, combined_spectrum = coef  
    
    # Combine the pure spectra and background according to the provided strengths
    combined_spectrum_model = (
        spectrum1_strength * fluorescein 
        + spectrum2_strength * nile_red 
        + background_strength
    )

    # Calculate chi-squared value
    chi2 = chi_squared(combined_spectrum, combined_spectrum_model, np.sqrt(np.abs(combined_spectrum[1])))

    
    return chi2


def monte_carlo_estimate_errors(fitting_function, initial_guess, data, num_simulations=1000, noise_scale=1000):
    optimized_parameters = np.zeros((num_simulations, 3))

    for i in range(num_simulations):
        perturbed_data = data + np.random.normal(0, np.sqrt(np.abs(data)), size=data.shape)
        coef = np.array([fluorescein, nile_red, perturbed_data])

        result = minimize(fitting_function, initial_guess, args=coef,
                          method='SLSQP', bounds=[(0, 1)] * 3, tol=1e-6, 
                          constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                          options={'disp': False})

        optimized_parameters[i] = result.x

    return np.std(optimized_parameters, axis=0)

# Constraint function to ensure the sum of spectrum strengths equals 1

def constraint_sum_to_one(variables):
    return sum(variables) - 1



fluorescein_x = fluorescein[0]
fluorescein_y = fluorescein[1]
nile_red_x = nile_red[0]
nile_red_y = nile_red[1]
combined_dataset_1x = combined_dataset_1[0]
combined_dataset_1y = combined_dataset_1[1]
combined_dataset_2x = combined_dataset_2[0]
combined_dataset_2y = combined_dataset_2[1]

# Normalise
fluorescein_y_normalised = (fluorescein_y - min(fluorescein_y)) / (max(fluorescein_y) - min(fluorescein_y))
nile_red_y_normalised = (nile_red_y - min(nile_red_y)) / (max(nile_red_y) - min(nile_red_y))
combined_dataset_1y_normalised = (combined_dataset_1y - min(combined_dataset_1y)) / (max(combined_dataset_1y) - min(combined_dataset_1y))
combined_dataset_2y_normalised = (combined_dataset_2y - min(combined_dataset_2y)) / (max(combined_dataset_2y) - min(combined_dataset_2y))


# Initial guess for the strength of the first spectrum
initial_guess = np.array([0.4,0.4,0.1])

coef = np.array([fluorescein_y_normalised, nile_red_y_normalised, combined_dataset_1y_normalised])
#print(coef)
coef_2 = np.array([fluorescein_y_normalised, nile_red_y_normalised, combined_dataset_2y_normalised])

num_simulations = 1000
noise_scale  = 100

# Perform the fitting using the minimize function from scipy.optimize for combined_dataset_1

result_1 = minimize(fit_spectra_strength, initial_guess, args=coef,
                    method='SLSQP', bounds = [(0, 1), (0, 1), (0, 1)], tol=1e-6,
                    constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                    options={'disp': True}
                    )

# Perform Monte Carlo error estimation for combined_dataset_1
parameter_errors_1 = monte_carlo_estimate_errors(fit_spectra_strength, initial_guess, combined_dataset_1,
                                                 num_simulations=num_simulations, noise_scale=noise_scale)

# Perform the fitting using the minimize function from scipy.optimize for combined_dataset_2

result_2 = minimize(fit_spectra_strength, initial_guess, args=coef_2,
                    method='SLSQP', bounds = [(0, 1), (0, 1), (0, 1)], tol=1e-6,
                    constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                    options={'disp': True}
                    )

# Perform Monte Carlo error estimation for combined_dataset_2
parameter_errors_2 = monte_carlo_estimate_errors(fit_spectra_strength, initial_guess, combined_dataset_2,
                                                 num_simulations=num_simulations, noise_scale=noise_scale)

# Extract the optimized strengths for each dataset
optimized_strength_spectrum1_1 = result_1.x[0] # fluorescein dataset 1
optimized_strength_spectrum2_1 = result_1.x[1]  # nile red dataset 1
optimized_strength_spectrum3_1 = result_1.x[2]  # background dataset 1


optimized_strength_spectrum1_2 = result_2.x[0] # fluorescein dataset 2
optimized_strength_spectrum2_2 = result_2.x[1]  # nile red dataset 2
optimized_strength_spectrum3_2 = result_2.x[2]  # background dataset 2
 
# Print the optimized strengths and their errors for each dataset
print(f'Optimized strength for dataset 1 - Fluorescein: {optimized_strength_spectrum1_1} ± {parameter_errors_1[0]}, '
      f'Nile red: {optimized_strength_spectrum2_1} ± {parameter_errors_1[1]}, '
      f'Background: {optimized_strength_spectrum3_1} ± {parameter_errors_1[2]}')

print(f'Optimized strength for dataset 2 - Fluorescein: {optimized_strength_spectrum1_2} ± {parameter_errors_2[0]}, '
      f'Nile red: {optimized_strength_spectrum2_2} ± {parameter_errors_2[1]}, '
      f'Background: {optimized_strength_spectrum3_2} ± {parameter_errors_2[2]}')


fluorescein, nile_red, combined_spectrum = coef 
#print('fluoro', fluorescein)
# Combine the pure spectra according to the provided strengths
combined_spectrum_model = optimized_strength_spectrum1_1 * fluorescein + optimized_strength_spectrum2_1 * nile_red
combined_spectrum_model_2 = optimized_strength_spectrum1_2 * fluorescein + optimized_strength_spectrum2_2 * nile_red
#print('model', combined_spectrum_model)
# Calculate chi-squared value
chi2 = chi_squared(combined_spectrum, combined_spectrum_model, (combined_spectrum))
chi2_2 = chi_squared(combined_spectrum, combined_spectrum_model_2, (combined_spectrum))
#print('comb', combined_spectrum)

print(f'chi squared dataset 1 = {chi_squared(combined_spectrum, combined_spectrum_model, combined_spectrum[10]**(0.5))}')
print(f'reduced chi squared = {chi_squared_reduced(combined_spectrum, combined_spectrum_model, combined_spectrum[10]**(0.5))}')

print(f'chi squared dataset 2 = {chi_squared(combined_spectrum, combined_spectrum_model_2, combined_spectrum[10]**(0.5))}')
print(f'reduced chi squared = {chi_squared_reduced(combined_spectrum, combined_spectrum_model_2, combined_spectrum[10]**(0.5))}')
