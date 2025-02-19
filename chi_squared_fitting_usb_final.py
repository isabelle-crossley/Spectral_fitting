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
import glob

epsilon = 1e-10


def chi_squared(observed, expected, error):
    return np.sum((observed - expected) ** 2 / (error**2 + epsilon))

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
    if 'Wavelength' not in df.columns or 'Intensity' not in df.columns or 'Error' not in df.columns:
        raise ValueError(f"File {filename} does not have the expected columns: 'Wavelength' and 'Intensity' and 'Error'")

    # Convert to numpy array (transpose to match previous format)
    return df[['Wavelength', 'Intensity', 'Error']].to_numpy()

# test data 
fluorescein = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra_files\fluorescein_total_250218.csv")  
nile_red = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra_files\nile_red_total_250218.csv")  
combined_dataset_1 = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra_files\mixed_spectrum_29_fluorescein_250218.csv")  
combined_dataset_2 = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra_files\mixed_spectrum_60_fluorescein_250218.csv")


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
        perturbed_data = data[2] + np.random.normal(0, 0.1 * np.sqrt(np.abs(data[2])), size=data[2].shape)
        result = minimize(fitting_function, initial_guess, args= data,
                          method='SLSQP', bounds=[(0, 1)] * 3, tol=1e-6, 
                          constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                          options={'disp': False})

        optimized_parameters[i] = result.x

    return np.std(optimized_parameters, axis=0)

# Constraint function to ensure the sum of spectrum strengths equals 1

def constraint_sum_to_one(variables):
    return np.abs(sum(variables) - 1) - 1e-6 # relaxed constraint


def process_spectra(directory, data_type="simulated", num_simulations = 1000, noise_scale = 100):
    """
    Processes spectra data from a directory
    """
    
    if data_type == "simulated":
        files = glob.glob(f"{directory}/mixed_spectrum_*.csv")
        fluorescein_file = glob.glob(f"{directory}/fluorescein_total_*.csv")
        nile_red_file = glob.glob(f"{directory}/nile_red_total_*.csv")
        read_function = read_csv  # Use read_csv for simulated data
    elif data_type == "experimental":
        files = glob.glob(f"{directory}/mixed_spectrum*.txt")
        fluorescein_file = glob.glob(f"{directory}/fluorescein*.txt")
        nile_red_file = glob.glob(f"{directory}/nile_red*.txt")
        read_function = import_arrays  # Use import_arrays for experimental data
    else:
        raise ValueError("Invalid data_type. Choose 'simulated' or 'experimental'.")

    if not fluorescein_file or not nile_red_file:
        raise FileNotFoundError("Fluorescein or Nile Red spectrum file not found.")

    fluorescein = read_function(fluorescein_file[0]) #- take first file in list found via glob.glob
    nile_red = read_function(nile_red_file[0])
    fluorescein_y = fluorescein[:,1]
    nile_red_y = nile_red[:,1]


    # Normalize spectra
    fluorescein_y_normalised = (fluorescein_y - min(fluorescein_y)) / (max(fluorescein_y) - min(fluorescein_y))
    nile_red_y_normalised = (nile_red_y - min(nile_red_y)) / (max(nile_red_y) - min(nile_red_y))

    initial_guess = np.array([0.45, 0.45, 0.1])
    
    results = [] 

    for file in files:
        dataset = read_function(file)
        combined_y = dataset[:,1]
        combined_y_normalised = (combined_y - min(combined_y)) / (max(combined_y) - min(combined_y) + epsilon)
  
        coef = np.array([fluorescein_y_normalised, nile_red_y_normalised, combined_y_normalised])

        #print(coef)

        result = minimize(fit_spectra_strength, initial_guess, args=(coef,),
                          method='SLSQP', bounds=[(0, 1)] * 3, tol=1e-6,
                          constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                          options={'disp': True})

        parameter_errors = monte_carlo_estimate_errors(fit_spectra_strength, initial_guess, coef,
                                                         num_simulations=num_simulations, noise_scale=noise_scale)
        
        combined_spectrum_model = result.x[0] * fluorescein_y_normalised + result.x[1] * nile_red_y_normalised
        
        reduced_chi_sq = chi_squared_reduced(combined_y_normalised, combined_spectrum_model, combined_y_normalised[10]**0.5)
        
        spectrum_number = os.path.basename(file).split("_")[2].split(".")[0]

        print(f'Optimized strength for {file} - Fluorescein: {result.x[0]} ± {parameter_errors[0]}, '
              f'Nile red: {result.x[1]} ± {parameter_errors[1]}, '
              f'Background: {result.x[2]} ± {parameter_errors[2]}')
        print(f'chi squared dataset 1 = {chi_squared(combined_y, combined_spectrum_model, combined_y[10]**(0.5))}')
        print(f'reduced chi squared = {chi_squared_reduced(combined_y_normalised, combined_spectrum_model, combined_y_normalised[10]**(0.5))}')
        
        # Store results
        results.append([
            spectrum_number,
            f"{result.x[0]:.4f} ± {parameter_errors[0]:.4f}",
            f"{result.x[1]:.4f} ± {parameter_errors[1]:.4f}",
            f"{reduced_chi_sq:.4f}"
        ])

        # Convert results to a DataFrame and save as CSV
        results_df = pd.DataFrame(results, columns=[
            "Fraction of Fluorescein (%)",
            "Fluorescein Value ± Error",
            "Nile Red Value ± Error",
           "Reduced Chi Squared"
        ])
        
        output_file = os.path.join(directory, "spectra_analysis_results.csv")
        results_df.to_csv(output_file, index=False)

# Example usage:
process_spectra("C:\\Users\\wlmd95\\OneDrive - Durham University\\Documents\\PhD\\microscope\\data_analysis\\spectra_files")
