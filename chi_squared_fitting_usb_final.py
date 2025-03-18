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

epsilon = 0.001


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
fluorescein = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra_files\fluorescein_total_250312.csv")  
nile_red = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra_files\nile_red_total_250312.csv")  
combined_dataset_1 = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra_files\mixed_spectrum_29_fluorescein_250312.csv")  
combined_dataset_2 = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra_files\mixed_spectrum_60_fluorescein_250312.csv")



#fitting process
def fit_spectra_strength(variables, coef, error):
    #spectrum1_strength, spectrum2_strength, background_strength = variables  # Added background_strength
    spectrum1_strength, spectrum2_strength = variables  # Added background_strength

    fluorescein, nile_red, combined_spectrum = coef  
    
    # Combine the pure spectra and background according to the provided strengths
    combined_spectrum_model = (
        spectrum1_strength * fluorescein 
        + spectrum2_strength * nile_red 
        #+ background_strength       # need to sort out background strength
    )

    # Calculate chi-squared value
    #print('here it is!', combined_spectrum)
    #print(np.sqrt(combined_spectrum))
    #chi2 = chi_squared(combined_spectrum, combined_spectrum_model, np.sqrt(combined_spectrum[1]))   # doesnt work coz of square root, error bigger than data
    chi2 = chi_squared(combined_spectrum, combined_spectrum_model, error)
    
    return chi2


def monte_carlo_estimate_errors(fitting_function, initial_guess, data, error, fluorescein, nile_red, num_simulations=1000):
    parameter_samples = []


    for i in range(num_simulations):
        #perturbed_data = np.random.normal(loc=0, scale = noise_scale, size=data[2].shape)
        noise = np.random.normal(loc=0, scale=np.sqrt(np.abs(data)), size= data.shape  )
        perturbed_data = data + noise
        perturbed_data = np.array(perturbed_data)
    
        min_length = min(len(fluorescein[:]), len(nile_red[:]), len(perturbed_data))
        fluorescein = fluorescein[:min_length]
        nile_red = nile_red[:min_length]
        perturbed_data = perturbed_data[:min_length]
        coef = np.array([fluorescein[:], nile_red[:], perturbed_data])
        
        result = minimize(fitting_function, initial_guess, args=(coef, error),
                          method='L-BFGS-B', bounds=[(0, 1)] * 2, tol=1e-6, 
                          #constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                          options={'disp': False})


        parameter_samples.append(result.x)
    parameter_samples = np.array(parameter_samples)
    parameter_errors = np.std(parameter_samples, axis=0)

    return parameter_errors

# Constraint function to ensure the sum of spectrum strengths equals 1

def constraint_sum_to_one(variables):
    return np.abs(sum(variables) - 1) - 1e-2 # relaxed constraint


def process_spectra(directory, data_type="simulated", num_simulations = 1000, detector = 'mkid', date = '250306'):
    """
    processes spectra data from a directory
    """
    
    #determine whether mkid or usb
    suffix = "_mkid" if detector == "mkid" else ""
    
    # Add date filtering to the glob pattern
    date_pattern = f"*_{date}" if date else "*"
    
    if data_type == "simulated":
        files = glob.glob(f"{directory}/mixed_spectrum{suffix}_{date_pattern}.csv")
        fluorescein_file = glob.glob(f"{directory}/fluorescein{suffix}_{date_pattern}.csv")
        nile_red_file = glob.glob(f"{directory}/nile_red{suffix}_{date_pattern}.csv")
        read_function = read_csv  # Use read_csv for simulated data
    elif data_type == "experimental":
        files = glob.glob(f"{directory}/mixed_spectrum{suffix}_{date_pattern}.txt")
        fluorescein_file = glob.glob(f"{directory}/fluorescein{suffix}_{date_pattern}.txt")
        nile_red_file = glob.glob(f"{directory}/nile_red{suffix}_{date_pattern}.txt")
        read_function = import_arrays  # Use import_arrays for experimental data
    else:
        raise ValueError("Invalid data_type. Choose 'simulated' or 'experimental'.")

    if not fluorescein_file or not nile_red_file:
        raise FileNotFoundError("Fluorescein or Nile Red spectrum file not found.")

    fluorescein = read_function(fluorescein_file[0]) #- take first file in list found via glob.glob
    nile_red = read_function(nile_red_file[0])
    fluorescein_y = fluorescein[:,1]
    fluorescein_err = fluorescein[:,2]
    nile_red_y = nile_red[:,1]
    nile_red_err = nile_red[:,2]

    #initial_guess = np.array([0.475, 0.475, 0.05])
    initial_guess = np.array([0.5, 0.5])

    
    results = [] 

    for file in files:
        dataset = read_function(file)
        combined_y = dataset[:,1]
        error = dataset[:,2]
        
  
        coef = np.array([fluorescein_y, nile_red_y, combined_y])

        #print(coef)

        result = minimize(fit_spectra_strength, initial_guess, args=(coef, error),
                          method='L-BFGS-B', bounds=[(0, 1)] * 2, tol=1e-6,
                          constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                          options={'disp': True})

        parameter_errors = monte_carlo_estimate_errors(fit_spectra_strength, initial_guess, combined_y, error, fluorescein_y, nile_red_y,
                                                         num_simulations=num_simulations)
        
        combined_spectrum_model = result.x[0] * fluorescein_y + result.x[1] * nile_red_y
        
        reduced_chi_sq = chi_squared_reduced(combined_y, combined_spectrum_model, error)  # fix this
        
        spectrum_number = os.path.basename(file).split("_")[2].split(".")[0]

        print(f'Optimized strength for {file} - Fluorescein: {result.x[0]} ± {parameter_errors[0]}, '
              f'Nile red: {result.x[1]} ± {parameter_errors[1]}, '
              #f'Background: {result.x[2]} ± {parameter_errors[2]}')
              )
        print(f'chi squared dataset 1 = {chi_squared(combined_y, combined_spectrum_model,  error)}')
        print(f'reduced chi squared = {reduced_chi_sq}')
        
        #store results
        results.append([
            spectrum_number,
            f"{result.x[0]:.4f} ± {parameter_errors[0]:.4f}",
            f"{result.x[1]:.4f} ± {parameter_errors[1]:.4f}",
           # f"{result.x[2]:.4f} ± {parameter_errors[2]:.4f}",
            f"{reduced_chi_sq:.4f}"
        ])

        # convert results to df and save as csv
        results_df = pd.DataFrame(results, columns=[
            "Fraction of Fluorescein (%)",
            "Fluorescein Value ± Error",
            "Nile Red Value ± Error",
            #"Background Value ± Error",
           "Reduced Chi Squared"
        ])
        
        output_file = f"spectra_analysis_results_{date}.csv" if date else "spectra_analysis_results.csv"
        results_df.to_csv(output_file, index=False)
        
        spectrum_number = os.path.basename(file).split("_")[2].split(".")[0]

        # Contribution to chi-squared for each wavelength
        chi_sq_contribution = ((combined_y - combined_spectrum_model) ** 2) / (error**2 + epsilon)

        # Plot chi-squared contributions
        plt.figure(figsize=(8, 5))
        plt.scatter(dataset[:, 0], chi_sq_contribution, marker='x', linestyle='-', color='r', label='Chi-squared Contribution')
        plt.axhline(y=1, color='darkblue', linestyle='--')
        plt.xlabel('Wavelength')
        plt.ylabel('Chi-squared Contribution')
        plt.title(f'Chi-squared Contribution per Wavelength (Dataset {spectrum_number})')
        plt.legend()
        plt.grid()
        plt.show()
        #plt.savefig(os.path.join(directory, f'chi_squared_contribution_{spectrum_number}.png'))
        plt.close()
        
        #print("Combined Y :", combined_y)
        #print("Error :", error)
        #print("Combined Spectrum Model:", combined_spectrum_model)
        #print("Chi-Squared Contribution:", ((combined_y - combined_spectrum_model) ** 2) / (error**2 + epsilon))


#example:
process_spectra("C:\\Users\\wlmd95\\OneDrive - Durham University\\Documents\\PhD\\microscope\\data_analysis\\spectra_files", detector = 'usb', date = '250317')


#chi squared contribution
