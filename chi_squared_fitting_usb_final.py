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
from scipy.stats import chi2


epsilon = 0.001


def chi_squared(observed, expected, error):
    return np.sum((observed - expected) ** 2 / (error**2 + epsilon))

def chi_squared_reduced(observed, expected, error):
    return chi_squared(observed, expected, error)/np.size(observed)

# Load pure spectra and combined datasets
def import_arrays(filename):
    # Read the text from file
    with open(filename, 'r') as file:
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
    return np.abs(sum(variables) - 1) - 1e-3 # relaxed constraint

def plot_chi_squared_contours(best_fit_params, hess_inv_matrix, param_names, confidence_levels=[0.68, 0.95, 0.997]):
    """
    Plots chi-squared confidence contours for two parameters.

    Parameters:
    - result: The result object from scipy.optimize.minimize.
    - param_names: A list of two parameter names for axis labels.
    - confidence_levels: Confidence levels corresponding to 1σ, 2σ, and 3σ.
    """
    
    param_uncertainties = np.sqrt(np.diag(hess_inv_matrix))  # Standard errors

    # Create a grid around the best-fit parameters
    num_points = 100
    p1_range = np.linspace(best_fit_params[0] - 3*param_uncertainties[0], 
                           best_fit_params[0] + 3*param_uncertainties[0], num_points)
    p2_range = np.linspace(best_fit_params[1] - 3*param_uncertainties[1], 
                           best_fit_params[1] + 3*param_uncertainties[1], num_points)
    
    P1, P2 = np.meshgrid(p1_range, p2_range)

    # Compute chi-squared values
    delta_chi2 = np.zeros_like(P1)
    cov_inv = np.linalg.inv(hess_inv_matrix[:2, :2])  # Inverse of 2x2 covariance submatrix

    for i in range(num_points):
        for j in range(num_points):
            delta = np.array([P1[i, j] - best_fit_params[0], 
                              P2[i, j] - best_fit_params[1]])
            delta_chi2[i, j] = delta.T @ cov_inv @ delta  # Quadratic form

    # Confidence levels based on chi-squared distribution for 2 parameters
    chi2_levels = chi2.ppf(confidence_levels, df=2)

    # Plot contours
    plt.figure(figsize=(7, 6))
    contour = plt.contour(P1, P2, delta_chi2, levels=chi2_levels, colors=['blue', 'green', 'red'])
    plt.clabel(contour, fmt={chi2_levels[0]: '1σ', chi2_levels[1]: '2σ', chi2_levels[2]: '3σ'}, inline=True)
    
    # Plot best-fit point
    plt.scatter(best_fit_params[0], best_fit_params[1], color='black', marker='x', label='Best Fit')
    
    # Labels and title
    plt.xlabel(param_names[0])
    plt.ylabel(param_names[1])
    plt.title("Chi-Squared Confidence Contours")
    plt.legend()
    plt.grid(True)
    plt.show()


#simulated rn
def process_spectra(directory, data_type="simulated", num_simulations = 1000, detector = 'mkid', date = '250306', plot_contours=False):
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
        files = glob.glob(f"{directory}/mixed_{suffix}.txt")
        fluorescein_file = glob.glob(f"{directory}/fluorescein{suffix}.txt")
        nile_red_file = glob.glob(f"{directory}/nile_red{suffix}.txt")
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
                          options={'disp': False})

        parameter_errors = monte_carlo_estimate_errors(fit_spectra_strength, initial_guess, combined_y, error, fluorescein_y, nile_red_y,
                                                         num_simulations=num_simulations)
        
        param_uncertainty = np.sqrt(np.diag(result.hess_inv.todense()))
        hess_inv_matrix = result.hess_inv.todense()  # Convert to dense if it's sparse
        cond_number = np.linalg.cond(hess_inv_matrix)

        
        combined_spectrum_model = result.x[0] * fluorescein_y + result.x[1] * nile_red_y
        
        reduced_chi_sq = chi_squared_reduced(combined_y, combined_spectrum_model, error)    

        
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
            f"{result.x[0]:.4f}",
            f"{parameter_errors[0]:.4f}",
            f"{param_uncertainty[0]:.4f}",
            f"{cond_number:.4f}",
            f"{result.x[1]:.4f}",
            f"{parameter_errors[1]:.4f}",
            f"{param_uncertainty[1]:.4f}",
            f"{cond_number:.4f}",
            f"{reduced_chi_sq:.4f}"
        ])

        # convert results to df and save as csv
        results_df = pd.DataFrame(results, columns=[
            "Fraction of Fluorescein (%)",
            "Fluorescein Value",
            "Monte Carlo Error",
            "minimize error",
            "Condition Number",
            "Nile Red Value",
            "Monte Carlo Error",
            "minimize error",
            "Condition Number",
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
        plt.close()
        
        if plot_contours:
            #plot_chi_squared_contours(output_file, result, param_names=["Fluorescein", "Nile Red"])
            plot_chi_squared_contours(
                best_fit_params=result.x,
                hess_inv_matrix=result.hess_inv.todense(),
                param_names=["Fluorescein", "Nile Red"]
            )
        



#example:
process_spectra("C:\\Users\\wlmd95\\OneDrive - Durham University\\Documents\\PhD\\microscope\\data_analysis\\spectra_files", detector = 'usb', date = '250318', plot_contours=True)

#experimental
def process_single_spectrum(file, fluorescein_file, nile_red_file):
    fluorescein = import_arrays(fluorescein_file)
    nile_red = import_arrays(nile_red_file)
    dataset = import_arrays(file)
    
    fluorescein_y, nile_red_y = fluorescein[1], nile_red[1]
    #combined_y, error = dataset[1], dataset[2]
    combined_y = dataset[1]

    
    initial_guess = [0.5, 0.5]
    coef = np.array([fluorescein_y, nile_red_y, combined_y])
    
    result = minimize(fit_spectra_strength, initial_guess, args=(coef, combined_y**0.5),
                      method='L-BFGS-B', bounds=[(0, 1)] * 2, tol=1e-6, options={'disp': False})
    
    optimized_strengths = result.x
    combined_spectrum_model = optimized_strengths[0] * fluorescein_y + optimized_strengths[1] * nile_red_y
    reduced_chi_sq = chi_squared_reduced(combined_y, combined_spectrum_model, combined_y**0.5)
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(dataset[0], combined_y, 'b-', label='Experimental Data')
    plt.plot(dataset[0], combined_spectrum_model, 'r--', label='Fitted Model')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(f'Spectral Fit (Reduced Chi-Sq: {reduced_chi_sq:.4f})')
    plt.legend()
    plt.grid()
    plt.show()
    
    print(f'Optimized strengths: Fluorescein = {optimized_strengths[0]:.4f}, Nile Red = {optimized_strengths[1]:.4f}')
    print(f'Reduced Chi-Squared: {reduced_chi_sq:.4f}')
    
#experimental 
def process_spectrum_folder(folder_path, fluorescein_file, nile_red_file):
    fluorescein = import_arrays(fluorescein_file)
    nile_red = import_arrays(nile_red_file)
    fluorescein_y, nile_red_y = fluorescein[1], nile_red[1]
    
    files = sorted([f for f in os.listdir(folder_path) if f.startswith("mixed_") and f.endswith(".txt")])
    
    for file in files:
       file_path = os.path.join(folder_path, file)
       dataset = import_arrays(file_path)
       combined_y = dataset[1]
        
       initial_guess = [0.5, 0.5]
       coef = np.array([fluorescein_y, nile_red_y, combined_y])
        
       result = minimize(fit_spectra_strength, initial_guess, args=(coef, combined_y**0.5),
                          method='SLSQP', bounds=[(0, 1)] * 2, tol=1e-6,
                          constraints={'type': 'eq', 'fun': constraint_sum_to_one}, options={'disp': False})
       
        
       optimized_strengths = result.x
       combined_spectrum_model = optimized_strengths[0] * fluorescein_y + optimized_strengths[1] * nile_red_y
       reduced_chi_sq = chi_squared_reduced(combined_y, combined_spectrum_model, combined_y**0.5)
       
       chi_squared_contributions = ((combined_y - combined_spectrum_model) ** 2) / (np.maximum(combined_y**0.5, 1e-4)**2)

       '''
       # Plot results
       plt.figure(figsize=(8, 5))
       plt.plot(dataset[0], combined_y, 'b-', label='Experimental Data')
       plt.plot(dataset[0], combined_spectrum_model, 'r--', label='Fitted Model')
       plt.xlabel('Wavelength')
       plt.ylabel('Intensity')
       plt.title(f'{file} (Reduced Chi-Sq: {reduced_chi_sq:.4f})')
       plt.legend()
       plt.grid()
       plt.show()
       '''
       # Plot results
       fig, ax = plt.subplots(2, 1, figsize=(8, 8))

        # Spectrum fit plot
       ax[0].plot(dataset[0], combined_y, 'b-', label='Experimental Data')
       ax[0].plot(dataset[0], combined_spectrum_model, 'r--', label='Fitted Model')
       ax[0].set_xlabel('Wavelength')
       ax[0].set_ylabel('Intensity')
       ax[0].set_title(f'{file} (Reduced Chi-Sq: {reduced_chi_sq:.4f})')
       ax[0].legend()
       ax[0].grid()
       # Chi-squared contribution plot
       ax[1].plot(dataset[0], chi_squared_contributions, 'g-', label='Chi-Squared Contribution')
       ax[1].set_xlabel('Wavelength')
       ax[1].set_ylabel('Chi-Squared Contribution')
       ax[1].set_title('Per-Point Contribution to Chi-Squared')
       ax[1].legend()
       ax[1].grid()
       plt.show()
       

        
       print(f'File: {file}')
       print(f'Optimized strengths: Fluorescein = {optimized_strengths[0]:.4f}, Nile Red = {optimized_strengths[1]:.4f}')
       print(f'Reduced Chi-Squared: {reduced_chi_sq:.4f}\n')

# test data 
nile_red = r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240122\pink_fluoro_txt.txt"
fluorescein = r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240122\fluorescein_test.txt"  
loc_2 = r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\240125\mixed_1_txt.txt"
loc_3 = r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\240125\mixed_2_txt.txt"
folder_path = r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240125"

#process_single_spectrum(loc_2, fluorescein, nile_red)

#process_spectrum_folder(folder_path, fluorescein, nile_red)



