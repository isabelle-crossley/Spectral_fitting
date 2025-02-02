# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:58:50 2024

@author: isabe
"""

import os 
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import re
import csv


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
    data = []
    
    # Open the file and read the data
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        
        for row in reader:
            try:
                data.append([float(row[0]), float(row[1])])  # Convert to float and append
            except ValueError:
                continue  # Skip rows with non-numeric values

    # Convert to a numpy array and transpose it to have 2 rows
    data_array = np.array(data).T
    
    return data_array

# test data 
fluorescein = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra files\fluorescein_total_txt_csv")  
nile_red = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra files\nile_red_total_txt_csv")  
combined_dataset_1 = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra files\mixed_spectrum_21_fluorescein_txt_csv")  
combined_dataset_2 = read_csv(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\spectra files\mixed_spectrum_32_fluorescein_txt_csv")

print(fluorescein)

#fluorescein = import_arrays(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240122\fluorescein_test.txt")  # Replace with the actual file path
#nile_red = import_arrays(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240122\pink_fluoro_txt")  # Replace with the actual file path
#combined_dataset_2 = import_arrays(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240123\nile_red_2_txt")  # Replace with the actual file path
#combined_dataset_1 = import_arrays(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\isabelle_microscope_test\240125\mixed_13_txt")
#kidney = import_arrays(r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\microscope_data\\240311\kidney.txt")  # Replace with the actual file pat

#print(fluorescein)
#print(nile_red)
#print(combined_dataset_2)
#print(combined_dataset_1)

#fitting process
def fit_spectra_strength(variables, coef):
    spectrum1_strength, spectrum2_strength = variables  # Unpacking both variables

    fluorescein, nile_red, combined_spectrum = coef  # Unpack coefficients
    
    # Combine the pure spectra according to the provided strengths
    combined_spectrum_model = spectrum1_strength * fluorescein + spectrum2_strength * nile_red

    # Calculate chi-squared value
    chi2 = chi_squared(combined_spectrum, combined_spectrum_model, combined_spectrum[1]**(0.5))
    
    return chi2

#CHOOSE EITHER BOOTSTRAP OR MONTE CARLO

def monte_carlo_estimate_errors(fitting_function, initial_guess, data, num_simulations=1000, noise_scale = 1000):
    optimized_parameters = []
    for _ in range(num_simulations):
        # Perturb the data by adding random noise
        noise = np.random.normal(loc=0, scale=noise_scale, size=data.shape)  # Adjust scale as needed
        perturbed_data = data + noise
        perturbed_data = np.array(perturbed_data)
        coef = np.array([fluorescein, nile_red, perturbed_data])
        
        # Perform the optimization process on the perturbed data
        result = minimize(fitting_function, initial_guess, args=coef,
                          method='SLSQP', bounds = [(0, 1), (0, 1)], tol=1e-6, #TNC needed for bounding
                          constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                          options={'disp': False})
        
        # Store the optimized parameters
        optimized_parameters.append(result.x[0])  # Assuming there's only one parameter
        
    # Calculate the standard deviation of the optimized parameters
    parameter_errors = np.std(optimized_parameters)
    
    return parameter_errors

# Constraint function to ensure the sum of spectrum strengths equals 1
'''
def constraint_sum_to_one(variables):
    spectrum1_strength, spectrum2_strength = variables
    return spectrum1_strength + spectrum2_strength - 1
'''
def constraint_sum_to_one(variables):
    return sum(variables) - 1

#print('fluoro', fluorescein)
plt.plot(fluorescein[0], fluorescein[1], color = 'g')
plt.plot(nile_red[0], nile_red[1], color = 'r')
plt.plot(combined_dataset_1[0], combined_dataset_1[1], color = 'b')
plt.plot(combined_dataset_2[0], combined_dataset_2[1], color = 'darkblue')
plt.show()

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
initial_guess = np.array([0.5,0.5])

coef = np.array([fluorescein_y_normalised, nile_red_y_normalised, combined_dataset_1y_normalised])
#print(coef)
coef_2 = np.array([fluorescein_y_normalised, nile_red_y_normalised, combined_dataset_2y_normalised])

num_simulations = 1000
noise_scale  = 100

# Perform the fitting using the minimize function from scipy.optimize for combined_dataset_1
'''
result_1 = minimize(fit_spectra_strength, initial_guess, args=coef,
                    method='TNC', bounds = [(0, 1), (0, 1)], tol=1e-6,  
                    constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                    options={'disp': True}
                    )
'''
# Perform the fitting using the minimize function from scipy.optimize for combined_dataset_1


result_1 = minimize(fit_spectra_strength, initial_guess, args=coef,
                    method='SLSQP', bounds = [(0, 1), (0, 1)], tol=1e-6,
                    constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                    options={'disp': True}
                    )


#print('result 1', result_1.x)

# Perform Monte Carlo error estimation for combined_dataset_1
parameter_errors_1 = monte_carlo_estimate_errors(fit_spectra_strength, initial_guess, combined_dataset_1,
                                                 num_simulations=num_simulations, noise_scale=noise_scale)

# Perform the fitting using the minimize function from scipy.optimize for combined_dataset_2
'''
result_2 = minimize(fit_spectra_strength, initial_guess, args=coef_2,
                    method='TNC', bounds = [(0, 1), (0, 1)], tol=1e-6,
                    constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                    options={'disp': True}
                    )
'''

result_2 = minimize(fit_spectra_strength, initial_guess, args=coef_2,
                    method='SLSQP', bounds = [(0, 1), (0, 1)], tol=1e-6,
                    constraints={'type': 'eq', 'fun': constraint_sum_to_one},
                    options={'disp': True}
                    )



# Perform Monte Carlo error estimation for combined_dataset_2
parameter_errors_2 = monte_carlo_estimate_errors(fit_spectra_strength, initial_guess, combined_dataset_2,
                                                 num_simulations=num_simulations, noise_scale=noise_scale)

# Extract the optimized strengths for each dataset
optimized_strength_spectrum1_1 = result_1.x[0]
optimized_strength_spectrum2_1 = 1 - optimized_strength_spectrum1_1  # Calculating the second strength for dataset 1

optimized_strength_spectrum1_2 = result_2.x[0]
optimized_strength_spectrum2_2 = 1 - optimized_strength_spectrum1_2  # Calculating the second strength for dataset 2

# Print the optimized strengths and their errors for each dataset
print(f'Optimized strength for dataset 1 - Fluorescein: {optimized_strength_spectrum1_1} ± {parameter_errors_1}, Nile red: {optimized_strength_spectrum2_1} ± {parameter_errors_1}')
print(f'Optimized strength for dataset 2 - Fluroescein: {optimized_strength_spectrum1_2} ± {parameter_errors_2}, Nile red: {optimized_strength_spectrum2_2} ± {parameter_errors_2}')


fluorescein, nile_red, combined_spectrum = coef 
#print('fluoro', fluorescein)
# Combine the pure spectra according to the provided strengths
combined_spectrum_model = optimized_strength_spectrum1_1 * fluorescein + optimized_strength_spectrum2_1 * nile_red
#print('model', combined_spectrum_model)
# Calculate chi-squared value
chi2 = chi_squared(combined_spectrum, combined_spectrum_model, (combined_spectrum))
#print('comb', combined_spectrum)

print(f'chi squared = {chi_squared(combined_spectrum, combined_spectrum_model, combined_spectrum[10]**(0.5))}')
print(f'reduced chi squared = {chi_squared_reduced(combined_spectrum, combined_spectrum_model, combined_spectrum[10]**(0.5))}')


#new version for github desktop