# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:16:43 2025

@author: wlmd95
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import os
from datetime import datetime
import pandas as pd

#https://omlc.org/spectra/PhotochemCAD/html/007.html source for spectra

df_f = np.loadtxt(r'C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\fluorescein_emission.txt')
df_nr = np.loadtxt(r'C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\microscope\data_analysis\nile_red_emission.txt')

wavelength_nr = df_nr[:, 0][20:-200]  
intensity_nr = df_nr[:, 1][20:-200] 


wavelength_f = df_f[:, 0]  
intensity_f = df_f[:, 1]



# Function to add noise to spectra
def add_noise(spectrum, quantum_efficiency=0.8, dark_current=0.1, read_noise=0.15, integration_time=10, background = 20):
    """
    Adds simulated noise to a spectrum

    Parameters:
    spectrum (array): original spectrum 
    quantum_efficiency (float): Detector quantum efficiency 0.8
    dark_current (float): dark current noise level, electrons per sec per pixel 0.1
    read_noise (float): readout noise 0.15
    integration_time (float): int time, seconds 100

    Returns:
    array: Noisy spectrum
    """
    # Add background contribution
    signal_with_background = spectrum + background

    # Convert to detected photon counts based on quantum efficiency
    detected_signal = np.random.poisson(lam=quantum_efficiency * np.maximum(signal_with_background, 1))

    # Dark current noise (Poisson distributed)
    dark_counts = np.random.poisson(lam=dark_current * integration_time, size=spectrum.shape)

    # Readout noise (Gaussian)
    readout_noise = np.random.normal(scale=read_noise, size=spectrum.shape)

    # Total noisy signal
    noisy_spectrum = detected_signal + dark_counts + readout_noise

    # Expected noise estimate
    expected_shot_noise = np.sqrt(quantum_efficiency * np.maximum(signal_with_background, 1))
    #print('shot noise', expected_shot_noise)
    expected_dark_noise = np.sqrt(dark_current * integration_time)
    print('dark', expected_dark_noise)
    expected_background_noise = np.sqrt(background)  # Poisson noise from background
    print(expected_background_noise)
    expected_noise = np.sqrt(expected_shot_noise**2 + expected_dark_noise**2 + read_noise**2 + expected_background_noise**2)

    return noisy_spectrum, expected_noise

# Wavelength range (in nm)
wavelengths = np.linspace(480, 700, 441)

#print(wavelengths_f)

# Normalize the spectra to make the excitation peaks 100
fluorescein_emission = intensity_f / np.max(intensity_f) * 10000
nile_red_emission = intensity_nr / np.max(intensity_nr) * 10000

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the noisy spectra
ax.plot(wavelengths, fluorescein_emission, label="Fluorescein", color='green')
ax.plot(wavelengths, nile_red_emission, label="Nile Red", color='red')


fluorescein_error = add_noise(fluorescein_emission)[1]*1.5
nile_red_error = add_noise(nile_red_emission)[1]*1.5


# Formatting
ax.set_ylim(0, 12000)
ax.set_title('Noisy Emission Spectra')
ax.set_xlabel('Wavelength (nm)')
ax.set_xlim(400, 800)
ax.set_ylabel('Intensity')
ax.legend()
ax.grid(True)

plt.show()

fluorescein_noisy = add_noise(fluorescein_emission)[0]
nile_red_noisy = add_noise(nile_red_emission)[0]

# --- Third Plot: Mixtures of Fluorescein and Nile Red ---
ratios = np.linspace(1.0, 0, 11)  # 100% fluorescein to 0%
mixed_spectra_noisy = [(r * fluorescein_noisy + (1 - r) * nile_red_noisy) for r in ratios]
mixed_error = [np.sqrt((r * fluorescein_error) ** 2 + ((1 - r) * nile_red_error) ** 2) for r in ratios]
#print(mixed_spectra_noisy[1])
#print(mixed_error[1])

# Plot 3x3 grid of noisy mixed spectra
fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
for ax, ratio, spectrum, error in zip(axes.ravel(), ratios, mixed_spectra_noisy, mixed_error):
    ax.plot(wavelengths, spectrum, color='purple')
    ax.fill_between(wavelengths, spectrum - error, spectrum + error, color='purple', alpha=0.3, label='Error')
    ax.set_title(f'{int(ratio * 100)}% Fluorescein')
    ax.grid(True)

fig.suptitle('Noisy Fluorescein-Nile Red Mixed Spectra', fontsize=16)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


def save_spectrum(filename, wavelengths, intensity, error): #add error as a variable soon
    """
    Saves wavelength & intensity arrays to a CSV file inside the 'spectra files' folder,
    appending the current date (YYMMDD) to the filename.

    Parameters:
    - filename (str): Name of the output CSV file (automatically appends .csv if missing).
    - wavelengths (array-like): Wavelength values.
    - intensity (array-like): Corresponding intensity values.
    """
    # Get today's date in YYMMDD format
    date_str = datetime.today().strftime("%y%m%d")

    # Ensure filename has a .csv extension
    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    # Insert date before file extension
    name, ext = os.path.splitext(filename)
    filename = f"{name}_{date_str}{ext}"

    # Create directory if it doesn't exist
    folder_name = "spectra_files"
    os.makedirs(folder_name, exist_ok=True)

    # Sanitize filename to prevent invalid characters
    filename = "".join(c if c.isalnum() or c in (" ", "_", "-", ".") else "_" for c in filename)

    # Define file path
    file_path = os.path.join(folder_name, filename)

    try:
        # Stack and save as CSV
        data = np.column_stack((wavelengths, intensity, error))
        np.savetxt(file_path, data, delimiter=",", header="Wavelength,Intensity,Error", comments='')

        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

# Save summed spectra
save_spectrum("fluorescein_total.csv", wavelengths, fluorescein_emission * 0.8, fluorescein_error)
save_spectrum("nile_red_total.csv", wavelengths, nile_red_emission *0.8, nile_red_error)

# Save mixed spectra with labels
for i, (ratio, spectrum, error) in enumerate(zip(ratios, mixed_spectra_noisy, mixed_error)):
    filename = f"mixed_spectrum_{int(ratio * 100)}_fluorescein.csv"
    save_spectrum(filename, wavelengths, spectrum, error)

print("Spectra saved successfully.")