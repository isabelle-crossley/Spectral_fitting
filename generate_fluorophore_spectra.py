# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:16:43 2025

@author: wlmd95
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import os

# Define function to generate Voigt profiles (Gaussian + Lorentzian)
def voigt(x, center, fwhm_g, fwhm_l, amplitude=1):
    sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))  # Convert Gaussian FWHM to standard deviation
    gamma = fwhm_l / 2  # Lorentzian half-width at half-maximum
    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

# Function to add noise to spectra
def add_noise(spectrum, quantum_efficiency=0.1, dark_current=0, read_noise=0, integration_time=1, background = 5):
    """
    Adds simulated noise to a spectrum.

    Parameters:
    spectrum (array): The original spectrum (photon counts or arbitrary units).
    quantum_efficiency (float): Detector quantum efficiency (fraction, e.g., 0.9).
    dark_current (float): Dark current noise level (electrons per second per pixel).
    read_noise (float): Readout noise (electrons per pixel).
    integration_time (float): Integration time in seconds.

    Returns:
    array: Noisy spectrum
    """
    # Add background contribution
    signal_with_background = spectrum + background
    
    # Shot noise (Photon noise, poisson)
    shot_noise = np.random.normal(scale=np.sqrt(quantum_efficiency * np.abs(spectrum)))

    # Dark current noise (Poisson-like, approximated as Gaussian)
    dark_noise = np.random.normal(scale=np.sqrt(dark_current * integration_time), size=spectrum.shape)

    # Readout noise (Fixed Gaussian noise)
    readout_noise = np.random.normal(scale=read_noise, size=spectrum.shape)

    # Total noise
    total_noise = np.sqrt(shot_noise**2 + dark_noise**2 + readout_noise**2)

    # Add noise to the original spectrum
    noisy_spectrum = signal_with_background + total_noise

    return noisy_spectrum

# Wavelength range (in nm)
wavelengths = np.linspace(400, 800, 201) 
# Fluorescein parameters
fluorescein_excitation_peak = 498
fluorescein_emission_peak = 517
fluorescein_exc_fwhm_g = 15  # Gaussian FWHM for excitation
fluorescein_exc_fwhm_l = 10  # Lorentzian FWHM for excitation
fluorescein_em_fwhm_g = 23  # Gaussian FWHM for emission
fluorescein_em_fwhm_l = 15  # Lorentzian FWHM for emission
fluorescein_qy = 0.79  # Quantum yield EtOH

# Nile Red parameters
nile_red_excitation_peak = 559
nile_red_emission_peak = 635
nile_red_exc_fwhm_g = 70  # Gaussian FWHM for excitation
nile_red_exc_fwhm_l = 30  # Lorentzian FWHM for excitation
nile_red_em_fwhm_g = 50  # Gaussian FWHM for emission
nile_red_em_fwhm_l = 25  # Lorentzian FWHM for emission
nile_red_qy = 0.7  # Quantum yield

# Simulate the spectra using Voigt profiles
fluorescein_excitation = voigt(wavelengths, fluorescein_excitation_peak, fluorescein_exc_fwhm_g, fluorescein_exc_fwhm_l)
fluorescein_emission = voigt(wavelengths, fluorescein_emission_peak, fluorescein_em_fwhm_g, fluorescein_em_fwhm_l) 
nile_red_excitation = voigt(wavelengths, nile_red_excitation_peak, nile_red_exc_fwhm_g, nile_red_exc_fwhm_l)
nile_red_emission = voigt(wavelengths, nile_red_emission_peak, nile_red_em_fwhm_g, nile_red_em_fwhm_l) 

# Normalize the spectra to make the excitation peaks 100
fluorescein_excitation = fluorescein_excitation / np.max(fluorescein_excitation) * 100
fluorescein_emission = fluorescein_emission / np.max(fluorescein_emission) * 100 * fluorescein_qy
nile_red_excitation = nile_red_excitation / np.max(nile_red_excitation) * 100
nile_red_emission = nile_red_emission / np.max(nile_red_emission) * 100 * nile_red_qy

# Apply noise to all the spectra
fluorescein_excitation_noisy = add_noise(fluorescein_excitation)
fluorescein_emission_noisy = add_noise(fluorescein_emission)
nile_red_excitation_noisy = add_noise(nile_red_excitation)
nile_red_emission_noisy = add_noise(nile_red_emission)

# Compute the sum of noisy excitation and emission spectra
fluorescein_total_noisy = fluorescein_excitation_noisy + fluorescein_emission_noisy
nile_red_total_noisy = nile_red_excitation_noisy + nile_red_emission_noisy

# Create the figure and subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# --- First Plot: Individual Spectra ---
axes[0].plot(wavelengths, fluorescein_excitation_noisy, label="Fluorescein Excitation (498 nm)", color='green', linestyle='--')
axes[0].plot(wavelengths, fluorescein_emission_noisy, label="Fluorescein Emission (517 nm)", color='green')
axes[0].plot(wavelengths, nile_red_excitation_noisy, label="Nile Red Excitation (559 nm)", color='red', linestyle='--')
axes[0].plot(wavelengths, nile_red_emission_noisy, label="Nile Red Emission (635 nm)", color='red')

axes[0].set_ylim(0, 125)
axes[0].set_title('Noisy Fluorescence Spectra of Fluorescein and Nile Red (Voigt Profile)')
axes[0].set_xlabel('Wavelength (nm)')
axes[0].set_xlim(400, 800)
axes[0].set_ylabel('Intensity')
axes[0].legend()
axes[0].grid(True)

# --- Second Plot: Summed Noisy Spectra ---
axes[1].plot(wavelengths, fluorescein_total_noisy, label="Fluorescein (Exc + Em)", color='green')
axes[1].plot(wavelengths, nile_red_total_noisy, label="Nile Red (Exc + Em)", color='red')

axes[1].set_ylim(0, 200)
axes[1].set_title('Noisy Summed Excitation and Emission Spectra')
axes[1].set_xlabel('Wavelength (nm)')
axes[1].set_xlim(400, 800)
axes[1].set_ylabel('Summed Intensity')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# --- Third Plot: Mixtures of Fluorescein and Nile Red ---
ratios = np.linspace(1.0, 0, 9)  # 100% fluorescein to 10%
mixed_spectra_noisy = [(r * fluorescein_total_noisy + (1 - r) * nile_red_total_noisy) for r in ratios]

# Plot 3x3 grid of noisy mixed spectra
fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
for ax, ratio, spectrum in zip(axes.ravel(), ratios, mixed_spectra_noisy):
    ax.plot(wavelengths, spectrum, color='purple')
    ax.set_title(f'{int(ratio * 100)}% Fluorescein')
    ax.grid(True)

fig.suptitle('Noisy Fluorescein-Nile Red Mixed Spectra', fontsize=16)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Summed Intensity')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


def save_spectrum(filename, wavelengths, intensity):
    """
    Saves wavelength & intensity arrays to a CSV file inside the 'spectra files' folder.

    Parameters:
    - filename (str): Name of the output CSV file (automatically appends .csv if missing).
    - wavelengths (array-like): Wavelength values.
    - intensity (array-like): Corresponding intensity values.
    """
    # Ensure filename has a .csv extension
    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    # Create directory if it doesn't exist
    folder_name = "spectra files"
    os.makedirs(folder_name, exist_ok=True)

    # Sanitize filename to prevent invalid characters
    #filename = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in filename)
    filename = "".join(c if c.isalnum() or c in (" ", "_", "-", ".") else "_" for c in filename)

    
    # Define file path
    file_path = os.path.join(folder_name, filename)

    try:
        # Stack and save as CSV
        data = np.column_stack((wavelengths, intensity))
        np.savetxt(file_path, data, delimiter=",", header="Wavelength,Intensity", comments='')

        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        
# Save summed spectra
save_spectrum("fluorescein_total.csv", wavelengths, fluorescein_total_noisy)
save_spectrum("nile_red_total.csv", wavelengths, nile_red_total_noisy)

# Save mixed spectra with labels
for i, (ratio, spectrum) in enumerate(zip(ratios, mixed_spectra_noisy)):
    filename = f"mixed_spectrum_{int(ratio * 100)}_fluorescein.csv"
    save_spectrum(filename, wavelengths, spectrum)

print("Spectra saved successfully.")