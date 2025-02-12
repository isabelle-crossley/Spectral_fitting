# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:28:48 2025

@author: wlmd95
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.ndimage import gaussian_filter1d
import os

# Define function to generate Voigt profiles (Gaussian + Lorentzian)
def voigt(x, center, fwhm_g, fwhm_l, amplitude=1):
    sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))  # Convert Gaussian FWHM to standard deviation
    gamma = fwhm_l / 2  # Lorentzian half-width at half-maximum
    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

# Function to add MKID-style noise
def add_mkid_noise(spectrum, background=10, wavelength_range=None, read_noise=0, tls_noise_level=0.02, temp_fluct=0.005, gr_noise_factor=0.1):
    """
    Adds noise characteristic of an MKID detector.
    
    Parameters:
    - spectrum: Input spectrum (array-like)
    - background: Background noise level (default: 10)
    - wavelength_range: Wavelength values corresponding to the spectrum (optional)
    - read_noise: Fixed Gaussian readout noise
    - tls_noise_level: Amplitude of two-level system (TLS) noise
    - temp_fluct: Amplitude of thermal fluctuation noise
    - gr_noise_factor: Scaling factor for generation-recombination (GR) noise
    
    Returns:
    - Noisy spectrum with added MKID-style noise
    """
    if wavelength_range is None:
        wavelength_range = np.linspace(400, 800, len(spectrum))
    
    # Background noise increases with wavelength
    background_noise = background * (1 + (wavelength_range - 600) / 500) 
    background_noise = np.clip(background_noise, 0, None)  

    # Shot noise (Poisson-like)
    shot_noise = np.random.normal(scale=np.sqrt(np.abs(spectrum)))
    
    # Readout noise (Fixed Gaussian noise)
    readout_noise = np.random.normal(scale=read_noise, size=spectrum.shape)

    # Thermal fluctuation noise (low-amplitude random variations)
    thermal_noise = temp_fluct * np.random.normal(size=spectrum.shape)

    # Two-Level System (TLS) Noise (1/f-like fluctuations)
    tls_noise = tls_noise_level * np.random.normal(size=spectrum.shape) * (wavelength_range / np.min(wavelength_range))**-0.5  

    # Generation-Recombination (GR) Noise (scales with sqrt(N_qp))
    gr_noise = gr_noise_factor * np.random.normal(scale=np.sqrt(np.abs(spectrum + background_noise)))

    # Total noise (quadrature sum)
    total_noise = np.sqrt(shot_noise**2 + readout_noise**2 + thermal_noise**2 + tls_noise**2 + gr_noise**2) + background_noise

    # Apply noise
    noisy_spectrum = spectrum + total_noise 
    return noisy_spectrum

# Function to add MKID-like noise and resolution effects
def apply_mkid_resolution(spectrum, wavelengths, base_R=10):
    """
    Applies MKID spectral resolution, dynamically setting FWHM based on R = λ / Δλ.
    """
    fwhm_values = wavelengths / base_R  # Calculate FWHM at each wavelength
    std_dev = fwhm_values / (2 * np.sqrt(2 * np.log(2)))  # Convert to standard deviation
    spectrum_mkid = gaussian_filter1d(spectrum, std_dev.mean())  # Apply Gaussian smoothing
    
    # Add a wavelength-dependent background tail at longer wavelengths
    background_tail = 5 * (wavelengths / wavelengths.max()) ** 2
    return spectrum_mkid + background_tail

# Wavelength range (in nm)
wavelengths = np.linspace(400, 800, 200)

# Define Fluorescein spectrum parameters
fluorescein_params = {
    'excitation_peak': 498, 'emission_peak': 517,
    'exc_fwhm_g': 15, 'exc_fwhm_l': 10,
    'em_fwhm_g': 23, 'em_fwhm_l': 15,
    'qy': 0.79
}

# Generate Fluorescein excitation and emission spectra
fluorescein_exc = voigt(wavelengths, fluorescein_params['excitation_peak'], fluorescein_params['exc_fwhm_g'], fluorescein_params['exc_fwhm_l'])
fluorescein_em = voigt(wavelengths, fluorescein_params['emission_peak'], fluorescein_params['em_fwhm_g'], fluorescein_params['em_fwhm_l'])
fluorescein_exc = fluorescein_exc / np.max(fluorescein_exc) * 100
fluorescein_em = fluorescein_em / np.max(fluorescein_em) * 100 * fluorescein_params['qy']
fluorescein_spectrum = fluorescein_exc + fluorescein_em

# Apply MKID noise and resolution effects
fluorescein_spectrum_noisy = add_mkid_noise(fluorescein_spectrum, wavelength_range=wavelengths)
fluorescein_spectrum_mkid = apply_mkid_resolution(fluorescein_spectrum_noisy, wavelengths)

# Define Nile Red spectrum parameters
nile_red_params = {
    'excitation_peak': 559, 'emission_peak': 635,
    'exc_fwhm_g': 70, 'exc_fwhm_l': 30,
    'em_fwhm_g': 50, 'em_fwhm_l': 25,
    'qy': 0.7
}

# Generate Nile Red excitation and emission spectra
nile_red_exc = voigt(wavelengths, nile_red_params['excitation_peak'], nile_red_params['exc_fwhm_g'], nile_red_params['exc_fwhm_l'])
nile_red_em = voigt(wavelengths, nile_red_params['emission_peak'], nile_red_params['em_fwhm_g'], nile_red_params['em_fwhm_l'])
nile_red_exc = nile_red_exc / np.max(nile_red_exc) * 100
nile_red_em = nile_red_em / np.max(nile_red_em) * 100 * nile_red_params['qy']
nile_red_spectrum = nile_red_exc + nile_red_em

# Apply MKID noise and resolution effects
nile_red_spectrum_noisy = add_mkid_noise(nile_red_spectrum, wavelength_range=wavelengths)
nile_red_spectrum_mkid = apply_mkid_resolution(nile_red_spectrum_noisy, wavelengths)

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))

# Fluorescein
ax.plot(wavelengths, fluorescein_spectrum, label='Fluorescein Original Spectrum', linestyle='--', color='green')
ax.plot(wavelengths, fluorescein_spectrum_mkid, label='Fluorescein MKID-Simulated Spectrum', color='blue')

# Nile Red
ax.plot(wavelengths, nile_red_spectrum, label='Nile Red Original Spectrum', linestyle='--', color='red')
ax.plot(wavelengths, nile_red_spectrum_mkid, label='Nile Red MKID-Simulated Spectrum', color='orange')

# Labels and title
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity')
ax.set_title('Simulated Fluorescence Spectra with MKID Resolution')
ax.legend()
ax.grid(True)
plt.show()

# Define mixture ratios (from 100% Fluorescein to 10%)
ratios = np.linspace(1.0, 0, 11)

# Generate mixed spectra for MKID resolution
mixed_spectra_mkid = [(r * fluorescein_spectrum_mkid + (1 - r) * nile_red_spectrum_mkid) for r in ratios]

# Plot 3x3 grid of MKID-simulated mixed spectra
fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
for ax, ratio, spectrum in zip(axes.ravel(), ratios, mixed_spectra_mkid):
    ax.plot(wavelengths, spectrum, color='purple')
    ax.set_title(f'{int(ratio * 100)}% Fluorescein')
    ax.grid(True)

fig.suptitle('MKID-Simulated Fluorescein-Nile Red Mixed Spectra', fontsize=16)
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
save_spectrum("fluorescein_mkid.csv", wavelengths, fluorescein_spectrum_mkid)
save_spectrum("nile_red_mkid.csv", wavelengths, nile_red_spectrum_mkid)

# Save mixed spectra with labels
for i, (ratio, spectrum) in enumerate(zip(ratios, mixed_spectra_mkid)):
    filename = f"mixed_spectrum_{int(ratio * 100)}_fluorescein_mkid.csv"
    save_spectrum(filename, wavelengths, spectrum)

print("Spectra saved successfully.")

