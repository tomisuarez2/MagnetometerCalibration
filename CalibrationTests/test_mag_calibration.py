"""
Magnetometer calibration test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "Iterative calibration method for inertial and magnetic sensors" - Dorveaux et al., (2009)
"""

import numpy as np

from MagnetometerCalibrationModules import magnetometer_calibration as mag
from MagnetometerCalibrationModules.utils import extract_mag_data

# Save data flag
save = True

# Read data
file_name = "calibration data/filtered_raw_data.csv" 
params, raw_mag_data = extract_mag_data(file_name)
sampling_freq, t_init = params

#sampling_freq = 100.0
#t_init = 60.0

# 1) Create a distortion
#A_distort = mag.random_soft_iron(scale_min=0.6, scale_max=1.6, rotation_magnitude=1.0, seed=42)
#b_distort = np.array([1.3, -1.2, 1.1])   # big-ish hard-iron bias

# 2) Generate synthetic raw data
#m_true, raw_mag_data = mag.generate_raw_data(17000, A_distort, b_distort, noise_std=0.02, outlier_fraction=0.01)

n_samples = raw_mag_data.shape[0]
print(f"Number of samples in the file: {n_samples}")

# Calibrate magnetometer
theta_opt_mag, c = mag.calibrate_mag_from_data(t_init, raw_mag_data, sampling_freq, n_max_iteration=5000, tol=0.0001)

# 4) Evaluate
A_cal = theta_opt_mag[:9].reshape(3,3)
b_cal = theta_opt_mag[9:]
#mag.evaluate_calibration(A_cal, b_cal, A_distort, b_distort, m_true, raw_mag_data)

# Optimization parameters
a_params = theta_opt_mag[:9]
b_params = theta_opt_mag[9:]

# Show results
print(f"Magnetometer optimized A matrix elements: {a_params}")
print(f"Magnetometer optimized B vector elements: {b_params}")

# Save data if required
if save:
    np.savetxt("optimization result data/paramters_mag.csv", theta_opt_mag, delimiter=',')

# Plots
mag.show_data(raw_mag_data, sampling_freq, ylabel="Raw vector components [-]")
mag.show_data(mag.apply_mag_calibration(theta_opt_mag, raw_mag_data), sampling_freq, ylabel="Calibrated vector components [-]")
mag.plot_data_sphere(raw_mag_data, title="Uncalibrated Magnetometer Data", u_sphere=False)
mag.plot_data_sphere(mag.apply_mag_calibration(theta_opt_mag, raw_mag_data), title="Calibrated Magnetometer Data")