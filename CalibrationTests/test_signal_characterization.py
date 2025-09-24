"""
Magnetometer signal characterization test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
"""

import numpy as np
from MagnetometerCalibrationModules import magnetometer_calibration as mag
from MagnetometerCalibrationModules.utils import extract_mag_data, show_time_data

# Save data flag
save = True

# Read data
file_name = "characterization data/mag_static_data_6h.csv" 
params, raw_mag_data = extract_mag_data(file_name)
sampling_freq, t_init = params
n_samples = raw_mag_data.shape[0]

print(f"Number of samples in the file: {n_samples}")

# Calibrate data
params_mag = np.loadtxt("optimization result data/paramters_mag.csv", delimiter=',')

cal_mag_data = mag.apply_mag_calibration(params_mag, raw_mag_data)

# Recorded data
mag.show_data(cal_mag_data, sampling_freq, ylabel="Magnetometer data [-]", title="Magnetometer data")

time_vector = np.arange(0, n_samples, 1) / sampling_freq

# Compute Allan Variance
mag_tau, mag_avar = mag.compute_allan_variance(cal_mag_data, sampling_freq, m_steps='exponential')
mag_a_dev = np.sqrt(mag_avar)


# Estimate R and q values
R_mx, q_mx, tauwn_mx, taurw_mx = mag.auto_estimate_R_q_from_allan(mag_tau, mag_a_dev[:,0], sampling_freq, plot=True, u='(-)', title='X Mag Allan Deviation')
R_my, q_my, tauwn_my, taurw_my = mag.auto_estimate_R_q_from_allan(mag_tau, mag_a_dev[:,1], sampling_freq, plot=True, u='(-)', title='Y Mag Allan Deviation')
R_mz, q_mz, tauwn_mz, taurw_mz = mag.auto_estimate_R_q_from_allan(mag_tau, mag_a_dev[:,2], sampling_freq, plot=True, u='(-)', title='Z Mag Allan Deviation')

# Show results
print(f">>> X axis magnetometer white measurement–noise variance [-]: {R_mx}")
print(f">>> X axis magnetometer bias random–walk intensity [(-)/s]: {q_mx}")

print(f">>> Y axis magnetometer white measurement–noise variance [-]: {R_my}")
print(f">>> Y axis magnetometer bias random–walk intensity [(-)/s]: {q_my}")

print(f">>> Z axis magnetometer white measurement–noise variance [-]: {R_mz}")
print(f">>> Z axis magnetometer bias random–walk intensity [(-)/s]: {q_mz}")

# Save data if required
if save:
    np.savetxt("characterization result data/R_q_mx.csv", (R_mx, q_mx), delimiter=',')
    np.savetxt("characterization result data/R_q_my.csv", (R_my, q_my), delimiter=',')
    np.savetxt("characterization result data/R_q_mz.csv", (R_mz, q_mz), delimiter=',')
    
# Show time data and simulated data.
sim_data_mx = mag.simulate_sensor_data(n_samples, sampling_freq, R_mx, q_mx, np.mean(cal_mag_data[:,0]))
sim_data_my = mag.simulate_sensor_data(n_samples, sampling_freq, R_my, q_my, np.mean(cal_mag_data[:,1]))
sim_data_mz = mag.simulate_sensor_data(n_samples, sampling_freq, R_mz, q_mz, np.mean(cal_mag_data[:,2]))

show_time_data(np.vstack([cal_mag_data[:,0], sim_data_mx]).T, sampling_freq, ["Logged Mx Signal", "Simulated Mx Signal"], ylabel="Direction component [-]")
show_time_data(np.vstack([cal_mag_data[:,1], sim_data_my]).T, sampling_freq, ["Logged My Signal", "Simulated My Signal"], ylabel="Direction component [-]")
show_time_data(np.vstack([cal_mag_data[:,2], sim_data_mz]).T, sampling_freq, ["Logged Mz Signal", "Simulated Mz Signal"], ylabel="Direction component [-]")
