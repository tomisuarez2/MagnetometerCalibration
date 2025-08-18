import MagnetometerCalibrationModules.magnetometer_calibration as mag
import MagnetometerCalibrationModules.utils as utils

# Read data
file_name = "calibration data/filtered_raw_data.csv" 
params, raw_mag_data = utils.extract_mag_data(file_name)
sampling_freq, t_init = params

mag.show_data(raw_mag_data, sampling_freq)


