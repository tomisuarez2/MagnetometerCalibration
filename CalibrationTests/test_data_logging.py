"""
Magnetometer logging data test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "Iterative calibration method for inertial and magnetic sensors" - Dorveaux et al., 2009
For this example we use an arduino UNO connected with a GY-273 magnetometer module, containing a QMC5883L magnetomer, as indicated in "connection.jpeg" with 
"QMC5883L_raw.ino" code, both found in "arduiono code" folder.
"""

from MagnetometerCalibrationModules.utils import log_data_from_magnetometer

file_name = log_data_from_magnetometer('COM7', 38400, t_init=60, t_log=400) # calibration data log
print(f"\nData has been saved in the following file: {file_name}")

