"""
utils module from Magnetometer Calibration Module
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "Iterative calibration method for inertial and magnetic sensors" - Dorveaux et al., 2009
"""

import csv
from datetime import datetime
import serial
import time
from typing import Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

def log_data_from_magnetometer(
    port: str,
    baud_rate: int,
    t_init: Union[int, float] = 60,
    t_log: Union[int, float] = 120
) -> str:
    """
    Log data from an QMC5883L magnetometer via UART communication. 
    You will need "QMC5883L_raw.ino" code in Arduino UNO and connected as shown in "connections.jpeg",
    you can find both files in "arduino code" folder.
    You must move your magnetometer in order to log as much magnetic field components in every orientation as you can in the specified time.
    Be sure to avoid external magnetic fields other than earth's.

    Args:
        port: Serial port for communication (e.g., 'COM3')
        baud_rate: Baud rate for serial communication (e.g., 38400)
        t_init: Initial static time interval for magnetomer warm up (seconds)
        t_log: Data logging time interval (seconds)

    Returns:
        str: Filename of the generated CSV file containing logged data

    Raises:
        serial.SerialException: If serial communication fails

    Output CSV format:
        - First row: Fs, sampling_frequency (Hz)
        - Second row: initialization time, t_init 
        - Third row: mx,my,mz (column headers)
        - Subsequent rows: [mx,my,mz] (sensor readings)

    Notes:
        - You must reset your Arduino UNO every time you want to use this function
    """
    error_data = 0 # Counter for corrupt data lines.

    def save_line(ser: serial.Serial, writer: csv.writer) -> None:
        """
        Read, validate, and save a single line of magnetometer data.

        Args:
            ser: Active serial connection
            writer: CSV writer object
        """
        nonlocal error_data
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                data = [int(val) for val in line.split(',')]
                if len(data) == 3:
                    writer.writerow(data)   
        except (ValueError, UnicodeDecodeError) as e:
            error_data += 1
            if error_data % 100 == 0: # Only print periodic error to avoid flooding
                print(f"Corrupt data (total errors: {error_data}): {e}")

    # Generate output file with timestamp
    time_stamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    output_file = f"magnetometer_data_{time_stamp}_cal.csv"

    try:
        with serial.Serial(port, baud_rate, timeout=1) as ser, open(output_file, 'w', newline='') as csvfile:
            print(f"Connected to {port}.")
            print(">>> Waiting for magnetometer initialization...")

            # Wait for MPU6050 connection confirmation
            while True:
                response = ser.readline().decode('utf-8').strip()
                if response == "QMC5883L connection succesful":
                    break
                elif response == "QMC5883L connection failed":
                    raise RuntimeError("QMC5883L connection failed")

            # Get sampling frecuency
            while True:
                response = ser.readline().decode('utf-8').strip()
                if response == "Selected sampling frequency:":
                    sampling_freq = ser.readline().decode('utf-8').strip().split()[1]
                    break

            print(f">>> Sampling frecuency: {sampling_freq} Hz")

            # Initialize CSV file
            writer = csv.writer(csvfile)
            writer.writerow(["Fs", sampling_freq])
            writer.writerow(["Initialization time", t_init])
            writer.writerow(["mx", "my", "mz"])

            print(f">>> Hold the magnetometer steady for {t_init} seconds")
            print(">>> Press any letter to start: ")
            input() # Wait for user input
            ser.write(b' ') # Send any byte to start

            # Wait for data collection to start
            while True:
                if ser.readline().decode('utf-8').strip() == "Getting raw data...":
                    break

            print(">>> Data logging process has started")
      
            # Warm up data collection
            t_static_start = time.time()
            while time.time() - t_static_start < t_init:
                save_line(ser, writer)

            print(f"\n>>> Move your magnetomer in all orientations")
            t_moving_start = time.time()
            while time.time() - t_moving_start < t_log:
                save_line(ser, writer)

            print("\n>>> Capture completed")
            print(f"\n>>> Total corrupt data lines: {error_data}")

    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
        raise

    return output_file

def extract_mag_data(
    file_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract magnetometer data and parameters from a CSV log file.

    Args:
        file_name: Path to the CSV file containing logged magnetometer data.

    Returns:
        A tuple containing:
        - params: Array of parameters:
            [fs, t_init]
        - data: Array of shape (N, 3) containing IMU measurements:
            [mx, my, mz]

    Raises:
        ValueError: If the file format is invalid
        FileNotFoundError: If the file doesn't exist

    Notes:
        - First 2 rows must contain metadata:
          1. fs: sampling frequency
          2. t_init: initial time for warm up 
        - Subsequent rows contain magnetometer data
    """

    try:
        # Read and parse metadata in one pass
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            metadata = [next(reader) for _ in range(2)]
            
        # Convert metadata with proper error handling
        fs = float(metadata[0][1])
        t_init = int(metadata[1][1])
        
        params = [fs, t_init]

        # Load data efficiently with numpy
        data = np.loadtxt(file_name, delimiter=',', skiprows=3, dtype=np.float32)

        return np.array(params, dtype=np.float32), data

    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid file format in {file_name}") from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_name}") from e
    
def show_loglog_data(
    x_data: np.ndarray,
    y_data: np.ndarray, 
    legend: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """
    Show data plot in double logaritmic axes.

    Args:
        x_data: X data plot.
        y_data: Y data plot.
        legend: Data legend.
        xlabel: X axis label.
        ylabel: Y axis label.
        title: Figure title.

    Returns:
        None
    """

    # Visualization
    plt.loglog(x_data, y_data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.title(title)
    plt.grid(True, which="both")
    plt.show()

def show_time_data(
    data: np.ndarray, 
    fs: Union[int,float], 
    legend: list,
    xlabel: str="Time [s]",
    ylabel: str="Sensor measurement data [u]",
    title: str="Sensor data"
) -> None:
    """
    Show sensor data as a function of time.

    Args:
        data: Data array of shape (N,) where N is number of samples.
        fs: Sampling rate in Hz.
        legend: Plot legend.
        xlabel: X axis label.
        ylabel: Y axis label.
        title: Figure title.

    Returns:
        None
    """
    n_samples = data.shape[0]

    # Time vector
    time_vector = np.arange(0, n_samples, 1) / fs

    # Completed barometer data over time
    _, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(time_vector, data)
    ax1.grid(True)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend(legend)

    plt.show()