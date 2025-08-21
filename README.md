# 🧭 Magnetometer Auto Calibration Module (QMC5883L)

This repository provides a Python implementation of the calibration algorithm from the paper:

> **Iterative calibration method for inertial and magnetic sensors**  
> Eric Dorveaux, David Vissière, Alain-Pierre Martin, Nicolas Petit – *2019*

The method is designed for **low-cost 3-axis magnetometers** (e.g., QMC5883L on the GY-63 module) and requires **no external calibration equipment**.
It automatically compensates **hard-iron and soft-iron distortions**, correcting sensor **bias, scaling, and misalignment**.

---

## ⚙️ How It Works

The calibration method follows the approach of Dorveaux et al. (2009) and combines two main stages:

## 1. 🟢 Coarse Bias Estimation (Sphere Fitting)

- Raw magnetometer measurements ideally lie on a sphere centered at the origin.

- Due to hard-iron effects, the sphere is displaced (biased).

- A least-squares sphere fitting algorithm is applied first to approximate the center of this sphere → giving a coarse estimate of the bias vector B.

- This initial step improves convergence of the more advanced iterative calibration.

## 2. 🔵 Iterative Least-Squares Refinement

- After bias correction, measurements should lie on an ellipsoid (due to scaling, misalignment, and soft-iron effects).

- The algorithm applies an iterative affine correction:

  $$
  m^{(k+1)}=A_k m^{(k)}+B_k	​
  $$

  where **A** is a calibration matrix and **B** a bias vector.

- At each iteration, a **least-squares optimization** enforces that the calibrated data remain close to the **unit sphere** (‖m‖ = 1).

- Successive iterations refine the calibration parameters, progressively reducing distortion.

## 3. ✅ Convergence

- With each iteration, the error metric decreases monotonically (the algorithm guarantees “better” calibration step by step).

- After a few iterations, data points converge toward a **unit sphere**, meaning the magnetometer is effectively calibrated.

## 📌 In summary:

- **Sphere fitting** quickly estimates the coarse bias.

- **Iterative refinement** corrects scaling, misalignment, and residual bias.

- The final model outputs:

  - **A** → 3×3 calibration matrix (scaling + misalignment)

  - **B** → 3×1 bias vector (hard-iron correction)

---

## ✨ Features

- 🔄 Fully automatic magnetometer calibration — no manual intervention required

- 📐 Estimates:

  - Bias vector (hard-iron error)

  - Misalignment and non-orthogonality

  - Scale factors (soft-iron error)

- 📊 Visualization tools to validate calibration results (time-domain and 3D space)

- 🧩 Clean, modular Python implementation

- 🔌 Includes Arduino sketch for raw data acquisition via I2C/UART

---

## 👨‍💻 Authors

**Tomás Suárez, Agustín Corazza, Rodrigo Pérez**  
Mechatronics Engineering Students 
Universidad Nacional de Cuyo  
📧 suareztomasm@gmail.com
📧 corazzaagustin@gmail.com
📧 rodrigoperez2110@gmail.com

---

## 📁 Project Structure

```text
MagnetometerCalibration/
├── arduino code/                  # Arduino interface for QMC5883L
│   ├── connection.jpeg             # Wiring diagram (Arduino UNO ↔ GY-63)
│   ├── QMC5883L/                   # Arduino library (C++ .h/.cpp)
│   ├── QMC5883L_raw.ino            # Arduino sketch for UART streaming
│   └── QMC5883L-Datasheet-1.0.pdf  # Sensor datasheet
├── MagnetometerCalibrationModules/ # Core Python modules
│   ├── magnetometer_calibration.py # Main calibration logic
│   └── utils.py                    # Helpers and data loaders
├── calibration data/               # Example raw CSV datasets
├── optimization result images/     # Sample plots (calibrated vs raw)
├── optimization result data/       # CSV of computed calibration parameters
├── CalibrationTests/               # Example test scripts
├── LICENSE                         # MIT License
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```
--

## 🚀 Quick Start

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/tomisuarez2/MagnetometerCalibration
cd MagnetometerCalibration
```

--

### 2. 📦 Install Requirements

```bash
pip install -r requirements.txt
```

--

### 3. ▶️ Run example tests (e.g. calibration test)

```bash
cd MagnetometerCalibration
python -m CalibrationTests.test_mag_calibration
```
--

## 📊 Example Output

The calibration outputs an **affine correction model**: 

```bash
Magnetometer optimized A matrix: 
[ 9.23963302e-04 -2.28276038e-05 -1.26491570e-05 
 -3.26463689e-05  1.39419601e-03 -8.31640876e-05  
  9.24662679e-07  5.04342440e-05  6.25530654e-04 ]

Magnetometer optimized B vector elements: 
[-5.6742573   0.1399525  -2.19375634]
```
![Non calibrated magnetometer data over time](optimization%20result%20images/non_cal_mag.png)

![Calibrated magnetometer data over time](optimization%20result%20images/cal_mag.png)

![Non calibrated magnetometer data in space](optimization%20result%20images/non_cal_mag_spa.png)

![Calibrated magnetometer data in space](optimization%20result%20images/cal_mag_spa.png)
--

## 📈 Input Data Format

CSV with raw magnetometer values:
```bash
mx, my, mz
```

- mx, my, mz: raw QMC5883L readings
- Consistent sampling rate recommended (default Arduino code: 50 Hz)
--

## 📟 Arduino Data Logger

The repository includes an Arduino sketch (QMC5883L_raw.ino) to acquire raw data:
- Configurable sampling frequency (10–200 Hz)
- I2C communication (Wire.h)
- Data-ready interrupt support
- UART output:
```bash
mx, my, mz
```

👉 Install the included **QMC5883L Arduino library** by copying the folder to your Arduino libraries/ directory.


### 👏 Acknowledgements

This Arduino library for sensor comunnication is based on the excellent open-source library provided by [**jarzebski**](https://github.com/jarzebski/Arduino-HMC5883L).  

---

## 🧪 Validation

Calibration can be validated via included test scripts:

✅ Sphere fitting of normalized magnetic field

✅ Unit norm preservation over time

✅ Synthetic data test mode (synthetic=True)

--

## ⚠️ Limitations

- Requires motion covering diverse orientations

- Poor excitation may lead to unobservable parameters or spurious solutions

- Works best in stable thermal environments

- Always inspect results visually to confirm calibration quality

--

## 📚 Citation

If you use this module or code, please cite the original work:

Dorveaux, E., Vissière, D., Martin, A.-P., & Petit, N. (2009). Iterative calibration method for inertial and magnetic sensors. 48th IEEE Conference on Decision and Control (CDC). https://doi.org/10.1109/CDC.2009.5400594
--

## 🤝 Contributing

Contributions are welcome!
Fork, improve, and open a pull request 🚀

(Also check out our related project: [ImuCalibration](https://github.com/tomisuarez2/ImuCalibration))


--

## 🛰️ Contact

If you have questions or want to collaborate, feel free to reach out:
📧 suareztomasm@gmail.com









