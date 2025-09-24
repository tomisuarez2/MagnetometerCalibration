# 🧭 Magnetometer Auto Calibration Module (QMC5883L)

This repository provides a Python implementation of the calibration algorithm from the paper:

> **Iterative calibration method for inertial and magnetic sensors**  
> Eric Dorveaux, David Vissière, Alain-Pierre Martin, Nicolas Petit – *2019*

The method is designed for **low-cost 3-axis magnetometers** (e.g., QMC5883L on the GY-63 module) and requires **no external calibration equipment**.
It automatically compensates **hard-iron and soft-iron distortions**, correcting sensor **bias, scaling, and misalignment**.

It also provides tools to **characterize and analyze the noise of all the magnetometer sensors** using Allan Deviation analysis.

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

  - Sensor measurement noises

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
├── characterization data/          # Data for signal characterization
│ └── mag_static_data_6h.csv        # 6h hour static data recording
├── characterization result data/   # Magnetometer sensors noise variances 
├── characterization result images/ # Magnetometer sensors noise characterization images, includes simulated data
├── optimization result images/     # Sample plots (calibrated vs raw)
├── optimization result data/       # CSV of computed calibration parameters
├── CalibrationTests/               # Example test scripts
├── LICENSE                         # MIT License
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```
---

## 🚀 Quick Start

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/tomisuarez2/MagnetometerCalibration
cd MagnetometerCalibration
```

---

### 2. 📦 Install Requirements

```bash
pip install -r requirements.txt
```

---

### 3. ▶️ Run example tests (e.g. calibration test)

```bash
cd MagnetometerCalibration
python -m CalibrationTests.test_mag_calibration
```
---

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

---

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

---

## ⚠️ Limitations

- Requires motion covering diverse orientations

- Poor excitation may lead to unobservable parameters or spurious solutions

- Works best in stable thermal environments

- Always inspect results visually to confirm calibration quality

---

## 📚 Citation

If you use this module or code, please cite the original work:

Dorveaux, E., Vissière, D., Martin, A.-P., & Petit, N. (2009). Iterative calibration method for inertial and magnetic sensors. 48th IEEE Conference on Decision and Control (CDC). https://doi.org/10.1109/CDC.2009.5400594

---

## ⚙️ Signal characterization working principle

The repository also implements a workflow to characterize magnetometer measured data:

1. **Allan Deviation Analysis**

   * From the altitude time series, Allan deviation (ADEV) is computed across multiple averaging times (τ).
   * This reveals how different noise sources dominate at different time scales:

     * **White noise (σ ∝ 1/√τ)**
     * **Random walk bias (σ ∝ √τ)**

2. **Noise Parameter Estimation**

   * The slopes of the Allan deviation curve are fitted to extract:

     * **R** → Measurement noise variance (white noise level).
     * **q** → Random walk bias intensity.

---

## 📐 Mathematical Background

Given a discrete-time sensor model:

* **Bias evolution**

$${b_{k+1} = b_k + w_k,\quad w_k \sim \mathcal{N}(0, Q)}$$ 

where

* $Q = qT_s$

* **Measurement equation**

$$d_k = p_k + b_k + v_k,\quad v_k \sim \mathcal{N}(0, R)$$

where

* $d_k$ = sensor measurement,
* $p_k$ = true measurement,
* $b_k$ = bias (random walk),
* $v_k$ = white measurement noise,
* $q$ = bias random walk intensity \[u²/s],
* $R$ = measurement noise variance \[u²],
* $T_s$ = sampling period \[s].

We form cluster $i$ (block) averages of length $m$ samples: $\tau = mT_s$, then from Allan variance definition (discrete sampling):

$$
\bar d^{(m)}_i = \frac{1}{m}\sum_{k=0}^{m-1} d_{i m + k}
\qquad \tau = m T_s
$$

The Allan variance at averaging time $\tau$ is

$$
\sigma^2(\tau) = \frac{1}{2}\mathbb{E}\Big[ \big(\bar d^{(m)}_{i+1}-\bar d^{(m)}_{i}\big)^2 \Big]
$$

We will evaluate $\sigma^2(\tau)$ for the two noise types mentioned above.

### White measurement noise $v_k$

Assume $d_k = p_0 + v_k$ (ignore bias for the moment). For the block average,

$$
\bar v_i = \frac{1}{m}\sum_{j=0}^{m-1} v_{im+j}
$$

Because the $v$'s are independent with $\mathrm{Var}(v_k)=R$,

$$
\mathrm{Var}(\bar v_i) = \frac{1}{m^2}\sum_{j=0}^{m-1}\mathrm{Var}(v_{im+j})
= \frac{mR}{m^2} = \frac{R}{m}
$$

Now

$$
\mathrm{Var}(\bar v_{i+1}-\bar v_i) = \mathrm{Var}(\bar v_{i+1})+\mathrm{Var}(\bar v_i)
= 2\frac{R}{m}
$$

(averages from disjoint blocks are independent), so Allan variance

$$
\sigma^2(\tau) = \frac{1}{2}\cdot 2\frac{R}{m} = \frac{R}{m}
$$

Substitute $m=\tau/T_s$:

$$
\sigma^2(\tau) = \dfrac{R}{m} = \dfrac{RT_s}{\tau}
$$

Equivalently,

$$
\sigma(\tau) = \sqrt{\dfrac{RT_s}{\tau}}
$$

So on a log–log Allan plot the white measurement noise region appears as a straight line of slope $-\tfrac{1}{2}$. From the intercept $a_{\text{wn}}$ of the fit:

$$
\log_{10}\sigma(\tau) = -\tfrac12\log_{10}\tau + a_{\text{wn}}
$$

we get 

$$
R = \tfrac{\big(10^{a_\text{wn}}\big) ^2}{T_s}
$$

### Random-walk bias $b_k$

Bias evolves $b_{k+1} = b_k + w_k$ with increments $w_k$ independent and $\mathrm{Var}(w_k)=qT_s$.

We want $\sigma^2(\tau)=\tfrac12\mathbb{E}[(\overline{b}_{i+1}-\overline{b}_i)^2]$ for block averages $\overline b_i$ over $m$ samples.

We need to:

* Write $b_{k}$ as cumulative sum of increments: $b_{k} = b_0 + \sum_{j=0}^{k-1} w_j$.
* Express block average $\overline b_i = \frac1m \sum_{n=0}^{m-1} b_{im+n}$ as a double sum of increments $w_j$ with triangular weights.

Then:

$$\overline b_i = \frac1m \sum_{n=0}^{m-1} \big(b_0 + \sum_{j=0}^{im+n-1} w_j\big)$$

we can assume for the derivation $b_0 = 0$, then:

$$\overline b_i = \frac1m \sum_{n=0}^{m-1} \big(\sum_{j=0}^{im-1} w_j + \sum_{t=0}^{n} w_{im+t}\big)$$

$$\overline b_i = \frac1m \sum_{n=0}^{m-1} \sum_{j=0}^{im-1} w_j + \frac1m \sum_{n=0}^{m-1} \sum_{t=0}^{n} w_{im+t}$$

$$\overline b_i = \sum_{j=0}^{im-1} w_j + \frac1m \sum_{n=0}^{m-1} (m-n) w_{im+n}$$

### Expression for $\bar b_{i+1}-\bar b_i$

Compute similarly $\bar b_{i+1}$ and subtract:

$$
\bar b_{i+1} = \sum_{j=0}^{(i+1)m-1} w_j + \frac{1}{m}\sum_{n=0}^{m-1} (m-n)w_{(i+1)m+n}
$$

Subtract $\bar b_i$. The common sum $\sum_{j=0}^{im-1} w_j$ cancels. Collect terms:

* Terms with indices $j=im + n$ (the middle block) appear from the expansion of $\bar b_{i+1}$ as full sum and from $\bar b_i$ with coefficient $(m-n)/m$. Their net coefficient is

$$1 - \frac{m-n}{m} = \frac{n}{m}$$
  
* Terms with indices $j=(i+1)m + n$ (the next block) appear only in $\bar b_{i+1}$ with coefficient $(m-n)/m$.

Thus

$$
\bar b_{i+1}-\bar b_i = \frac{1}{m}\sum_{n=0}^{m-1} \bigg( nw_{im+n} + (m-n)w_{(i+1)m+n}\bigg)
$$

This is a linear combination of $2m$ independent increments $w$ with known deterministic coefficients.

### Variance of the difference (exact finite-m expression)

Because the $w$'s are independent, the variance of the linear combination equals $Q$ times the sum of squared coefficients:

$$
\begin{aligned}
\mathrm{Var}(\bar b_{i+1}-\bar b_i)
&= \frac{Q}{m^2}\sum_{n=0}^{m-1} \big( n^2 + (m-n)^2 \big) \\
&= \frac{Q}{m^2}\Big( \sum_{n=0}^{m-1} n^2 + \sum_{n=0}^{m-1} (m-n)^2 \Big)
\end{aligned}
$$

Evaluate the sums. Use the known formula:

$$
\sum_{n=0}^{m-1} n^2 = \frac{(m-1)m(2m-1)}{6},\qquad
\sum_{k=1}^{m} k^2 = \frac{m(m+1)(2m+1)}{6}
$$

Noting $\sum_{n=0}^{m-1}(m-n)^2 = \sum_{k=1}^{m} k^2$, sum them:

$$
\begin{aligned}
S &= \sum_{n=0}^{m-1} n^2 + \sum_{k=1}^{m} k^2
= \tfrac{(m-1)m(2m-1)}{6} + \tfrac{m(m+1)(2m+1)}{6} \\
&= \tfrac{m}{6}\Big[ (m-1)(2m-1) + (m+1)(2m+1)\Big] \\
&= \tfrac{m}{6}\Big[(2m^2-3m+1) + (2m^2+3m+1)\Big] \\
&= \tfrac{m}{6}(4m^2 + 2) = \frac{m(2m^2+1)}{3}
\end{aligned}
$$

Therefore

$$
\mathrm{Var}(\bar b_{i+1}-\bar b_i)
= \frac{Q}{m^2}\cdot \frac{m(2m^2+1)}{3}
= Q\cdot \frac{2m^2+1}{3m}
$$

### Allan variance (exact discrete expression)

Recall Allan variance is one half of the expected squared difference:

$$
\sigma^2(\tau) = \tfrac12\mathrm{Var}(\bar b_{i+1}-\bar b_i)
= \frac{Q}{2}\cdot\frac{2m^2+1}{3m}
= Q\cdot \frac{2m^2+1}{6m}
$$

Replace $Q=qT_s$ and $m=\tau/T_s$ to express in $\tau$ and $T_s$. Two algebraically equivalent forms are useful:

1. Expand to isolate the dominant and correction terms:

$$
\sigma^2(\tau)
= \frac{q}{3}\tau + \frac{qT_s^{2}}{6\tau}
$$

(derivation: substitute $Q=qT_s$ and simplify).

2. Or as a single fraction:

$$
\sigma^2(\tau)
= \frac{6\tau\sigma^2(\tau)}{2\tau^2 + T_s^2}\quad\text{(rearranged when solving for }q\text{)}
$$

The first form is very instructive: it is the exact discrete formula and clearly shows the **leading term** $(q/3)\tau$ and the **finite-sample correction** $\dfrac{qT_s^2}{6\tau}$.

### Asymptotic (continuous / large-m) approximation

For $m\gg 1$ (i.e. $\tau \gg T_s$), the correction term is negligible. Then

$$
\sigma^2(\tau) \approx \frac{q}{3}\tau
\qquad\Longrightarrow\qquad
\sigma(\tau) \approx \sqrt{\frac{q}{3}}\sqrt{\tau}
$$

So on a log–log Allan plot the random-walk region appears as a straight line of slope $+\tfrac{1}{2}$. From the intercept $a_{\text{rw}}$ of the fit

$$
\log_{10}\sigma(\tau) = \tfrac12\log_{10}\tau + a_{\text{rw}}
$$

we get (neglecting finite-sample correction)

$$
q \approx 3\cdot\big(10^{a_{\text{rw}}}\big)^2
$$

This is the common practical formula used when $\tau$ is comfortably larger than $T_s$.

---
### Summarazing:

* **White noise region**

  $\sigma(\tau) = \sqrt{\frac{RT_s}{\tau}}$

* **Random walk bias region**

  $\sigma(\tau) = \sqrt{\frac{q}{3}}\sqrt{\tau}$

These relationships allow estimation of $R$ and $q$ directly from logged data.

---

## 📊 Signal characterization example output

* **Allan deviation curve** with fitted slopes
* Estimated noise parameters:

 ```bash
 >>> Z axis magnetometer white measurement–noise variance [-]: 0.007715742865748797
 >>> Z axis magnetometer bias random–walk intensity [(-)/s]: nan 

 ```
* Visualization of white noise (−½ slope) and random walk (+½ slope) regions

![Allan Deviation Plot](characterization%20result%20images/allan_dev_plot_mz.png)

![Real vs Simulated data](characterization%20result%20images/real_vs_sim_mz.png)

As can be seen in the above pictures, all the sensor noise in due to white gaussian measurement noise.
---

## 🤝 Contributing

Contributions are welcome!
Fork, improve, and open a pull request 🚀

(Also check out our other related projects: [TimeOfFlightCalibration](https://github.com/tomisuarez2/TimeOfFlightCalibration), [ImuCalibration](https://github.com/tomisuarez2/ImuCalibration) and [BarometricAltimeterCalibration](https://github.com/tomisuarez2/BarometricAltimeterCalibration))

---

## 🛰️ Contact

If you have questions or want to collaborate, feel free to reach out:
**Tomás Suárez**
Mechatronics Engineering Student
📧 [suareztomasm@gmail.com](mailto:suareztomasm@gmail.com)









