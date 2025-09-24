"""
Magnetometer Calibration Module
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "Iterative calibration method for inertial and magnetic sensors" - Dorveaux et al., (2009)
"""
from typing import Literal, Union, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from . import utils

#==========================================================
#----------Functions for magnetometer calibration----------
#==========================================================

def fit_sphere_ls(
    raw_mag_data: np.ndarray
) -> Tuple[float, float]:
    """
    Algebraic sphere fit: ||x - c||^2 = R^2. It can be used to get coarse bias c.
    Solves linear LS for [c_x, c_y, c_z, d], with R^2 = c·c + d.anac

    Args:
        raw_data: Raw magnetometer data array of shape (N, 3)

    Returns:
        center c (bias) and radius R.
    """
    mx, my, mz = raw_mag_data[:,0], raw_mag_data[:,1], raw_mag_data[:,2]
    A = np.column_stack([2*mx, 2*my, 2*mz, np.ones_like(mx)])
    b = mx*mx + my*my + mz*mz
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cz, d = theta
    c = np.array([cx, cy, cz])
    R2 = np.dot(c, c) + d
    R  = np.sqrt(max(R2, 0.0))
    return c, R
    
def apply_mag_calibration(
    params: np.ndarray, 
    raw_data: np.ndarray
) -> np.ndarray:
    """
    Applies the Dorveaux et al. (2009) magnetometer calibration model to raw magnetometer data.
    
    Args:
        params: Calibration parameters array of shape (12,)
            [a11, a12, a13, a21, a22, a23, a31, a32, a33, b1, b2, b3]
        raw_data: Raw magnetometer data array of shape (N, 3)

    Returns:
        Calibrated magnetometer data of shape (N, 3)

    Notes:
        - Implements the proposed calibration model from: 
          "Iterative calibration method for inertial and magnetic sensors"
        - The calibration model is: A @ raw + B
          where A is a general matrix and B is the zero bias vector
    """
    # Split parameters
    a_params, B = np.split(params, [9])

    # Construct general matrix (A)
    A = a_params.reshape(3,3)

    # Apply calibration in vectorized form:
    # Equivalent to: (A @ raw_data.T + B).T
    return raw_data @ A.T + B

def design_matrix_and_target(
    iter_mag_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the linear LS system for Eq. (4) at iteration k as presented in Dorveaux et al. (2009):
        [A y_i,k + B] ≈ u_i,k  with  u_i = y_i,k / ||y_i,k||
    Stacks x,y,z components for all measurements i.

    Args:
        iter_mag_data: Current magnetometer data array of shape (N, 3) (y_k)

    Returns:
        M: (3N, 12) design matrix
        u: (3N,)   target vector
    """
    n_samples = iter_mag_data.shape[0]

    # Unit targets u_i,k = y_i,k / ||y_i,k||
    norms = np.linalg.norm(iter_mag_data, axis=1, keepdims=True)
    u = iter_mag_data / norms
    u = u.reshape(-1)

    # Design matrix blocks 12 parameters
    # Row for x-component: [y1 y2 y3  0  0  0  0  0  0  1 0 0]
    # Row for y-component: [ 0  0  0 y1 y2 y3  0  0  0  0 1 0]
    # Row for z-component: [ 0  0  0  0  0  0 y1 y2 y3  0 0 1]
    M = np.zeros((3 * n_samples, 12), dtype=np.float64)

    # x rows
    M[0::3, 0:3] = iter_mag_data
    M[0::3, 9]   = 1.0
    # y rows
    M[1::3, 3:6] = iter_mag_data
    M[1::3, 10]  = 1.0
    # z rows
    M[2::3, 6:9] = iter_mag_data
    M[2::3, 11]  = 1.0

    return M, u

def calibrate_mag_from_data(
    t_init: Union[int, float], 
    raw_mag_data: np.ndarray, 
    fs: Union[int, float], 
    tol: float=1e-3,
    n_max_iteration: int=2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized magnetometer calibration by means of iterations of least square problems 
    and successive partial calibration of data as presented in Dorveaux et al., (2009).
 
    Args:
        t_init: Initial static time interval for IMU initialization (s)
        raw_mag_data: Raw magnetometer data array of shape (N, 3)
        fs: Sampling rate in Hz.
        tol: Cost function relative reduction tolerance in each iteration
        n_iterations: Maximum number of iterations

    Returns:
        Optimal calibration parameters (12,) [A_tilde (row-wise 9), B_tilde (3)]
    """
    # t_init samples
    start = int(t_init * fs)

    # Sphere-fit to get coarse bias c
    #c, _ = fit_sphere_ls(raw_mag_data[start:])
    c = np.mean(raw_mag_data[start:], axis=0)
    print(c)

    # De-bias data, first calibration iteration also
    cal_mag_data = raw_mag_data[start:].copy() - c

    # Pre-allocate results storage
    thetas: list[np.ndarray] = []
    costs: list[float] = []

    print(">>> Magnetometer calibration in progress...")

    for k in range(n_max_iteration):
        M, u = design_matrix_and_target(cal_mag_data)
        # Solve M θ ≈ u
        theta, cost, _, _ = np.linalg.lstsq(M, u, rcond=None)

        # Check termination (needs at least 2 iters)
        if k > 0:
            rel_red = (costs[-1] - cost[0]) / max(costs[-1], 1e-16)
            if rel_red < tol:
                break

        thetas.append(theta)
        costs.append(cost[0])

        # Update dataset for next iteration: m^(k+1) = A_k m^(k) + B_k
        cal_mag_data = apply_mag_calibration(theta, cal_mag_data)
            
    # Find matrices by recursion
    n_iter = len(thetas)
    A_tilde_k = thetas[0][:9].reshape(3,3)
    B_tilde_k = thetas[0][9:]
    for i in range(1, n_iter):
         A_k = thetas[i][:9].reshape(3,3)
         B_k = thetas[i][9:]
         A_tilde_k =  A_k @ A_tilde_k 
         B_tilde_k = A_k @ B_tilde_k + B_k 

    # Compose with coarse bias c
    B_tilde_k = A_tilde_k @ (-c) + B_tilde_k

    # Get parameters from matrices
    params_mag = np.hstack((A_tilde_k.ravel(), B_tilde_k))

    print(">>> Magnetometer calibration finished")
    print(f">>> Number of interations: {n_iter}")
    return params_mag

def show_data(
    mag_data: np.ndarray, 
    fs: Union[int,float], 
    xlabel: str="Time [s]",
    ylabel: str="Vector Components [-]",
    title: str="Magnetic Field Orientation Vector"
) -> None:
    """
    Show magnetometer data as a function of time.

    Args:
        mag_data: Data array of shape (N, 3) where N is number of samples.
        fs: Sampling rate in Hz.
        xlabel: X axis label
        ylabel: Y axis label
        title: Figure title

    Returns:
        None
    """
    n_samples = mag_data.shape[0]

    # Time vector
    time_vector = np.arange(0, n_samples, 1) / fs

    # Completed magnetometer  data over time
    _, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(time_vector, mag_data)
    ax1.plot(time_vector, np.linalg.norm(mag_data, axis=1))
    ax1.grid(True)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend(["mx","my","mz","||m||"])

    plt.show()

def plot_data_sphere(
    mag_data: np.ndarray, 
    title: str = "3D Magnetometer Data",
    u_sphere: bool=True
) -> None:
    """
    Plots a magnetometer 3D data and unit sphere if specified.
    It can be used to visualize calibration results.

    Args:
        mag_data: Data array of shape (N, 3) where N is number of samples.
        title: Figure title
        u_sphere: If true plots unit sphere

    Returns:
        None
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.scatter(0,0,0, color='black')

    if u_sphere:
        # Create unit sphere mesh
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Plot sphere surface (light transparent)
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, linewidth=0)

    # Scatter calibrated data
    ax.scatter(mag_data[:, 0], mag_data[:, 1], mag_data[:, 2],
               c='r', s=5, alpha=0.6, label="Data points")

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

#===========================================================
#----------Functions for synthetic data generation----------
#===========================================================

def sample_unit_sphere(
    n_samples: int
) -> np.ndarray:
    """
    Sample points uniformly on the unit sphere.

    Args:
        n_samples: Number of desired samples.

    Returns: 
        Data array of shape (n_samples,3)  
    """

    x = np.random.normal(size=(n_samples, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x
   
def random_soft_iron(
    scale_min: float=0.7, 
    scale_max: float=1.5, 
    rotation_magnitude: float=0.5
) -> np.ndarray:
    """
    Build a random A_distort matrix

    Args:
        scale_min: minumun diagonal scale
        scale_max: maximun diagonal scale
        rotation_magnitud: A small random rotation (radians)

    Returns:
        Inverse of true general matrix A.
    """
    # Random diagonal scales
    scales = np.random.uniform(scale_min, scale_max, size=3)
    D = np.diag(scales)
    # Random rotation via axis-angle
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-rotation_magnitude, rotation_magnitude)
    # Rodrigues rotation
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    A = R @ D @ R.T
    return A

def generate_raw_data(
    n_samples: int,
    A_distort: np.ndarray,
    b_distort: np.ndarray,
    noise_std: float = 0.0,
    outlier_fraction: float = 0.0,
    outlier_scale: float = 1.1,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate measured (raw) magnetometer data from unit field vectors.

    Args:
        A_distort: Inverse of true general matrix A.
        b_distort: It relates with real zero bias vector as (A_distort)^(-1) @ b_distort approx. -b_cal
        noise_std: White gaussian noise standard deviation 
        outlier_fraction: Fraction of large random points in data set
        outlier_scale: Scale of the fraction odf large random points in data set
        clip_min: Minimun clipping value
        clip_max: Maximun clipping value

    Returns:
        m_true: Real unit sphere data array of shape (N,3).
        m_raw : Synthetic measured (distorted) data array of shape (N,3).
    """
    m_true = sample_unit_sphere(n_samples)
    m_raw = (A_distort @ m_true.T).T + b_distort

    # Add gaussian noise
    if noise_std > 0:
        m_raw += np.random.normal(scale=noise_std, size=m_raw.shape)

    # Outliers (large random points)
    if outlier_fraction > 0:
        n_out = int(np.round(outlier_fraction * n_samples))
        idx = np.random.choice(n_samples, size=n_out, replace=False)
        # replace by an amplified direction or random large vector
        out = sample_unit_sphere(n_out) * outlier_scale
        m_raw[idx] = out

    # Clipping (simulate ADC saturation)
    if clip_min is not None or clip_max is not None:
        if clip_min is not None:
            m_raw = np.maximum(m_raw, clip_min)
        if clip_max is not None:
            m_raw = np.minimum(m_raw, clip_max)

    return m_true, m_raw

def evaluate_calibration(
    A_cal: np.ndarray, 
    b_cal: np.ndarray, 
    A_distort: np.ndarray, 
    b_distort: np.ndarray, 
    m_true: np.ndarray, 
    m_raw: np.ndarray
) -> dict:
    """
    Evaluate calibration quality.
    Computes:
      - Norm error stats of calibrated points vs 1
      - Parameter differences (A_cal * A_distort ≈ I, A_cal @ b_distort + b_cal ≈ 0)

    Args:
        A_cal: General matrix A obtained by means of calibration.
        b_cal: Zero vias vector B obtained by means of calibration.
        A_distort: Inverse of true general matrix A.
        b_distort: It relates with real zero bias vector as (A_distort)^(-1) @ b_distort approx. -b_cal
        m_true: Real unit sphere data array of shape (N,3).
        m_raw : Synthetic measured (distorted) data array of shape (N,3).

    Returns:
        Calibration quality dictionary.

    """
    # Apply calibration
    m_est = (A_cal @ m_raw.T).T + b_cal
    norms = np.linalg.norm(m_est, axis=1)

    # Expected relationships
    compA = A_cal @ A_distort
    compb = A_cal @ b_distort + b_cal

    print(">>> Calibrated norms: mean {:.6f}, std {:.6f}, min {:.6f}, max {:.6f}".format(norms.mean(), norms.std(), norms.min(), norms.max()))
    print(">>> Composition A_cal @ A_distort (should be ~I):\n", compA)
    print(">>> Composition error vs Identity (Frobenius norm):", np.linalg.norm(compA - np.eye(3)))
    print(">>> Composition for bias (should be near zero):", compb, " norm:", np.linalg.norm(compb))

    # If m_true known, show the direct RMSE vs ground-truth directions/coords
    rmse_coords = np.sqrt(np.mean((m_est - m_true)**2))
    rmse_norms = np.sqrt(np.mean((np.linalg.norm(m_est,axis=1) - 1.0)**2))
    print(">>> RMSE (coords vs true):", rmse_coords)
    print(">>> RMSE (norm vs 1):", rmse_norms)

    return {"m_est": m_est, "compA": compA, "compb": compb, "rmse_coords": rmse_coords, "rmse_norms": rmse_norms}
    
#==================================================================
#----------Functions for simple synthetic data generation----------
#==================================================================

def simulate_sensor_data(
    N: int, 
    fs: float, 
    R: float, 
    q: float, 
    mean: float,
) -> np.ndarray:
    """
    Simulate synthetic static sensor data with white noise and bias random walk.

    Args:
        N: Number of samples.
        fs: Sampling frequency [Hz].
        R: White noise variance (per sample).
        q: Random walk variance (per sample).
        mean: Data mean.

    Returns:
        y: Synthetic sensor measurement array of length N.
    """
    # Initialization
    y = np.zeros(N)

    # White noise
    v = 0
    if not np.isnan(R):
        v = np.random.normal(0, np.sqrt(R), size=N)

    # Random walk increments for bias
    w = 0
    if not np.isnan(q):
        w = np.random.normal(0, np.sqrt(q/fs), size=N)  
    u = np.cumsum(w) 

    y = u + v + mean

    return y

#===========================================================
#-----------Functions for signal characterization-----------
#===========================================================

def compute_allan_variance(
    data: np.ndarray,
    fs: Union[int,float],
    m_steps: Literal['linear', 'exponential'] = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Allan Variance of the input data.

    Args:
        data: Input array of shape (N, n) where N is number of samples and n is number of signals.
        fs: Sampling rate in Hz.
        m_steps: Method for interval length variation:
                'linear' - linear spacing between intervals
                'exponential' - base-2 exponential spacing (default: 'linear')

    Returns:
        Tuple containing:
        - taus: Array of interval lengths in seconds
        - avar: Array of corresponding Allan Variance values

    Notes:
        - For 'linear', evaluates intervals from 2 samples to N//2 samples
        - For 'exponential', evaluates intervals as powers of 2 up to N//2
        - Minimum of 2 intervals required for variance calculation
    """
    n_samples = data.shape[0]

    # Generate interval lengths (tau in samples)
    if m_steps == 'linear':
        max_m = n_samples // 2
        taus = np.arange(2, max_m, dtype=int)
    elif m_steps == 'exponential':
        max_power = int(np.floor(np.log2(n_samples // 2)))
        taus = 2**np.arange(1, max_power + 1)
    else:
        raise ValueError("m_steps must be either 'linear' or 'exponential'")

    # Pre-allocate array for Allan Variance resuts
    avar = np.empty((len(taus), data.shape[1]))

    for i, tau in enumerate(taus):
        # Reshape data into intervals of length tau
        n_intervals = n_samples // tau
        reshaped = data[:n_intervals * tau].reshape(n_intervals, tau, -1)

        # Compute means and differences
        interval_means = reshaped.mean(axis=1)
        diffs = np.diff(interval_means, axis=0)

        # Compute the Allan Variance
        avar[i] = 0.5 * np.mean(diffs**2, axis=0)

    return taus / fs, avar

def auto_estimate_R_q_from_allan(
    tau: np.ndarray, 
    sigma: np.ndarray, 
    fs: np.float,
    slope_tol: float=0.1, 
    min_points: int=5,
    plot: bool=False,
    u: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[float, float, Tuple[int, int], Tuple[int, int]]:
    """
    Automatically estimate R and q from Allan deviation curve.
    It is assumed the standard 1-state random–walk + white-noise measurement model
    for a sensor signal.

    d_k = p_k + b_k + v_k ,    v_k ~ N(0, R)
    b_{k+1} = b_k + w_k ,      w_k ~ N(0, q·T_s)

    where:
        - d_k = sensor measurement 
        - p_k = true measurement
        - R   = white measurement–noise variance [u²]
        - b_k = barometer bias at step k
        - q   = bias random–walk intensity [u²/s]
        - T_s = sampling time [s]

    Args:
        tau: Array of interval lengths in seconds.
        sigma: Array of corresponding Allan Deviation values [u].
        fs: Sampling frecuency [Hz].
        slope_tol: Allowed deviation from ideal slopes (-0.5, +0.5).
        min_points: Minimum number of consecutive points to accept a region.
        plot: Plot flag.
        u: Plot units.
        title: Plot title.

    Returns:
        R: Measurement noise variance [u^2].
        q: Random walk intensity [u^2/s].
        tau_white_region: (min_tau, max_tau) used for white noise fit.
        tau_rw_region: (min_tau, max_tau) used for random walk fit.
    """
    logtau = np.log10(tau)
    logsig = np.log10(sigma)

    # Local slopes between adjacent points
    slopes = np.diff(logsig) / np.diff(logtau)
    
    def find_region(
        target_slope: float
    ) -> Optional[Tuple[int, int]]:
        """"
        Find regions whit an desired slope.

        Args:
            target_slop: Desired slope.
        
        Returns:
            A tuple containing the indices of the region.
        """
        mask = np.abs(slopes - target_slope) < slope_tol
        # Group consecutive True values
        regions = []
        start = None
        for i, m in enumerate(mask):
            if m and start is None:
                start = i
            elif not m and start is not None:
                if i - start + 1 >= min_points:
                    regions.append((start, i))
                start = None
        if start is not None and len(slopes)-start >= min_points:
            regions.append((start, len(slopes)-1))
        if not regions:
            return None
        # Choose longest region
        region = max(regions, key=lambda r: r[1]-r[0])
        return region
    
    # White noise region 
    reg_w = find_region(-0.5)
    if reg_w:
        idx = range(reg_w[0], reg_w[1]+1)
        _, intercept_w, *_ = linregress(logtau[idx], logsig[idx])
        # Model: sigma = sqrt(R)/sqrt(tau) => log10(sigma) = -0.5*log10(tau) + log10(sqrt(R/fs))
        sqrtR_fs = 10**intercept_w
        R = (sqrtR_fs**2) * fs
        tau_white = (tau[idx[0]], tau[idx[-1]])
    else:
        R, tau_white = np.nan, None

    # Random walk region 
    reg_rw = find_region(0.5)
    if reg_rw:
        idx = range(reg_rw[0], reg_rw[1]+1)
        _, intercept_rw, *_ = linregress(logtau[idx], logsig[idx])
        # Model: sigma = sqrt(q/3) * sqrt(tau) => log10(sigma) = 0.5*log10(tau) + log10(sqrt(q/3))
        sqrt_q_over_3 = 10**intercept_rw
        q = 3 * (sqrt_q_over_3**2)
        tau_rw = (tau[idx[0]], tau[idx[-1]])
    else:
        q, tau_rw = np.nan, None

    if plot:
        utils.show_loglog_data(tau, np.vstack([sigma,np.sqrt(R/fs)/np.sqrt(tau),np.sqrt(q/3)*np.sqrt(tau)]).T, 
                               legend=["Sensor measurement Allan Dev.","White-Gaussian Noise","Random-Walk bias"],
                               xlabel="Interval Length [s]", ylabel=f"Sensor signal Allan deviation {u}",
                               title=title)

    return R, q, tau_white, tau_rw