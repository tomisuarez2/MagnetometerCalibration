"""
Magnetometer Calibration Module
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "Iterative calibration method for inertial and magnetic sensors" - Dorveaux et al., (2009)
"""
from typing import Union, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

def fit_sphere_ls(
    raw_mag_data: np.ndarray
) -> Tuple[float, float]:
    """
    Algebraic sphere fit: ||x - c||^2 = R^2 in order to get coarse bias c.
    Solves linear LS for [c_x, c_y, c_z, d], with R^2 = c·c + d.

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

def compute_general_matrix(
    a_params: np.ndarray,
) -> np.ndarray:
    """
    Computes general A matrix according to Dorveaux et al. (2009).

    Args:
        a_params: Upper triangular general A matrix elements, array of shape (9,)

    Returns:
        Upper triangular general A of shape (3,3)
     """
    # Split parameters into components
    #a11, a12, a13, a22, a23, a33 = a_params
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = a_params
    
    
    # Return general matrix (A)
    #return np.array([[a11, a12, a13],[0, a22, a23],[0, 0, a33]], dtype=np.float64)
    return np.array([[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]], dtype=np.float32)
    
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
          where A is a general upper triangular matrix and B is the zero bias vector
    """
    # Split parameters into components
    #a_params, B = np.split(params, [6])
    a_params, B = np.split(params, [9])

    # Construct general matrix (A)
    A = compute_general_matrix(a_params)

    # Apply calibration in vectorized form:
    # Equivalent to: (A @ raw_data.T + B).T
    return raw_data @ A.T + B

def design_matrix_and_target(
    iter_mag_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the linear LS system for Eq. (4) at iteration k as presented in Dorveaux et al. (2009):
        [A y_i,k + B] ≈ u_i,k  with  u_i = y_i,k / ||y_i,k||
    Stacks 1,2,3 components for all i.

    Args:
        iter_mag_data: (N,3) current dataset y_i,k

    Returns:
        M: (3N, 12) design matrix
        u: (3N,)   target vector
    """
    n_samples = iter_mag_data.shape[0]
    #mx, my, mz = iter_mag_data[:,0], iter_mag_data[:,1], iter_mag_data[:,2] 

    # Unit targets u_i,k = y_i,k / ||y_i,k||
    norms = np.linalg.norm(iter_mag_data, axis=1, keepdims=True)
    u = iter_mag_data / norms

    # Design matrix blocks 9 parameters
    # Row for 1-component: [y1 y2 y3  0  0  0  1 0 0]
    # Row for 2-component: [ 0  0  0 y2 y3  0  0 1 0]
    # Row for 3-component: [ 0  0  0  0  0 y3  0 0 1]
    #M = np.zeros((3 * n_samples, 9), dtype=np.float64)
    # Design matrix blocks 12 parameters
    # Row for 1-component: [y1 y2 y3  0  0  0  0  0  0  1 0 0]
    # Row for 2-component: [ 0  0  0 y1 y2 y3  0  0  0  0 1 0]
    # Row for 3-component: [ 0  0  0  0  0  0 y1 y2 y3  0 0 1]
    M = np.zeros((3 * n_samples, 12), dtype=np.float64)

    # 1 rows
    #M[0::3, 0:3] = iter_mag_data
    #M[0::3, 6]   = 1.0
    # 2 rows
    #M[1::3, 3] = my
    #M[1::3, 4] = mz
    #M[1::3, 7]  = 1.0
    # 3 rows
    #M[2::3, 5] = mz
    #M[2::3, 8]  = 1.0

    # 1 rows
    M[0::3, 0:3] = iter_mag_data
    M[0::3, 9]   = 1.0
    # 2 rows
    M[1::3, 3:6] = iter_mag_data
    M[1::3, 10]  = 1.0
    # 3 rows
    M[2::3, 6:9] = iter_mag_data
    M[2::3, 11]  = 1.0

    u = u.reshape(-1)  # stack 1,2,3 for all i
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
        n_iterations: Maximum iterations

    Returns:
        Optimal calibration parameters (12,) [A_tilde (row-wise 9), B_tilde (3)]
    """
    # t_init samples
    start = int(t_init * fs)

    # Sphere-fit to get coarse bias c
    c, R = fit_sphere_ls(raw_mag_data[start:])
    print(f"Coarse bias c: {c}")
    print(f"Radius: {R}")

    # De-bias data
    raw_mag_data_unbiased = raw_mag_data[start:] - c
    #raw_mag_data_unbiased = raw_mag_data.copy()

    # Pre-allocate results storage
    thetas: list[np.ndarray] = []
    costs: list[float] = []

    print(">>> Magnetometer calibration in progress...")

    cal_mag_data = raw_mag_data_unbiased.copy()
    for k in range(n_max_iteration):
        M, y = design_matrix_and_target(cal_mag_data)
        # Solve M θ ≈ y
        theta, cost, _, _ = np.linalg.lstsq(M, y, rcond=None)

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
    A_tilde_k = compute_general_matrix(thetas[0][:9])
    B_tilde_k = thetas[0][9:]
    for i in range(1, n_iter):
         A_k = compute_general_matrix(thetas[i][:9])
         B_k = thetas[i][9:]
         A_tilde_k =  A_k @ A_tilde_k 
         B_tilde_k = A_k @ B_tilde_k + B_k 

    # Compose with coarse bias c
    B_tilde_k = A_tilde_k @ (-c) + B_tilde_k

    # Get parameters from matrices
    #params_mag = np.array([A_tilde_k[0,0],A_tilde_k[0,1],A_tilde_k[0,2],A_tilde_k[1,1],A_tilde_k[1,2],A_tilde_k[2,2],B_tilde_k[0], B_tilde_k[1], B_tilde_k[2]])
    params_mag = np.hstack((A_tilde_k.ravel(), B_tilde_k))
    #params_mag = thetas[0]

    print(">>> Magnetometer calibration finished")
    print(f">>> Number of interations: {n_iter}")
    return params_mag,c

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
    Plots a magnetometer 3D data and unitary sphere if specified.
    It can be used to visualize calibration results.

    Args:
        mag_data: Data array of shape (N, 3) where N is number of samples.
        title: Figure title
        u_sphere: If true plots unitary sphere
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.scatter(0,0,0)

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
#----------Functions for sinthetic data generation----------
#===========================================================

def sample_unit_sphere(n_samples: int, method: str = "rand"):
    """
    Sample points uniformly on the unit sphere.

    method:
      - "rand": random directions via normal sampling (fast, uniform on sphere)
      - "grid": deterministic sampling (not uniform) - uses random for now
    Returns: (N,3) array
    """
    if method == "rand":
        x = np.random.normal(size=(n_samples, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x
    else:
        # fallback
        x = np.random.normal(size=(n_samples, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x
    
def random_soft_iron(scale_min=0.7, scale_max=1.5, rotation_magnitude=0.5, seed=None):
    """
    Build a random A_distort matrix with:
      - diagonal scales in [scale_min, scale_max]
      - a small random rotation controlled by rotation_magnitude (radians)
    Returns (3,3) invertible matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    # random diagonal scales
    scales = np.random.uniform(scale_min, scale_max, size=3)
    D = np.diag(scales)
    # random rotation via axis-angle
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

def generate_raw_data(n_samples: int,
                      A_distort: np.ndarray,
                      b_distort: np.ndarray,
                      noise_std: float = 0.0,
                      outlier_fraction: float = 0.0,
                      outlier_scale: float = 5.0,
                      clip_min: Optional[float] = None,
                      clip_max: Optional[float] = None,
                      seed: Optional[int] = None):
    """
    Generate measured (raw) magnetometer data from unit field vectors.
    Returns:
      m_true: (N,3) unit sphere
      m_raw : (N,3) measured (distorted)
    """
    if seed is not None:
        np.random.seed(seed)
    m_true = sample_unit_sphere(n_samples)
    m_raw = (A_distort @ m_true.T).T + b_distort

    # Add gaussian noise
    if noise_std and noise_std > 0:
        m_raw += np.random.normal(scale=noise_std, size=m_raw.shape)

    # Outliers (large random points)
    if outlier_fraction and outlier_fraction > 0:
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

def evaluate_calibration(A_cal: np.ndarray, b_cal: np.ndarray, A_distort: np.ndarray, b_distort: np.ndarray, m_true: np.ndarray, m_raw: np.ndarray):
    """
    Evaluate calibration quality.
    Computes:
      - Norm error stats of calibrated points vs 1
      - Parameter differences (A_cal * A_distort ≈ I, A_cal @ b_distort + b_cal ≈ 0)
    """
    # Apply calibration
    m_est = (A_cal @ m_raw.T).T + b_cal
    norms = np.linalg.norm(m_est, axis=1)

    # Expected relationships
    compA = A_cal @ A_distort
    compb = A_cal @ b_distort + b_cal

    print("Calibrated norms: mean {:.6f}, std {:.6f}, min {:.6f}, max {:.6f}".format(norms.mean(), norms.std(), norms.min(), norms.max()))
    print("Composition A_cal @ A_distort (should be ~I):\n", compA)
    print("Composition error vs Identity (Frobenius norm):", np.linalg.norm(compA - np.eye(3)))
    print("Composition for bias (should be near zero):", compb, " norm:", np.linalg.norm(compb))

    # If m_true known, show the direct RMSE vs ground-truth directions/coords
    rmse_coords = np.sqrt(np.mean((m_est - m_true)**2))
    rmse_norms = np.sqrt(np.mean((np.linalg.norm(m_est,axis=1) - 1.0)**2))
    print("RMSE (coords vs true):", rmse_coords)
    print("RMSE (norm vs 1):", rmse_norms)

    return {"m_est": m_est, "compA": compA, "compb": compb, "rmse_coords": rmse_coords, "rmse_norms": rmse_norms}
    
