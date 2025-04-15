import numpy as np
import matplotlib.pyplot as plt

class ExtendedKalmanFilter:
    def __init__(self, Q, R, P0, x0):
        """
        Initialize Extended Kalman Filter.
        Jacobians F and H will be calculated dynamically if needed,
        or provided if constant.

        Parameters:
        Q: Process noise covariance
        R: Measurement noise covariance
        P0: Initial state covariance
        x0: Initial state
        """
        self.Q = Q
        self.R = R
        self.P = P0
        self.x = x0
        self.n_states = x0.shape[0]

    def predict(self, f_func, F_jacobian_func, dt, **kwargs):
        """
        Prediction step using state transition function f and its Jacobian F.

        Parameters:
        f_func: The non-linear state transition function x_k = f(x_{k-1}, dt, **kwargs)
        F_jacobian_func: Function to compute the Jacobian F = df/dx evaluated at self.x
        dt: Time step
        **kwargs: Additional arguments needed by f_func or F_jacobian_func
        """
        # Calculate Jacobian at current state estimate
        F = F_jacobian_func(self.x, dt, **kwargs)

        # State prediction using the (potentially non-linear) state transition function
        self.x = f_func(self.x, dt, **kwargs)

        # Covariance prediction
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

        return self.x

    def update(self, z, h_func, H_jacobian_func, **kwargs):
        """
        Update step using measurement z, measurement function h, and its Jacobian H.

        Parameters:
        z: Measurement
        h_func: The (potentially non-linear) measurement function z_pred = h(x_pred, **kwargs)
        H_jacobian_func: Function to compute the Jacobian H = dh/dx evaluated at predicted state self.x
        **kwargs: Additional arguments needed by h_func or H_jacobian_func
        """
        # Calculate Jacobian at predicted state
        H = H_jacobian_func(self.x, **kwargs)

        # Measurement prediction using the (potentially non-linear) measurement function
        z_pred = h_func(self.x, **kwargs)

        # Measurement residual
        y = z - z_pred

        # Residual covariance
        S = np.dot(np.dot(H, self.P), H.T) + self.R

        # Kalman gain
        S_inv = np.linalg.inv(S)
        K = np.dot(np.dot(self.P, H.T), S_inv)

        # State update
        self.x = self.x + np.dot(K, y)

        # Covariance update (Joseph form for stability)
        I = np.eye(self.n_states)
        ImKH = I - np.dot(K, H)
        self.P = np.dot(np.dot(ImKH, self.P), ImKH.T) + np.dot(np.dot(K, self.R), K.T)

        return self.x

# --- Functions copied and potentially modified from kalman_filter_sine_wave.py ---

def generate_bpsk_signal_with_doppler(t, data_bits, samples_per_bit, fc, A_d, f_d, noise_std_signal):
    """
    Generates a BPSK-like signal with sinusoidal Doppler shift on the carrier.
    (Identical to the previous version)
    """
    dt = t[1] - t[0]
    n_steps = len(t)
    samples_per_symbol = samples_per_bit # BPSK

    baseband = np.repeat(data_bits, samples_per_symbol)
    if len(baseband) > n_steps:
        baseband = baseband[:n_steps]
    elif len(baseband) < n_steps:
         padding = np.ones(n_steps - len(baseband)) * baseband[-1]
         baseband = np.concatenate((baseband, padding))

    omega_d = 2 * np.pi * f_d
    true_doppler_shift = A_d * np.sin(omega_d * t)
    true_inst_freq = fc + true_doppler_shift

    instantaneous_phase = 2 * np.pi * np.cumsum(true_inst_freq) * dt
    signal_clean = baseband * np.cos(instantaneous_phase)
    noise = np.random.normal(0, noise_std_signal, n_steps)
    signal = signal_clean + noise

    return signal, true_inst_freq

def estimate_frequency_simple(signal, fc_approx, fs):
    """
    Placeholder frequency estimator.
    (Identical to the previous version, relies on global variables for shortcut)
    """
    global t_global, A_d_global, f_d_global, noise_std_freq_global, fc_global
    omega_d = 2 * np.pi * f_d_global
    true_doppler_shift = A_d_global * np.sin(omega_d * t_global)
    # Use fc_global (true fc) here for calculating true freq, but add noise
    true_inst_freq = fc_global + true_doppler_shift
    noise = np.random.normal(0, noise_std_freq_global, len(t_global))
    # Return noisy version centered around fc_approx passed to function? No, center around true fc.
    return true_inst_freq + noise


# --- EKF Specific Functions ---

def state_transition_function(x, dt, omega_d):
    """
    State transition function f(x) for x = [f - fc, f_dot].
    x_k = F_doppler * x_{k-1}
    This is linear, so f(x) = F_doppler * x.
    """
    F_doppler = np.array([
        [np.cos(omega_d*dt), np.sin(omega_d*dt)/omega_d if omega_d != 0 else dt],
        [-omega_d*np.sin(omega_d*dt), np.cos(omega_d*dt)]
    ])
    return np.dot(F_doppler, x)

def state_transition_jacobian(x, dt, omega_d):
    """
    Jacobian F = df/dx for the state transition.
    Since f(x) is linear (f(x) = F_doppler * x), the Jacobian is constant.
    """
    F_doppler = np.array([
        [np.cos(omega_d*dt), np.sin(omega_d*dt)/omega_d if omega_d != 0 else dt],
        [-omega_d*np.sin(omega_d*dt), np.cos(omega_d*dt)]
    ])
    return F_doppler

def measurement_function(x, fc):
    """
    Measurement function h(x). Predicts measurement z based on state x.
    State x = [f - fc, f_dot]. Measurement z = f.
    h(x) = x[0] + fc
    """
    return x[0] + fc

def measurement_jacobian(x, fc):
    """
    Jacobian H = dh/dx for the measurement function.
    h(x) = x[0] + fc. dh/d(x[0]) = 1, dh/d(x[1]) = 0.
    """
    return np.array([[1.0, 0.0]])


# --- Main Simulation ---

def main_ekf_frequency_tracking():
    # --- Simulation Parameters (mostly same as before) ---
    data_rate = 16000
    samples_per_bit = 8
    fs = data_rate * samples_per_bit
    dt = 1.0 / fs
    duration = 4
    n_data_bits = int(duration * data_rate)
    T_data = duration
    barker13 = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1])
    n_preamble_bits = len(barker13)
    T_preamble = n_preamble_bits / data_rate
    T_total = T_preamble + T_data
    t = np.arange(0, T_total, dt)
    n_steps = len(t)

    global t_global # For simple estimator shortcut
    t_global = t

    # --- Signal Parameters ---
    fc = 10e9  # True nominal carrier frequency
    A_d = 5000
    f_d = 0.5
    noise_std_signal = 0.5
    noise_std_freq = 1000 # Measurement noise std dev for frequency

    # Globals for simple estimator shortcut
    global A_d_global, f_d_global, noise_std_freq_global, fc_global
    A_d_global = A_d
    f_d_global = f_d
    noise_std_freq_global = noise_std_freq
    fc_global = fc # Store true fc for estimator

    # --- Data Generation ---
    random_bits = np.random.randint(0, 2, n_data_bits) * 2 - 1
    data_bits = np.concatenate((barker13, random_bits))

    # --- Generate Signal and True Frequency ---
    signal, true_inst_freq = generate_bpsk_signal_with_doppler(
        t, data_bits, samples_per_bit, fc, A_d, f_d, noise_std_signal
    )

    # --- Generate Noisy Frequency Measurements (Simulation Shortcut) ---
    noisy_freq_measurements = estimate_frequency_simple(signal, fc, fs) # Pass true fc to estimator

    # --- EKF Setup ---
    # State: x = [f - fc, f_dot]'
    omega_d = 2 * np.pi * f_d

    # Process noise Q: How much variance in [f-fc, f_dot] change per step?
    # Need a 2x2 matrix. Let's assume some noise in both state elements.
    # Heuristic based on max rate of change.
    max_f_dot = A_d * omega_d
    # max_f_ddot = A_d * omega_d**2
    q1 = (max_f_dot * dt)**2 * 0.1
    q2 = (A_d * omega_d**2 * dt)**2 * 0.1 # Noise in acceleration term?
    Q = np.diag([q1, q2]) # Simplified: Assume uncorrelated process noise

    # Measurement noise covariance R
    R = np.array([[noise_std_freq**2]])

    # Initial state covariance P0 (high uncertainty)
    P0 = np.diag([1e6, 1e8]) # Uncertainty in initial f-fc and f_dot

    # Initial state x0 = [f(0)-fc, f_dot(0)]'
    # f(0) = fc + A_d*sin(0) = fc => f(0)-fc = 0
    # f_dot(0) = A_d*omega_d*cos(0) = A_d*omega_d
    x0 = np.array([[0.0], [A_d * omega_d]])

    # Create Extended Kalman filter
    ekf = ExtendedKalmanFilter(Q, R, P0, x0)

    # --- Run EKF ---
    ekf_state_estimates = np.zeros((n_steps, ekf.n_states, 1))
    ekf_freq_estimates = np.zeros(n_steps)

    for i in range(n_steps):
        # Prediction
        ekf.predict(state_transition_function, state_transition_jacobian, dt, omega_d=omega_d)

        # Update with measurement
        measurement = np.array([[noisy_freq_measurements[i]]])
        ekf.update(measurement, measurement_function, measurement_jacobian, fc=fc)

        # Store filtered state
        ekf_state_estimates[i] = ekf.x
        # Estimated frequency is x[0] + fc
        ekf_freq_estimates[i] = ekf.x[0, 0] + fc

    # --- Plot Results ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1) # Top plot: Frequencies
    plt.plot(t, true_inst_freq / 1e3, 'g-', label='True Instantaneous Freq (kHz)')
    plt.plot(t, noisy_freq_measurements / 1e3, 'b.', markersize=2, label='Noisy Measurements (kHz)')
    plt.plot(t, ekf_freq_estimates / 1e3, 'r-', linewidth=1.5, label='EKF Estimate (kHz)')
    plt.ylabel('Frequency (kHz)')
    plt.title(f'EKF Tracking Frequency (fc={fc/1e9:.1f}GHz, Doppler={A_d/1e3:.1f}kHz @ {f_d}Hz)')
    plt.legend()
    plt.grid(True)
    plt.ylim((fc - 1.5*A_d)/1e3, (fc + 1.5*A_d)/1e3)

    plt.subplot(2, 1, 2) # Bottom plot: Estimation Error
    error = (ekf_freq_estimates - true_inst_freq) / 1e3 # Error in kHz
    plt.plot(t, error, 'm-', label='EKF Estimation Error (kHz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (kHz)')
    plt.title('EKF Estimation Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('plots/ekf_frequency_track.png')
    print("Plot saved to plots/ekf_frequency_track.png")
    plt.show() # Display interactive plot window

if __name__ == "__main__":
    main_ekf_frequency_tracking()
