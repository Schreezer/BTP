import numpy as np
import matplotlib.pyplot as plt

# --- Standard Kalman Filter Class (from original script) ---
class KalmanFilter:
    def __init__(self, F, H, Q, R, P0, x0):
        """
        Initialize Kalman Filter

        Parameters:
        F: State transition matrix
        H: Measurement matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        P0: Initial state covariance
        x0: Initial state
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P0
        self.x = x0
        self.n_states = x0.shape[0] # Added for consistency

    def predict(self):
        """
        Prediction step
        """
        # State prediction
        self.x = np.dot(self.F, self.x)
        # Covariance prediction
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """
        Update step

        Parameters:
        z: Measurement (Note: Should correspond to H*x)
        """
        # Measurement residual (y = z - H*x_predicted)
        y = z - np.dot(self.H, self.x)

        # Residual covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain
        # Ensure S is treated as a scalar if it's 1x1
        S_inv = 1.0 / S[0,0] if S.shape == (1,1) else np.linalg.inv(S)
        K = np.dot(np.dot(self.P, self.H.T), S_inv)

        # State update
        self.x = self.x + np.dot(K, y)

        # Covariance update (Joseph form for stability)
        I = np.eye(self.n_states)
        ImKH = I - np.dot(K, self.H)
        self.P = np.dot(np.dot(ImKH, self.P), ImKH.T) + np.dot(np.dot(K, self.R), K.T)
        return self.x

# --- Functions copied from previous scripts ---

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

# --- Main Simulation ---

def main_kf_const_vel_frequency_tracking():
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
    A_d = 5000  # Doppler amplitude (Hz)
    f_d = 0.5   # Doppler frequency variation (Hz)
    noise_std_signal = 0.5 # Noise added to the baseband signal itself (arbitrary)

    # --- SNR Configuration ---
    target_snr_db = 11.0 # Target SNR for frequency measurement in dB
    signal_power = (A_d**2) / 2.0
    snr_linear = 10**(target_snr_db / 10.0)
    noise_variance = signal_power / snr_linear
    noise_std_freq = np.sqrt(noise_variance) # Calculated measurement noise std dev
    print(f"Target SNR: {target_snr_db:.1f} dB -> Noise Std Dev: {noise_std_freq:.2f} Hz")

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

    # --- KF Setup (Constant Velocity Model) ---
    # State: x = [f, f_dot]'
    omega_d = 2 * np.pi * f_d # Still needed for heuristics

    # State transition matrix F (Constant Velocity)
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])

    # Measurement matrix H
    # Measurement z = f = H * x
    H = np.array([[1.0, 0.0]])

    # Process noise Q: Uncertainty in constant velocity model
    # Heuristic: q_fdot related to max expected acceleration
    max_f_ddot = A_d * omega_d**2
    q_fdot = (max_f_ddot * dt)**2 * 1024# Variance for f_dot change
    q_f = (10 * dt)**2 # Small variance for frequency itself assuming model is okay
    Q = np.diag([q_f, q_fdot])

    # Measurement noise covariance R (using calculated noise variance)
    R = np.array([[noise_variance]])

    # Initial state covariance P0 (high uncertainty)
    P0 = np.diag([1e6, 1e8]) # Uncertainty in initial f and f_dot

    # Initial state x0 = [f(0), f_dot(0)]'
    # Guess nominal frequency and zero rate of change
    x0 = np.array([[fc], [0.0]])

    # Create Kalman filter
    kf = KalmanFilter(F, H, Q, R, P0, x0)

    # --- Run KF ---
    kf_state_estimates = np.zeros((n_steps, kf.n_states, 1))
    kf_freq_estimates = np.zeros(n_steps)

    for i in range(n_steps):
        # Prediction
        kf.predict()

        # Measurement is the total noisy frequency
        measurement = np.array([[noisy_freq_measurements[i]]])

        # Update with measurement
        kf.update(measurement)

        # Store filtered state
        kf_state_estimates[i] = kf.x
        # Estimated frequency is state x[0]
        kf_freq_estimates[i] = kf.x[0, 0]

    # --- Plot Results ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1) # Top plot: Frequencies
    plt.plot(t, true_inst_freq / 1e3, 'g-', label='True Instantaneous Freq (kHz)')
    plt.plot(t, noisy_freq_measurements / 1e3, 'b.', markersize=2, label='Noisy Measurements (kHz)')
    plt.plot(t, kf_freq_estimates / 1e3, 'r-', linewidth=1.5, label='KF Const Vel Estimate (kHz)')
    plt.ylabel('Frequency (kHz)')
    plt.title(f'KF (Const Vel) Tracking Frequency (fc={fc/1e9:.1f}GHz, Doppler={A_d/1e3:.1f}kHz @ {f_d}Hz, SNR={target_snr_db:.1f}dB)')
    plt.legend()
    plt.grid(True)
    plt.ylim((fc - 1.5*A_d)/1e3, (fc + 1.5*A_d)/1e3)

    plt.subplot(2, 1, 2) # Bottom plot: Estimation Error
    error = (kf_freq_estimates - true_inst_freq) / 1e3 # Error in kHz
    plt.plot(t, error, 'm-', label='KF Const Vel Estimation Error (kHz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (kHz)')
    plt.title('KF (Constant Velocity) Estimation Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('plots/kf_const_vel_frequency_track.png')
    print("Plot saved to plots/kf_const_vel_frequency_track.png")
    plt.show() # Display interactive plot window

if __name__ == "__main__":
    main_kf_const_vel_frequency_tracking()
