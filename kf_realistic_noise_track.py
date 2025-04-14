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

# --- Realistic Noise Model Functions ---

def generate_bpsk_signal_with_doppler_complex(t, data_bits, samples_per_bit, fc, A_d, f_d, waveform_snr_db):
    """
    Generates a complex BPSK-like signal with sinusoidal Doppler shift on the carrier,
    and adds complex AWGN based on waveform SNR.

    Returns:
    noisy_signal_complex: The noisy complex signal waveform
    true_inst_freq: The true instantaneous frequency over time
    """
    dt = t[1] - t[0]
    n_steps = len(t)
    samples_per_symbol = samples_per_bit # BPSK

    # Create Baseband Signal (+1/-1)
    baseband = np.repeat(data_bits, samples_per_symbol)
    if len(baseband) > n_steps:
        baseband = baseband[:n_steps]
    elif len(baseband) < n_steps:
         padding = np.ones(n_steps - len(baseband)) * baseband[-1]
         baseband = np.concatenate((baseband, padding))

    # Calculate Instantaneous Frequency and Phase
    omega_d = 2 * np.pi * f_d
    true_doppler_shift = A_d * np.sin(omega_d * t)
    true_inst_freq = fc + true_doppler_shift
    instantaneous_phase = 2 * np.pi * np.cumsum(true_inst_freq) * dt

    # Generate Clean Complex Signal (Baseband * Complex Exponential)
    signal_complex_clean = baseband * np.exp(1j * instantaneous_phase)

    # Calculate Signal Power
    signal_power = np.mean(np.abs(signal_complex_clean)**2)

    # Calculate Noise Variance from Waveform SNR
    snr_linear = 10**(waveform_snr_db / 10.0)
    noise_variance = signal_power / snr_linear

    # Generate Complex AWGN
    # Variance is split equally between real and imaginary parts
    noise_std_per_component = np.sqrt(noise_variance / 2.0)
    noise_real = np.random.normal(0, noise_std_per_component, n_steps)
    noise_imag = np.random.normal(0, noise_std_per_component, n_steps)
    complex_noise = noise_real + 1j * noise_imag

    # Add Noise to Signal
    noisy_signal_complex = signal_complex_clean + complex_noise

    print(f"Waveform SNR: {waveform_snr_db:.1f} dB -> Complex Noise Variance: {noise_variance:.4f}")

    return noisy_signal_complex, true_inst_freq

def estimate_frequency_phase_diff(noisy_signal_complex, dt, fc):
    """
    Estimates instantaneous frequency from complex signal using phase difference.
    """
    n_steps = len(noisy_signal_complex)
    # Calculate phase difference between consecutive samples
    phase_diff = np.angle(noisy_signal_complex[1:] * np.conj(noisy_signal_complex[:-1]))

    # Convert phase difference to frequency deviation
    freq_deviation_estimate = phase_diff / (2 * np.pi * dt)

    # Add nominal carrier frequency back
    freq_estimate = fc + freq_deviation_estimate

    # Handle the first sample - repeat the first valid estimate
    noisy_freq_estimates = np.zeros(n_steps)
    noisy_freq_estimates[0] = fc + freq_deviation_estimate[0] # Simple first estimate
    noisy_freq_estimates[1:] = freq_estimate

    return noisy_freq_estimates


# --- Simulation Core Function ---
def run_kf_simulation(q_fddot_multiplier, waveform_snr_db=10.0, R_variance_guess=1500**2):
    """Runs the KF simulation for a given Q multiplier and returns results."""
    # --- Simulation Parameters ---
    data_rate = 16000
    samples_per_bit = 8
    fs = data_rate * samples_per_bit # Sample rate
    dt = 1.0 / fs # Time step
    duration = 4 # Simulation duration (seconds)
    n_data_bits = int(duration * data_rate) # Number of random data bits
    T_data = duration # Duration of random data part
    barker13 = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1]) # Preamble
    n_preamble_bits = len(barker13)
    T_preamble = n_preamble_bits / data_rate # Duration of preamble
    T_total = T_preamble + T_data # Total simulation time
    t = np.arange(0, T_total, dt) # Time vector
    n_steps = len(t)

    # --- Signal Parameters ---
    fc = 10e9  # True nominal carrier frequency
    A_d = 5000  # Doppler amplitude (Hz)
    f_d = 0.5   # Doppler frequency variation (Hz)

    # --- Data Generation ---
    random_bits = np.random.randint(0, 2, n_data_bits) * 2 - 1 # Generate random +/-1 bits
    data_bits = np.concatenate((barker13, random_bits)) # Combine preamble and random data

    # --- Generate Signal with Waveform Noise ---
    # Note: Signal generation depends on waveform_snr_db passed to this function
    noisy_signal_complex, true_inst_freq = generate_bpsk_signal_with_doppler_complex(
        t, data_bits, samples_per_bit, fc, A_d, f_d, waveform_snr_db
    )

    # --- Estimate Frequency from Noisy Waveform ---
    noisy_freq_measurements = estimate_frequency_phase_diff(noisy_signal_complex, dt, fc)

    # --- KF Setup (Constant Acceleration Model) ---
    # State: x = [f, f_dot, f_ddot]'
    omega_d = 2 * np.pi * f_d # Doppler angular frequency

    # State transition matrix F (Constant Acceleration)
    dt2_half = 0.5 * dt**2
    F = np.array([[1.0, dt,  dt2_half],
                  [0.0, 1.0, dt       ],
                  [0.0, 0.0, 1.0      ]])

    # Measurement matrix H
    H = np.array([[1.0, 0.0, 0.0]])

    # Process noise Q: Use the provided multiplier
    max_f_dddot = A_d * omega_d**3
    q_fddot_variance = (max_f_dddot * dt)**2 * q_fddot_multiplier # Use passed multiplier
    q_fdot_base = (1 * dt)**2
    q_f_base = (1 * dt)**2
    Q = np.diag([q_f_base, q_fdot_base, q_fddot_variance])

    # Measurement noise covariance R: Use the provided guess
    R = np.array([[R_variance_guess]])

    # Initial state covariance P0
    P0 = np.diag([1e6, 1e8, 1e10])

    # Initial state x0
    x0 = np.array([[fc], [0.0], [0.0]])

    # Create Kalman filter
    kf = KalmanFilter(F, H, Q, R, P0, x0)

    # --- Run KF ---
    kf_state_estimates = np.zeros((n_steps, kf.n_states, 1))
    kf_freq_estimates = np.zeros(n_steps)

    for i in range(n_steps):
        kf.predict()
        measurement = np.array([[noisy_freq_measurements[i]]])
        kf.update(measurement)
        kf_state_estimates[i] = kf.x
        kf_freq_estimates[i] = kf.x[0, 0]

    # Return results needed for tuning and plotting
    results = {
        "t": t,
        "true_inst_freq": true_inst_freq,
        "noisy_freq_measurements": noisy_freq_measurements,
        "kf_freq_estimates": kf_freq_estimates,
        "fc": fc,
        "A_d": A_d,
        "waveform_snr_db": waveform_snr_db,
        "R": R,
        "Q": Q, # Store Q used for this run
        "q_fddot_multiplier": q_fddot_multiplier
    }
    return results


# --- Plotting Function ---
def plot_results(results, model_name="Const Accel", filename="kf_tuning_result.png"):
    """Generates the standard 3-panel plot from simulation results."""
    t = results["t"]
    true_inst_freq = results["true_inst_freq"]
    noisy_freq_measurements = results["noisy_freq_measurements"]
    kf_freq_estimates = results["kf_freq_estimates"]
    fc = results["fc"]
    A_d = results["A_d"]
    waveform_snr_db = results["waveform_snr_db"]
    R = results["R"]
    q_fddot_multiplier = results["q_fddot_multiplier"]

    plt.figure(figsize=(12, 10))

    # Top plot: KF Performance
    plt.subplot(3, 1, 1)
    plt.plot(t, true_inst_freq / 1e3, 'g-', label='True Instantaneous Freq (kHz)')
    plt.plot(t, kf_freq_estimates / 1e3, 'r-', linewidth=1.5, label=f'KF {model_name} Estimate (kHz)')
    plt.ylabel('Frequency (kHz)')
    title = (f'KF ({model_name}) Tracking (SNR={waveform_snr_db:.1f}dB, R={R[0,0]:.0f}, '
             f'Q_mult={q_fddot_multiplier:.2e})')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim((fc - 1.5*A_d)/1e3, (fc + 1.5*A_d)/1e3)
    plt.xticks([])

    # Middle plot: KF Estimation Error
    plt.subplot(3, 1, 2)
    error = (kf_freq_estimates - true_inst_freq) / 1e3 # Error in kHz
    plt.plot(t, error, 'm-', label=f'KF {model_name} Estimation Error (kHz)')
    plt.ylabel('KF Error (kHz)')
    plt.legend()
    plt.grid(True)
    plt.xticks([])

    # Bottom plot: Raw Estimator vs True Frequency
    plt.subplot(3, 1, 3)
    plt.plot(t, true_inst_freq / 1e3, 'g-', label='True Instantaneous Freq (kHz)')
    plt.plot(t, noisy_freq_measurements / 1e3, 'b.', markersize=2, alpha=0.7, label='Raw Freq Estimates (kHz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')
    plt.title('Frequency Estimator Output vs. True Frequency')
    plt.legend()
    plt.grid(True)
    plt.ylim((fc - 1.5*A_d)/1e3, (fc + 1.5*A_d)/1e3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    # plt.show() # Usually disabled during tuning


# --- Tuning Function ---
def tune_q_fddot_multiplier(num_iterations=50, start_multiplier=0.22):
    """Tunes the q_fddot_multiplier by minimizing max error after t=2s."""
    print(f"--- Starting Q Tuning ({num_iterations} iterations) ---")
    # Define search space (logarithmic around start_multiplier)
    # e.g., from start/100 to start*100
    min_log_mult = np.log10(start_multiplier / 100)
    max_log_mult = np.log10(start_multiplier * 100)
    multipliers_to_test = np.logspace(min_log_mult, max_log_mult, num_iterations)

    best_multiplier = None
    min_max_error = float('inf')

    # Base parameters (could be passed as args if needed)
    waveform_snr_db = 10.0
    R_variance_guess = 1500**2

    for i, multiplier in enumerate(multipliers_to_test):
        # Run simulation with current multiplier
        results = run_kf_simulation(
            q_fddot_multiplier=multiplier,
            waveform_snr_db=waveform_snr_db,
            R_variance_guess=R_variance_guess
        )

        # Calculate performance metric (max error after t=2s)
        t = results["t"]
        kf_freq_estimates = results["kf_freq_estimates"]
        true_inst_freq = results["true_inst_freq"]

        indices = np.where(t >= 2.0)[0]
        if len(indices) > 0:
            abs_error = np.abs(kf_freq_estimates[indices] - true_inst_freq[indices])
            current_max_error = np.max(abs_error)
        else:
            current_max_error = float('inf') # Handle case where duration < 2s

        print(f"Iter {i+1}/{num_iterations}: Q_mult={multiplier:.3e} -> Max Error (t>=2s)={current_max_error/1e3:.4f} kHz")

        # Update best result
        if current_max_error < min_max_error:
            min_max_error = current_max_error
            best_multiplier = multiplier

    print(f"--- Tuning Complete ---")
    if best_multiplier is not None:
        print(f"Best Q Multiplier: {best_multiplier:.3e}")
        print(f"Minimum Max Error (t>=2s): {min_max_error/1e3:.4f} kHz")
    else:
        print("No best multiplier found (check simulation duration and error calculation).")

    return best_multiplier


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Tune the Q multiplier
    best_q_mult = tune_q_fddot_multiplier(num_iterations=50, start_multiplier=0.22)

    # 2. Run the simulation one last time with the best multiplier
    if best_q_mult is not None:
        print("\n--- Running final simulation with best Q multiplier ---")
        final_results = run_kf_simulation(q_fddot_multiplier=best_q_mult)

        # 3. Plot the results of the final simulation
        plot_results(final_results,
                     model_name="Const Accel (Tuned Q)",
                     filename="kf_realistic_noise_track_tuned.png")
        plt.show() # Show the final plot
    else:
        print("\nSkipping final simulation and plot as no best multiplier was found.")
