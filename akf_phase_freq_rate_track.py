import numpy as np
import matplotlib.pyplot as plt

# --- Standard Kalman Filter Class (Adapted from original script) ---
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
        self.x = x0.reshape(-1, 1) # Ensure x is a column vector
        self.n_states = self.x.shape[0]

    def predict(self):
        """
        Prediction step
        """
        # State prediction: x_k|k-1 = F * x_k-1|k-1
        self.x = np.dot(self.F, self.x)
        # Covariance prediction: P_k|k-1 = F * P_k-1|k-1 * F.T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """
        Update step

        Parameters:
        z: Measurement (scalar phase difference in this case)
        """
        # Ensure z is treated as a scalar numpy array for consistency
        z_arr = np.array([[z]])

        # Measurement residual (innovation): y = z - H * x_k|k-1
        # Need to handle phase wrapping for y: normalize to [-pi, pi]
        predicted_measurement = np.dot(self.H, self.x)
        y_raw = z_arr - predicted_measurement
        y = np.arctan2(np.sin(y_raw), np.cos(y_raw)) # Normalize phase difference

        # Residual covariance (innovation covariance): S = H * P_k|k-1 * H.T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain: K = P_k|k-1 * H.T * S^-1
        # Ensure S is treated as a scalar if it's 1x1
        S_inv = 1.0 / S[0,0] if S.shape == (1,1) else np.linalg.inv(S)
        K = np.dot(np.dot(self.P, self.H.T), S_inv)

        # State update: x_k|k = x_k|k-1 + K * y
        self.x = self.x + np.dot(K, y)
        # Normalize the phase state (x[0]) to [-pi, pi] after update
        self.x[0, 0] = np.arctan2(np.sin(self.x[0, 0]), np.cos(self.x[0, 0]))


        # Covariance update (Joseph form for stability): P_k|k = (I - K*H) * P_k|k-1 * (I - K*H).T + K*R*K.T
        I = np.eye(self.n_states)
        ImKH = I - np.dot(K, self.H)
        self.P = np.dot(np.dot(ImKH, self.P), ImKH.T) + np.dot(np.dot(K, self.R), K.T)
        return self.x

# --- Realistic Noise Model Functions (from original script) ---

def generate_bpsk_signal_with_doppler_complex(t, data_bits, samples_per_bit, fc, A_d, f_d, waveform_snr_db):
    """
    Generates a complex BPSK-like signal with sinusoidal Doppler shift on the carrier,
    and adds complex AWGN based on waveform SNR.

    Returns:
    noisy_signal_complex: The noisy complex signal waveform
    true_inst_freq: The true instantaneous frequency over time
    true_phase: The true instantaneous phase over time (relative to initial phase 0)
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
    true_doppler_shift = A_d * np.sin(omega_d * t) # This is xDx (Doppler Freq Shift)
    true_inst_freq = fc + true_doppler_shift
    # Integrate frequency shift to get phase relative to nominal carrier phase (2*pi*fc*t)
    # This represents the phase difference component xDu
    true_phase_diff = 2 * np.pi * np.cumsum(true_doppler_shift) * dt
    # Total instantaneous phase
    instantaneous_phase = 2 * np.pi * fc * t + true_phase_diff

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

    # Also return true phase difference for comparison
    return noisy_signal_complex, true_inst_freq, true_phase_diff

def estimate_phase_difference(noisy_signal_complex):
    """
    Estimates instantaneous phase difference from complex signal.
    This acts as the measurement 'z' for the KF.
    Assumes the BPSK modulation has been removed or is handled.
    Here, we just take the angle, acknowledging BPSK flips will add noise.
    """
    # Simple phase extraction - BPSK flips will appear as +/- pi jumps
    # A more sophisticated approach might try to remove modulation first.
    noisy_phase_measurements = np.angle(noisy_signal_complex)
    return noisy_phase_measurements


# --- Simulation Core Function ---
def run_akf_simulation(waveform_snr_db=15):
    """Runs the AKF simulation with fixed parameters."""
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
    fc = 10e9  # True nominal carrier frequency (Hz) - Not directly used by KF state
    A_d = 5000  # Doppler amplitude (Hz)
    f_d = 0.5   # Doppler frequency variation (Hz)
    omega_d = 2 * np.pi * f_d # Doppler angular frequency

    # --- Data Generation ---
    random_bits = np.random.randint(0, 2, n_data_bits) * 2 - 1 # Generate random +/-1 bits
    data_bits = np.concatenate((barker13, random_bits)) # Combine preamble and random data

    # --- Generate Signal with Waveform Noise ---
    noisy_signal_complex, true_inst_freq, true_phase_diff = generate_bpsk_signal_with_doppler_complex(
        t, data_bits, samples_per_bit, fc, A_d, f_d, waveform_snr_db
    )

    # --- Calculate True State Values for Comparison ---
    true_freq_shift = true_inst_freq - fc # xDx
    true_freq_rate = np.gradient(true_freq_shift, dt) # xDa

    # --- Estimate Phase Measurements from Noisy Waveform ---
    noisy_phase_measurements = estimate_phase_difference(noisy_signal_complex)

    # --- KF Setup (Phase, Frequency Shift, Frequency Rate Model) ---
    # State: x = [phase_diff (xDu), freq_shift (xDx), freq_rate (xDa)]'
    # Units: [rad, Hz, Hz/s]

    # State transition matrix F (Constant Doppler Rate)
    dt2_half = 0.5 * dt**2
    F = np.array([[1.0, 2*np.pi*dt, np.pi*dt**2], # Phase update: phi_k = phi_k-1 + 2*pi*f_k-1*dt + pi*fdot_k-1*dt^2
                  [0.0, 1.0,        dt         ], # Freq shift update: f_k = f_k-1 + fdot_k-1*dt
                  [0.0, 0.0,        1.0        ]]) # Freq rate update: fdot_k = fdot_k-1

    # Measurement matrix H (We measure phase difference)
    H = np.array([[1.0, 0.0, 0.0]])

    # Process noise Q: Needs careful tuning, setting fixed values for now
    # Variances for [phase_diff_process, freq_shift_process, freq_rate_process]
    # These represent uncertainty in the model dynamics per step.
    # Heuristics:
    # q_phasediff: Small, as it's mostly driven by freq state noise. Maybe (0.01 rad)^2?
    # q_freqshift: Related to uncertainty in freq_rate. Maybe (10 Hz * dt)^2?
    # q_freqrate: Related to uncertainty in freq_jerk (rate of change of rate).
    #             Max freq jerk ~ A_d * omega_d^2. Let variance be (Max_jerk * dt)^2 * multiplier
    max_freq_jerk = A_d * omega_d**2
    q_freqrate_var = (max_freq_jerk * dt)**2 * 0.1 # Example value
    q_freqshift_var = (100 * dt)**2 # Example value (Hz^2)
    q_phasediff_var = (0.5 * dt)**2 # Example value (rad^2) - driven by freq noise mostly
    Q = np.diag([q_phasediff_var, q_freqshift_var, q_freqrate_var])
    print(f"Using fixed Q = diag([{Q[0,0]:.2e}, {Q[1,1]:.2e}, {Q[2,2]:.2e}])")


    # Measurement noise covariance R: Variance of the phase measurement noise
    # Related to SNR. High SNR -> low R. Low SNR -> high R.
    # Also affected by BPSK modulation if not removed.
    # Let's estimate based on SNR. For complex noise variance N0, phase variance ~ N0 / (2 * SignalPower)
    snr_linear = 10**(waveform_snr_db / 10.0)
    # Approx Signal Power = 1 (since baseband is +/-1)
    approx_noise_variance = 1.0 / snr_linear
    R_variance_guess = approx_noise_variance / 2.0 # Variance of phase noise
    # Add extra variance due to BPSK phase flips? Heuristic increase.
    R_variance_guess *= 5 # Increase R to account for BPSK noise etc.
    R = np.array([[R_variance_guess]])
    print(f"Using fixed R = [[{R[0,0]:.2e}]] based on SNR={waveform_snr_db}dB")


    # Initial state covariance P0 (High uncertainty)
    P0 = np.diag([np.pi**2, (A_d*2)**2, (A_d*omega_d*2)**2]) # Large initial variance

    # Initial state x0 [phase_diff, freq_shift, freq_rate]
    x0 = np.array([0.0, 0.0, 0.0]) # Start assuming zero initial state

    # Create Kalman filter
    kf = KalmanFilter(F, H, Q, R, P0, x0)

    # --- Run KF ---
    n_states = kf.n_states
    kf_state_estimates = np.zeros((n_steps, n_states))

    for i in range(n_steps):
        kf.predict()
        measurement = noisy_phase_measurements[i]
        kf.update(measurement)
        kf_state_estimates[i, :] = kf.x.flatten() # Store flattened state vector

    # Extract individual state estimates
    kf_phase_diff_est = kf_state_estimates[:, 0]
    kf_freq_shift_est = kf_state_estimates[:, 1]
    kf_freq_rate_est = kf_state_estimates[:, 2]

    # --- Wrap estimated phase for plotting continuity ---
    kf_phase_diff_est_unwrapped = np.unwrap(kf_phase_diff_est)
    true_phase_diff_unwrapped = np.unwrap(true_phase_diff) # Should already be unwrapped by cumsum

    # Return results needed for plotting
    results = {
        "t": t,
        "true_phase_diff": true_phase_diff_unwrapped,
        "true_freq_shift": true_freq_shift,
        "true_freq_rate": true_freq_rate,
        "noisy_phase_measurements": noisy_phase_measurements,
        "kf_phase_diff_est": kf_phase_diff_est_unwrapped,
        "kf_freq_shift_est": kf_freq_shift_est,
        "kf_freq_rate_est": kf_freq_rate_est,
        "fc": fc,
        "A_d": A_d,
        "f_d": f_d,
        "waveform_snr_db": waveform_snr_db,
        "R": R,
        "Q": Q,
    }
    return results


# --- Plotting Function ---
def plot_results(results, model_name="Phase/Freq/Rate KF", filename="plots/akf_phase_freq_rate_track.png"):
    """Generates plots for the phase/frequency/rate KF results."""
    t = results["t"]
    true_phase_diff = results["true_phase_diff"]
    true_freq_shift = results["true_freq_shift"]
    true_freq_rate = results["true_freq_rate"]
    noisy_phase_measurements = results["noisy_phase_measurements"]
    kf_phase_diff_est = results["kf_phase_diff_est"]
    kf_freq_shift_est = results["kf_freq_shift_est"]
    kf_freq_rate_est = results["kf_freq_rate_est"]
    fc = results["fc"]
    A_d = results["A_d"]
    f_d = results["f_d"]
    waveform_snr_db = results["waveform_snr_db"]
    R = results["R"]
    Q = results["Q"]

    plt.figure(figsize=(12, 12)) # Increased height for 4 plots

    # --- Plot 1: Phase Difference ---
    plt.subplot(4, 1, 1)
    plt.plot(t, true_phase_diff, 'g-', label='True Phase Diff (rad)')
    plt.plot(t, kf_phase_diff_est, 'r-', linewidth=1.5, label='KF Est Phase Diff (rad)')
    # Plot noisy measurements (wrapped) for context
    plt.plot(t, np.unwrap(noisy_phase_measurements), 'b.', markersize=1, alpha=0.3, label='Noisy Phase Meas (Unwrapped)')
    plt.ylabel('Phase Diff (rad)')
    title = (f'{model_name} Tracking (SNR={waveform_snr_db:.1f}dB, R={R[0,0]:.1e}, '
             f'Q=[{Q[0,0]:.1e},{Q[1,1]:.1e},{Q[2,2]:.1e}])')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks([]) # Remove x-axis labels for top plots

    # --- Plot 2: Frequency Shift ---
    plt.subplot(4, 1, 2)
    plt.plot(t, true_freq_shift / 1e3, 'g-', label='True Freq Shift (kHz)')
    plt.plot(t, kf_freq_shift_est / 1e3, 'r-', linewidth=1.5, label='KF Est Freq Shift (kHz)')
    plt.ylabel('Freq Shift (kHz)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks([])
    plt.ylim(-1.5*A_d/1e3, 1.5*A_d/1e3) # Set y-limits based on Doppler amplitude

    # --- Plot 3: Frequency Rate ---
    plt.subplot(4, 1, 3)
    plt.plot(t, true_freq_rate / 1e3, 'g-', label='True Freq Rate (kHz/s)')
    plt.plot(t, kf_freq_rate_est / 1e3, 'r-', linewidth=1.5, label='KF Est Freq Rate (kHz/s)')
    plt.ylabel('Freq Rate (kHz/s)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks([])
    max_rate = A_d * (2*np.pi*f_d) # Max true rate
    plt.ylim(-1.5*max_rate/1e3, 1.5*max_rate/1e3) # Set y-limits based on max rate

    # --- Plot 4: Estimation Errors ---
    plt.subplot(4, 1, 4)
    phase_error = kf_phase_diff_est - true_phase_diff
    freq_shift_error = (kf_freq_shift_est - true_freq_shift) / 1e3 # kHz
    freq_rate_error = (kf_freq_rate_est - true_freq_rate) / 1e3 # kHz/s

    # --- Filter data for error plot (t >= 0.5s) ---
    error_plot_start_time = 0.5
    mask = t >= error_plot_start_time

    plt.plot(t[mask], phase_error[mask], 'm-', label=f'Phase Diff Error (rad)')
    plt.plot(t[mask], freq_shift_error[mask], 'c-', label=f'Freq Shift Error (kHz)')
    plt.plot(t[mask], freq_rate_error[mask], 'y-', label=f'Freq Rate Error (kHz/s)')
    plt.xlabel(f'Time (s) [Starting from {error_plot_start_time}s]')
    plt.ylabel('Estimation Error')
    plt.legend(loc='upper left')
    plt.grid(True)
    # Adjust y-limits for error plot if needed, e.g., focus after initial transient
    # plt.ylim(-0.5, 0.5) # Example error limits

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    # plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # --- Set Simulation Parameters ---
    simulation_snr_db = 15.0 # Example SNR in dB

    print(f"--- Running AKF Simulation (Phase/Freq/Rate) with SNR = {simulation_snr_db} dB ---")

    # --- Run the simulation ---
    # Q and R are now set inside run_akf_simulation based on heuristics and SNR
    final_results = run_akf_simulation(
        waveform_snr_db=simulation_snr_db
    )

    # --- Plot final results ---
    plot_results(final_results,
                 model_name=f"AKF Phase/Freq/Rate (SNR={simulation_snr_db}dB)",
                 filename=f"plots/akf_phase_freq_rate_track_snr{int(simulation_snr_db)}db.png")

    print("--- Simulation Complete ---")
    plt.show() # Show the final plot interactively
