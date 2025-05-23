import numpy as np
import matplotlib.pyplot as plt
# (Removed unused scipy import)

# --- Standard Kalman Filter Class (Adapted from original script) ---
class KalmanFilter:
    def __init__(self, F, H, Q, R, P0, x0, A_d, omega_d):
        """
        Initialize Kalman Filter

        Parameters:
        F: State transition matrix
        H: Measurement matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        P0: Initial state covariance
        x0: Initial state
        A_d: Doppler amplitude (Hz)
        omega_d: Doppler angular frequency (rad/s)
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P0
        self.x = x0.reshape(-1, 1) # Ensure x is a column vector
        self.n_states = self.x.shape[0]
        self.A_d = A_d
        self.omega_d = omega_d

    def predict(self):
        """
        Prediction step
        """
        # State prediction: x_k|k-1 = F * x_k-1|k-1
        self.x = np.dot(self.F, self.x)
        # Covariance prediction: P_k|k-1 = F * P_k-1|k-1 * F.T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # Add state constraints
        max_freq_shift = self.A_d * 1.2  # 20% margin
        max_freq_rate = self.A_d * self.omega_d * 1.2

        # Constrain frequency shift
        self.x[1] = np.clip(self.x[1], -max_freq_shift, max_freq_shift)
        # Constrain frequency rate
        self.x[2] = np.clip(self.x[2], -max_freq_rate, max_freq_rate)

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
        y = np.arctan2(np.sin(y_raw), np.cos(y_raw))  # Normalize to [-pi, pi]

        # Residual covariance (innovation covariance): S = H * P_k|k-1 * H.T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain: K = P_k|k-1 * H.T * S^-1
        # Ensure S is treated as a scalar if it's 1x1
        S_inv = 1.0 / S[0,0] if S.shape == (1,1) else np.linalg.inv(S)
        K = np.dot(np.dot(self.P, self.H.T), S_inv)

        # State update with additional phase normalization
        self.x = self.x + np.dot(K, y)
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
    Estimates instantaneous phase difference from complex signal (measurement 'z').
    This acts as the measurement 'z' for the KF.
    Assumes the BPSK modulation has been removed or is handled.
    Here, we just take the angle, acknowledging BPSK flips will add noise.
    """
    # Simple phase extraction - BPSK flips will appear as +/- pi jumps
    # A more sophisticated approach might try to remove modulation first.
    noisy_phase_measurements = np.angle(noisy_signal_complex)
    return noisy_phase_measurements


# --- Signal Reconstruction Function ---
def reconstruct_signal(data_bits, kf_phase_diff_est, kf_freq_shift_est, fc, t, dt, samples_per_bit):
    """
    Reconstructs the complex signal using estimated phase and frequency.

    Args:
        data_bits: The original transmitted bits (+1/-1), including preamble.
        kf_phase_diff_est: Estimated phase difference trajectory (rad).
        kf_freq_shift_est: Estimated frequency shift trajectory (Hz).
        fc: Nominal carrier frequency (Hz).
        t: Time vector (s).
        dt: Time step (s).
        samples_per_bit: Number of samples per data bit.

    Returns:
        reconstructed_signal_complex: The reconstructed complex signal.
    """
    n_steps = len(t)
    # Create Baseband Signal (+1/-1) from original bits
    baseband = np.repeat(data_bits, samples_per_bit)
    if len(baseband) > n_steps:
        baseband = baseband[:n_steps]
    elif len(baseband) < n_steps:
         padding = np.ones(n_steps - len(baseband)) * baseband[-1]
         baseband = np.concatenate((baseband, padding))

    # Use the estimated phase difference directly from the KF state
    estimated_total_phase = 2 * np.pi * fc * t + kf_phase_diff_est

    # Reconstruct the signal
    reconstructed_signal_complex = baseband * np.exp(1j * estimated_total_phase)
    return reconstructed_signal_complex


# --- BER Calculation Function ---
def calculate_ber(original_bits, reconstructed_signal_complex, samples_per_bit, data_rate, window_size_bits=100, n_preamble_bits=0):
    """
    Calculates Bit Error Rate (BER) by comparing original bits to demodulated
    reconstructed signal. Also calculates BER over a sliding window.
    Excludes preamble bits from BER calculation.

    Args:
        original_bits: The original transmitted bits (+1/-1).
        reconstructed_signal_complex: The reconstructed complex signal.
        samples_per_bit: Number of samples per data bit.
        data_rate: The data rate in bits per second (Hz).
        window_size_bits: Size of the sliding window for BER calculation (in bits).
        n_preamble_bits: Number of preamble bits to exclude from BER calculation.

    Returns:
        overall_ber: The overall Bit Error Rate.
        ber_over_time: Array of BER values calculated over the sliding window.
        ber_time_vector: Time vector corresponding to the center of each BER window.
    """
    n_total_samples = len(reconstructed_signal_complex)
    n_total_bits = len(original_bits)

    # --- Demodulation ---
    # Sample the reconstructed signal at the center of each bit interval
    # Ensure we don't exceed signal length
    max_bits_in_signal = n_total_samples // samples_per_bit
    num_bits_to_demod = min(n_total_bits, max_bits_in_signal)

    # Calculate sample indices for the middle of each bit
    sample_indices = np.arange(samples_per_bit // 2, num_bits_to_demod * samples_per_bit, samples_per_bit)

    # Ensure indices are within bounds
    sample_indices = sample_indices[sample_indices < n_total_samples]
    num_bits_to_demod = len(sample_indices) # Update based on valid indices

    # Extract samples and demodulate (BPSK: check sign of real part)
    sampled_values = reconstructed_signal_complex[sample_indices]
    estimated_bits = np.sign(np.real(sampled_values))

    # Align original bits with the number of demodulated bits
    original_bits_aligned = original_bits[:num_bits_to_demod]

    # --- Exclude preamble from BER calculation ---
    if n_preamble_bits > 0 and n_preamble_bits < num_bits_to_demod:
        # Skip preamble bits for BER calculation
        data_original = original_bits_aligned[n_preamble_bits:]
        data_estimated = estimated_bits[n_preamble_bits:]
        num_data_bits = len(data_original)

        # Calculate errors only on data bits (excluding preamble)
        errors = np.sum(data_original != data_estimated)
        overall_ber = errors / num_data_bits if num_data_bits > 0 else 0
        print(f"Data-only BER (excluding {n_preamble_bits} preamble bits): {overall_ber:.4f} ({errors}/{num_data_bits} errors)")
    else:
        # If no preamble or preamble is too long, use all bits
        data_original = original_bits_aligned
        data_estimated = estimated_bits
        errors = np.sum(data_original != data_estimated)
        overall_ber = errors / num_bits_to_demod if num_bits_to_demod > 0 else 0
        print(f"Overall BER: {overall_ber:.4f} ({errors}/{num_bits_to_demod} errors)")

    # --- Sliding Window BER (on data bits only, excluding preamble) ---
    ber_over_time = []
    ber_bit_indices = [] # Store the index of the *last* bit in the window

    # Start the sliding window after the preamble
    start_idx = max(0, n_preamble_bits)
    num_data_bits = num_bits_to_demod - start_idx

    if num_data_bits >= window_size_bits:
        for i in range(num_data_bits - window_size_bits + 1):
            # i is relative to start_idx
            abs_i = i + start_idx  # absolute index in the full bit sequence
            window_original = original_bits_aligned[abs_i : abs_i + window_size_bits]
            window_estimated = estimated_bits[abs_i : abs_i + window_size_bits]
            window_errors = np.sum(window_original != window_estimated)
            window_ber = window_errors / window_size_bits
            ber_over_time.append(window_ber)
            ber_bit_indices.append(abs_i + window_size_bits - 1) # Index of last bit in window

    ber_over_time = np.array(ber_over_time)
    ber_bit_indices = np.array(ber_bit_indices)

    # Convert bit indices to time (time of the *end* of the window)
    ber_time_vector = (ber_bit_indices + 1) / data_rate # Time at the end of the window

    # Adjust time vector to represent the center of the window for plotting
    window_duration_time = window_size_bits / data_rate
    ber_time_vector_centered = ber_time_vector - (window_duration_time / 2.0)

    return overall_ber, ber_over_time, ber_time_vector_centered


# --- Preamble Processing Function ---
def estimate_initial_state_from_preamble(noisy_signal_complex, preamble, samples_per_bit, dt, fc):
    """
    Estimates initial phase difference, frequency shift, and frequency rate using the known preamble.

    Args:
        noisy_signal_complex: The complex signal containing the preamble
        preamble: The known preamble sequence (+1/-1 values)
        samples_per_bit: Number of samples per bit
        dt: Time step
        fc: Nominal carrier frequency

    Returns:
        initial_state: Array [phase_diff, freq_shift, freq_rate] for KF initialization
    """
    n_preamble_bits = len(preamble)
    n_preamble_samples = n_preamble_bits * samples_per_bit

    # Extract the preamble portion of the signal
    preamble_signal = noisy_signal_complex[:n_preamble_samples]

    # Create expected preamble waveform (without Doppler effects)
    preamble_baseband = np.repeat(preamble, samples_per_bit)

    # Remove BPSK modulation by multiplying by known preamble
    # This converts phase flips back to continuous phase
    demodulated_preamble = preamble_signal * preamble_baseband

    # Extract phase from demodulated signal
    demodulated_phase = np.angle(demodulated_preamble)
    unwrapped_phase = np.unwrap(demodulated_phase)

    # Estimate frequency shift using linear regression on unwrapped phase
    t_preamble = np.arange(len(unwrapped_phase)) * dt
    # Remove nominal carrier phase (2*pi*fc*t) to get just the Doppler component
    phase_diff = unwrapped_phase - 2*np.pi*fc*t_preamble

    # Fit a quadratic polynomial to the phase difference
    # phase_diff ≈ p0 + p1*t + p2*t^2
    # where p0 is initial phase, p1/(2π) is freq shift, p2/(π) is freq rate
    poly_coeffs = np.polyfit(t_preamble, phase_diff, 2)

    # Extract initial state estimates
    initial_phase_diff = poly_coeffs[2]  # Constant term (p0)
    initial_freq_shift = poly_coeffs[1] / (2*np.pi)  # Linear term (p1/(2π))
    initial_freq_rate = 2 * poly_coeffs[0] / np.pi  # Quadratic term (2*p2/π)

    print(f"Preamble-based initial state estimate:")
    print(f"  Phase diff: {initial_phase_diff:.4f} rad")
    print(f"  Freq shift: {initial_freq_shift:.2f} Hz")
    print(f"  Freq rate: {initial_freq_rate:.2f} Hz/s")

    return np.array([initial_phase_diff, initial_freq_shift, initial_freq_rate])

# --- Simulation Core Function ---
def run_akf_simulation(waveform_snr_db=15):
    """Runs the AKF simulation, including reconstruction and BER calculation."""
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

    # --- Use preamble to estimate initial state ---
    initial_state = estimate_initial_state_from_preamble(
        noisy_signal_complex, barker13, samples_per_bit, dt, fc
    )

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

    # Initial state x0 [phase_diff, freq_shift, freq_rate] from preamble
    x0 = initial_state  # Use preamble-based estimate instead of zeros

    # Create Kalman filter
    kf = KalmanFilter(F, H, Q, R, P0, x0, A_d, omega_d)

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

    # --- Signal Reconstruction ---
    reconstructed_signal = reconstruct_signal(
        data_bits, kf_phase_diff_est_unwrapped, kf_freq_shift_est, fc, t, dt, samples_per_bit
    )

    # --- BER Calculation (excluding preamble) ---
    overall_ber, ber_over_time, ber_time_vector = calculate_ber(
        data_bits, reconstructed_signal, samples_per_bit, data_rate,
        window_size_bits=200, n_preamble_bits=n_preamble_bits # Exclude preamble from BER
    )


    # Return results needed for plotting and analysis
    results = {
        "t": t,
        "data_bits": data_bits, # Added
        "samples_per_bit": samples_per_bit, # Added
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
        "reconstructed_signal": reconstructed_signal, # Added
        "overall_ber": overall_ber, # Added
        "ber_over_time": ber_over_time, # Added
        "ber_time_vector": ber_time_vector, # Added
    }
    return results


# --- Plotting Function ---
def plot_results(results, model_name="Phase/Freq/Rate KF", filename="plots/akf_phase_freq_rate_track.png"):
    """Generates plots for the AKF results, including BER."""
    t = results["t"]
    true_phase_diff = results["true_phase_diff"]
    true_freq_shift = results["true_freq_shift"]
    true_freq_rate = results["true_freq_rate"]
    # noisy_phase_measurements = results["noisy_phase_measurements"] # No longer plotted directly
    kf_phase_diff_est = results["kf_phase_diff_est"] # Used for error calculation
    kf_freq_shift_est = results["kf_freq_shift_est"]
    kf_freq_rate_est = results["kf_freq_rate_est"]
    fc = results["fc"]
    A_d = results["A_d"]
    f_d = results["f_d"]
    waveform_snr_db = results["waveform_snr_db"]
    R = results["R"]
    Q = results["Q"]
    ber_over_time = results["ber_over_time"]
    ber_time_vector = results["ber_time_vector"]
    overall_ber = results["overall_ber"]

    plt.figure(figsize=(12, 12)) # Keep height for 4 plots

    # --- Plot 1 (Previously 2): Frequency Shift ---
    plt.subplot(4, 1, 1)
    plt.plot(t, true_freq_shift / 1e3, 'g-', label='True Freq Shift (kHz)')
    plt.plot(t, kf_freq_shift_est / 1e3, 'r-', linewidth=1.5, label='KF Est Freq Shift (kHz)')
    plt.ylabel('Freq Shift (kHz)')
    plt.title(f'{model_name} Tracking (SNR={waveform_snr_db:.1f}dB, R={R[0,0]:.1e}, Q=[{Q[0,0]:.1e},{Q[1,1]:.1e},{Q[2,2]:.1e}])')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks([]) # Remove x-axis labels for top plots
    plt.ylim(-1.5*A_d/1e3, 1.5*A_d/1e3) # Set y-limits based on Doppler amplitude

    # --- Plot 2 (Previously 3): Frequency Rate ---
    plt.subplot(4, 1, 2)
    plt.plot(t, true_freq_rate / 1e3, 'g-', label='True Freq Rate (kHz/s)')
    plt.plot(t, kf_freq_rate_est / 1e3, 'r-', linewidth=1.5, label='KF Est Freq Rate (kHz/s)')
    plt.ylabel('Freq Rate (kHz/s)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks([]) # Remove x-axis labels
    max_rate = A_d * (2*np.pi*f_d) # Max true rate
    plt.ylim(-1.5*max_rate/1e3, 1.5*max_rate/1e3) # Set y-limits based on max rate

    # --- Plot 3 (Previously 4): Estimation Errors ---
    plt.subplot(4, 1, 3)
    phase_error = kf_phase_diff_est - true_phase_diff
    freq_shift_error = (kf_freq_shift_est - true_freq_shift) / 1e3 # kHz
    freq_rate_error = (kf_freq_rate_est - true_freq_rate) / 1e3 # kHz/s

    # --- Filter data for error plot (t >= 0.5s) ---
    error_plot_start_time = 0.5
    mask = t >= error_plot_start_time

    plt.plot(t[mask], phase_error[mask], 'm-', label=f'Phase Diff Error (rad)')
    plt.plot(t[mask], freq_shift_error[mask], 'c-', label=f'Freq Shift Error (kHz)')
    plt.plot(t[mask], freq_rate_error[mask], 'y-', label=f'Freq Rate Error (kHz/s)')
    plt.ylabel('Estimation Error')
    plt.title(f'Estimation Errors (Starting from {error_plot_start_time}s)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks([]) # Remove x-axis labels
    # Optional: Adjust y-limits for error plot if needed
    # error_max = max(np.max(np.abs(phase_error[mask])), np.max(np.abs(freq_shift_error[mask])), np.max(np.abs(freq_rate_error[mask]))) * 1.1
    # plt.ylim(-error_max, error_max)

    # --- Plot 4: BER Performance (excluding preamble) ---
    plt.subplot(4, 1, 4)
    if len(ber_time_vector) > 0:
        plt.semilogy(ber_time_vector, ber_over_time, 'b-', label=f'Sliding Window BER (Win={int(results.get("window_size_bits", 200))} bits)')
        # Add a horizontal line for overall BER for comparison
        plt.axhline(overall_ber, color='r', linestyle='--', label=f'Overall BER: {overall_ber:.2e} (excl. preamble)')
        plt.ylim(bottom=max(1e-5, overall_ber / 10) if overall_ber > 0 else 1e-5, top=1.0) # Adjust y-limits for log scale
    else:
        plt.text(0.5, 0.5, 'BER calculation requires more data\nor smaller window size',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.ylim(1e-5, 1.0) # Default limits even if no data

    plt.xlabel('Time (s) [Window Center]')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER Performance Over Time')
    plt.legend(loc='upper right')
    plt.grid(True, which='both') # Grid for major and minor ticks on log scale


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout slightly for main title
    plt.suptitle(f'{model_name} Results (SNR={waveform_snr_db:.1f}dB)', fontsize=14, y=0.99) # Add overall title
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    # plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # --- Set Simulation Parameters ---
    simulation_snr_db = 15.0 # Example SNR in dB

    print(f"--- Running AKF Simulation (Phase/Freq/Rate) with SNR = {simulation_snr_db} dB ---")

    # --- Run the simulation ---
    # Q and R are set inside run_akf_simulation based on heuristics and SNR
    final_results = run_akf_simulation(
        waveform_snr_db=simulation_snr_db
    )
    # Pass necessary parameters for BER time conversion if needed, or ensure they are calculated within
    # For now, assuming calculate_ber can get fs from dt within run_akf_simulation's scope or t vector.

    # --- Plot final results ---
    plot_results(final_results,
                 model_name=f"AKF Phase/Freq/Rate (SNR={simulation_snr_db}dB)",
                 filename=f"plots/akf_phase_freq_rate_track_snr{int(simulation_snr_db)}db.png")

    print("--- Simulation Complete ---")
    plt.show() # Show the final plot interactively
