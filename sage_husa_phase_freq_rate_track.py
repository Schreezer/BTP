import numpy as np
import matplotlib.pyplot as plt
# (Removed unused scipy import)

# --- Sage-Husa Adaptive Kalman Filter Class ---
class SageHusaKalmanFilter:
    def __init__(self, F, H, Q0, R0, P0, x0, A_d, omega_d, b=0.98, epsilon=1e-6):
        """
        Initialize Sage-Husa Adaptive Kalman Filter

        Parameters:
        F: State transition matrix
        H: Measurement matrix
        Q0: Initial process noise covariance estimate
        R0: Initial measurement noise covariance estimate
        P0: Initial state covariance
        x0: Initial state
        A_d: Doppler amplitude (Hz)
        omega_d: Doppler angular frequency (rad/s)
        b: Forgetting factor (0 < b < 1)
        epsilon: Small value to ensure positive definite covariance estimates
        """
        self.F = F
        self.H = H
        self.P = P0
        self.x = x0.reshape(-1, 1) # Ensure x is a column vector
        self.n_states = self.x.shape[0]
        self.n_measurements = self.H.shape[0] # Should be 1 for phase measurement
        self.A_d = A_d
        self.omega_d = omega_d
        self.b = b
        self.epsilon = epsilon # For numerical stability

        # Initialize noise estimates
        self.q_hat = np.zeros((self.n_states, 1))
        self.Q_hat = Q0.copy()
        self.r_hat = np.zeros((self.n_measurements, 1)) # Scalar in this case
        self.R_hat = R0.copy()

        self.k = 0 # Step counter
        self.x_prev_updated = self.x.copy() # Store previous updated state for q_hat update
        self.P_prev_updated = self.P.copy() # Store previous updated covariance for Q_hat update

    def predict(self):
        """
        Prediction step using Sage-Husa equations (Eq. 3, 4)
        """
        # Store previous updated state and covariance before prediction
        # Note: These are already stored at the end of the previous update step
        # self.x_prev_updated = self.x.copy() # No, self.x is already updated state from previous step
        # self.P_prev_updated = self.P.copy() # No, self.P is already updated cov from previous step

        # State prediction: x_k|k-1 = F * x_k-1|k-1 + q_hat_k-1 (Eq. 3)
        # self.x here is x_k-1|k-1 from the previous step
        x_predicted = np.dot(self.F, self.x) + self.q_hat
        # Covariance prediction: P_k|k-1 = F * P_k-1|k-1 * F.T + Q_hat_k-1 (Eq. 4)
        # self.P here is P_k-1|k-1 from the previous step
        P_predicted = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q_hat

        # Apply state constraints (Keep these as they are physical limits)
        max_freq_shift = self.A_d * 1.2  # 20% margin
        max_freq_rate = self.A_d * self.omega_d * 1.2
        x_predicted[1] = np.clip(x_predicted[1], -max_freq_shift, max_freq_shift)
        x_predicted[2] = np.clip(x_predicted[2], -max_freq_rate, max_freq_rate)

        # Update internal state to predicted state for the update step
        self.x = x_predicted
        self.P = P_predicted

        return self.x

    def update(self, z):
        """
        Update step using Sage-Husa equations (Eq. 5-11)

        Parameters:
        z: Measurement (scalar phase difference in this case)
        """
        # Ensure z is treated as a scalar numpy array for consistency
        z_arr = np.array([[z]]) # Measurement vector (1x1)
        x_predicted = self.x.copy() # Predicted state x_k|k-1 (from predict step)
        P_predicted = self.P.copy() # Predicted covariance P_k|k-1 (from predict step)

        # Store the state and covariance from the *previous* update step (k-1)
        # These are needed for noise updates (Eq 8, 9)
        x_prev_upd = self.x_prev_updated.copy()
        P_prev_upd = self.P_prev_updated.copy()

        # Calculate time-varying factor dk
        # Avoid potential division by zero if b is very close to 1 and k is large
        # or if b=1 (though we assume b<1)
        if self.b < 1.0:
            b_k_plus_1 = self.b**(self.k + 1)
            # Use a reasonable lower bound for the denominator to prevent overflow/division by zero
            denominator = 1.0 - b_k_plus_1
            if abs(denominator) < self.epsilon:
                 dk = 0.0 # Effectively stop adapting if b^k+1 is too small or b=1
            else:
                 dk = (1.0 - self.b) / denominator
        else: # Should not happen based on constraint 0 < b < 1
            dk = 1.0 / (self.k + 1.0) # Standard averaging if b=1

        # --- Standard Update Steps (using estimated noise stats) ---
        # Measurement residual (innovation): ek = z - H * x_k|k-1 - r_hat_k-1
        predicted_measurement = np.dot(self.H, x_predicted)
        e_raw = z_arr - predicted_measurement - self.r_hat
        ek = np.arctan2(np.sin(e_raw), np.cos(e_raw)) # Normalize phase residual to [-pi, pi] (1x1)

        # Residual covariance (innovation covariance): S = H * P_k|k-1 * H.T + R_hat_k-1
        S = np.dot(np.dot(self.H, P_predicted), self.H.T) + self.R_hat
        # Ensure S is positive definite (scalar case)
        if S[0,0] < self.epsilon:
            S[0,0] = self.epsilon

        # Kalman gain: Kk = P_k|k-1 * H.T * S^-1 (Eq. 5)
        S_inv = 1.0 / S[0,0] # Since S is 1x1
        Kk = np.dot(P_predicted, self.H.T) * S_inv # Kk is n_states x 1

        # State update: xk = x_k|k-1 + Kk * ek (Eq. 6)
        x_updated = x_predicted + np.dot(Kk, ek)
        # Normalize phase state
        x_updated[0, 0] = np.arctan2(np.sin(x_updated[0, 0]), np.cos(x_updated[0, 0]))

        # Covariance update: Pk = (I - Kk*H) * P_k|k-1 (Eq. 7)
        I = np.eye(self.n_states)
        P_updated = np.dot(I - np.dot(Kk, self.H), P_predicted)
        # Ensure P remains symmetric and positive semi-definite
        P_updated = (P_updated + P_updated.T) / 2.0
        # Optional: Add small diagonal term if needed for stability
        # P_updated += np.eye(self.n_states) * self.epsilon

        # --- Noise Statistics Update ---
        # Update measurement noise mean: r_hat_k = (1-dk)*r_hat_k-1 + dk*(z - H*x_k|k-1) (Eq. 10)
        # Note: Use the raw residual before phase wrapping for mean estimation
        raw_residual_for_mean = z_arr - predicted_measurement
        self.r_hat = (1.0 - dk) * self.r_hat + dk * raw_residual_for_mean

        # Update measurement noise covariance: R_hat_k = (1-dk)*R_hat_k-1 + dk*(ek*ek.T - H*P_k|k-1*H.T) (Eq. 11)
        # Ensure R_hat remains positive (scalar case)
        R_hat_new_term = dk * (np.dot(ek, ek.T) - np.dot(np.dot(self.H, P_predicted), self.H.T))
        self.R_hat = (1.0 - dk) * self.R_hat + R_hat_new_term
        if self.R_hat[0,0] < self.epsilon:
            self.R_hat[0,0] = self.epsilon

        # Update process noise mean: q_hat_k = (1-dk)*q_hat_k-1 + dk*(xk - F*x_k-1) (Eq. 8)
        # Uses xk (current updated state) and x_k-1 (previous updated state)
        self.q_hat = (1.0 - dk) * self.q_hat + dk * (x_updated - np.dot(self.F, x_prev_upd))

        # Update process noise covariance: Q_hat_k = (1-dk)*Q_hat_k-1 + dk*(Kk*ek*ek.T*Kk.T + Pk - F*P_k-1*F.T) (Eq. 9)
        # Uses Kk, ek, Pk (current updated covariance), P_k-1 (previous updated covariance)
        # ek is 1x1, Kk is nx1 -> Kk*ek*ek.T*Kk.T = Kk * Kk.T * ek^2
        ek_scalar_sq = ek[0,0]**2
        Q_hat_new_term = dk * (np.outer(Kk, Kk) * ek_scalar_sq + P_updated - np.dot(np.dot(self.F, P_prev_upd), self.F.T))
        self.Q_hat = (1.0 - dk) * self.Q_hat + Q_hat_new_term
        # Ensure Q_hat remains symmetric and positive semi-definite
        self.Q_hat = (self.Q_hat + self.Q_hat.T) / 2.0
        # Add small value to diagonal to maintain positive definiteness if needed
        min_diag = np.min(np.diag(self.Q_hat))
        if min_diag < self.epsilon:
             self.Q_hat += np.eye(self.n_states) * (self.epsilon - min_diag)


        # --- Prepare for next step ---
        # Store current updated state and covariance for the *next* prediction step
        self.x = x_updated
        self.P = P_updated
        # Store current updated state and covariance to be used as "previous" in the *next* update step's noise calcs
        self.x_prev_updated = x_updated.copy()
        self.P_prev_updated = P_updated.copy()

        # Increment step counter
        self.k += 1

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
def calculate_ber(original_bits, reconstructed_signal_complex, samples_per_bit, data_rate, window_size_bits=100):
    """
    Calculates Bit Error Rate (BER) by comparing original bits to demodulated
    reconstructed signal. Also calculates BER over a sliding window.

    Args:
        original_bits: The original transmitted bits (+1/-1).
        reconstructed_signal_complex: The reconstructed complex signal.
        samples_per_bit: Number of samples per data bit.
        data_rate: The data rate in bits per second (Hz).
        window_size_bits: Size of the sliding window for BER calculation (in bits).

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

    # --- Overall BER ---
    errors = np.sum(original_bits_aligned != estimated_bits)
    overall_ber = errors / num_bits_to_demod if num_bits_to_demod > 0 else 0

    # --- Sliding Window BER ---
    ber_over_time = []
    ber_bit_indices = [] # Store the index of the *last* bit in the window

    if num_bits_to_demod >= window_size_bits:
        for i in range(num_bits_to_demod - window_size_bits + 1):
            window_original = original_bits_aligned[i : i + window_size_bits]
            window_estimated = estimated_bits[i : i + window_size_bits]
            window_errors = np.sum(window_original != window_estimated)
            window_ber = window_errors / window_size_bits
            ber_over_time.append(window_ber)
            ber_bit_indices.append(i + window_size_bits - 1) # Index of last bit in window

    ber_over_time = np.array(ber_over_time)
    ber_bit_indices = np.array(ber_bit_indices)

    # Convert bit indices to time (time of the *end* of the window)
    # bit_duration = 1.0 / data_rate # data_rate is now passed as an argument
    ber_time_vector = (ber_bit_indices + 1) / data_rate # Time at the end of the window

    # Adjust time vector to represent the center of the window for plotting
    window_duration_time = window_size_bits / data_rate
    ber_time_vector_centered = ber_time_vector - (window_duration_time / 2.0)

    print(f"Overall BER: {overall_ber:.4f} ({errors}/{num_bits_to_demod} errors)")

    return overall_ber, ber_over_time, ber_time_vector_centered

# --- Simulation Core Function ---
def run_sage_husa_simulation(waveform_snr_db=15, forgetting_factor_b=1):
    """Runs the Sage-Husa AKF simulation, including reconstruction and BER calculation."""
    # --- Simulation Parameters ---
    data_rate = 16000
    samples_per_bit = 8
    fs = data_rate * samples_per_bit # Sample rate
    dt = 1.0 / fs # Time step
    duration = 6 # Simulation duration (seconds)
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

    # Create Sage-Husa Kalman filter
    # Pass initial Q and R as Q0 and R0
    kf = SageHusaKalmanFilter(F, H, Q, R, P0, x0, A_d, omega_d, b=forgetting_factor_b)

    # --- Run Sage-Husa KF ---
    n_states = kf.n_states
    kf_state_estimates = np.zeros((n_steps, n_states))
    # Store noise estimates for analysis
    q_hat_history = np.zeros((n_steps, n_states))
    Q_hat_diag_history = np.zeros((n_steps, n_states)) # Store only diagonal for simplicity
    r_hat_history = np.zeros((n_steps, kf.n_measurements))
    R_hat_history = np.zeros((n_steps, kf.n_measurements, kf.n_measurements))

    for i in range(n_steps):
        kf.predict()
        measurement = noisy_phase_measurements[i]
        kf.update(measurement)
        kf_state_estimates[i, :] = kf.x.flatten() # Store flattened state vector
        # Store noise estimates
        q_hat_history[i, :] = kf.q_hat.flatten()
        Q_hat_diag_history[i, :] = np.diag(kf.Q_hat)
        r_hat_history[i, :] = kf.r_hat.flatten()
        R_hat_history[i, :, :] = kf.R_hat

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

    # --- BER Calculation ---
    overall_ber, ber_over_time, ber_time_vector = calculate_ber(
        data_bits, reconstructed_signal, samples_per_bit, data_rate, window_size_bits=200 # Pass data_rate
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
        "R0": R, # Initial R used as R0
        "Q0": Q, # Initial Q used as Q0
        "b": forgetting_factor_b, # Added
        "q_hat_history": q_hat_history, # Added
        "Q_hat_diag_history": Q_hat_diag_history, # Added
        "r_hat_history": r_hat_history, # Added
        "R_hat_history": R_hat_history, # Added
        "reconstructed_signal": reconstructed_signal, # Added
        "overall_ber": overall_ber, # Added
        "ber_over_time": ber_over_time, # Added
        "ber_time_vector": ber_time_vector, # Added
    }
    return results

# --- Plotting Function ---
def plot_results(results, model_name="Sage-Husa Phase/Freq/Rate KF", filename="plots/sage_husa_phase_freq_rate_track.png"):
    """Generates plots for the Sage-Husa AKF results, including BER and noise estimates."""
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
    R0 = results["R0"] # Initial R
    Q0 = results["Q0"] # Initial Q
    b = results["b"]
    # Noise estimate histories
    q_hat_history = results["q_hat_history"]
    Q_hat_diag_history = results["Q_hat_diag_history"]
    r_hat_history = results["r_hat_history"]
    R_hat_history = results["R_hat_history"]
    # BER results
    ber_over_time = results["ber_over_time"]
    ber_time_vector = results["ber_time_vector"]
    overall_ber = results["overall_ber"]

    plt.figure(figsize=(12, 15)) # Increased height for 5 plots

    # --- Plot 1: Frequency Shift ---
    plt.subplot(5, 1, 1)
    plt.plot(t, true_freq_shift / 1e3, 'g-', label='True Freq Shift (kHz)')
    plt.plot(t, kf_freq_shift_est / 1e3, 'r-', linewidth=1.5, label='Est Freq Shift (kHz)')
    plt.ylabel('Freq Shift (kHz)')
    plt.title(f'{model_name} Tracking (SNR={waveform_snr_db:.1f}dB, b={b:.2f})')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks([]) # Remove x-axis labels
    plt.ylim(-1.5*A_d/1e3, 1.5*A_d/1e3) # Set y-limits based on Doppler amplitude

    # --- Plot 2: Frequency Rate ---
    plt.subplot(5, 1, 2)
    plt.plot(t, true_freq_rate / 1e3, 'g-', label='True Freq Rate (kHz/s)')
    plt.plot(t, kf_freq_rate_est / 1e3, 'r-', linewidth=1.5, label='Est Freq Rate (kHz/s)')
    plt.ylabel('Freq Rate (kHz/s)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks([]) # Remove x-axis labels
    max_rate = A_d * (2*np.pi*f_d) # Max true rate
    plt.ylim(-1.5*max_rate/1e3, 1.5*max_rate/1e3) # Set y-limits based on max rate

    # --- Plot 3: Estimation Errors ---
    plt.subplot(5, 1, 3)
    # Calculate wrapped phase error
    phase_error_unwrapped = kf_phase_diff_est - true_phase_diff
    phase_error_wrapped = np.arctan2(np.sin(phase_error_unwrapped), np.cos(phase_error_unwrapped))
    freq_shift_error = (kf_freq_shift_est - true_freq_shift) / 1e3 # kHz
    freq_rate_error = (kf_freq_rate_est - true_freq_rate) / 1e3 # kHz/s

    # Filter data for error plot (e.g., start after initial convergence)
    error_plot_start_time = 0.5
    mask = t >= error_plot_start_time

    plt.plot(t[mask], phase_error_wrapped[mask], 'm-', label=f'Wrapped Phase Diff Error (rad)') # Use wrapped error
    plt.plot(t[mask], freq_shift_error[mask], 'c-', label=f'Freq Shift Error (kHz)')
    plt.plot(t[mask], freq_rate_error[mask], 'y-', label=f'Freq Rate Error (kHz/s)')
    plt.ylabel('Estimation Error')
    plt.title(f'Estimation Errors (from {error_plot_start_time}s)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks([]) # Remove x-axis labels
    # Set y-limits appropriate for wrapped phase error
    plt.ylim(-np.pi * 1.1, np.pi * 1.1)

    # --- Plot 4: Adaptive Noise Covariances (R_hat, Q_hat diagonals) ---
    plt.subplot(5, 1, 4)
    # Plot R_hat (measurement noise variance)
    plt.semilogy(t, R_hat_history[:, 0, 0], 'b-', label=r'$\hat{R}_k$ (Phase Var)')
    plt.axhline(R0[0,0], color='b', linestyle='--', label=r'$R_0$ (Initial)')

    # Plot Q_hat diagonals (process noise variances)
    colors = ['r', 'g', 'm']
    labels = [r'$\hat{Q}_k(1,1)$ (Phase)', r'$\hat{Q}_k(2,2)$ (Freq)', r'$\hat{Q}_k(3,3)$ (Rate)']
    initial_labels = [r'$Q_0(1,1)$', r'$Q_0(2,2)$', r'$Q_0(3,3)$']
    for i in range(Q_hat_diag_history.shape[1]):
        plt.semilogy(t, Q_hat_diag_history[:, i], color=colors[i], linestyle='-', label=labels[i])
        plt.axhline(Q0[i,i], color=colors[i], linestyle='--', label=initial_labels[i])

    plt.ylabel('Noise Variance Est.')
    plt.title('Adaptive Noise Covariance Estimates (Diagonal)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Place legend outside
    plt.grid(True, which='both')
    plt.xticks([]) # Remove x-axis labels
    plt.ylim(bottom=1e-9) # Adjust lower limit if needed

    # --- Plot 5: BER Performance ---
    plt.subplot(5, 1, 5)
    if len(ber_time_vector) > 0:
        plt.semilogy(ber_time_vector, ber_over_time, 'b-', label=f'Sliding Window BER (Win={int(results.get("window_size_bits", 200))} bits)')
        plt.axhline(overall_ber, color='r', linestyle='--', label=f'Overall BER: {overall_ber:.2e}')
        plt.ylim(bottom=max(1e-5, overall_ber / 10) if overall_ber > 0 else 1e-5, top=1.0)
    else:
        plt.text(0.5, 0.5, 'BER requires more data/smaller window',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.ylim(1e-5, 1.0)

    plt.xlabel('Time (s)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER Performance Over Time')
    plt.legend(loc='upper right')
    plt.grid(True, which='both')

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.97]) # Adjust layout for main title and legend
    plt.suptitle(f'{model_name} Results (SNR={waveform_snr_db:.1f}dB, b={b:.2f})', fontsize=14, y=0.99)
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    # plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # --- Set Simulation Parameters ---
    simulation_snr_db = 15.0 # Example SNR in dB
    forgetting_factor = 0.99 # Set forgetting factor b

    print(f"--- Running Sage-Husa Simulation (Phase/Freq/Rate) with SNR = {simulation_snr_db} dB, b = {forgetting_factor} ---")

    # --- Run the simulation ---
    # Initial Q and R are set inside run_sage_husa_simulation based on heuristics and SNR
    final_results = run_sage_husa_simulation(
        waveform_snr_db=simulation_snr_db,
        forgetting_factor_b=forgetting_factor
    )

    # --- Plot final results ---
    plot_results(final_results,
                 model_name=f"Sage-Husa Phase/Freq/Rate (b={forgetting_factor})",
                 filename=f"plots/sage_husa_phase_freq_rate_track_snr{int(simulation_snr_db)}db_b{int(forgetting_factor*100)}.png")

    print("--- Simulation Complete ---")
    plt.show() # Show the final plot interactively
