import numpy as np
import matplotlib.pyplot as plt

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
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P0  # Initial state covariance
        self.x = x0  # Initial state

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
        z: Measurement
        """
        # Measurement residual
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
        I = np.eye(self.P.shape[0])
        ImKH = I - np.dot(K, self.H)
        self.P = np.dot(np.dot(ImKH, self.P), ImKH.T) + np.dot(np.dot(K, self.R), K.T)

        return self.x

def generate_bpsk_signal_with_doppler(t, data_bits, samples_per_bit, fc, A_d, f_d, noise_std_signal):
    """
    Generates a BPSK-like signal with sinusoidal Doppler shift on the carrier.
    Note: Simulates phase modulation based on integrated instantaneous frequency.

    Parameters:
    t: Time vector
    data_bits: Array of +1/-1 representing bits (including preamble)
    samples_per_bit: Number of samples for each bit duration
    fc: Nominal carrier frequency (Hz)
    A_d: Amplitude of Doppler shift (Hz)
    f_d: Frequency of Doppler shift variation (Hz)
    noise_std_signal: Standard deviation of AWGN added to the signal

    Returns:
    signal: The noisy modulated signal
    true_inst_freq: The true instantaneous frequency over time
    """
    dt = t[1] - t[0]
    n_steps = len(t)
    samples_per_symbol = samples_per_bit # BPSK

    # 1. Create Baseband Signal (+1/-1)
    baseband = np.repeat(data_bits, samples_per_symbol)
    # Ensure baseband signal matches length of t (truncate if necessary)
    if len(baseband) > n_steps:
        baseband = baseband[:n_steps]
    elif len(baseband) < n_steps:
         # Pad if necessary, though ideally t should match data length
         padding = np.ones(n_steps - len(baseband)) * baseband[-1]
         baseband = np.concatenate((baseband, padding))

    # 2. Calculate Instantaneous Frequency
    omega_d = 2 * np.pi * f_d
    true_doppler_shift = A_d * np.sin(omega_d * t)
    true_inst_freq = fc + true_doppler_shift

    # 3. Calculate Phase by Integrating Frequency
    # phi(t) = 2 * pi * integral(f_inst(tau) dtau) from 0 to t
    # Approximate integral using cumulative sum: cumsum(f_inst) * dt
    instantaneous_phase = 2 * np.pi * np.cumsum(true_inst_freq) * dt

    # 4. Generate Modulated Signal (using phase directly)
    # Using complex envelope representation is often better, but for simplicity:
    signal_clean = baseband * np.cos(instantaneous_phase)

    # 5. Add Noise
    noise = np.random.normal(0, noise_std_signal, n_steps)
    signal = signal_clean + noise

    return signal, true_inst_freq

def estimate_frequency_simple(signal, fc_approx, fs):
    """
    Placeholder for a real frequency estimator.
    For now, it just returns a noisy version of the true frequency
    (cheating for simulation purposes). In reality, this would involve
    phase differencing, STFT, or other DSP techniques on the 'signal'.

    Parameters:
    signal: The received noisy signal (currently unused in this placeholder)
    fc_approx: Approximate carrier frequency (used for the Kalman filter)
    fs: Sampling frequency

    Returns:
    noisy_freq_estimate: A simulated noisy frequency measurement
    """
    # This function needs the true frequency to add noise to it,
    # which isn't realistic but allows testing the KF.
    # We'll calculate the true frequency again here based on global params.
    # THIS IS A SIMULATION SHORTCUT.
    global t_global, A_d_global, f_d_global, noise_std_freq_global
    omega_d = 2 * np.pi * f_d_global
    true_doppler_shift = A_d_global * np.sin(omega_d * t_global)
    true_inst_freq = fc_approx + true_doppler_shift # Use fc_approx as base
    noise = np.random.normal(0, noise_std_freq_global, len(t_global))
    return true_inst_freq + noise


def main_frequency_tracking():
    # --- Simulation Parameters ---
    data_rate = 16000  # bits per second (16 kbps)
    samples_per_bit = 8
    fs = data_rate * samples_per_bit  # Sampling frequency (128 kHz)
    dt = 1.0 / fs
    duration = 2 # Duration of random data part in seconds
    n_data_bits = int(duration * data_rate) # Calculate number of data bits
    T_data = duration # Duration of data part
    barker13 = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1])
    n_preamble_bits = len(barker13)
    T_preamble = n_preamble_bits / data_rate
    T_total = T_preamble + T_data # Total simulation time
    t = np.arange(0, T_total, dt)
    n_steps = len(t)

    # Make t global for the simple estimator function (simulation shortcut)
    global t_global
    t_global = t

    # --- Signal Parameters ---
    fc = 10e9  # Nominal carrier frequency (10 GHz) - Note: High freq, simulation is conceptual
    A_d = 5000  # Doppler amplitude (5 kHz)
    f_d = 0.5   # Doppler frequency variation (0.5 Hz)
    noise_std_signal = 0.5 # Noise added to the modulated signal itself
    noise_std_freq = 1000 # Noise added to the frequency *measurement* (Hz) - Represents estimator error

    # Make Doppler params global for simple estimator
    global A_d_global, f_d_global, noise_std_freq_global
    A_d_global = A_d
    f_d_global = f_d
    noise_std_freq_global = noise_std_freq

    # --- Data Generation ---
    random_bits = np.random.randint(0, 2, n_data_bits) * 2 - 1 # Generate 0,1 then map to -1,+1
    data_bits = np.concatenate((barker13, random_bits))

    # --- Generate Signal and True Frequency ---
    # Note: We generate the signal but the simple estimator doesn't use it yet
    signal, true_inst_freq = generate_bpsk_signal_with_doppler(
        t, data_bits, samples_per_bit, fc, A_d, f_d, noise_std_signal
    )

    # --- Generate Noisy Frequency Measurements (Simulation Shortcut) ---
    # In a real system, this would come from processing 'signal'
    noisy_freq_measurements = estimate_frequency_simple(signal, fc, fs) # Pass fc as approx

    # --- Kalman Filter Setup for Frequency Tracking ---
    # State: x = [frequency, frequency_rate_of_change]'
    # Model: f(t) = fc + A_d*sin(omega_d*t) => f_dot(t) = A_d*omega_d*cos(omega_d*t)
    # This is a harmonic oscillator centered at fc for the Doppler part.
    omega_d = 2 * np.pi * f_d

    # State transition matrix F for [f, f_dot] of the *Doppler component*
    # We assume fc is constant and track the deviation f - fc.
    # Let state be y = [f - fc, f_dot] = [A_d*sin(omega_d*t), A_d*omega_d*cos(omega_d*t)]
    # y[k+1] = [[cos(wd*dt), sin(wd*dt)/wd], [-wd*sin(wd*dt), cos(wd*dt)]] * y[k]
    # However, the filter estimates the *total* frequency f.
    # Let state be x = [f, f_dot].
    # Prediction: x_pred = F * x_est + G * u
    # where u is fc, G = [[1], [0]]? No, simpler: predict Doppler, add fc.
    # Let's define the state x = [ f - fc, f_dot ]. Predict this, then add fc for output.
    F_doppler = np.array([
        [np.cos(omega_d*dt), np.sin(omega_d*dt)/omega_d if omega_d != 0 else dt],
        [-omega_d*np.sin(omega_d*dt), np.cos(omega_d*dt)]
    ])

    # Measurement matrix H: We measure f, so H relates [f-fc, f_dot] to f.
    # Measurement z = f = (f - fc) + fc.
    # z = [1  0] * [f-fc, f_dot]' + fc
    # The standard KF update is y = z - H*x_pred.
    # If x is [f-fc, f_dot], then H*x_pred = predicted (f-fc).
    # We need y = measured_f - predicted_f = z - (predicted(f-fc) + fc)
    # So, let the filter state be x = [f, f_dot].
    # How does f_dot evolve? f_dot = d/dt (fc + A_d*sin(wd*t)) = A_d*wd*cos(wd*t)
    # f_ddot = -A_d*wd^2*sin(wd*t) = -wd^2 * (f - fc)
    # State model: d/dt [f, f_dot]' = [[0, 1], [-wd^2, 0]] * [f, f_dot]' + [[0], [wd^2*fc]]'
    # This is complex due to the input term wd^2*fc.

    # Alternative: Simpler state model (Random Walk for frequency)
    # Assume frequency changes slowly. State x = [frequency]
    # F = np.array([[1]])
    # H = np.array([[1]])
    # Q = np.array([[q_rw]]) # Process noise variance - how much f can change per step
    # R = np.array([[noise_std_freq**2]])
    # Let's try this simpler model first.

    # --- Kalman Filter Setup (Random Walk Frequency Model) ---
    F = np.array([[1]])
    H = np.array([[1]])

    # Process noise: How much variance do we expect in frequency change per step?
    # Let's assume max change is related to Doppler dynamics. Max f_dot = A_d * omega_d
    # Variance could be (max_change * dt)^2 ? Let's guess.
    max_f_dot = A_d * omega_d
    q_rw = (max_f_dot * dt)**2 * 0.1 # Heuristic scaling factor
    Q = np.array([[q_rw]])

    # Measurement noise covariance
    R = np.array([[noise_std_freq**2]])

    # Initial state covariance (high uncertainty)
    P0 = np.array([[1e6]]) # High uncertainty in initial frequency

    # Initial state (guess nominal carrier frequency)
    x0 = np.array([[fc]])

    # Create Kalman filter
    kf = KalmanFilter(F, H, Q, R, P0, x0)

    # --- Run Kalman Filter ---
    filtered_freq_estimates = np.zeros(n_steps)

    for i in range(n_steps):
        # Prediction
        kf.predict()

        # Update with measurement
        measurement = np.array([[noisy_freq_measurements[i]]])
        kf.update(measurement)

        # Store filtered state (frequency)
        filtered_freq_estimates[i] = kf.x[0, 0]

    # --- Plot Results ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1) # Top plot: Frequencies
    plt.plot(t, true_inst_freq / 1e3, 'g-', label='True Instantaneous Freq (kHz)') # Plot in kHz
    plt.plot(t, noisy_freq_measurements / 1e3, 'b.', markersize=2, label='Noisy Measurements (kHz)')
    plt.plot(t, filtered_freq_estimates / 1e3, 'r-', linewidth=1.5, label='Kalman Filter Estimate (kHz)')
    plt.ylabel('Frequency (kHz)')
    plt.title(f'Kalman Filter Tracking Frequency (fc={fc/1e9:.1f}GHz, Doppler={A_d/1e3:.1f}kHz @ {f_d}Hz)')
    plt.legend()
    plt.grid(True)
    # Zoom y-axis around the variation
    plt.ylim((fc - 1.5*A_d)/1e3, (fc + 1.5*A_d)/1e3)


    plt.subplot(2, 1, 2) # Bottom plot: Estimation Error
    error = (filtered_freq_estimates - true_inst_freq) / 1e3 # Error in kHz
    plt.plot(t, error, 'm-', label='Estimation Error (kHz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (kHz)')
    plt.title('Kalman Filter Estimation Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust spacing between plots

    plt.savefig('plots/random_walk_freq_track.png')
    print("Plot saved to plots/random_walk_freq_track.png")
    plt.show() # Display interactive plot window

if __name__ == "__main__":
    # main() # Keep the old main if needed, or remove
    main_frequency_tracking()
