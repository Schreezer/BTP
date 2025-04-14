# Project Memory Bank: Kalman Filter for Frequency Tracking

This document summarizes the development and experimentation process for using Kalman filters to track the frequency of a simulated signal with Doppler shift.

## Initial Goal

*   Track the amplitude of a noisy sine wave using a standard Kalman Filter.
    *   File: `kalman_filter_sine_wave.py` (original version, later overwritten).

## Evolution to Frequency Tracking

*   **Task:** Track the instantaneous frequency of a BPSK-modulated signal with a 10 GHz carrier frequency affected by a sinusoidal Doppler shift (Amplitude `A_d=5kHz`, Frequency `f_d=0.5Hz`).
*   **Initial Frequency Tracking KF (Random Walk):**
    *   File: `kalman_filter_sine_wave.py` (modified version).
    *   Model: Standard KF, 1-state (`x = [frequency]`), Random Walk model (`F=[[1]]`).
    *   Noise Model: AWGN added directly to the true frequency to create noisy measurements. `noise_std_freq` parameter.
    *   Observation: Showed significant lag due to the model not matching the sinusoidal dynamics.
*   **EKF with Harmonic Oscillator Model:**
    *   File: `ekf_frequency_track.py`.
    *   Model: EKF structure, 2-state (`x = [f - fc, f_dot]`), Harmonic Oscillator model based on `omega_d = 2*pi*f_d`.
    *   Noise Model: Same as initial frequency tracking KF (noise added to frequency).
    *   Observation: Performed significantly better, accurately tracking the sinusoidal frequency due to the matching model. Showed that the EKF structure itself wasn't strictly necessary as the model was linear.
*   **Standard KF with Harmonic Oscillator Model:**
    *   File: `kf_2state_frequency_track.py`.
    *   Model: Standard KF, 2-state (`x = [f - fc, f_dot]`), Harmonic Oscillator model.
    *   Noise Model: Same as above.
    *   Observation: Performed virtually identically to the EKF version, confirming the performance gain came from the 2-state harmonic model, not the EKF algorithm itself for this linear case.
*   **Standard KF with Constant Velocity Model:**
    *   File: `kf_const_vel_freq_track.py`.
    *   Model: Standard KF, 2-state (`x = [f, f_dot]`), Constant Velocity model (`F = [[1, dt], [0, 1]]`).
    *   Noise Model: Initially same as above. Later modified to calculate `noise_std_freq` based on a target frequency SNR (`target_snr_db`).
    *   Observation: Initially showed significant lag compared to the harmonic model. Aggressively increasing the process noise `Q` (specifically `q_fdot`) dramatically improved tracking by forcing the filter to rely more heavily on measurements, but at the potential cost of increased sensitivity to measurement noise.
*   **Standard KF with Constant Velocity Model & Realistic Noise:**
    *   File: `kf_realistic_noise_track.py`.
    *   Model: Standard KF, 2-state (`x = [f, f_dot]`), Constant Velocity model (with aggressive `Q` tuning from previous step).
    *   Noise Model: AWGN added to the *complex signal waveform* based on `waveform_snr_db`. A phase difference frequency estimator (`estimate_frequency_phase_diff`) was implemented to generate noisy frequency measurements from the noisy waveform.
    *   KF Tuning: The measurement noise covariance `R` became a tuning parameter, representing the variance of the frequency estimator's output (dependent on `waveform_snr_db` and estimator specifics).
    *   Observation: Simulates the process more realistically. Performance depends on `waveform_snr_db` and the tuning of `Q` and `R`.

## Key Parameters & Settings (Last Version - `kf_realistic_noise_track.py`)

*   Carrier Frequency (`fc`): 10 GHz
*   Doppler Amplitude (`A_d`): 5 kHz
*   Doppler Frequency (`f_d`): 0.5 Hz
*   Data Rate: 16 kbps
*   Samples per Bit: 8
*   Waveform SNR (`waveform_snr_db`): 10.0 dB (tunable)
*   KF Model: Constant Velocity (`F = [[1, dt], [0, 1]]`)
*   KF State: `x = [frequency, frequency_rate_of_change]`
*   KF Measurement Noise (`R`): Tunable parameter (e.g., `[[1500**2]]`)
*   KF Process Noise (`Q`): Tuned aggressively based on max acceleration (`q_fdot = (max_f_ddot * dt)**2 * 1024`).

## Next Steps

*   Further tuning of `Q` and `R` in `kf_realistic_noise_track.py` for different `waveform_snr_db` values.
*   Implementing the harmonic oscillator model within the realistic noise framework (`kf_realistic_noise_track.py` structure but with the 2-state harmonic `F` and `H` from `kf_2state_frequency_track.py`).
*   Implementing more sophisticated frequency estimation algorithms.
*   Exploring the augmented state model (sinusoidal + random walk) discussed previously.
