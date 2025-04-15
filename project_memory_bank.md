# Project Memory Bank: Kalman Filter Implementations for Frequency/Phase Tracking

This document summarizes the development and experimentation process for using Kalman filters (KF) and Extended Kalman Filters (EKF) to track the frequency and phase characteristics of a simulated BPSK signal affected by sinusoidal Doppler shift.

## Initial Goal (Conceptual)

*   Track the amplitude of a noisy sine wave using a standard Kalman Filter.
    *   *Note: The original script `kalman_filter_sine_wave.py` was likely overwritten during development.*

## Evolution of Frequency Tracking Filters

*   **KF - Random Walk Frequency Model:**
    *   File: `kf_random_walk_freq_track.py`
    *   Model: Standard KF, 1-state (`x = [frequency]`), Random Walk model (`F=[[1]]`).
    *   Noise Model: AWGN added directly to the true frequency to create noisy measurements (`noise_std_freq` parameter). This is a simulation shortcut.
    *   Observation: Showed significant lag and inability to track the sinusoidal Doppler dynamics due to the overly simplistic model.

*   **EKF - Harmonic Oscillator Model (Frequency Deviation):**
    *   File: `ekf_frequency_track.py`
    *   Model: EKF structure, 2-state (`x = [f - fc, f_dot]`), Harmonic Oscillator model based on `omega_d = 2*pi*f_d`. State represents frequency *deviation* from carrier and its rate.
    *   Noise Model: Same simulation shortcut as the Random Walk model (noise added directly to frequency).
    *   Observation: Performed significantly better than the Random Walk model, accurately tracking the sinusoidal frequency due to the matching dynamics model. It was noted that since the underlying model is linear, the EKF structure itself wasn't strictly necessary for this specific case.

*   **KF - Harmonic Oscillator Model (Frequency Deviation):**
    *   File: `kf_2state_frequency_track.py`
    *   Model: Standard KF, 2-state (`x = [f - fc, f_dot]`), Harmonic Oscillator model (same as EKF version).
    *   Noise Model: Same simulation shortcut as above.
    *   Observation: Performed virtually identically to the EKF version, confirming the performance gain came from the 2-state harmonic model, not the EKF algorithm itself for this linear system.

*   **KF - Constant Velocity Model (Frequency):**
    *   File: `kf_const_vel_freq_track.py`
    *   Model: Standard KF, 2-state (`x = [f, f_dot]`), Constant Velocity model (`F = [[1, dt], [0, 1]]`). State represents absolute frequency and its rate.
    *   Noise Model: Simulation shortcut, but `noise_std_freq` was calculated based on a target frequency SNR (`target_snr_db`).
    *   Observation: Initially showed significant lag compared to the harmonic model. Aggressively increasing the process noise `Q` (specifically `q_fdot`) dramatically improved tracking by forcing the filter to rely more heavily on measurements, highlighting the trade-off between model fidelity and noise sensitivity.

*   **KF - Constant Acceleration Model (Frequency) with Realistic Noise & Iterative Q/R Tuning:**
    *   File: `kf_random_walk_doppler_track.py` (*Note: Filename is misleading*)
    *   Model: Standard KF, 3-state (`x = [f, f_dot, f_ddot]`), Constant Acceleration model (`F = [[1, dt, 0.5*dt^2], [0, 1, dt], [0, 0, 1]]`).
    *   Noise Model: AWGN added to the *complex signal waveform* based on `waveform_snr_db`. A phase difference frequency estimator (`estimate_frequency_phase_diff`) was implemented to generate noisy frequency measurements from the noisy waveform.
    *   KF Tuning: Implemented iterative tuning for the main process noise component (`q_fddot_multiplier`) and the measurement noise variance (`R_variance_guess`).
    *   Observation: Simulates the process more realistically. Performance depends heavily on `waveform_snr_db` and the tuning of `Q` and `R`. The Constant Acceleration model attempts to better capture the sinusoidal dynamics than Constant Velocity.

*   **KF - Constant Acceleration Model (Frequency) with Realistic Noise & Iterative Base Q Tuning:**
    *   File: `kf_realistic_noise_track.py`
    *   Model: Standard KF, 3-state (`x = [f, f_dot, f_ddot]`), Constant Acceleration model (same as above).
    *   Noise Model: Realistic noise added to the complex waveform (`waveform_snr_db`), phase difference frequency estimator used for measurements.
    *   KF Tuning: Extends the iterative tuning from the previous script (`kf_random_walk_doppler_track.py`) to also tune the *base* process noise components (`q_f_base_mult`, `q_fdot_base_mult`) in addition to `q_fddot_multiplier` and `R_variance_guess`. This represents the most refined *frequency tracking* KF in this set.
    *   Observation: Provides a more thorough tuning approach for the Constant Acceleration model under realistic noise conditions.

*   **KF - Phase/Frequency Shift/Frequency Rate Model with Realistic Noise:**
    *   File: `akf_phase_freq_rate_track.py`
    *   Model: Standard KF, 3-state (`x = [phase_diff, freq_shift, freq_rate]`). Tracks phase difference relative to carrier, frequency shift from carrier, and rate of frequency shift. Uses a Constant Doppler Rate dynamic model.
    *   Noise Model: Realistic noise added to the complex waveform (`waveform_snr_db`).
    *   Measurement: Uses the *phase angle* of the noisy complex signal samples directly as the measurement (`z`). Includes logic for handling phase wrapping in the update step.
    *   KF Tuning: `Q` and `R` are set based on heuristics and the `waveform_snr_db`, not iteratively tuned within this script. `R` is estimated based on SNR, with a heuristic increase to account for BPSK modulation effects.
    *   Observation: Directly incorporates phase measurements, potentially offering advantages in low SNR or when phase coherence is important. Performance depends on the fixed `Q` and `R` values chosen.

## Key Parameters & Settings (Latest Implementations)

**`kf_realistic_noise_track.py` (Const. Accel. Freq. Tracking):**

*   Carrier Frequency (`fc`): 10 GHz
*   Doppler Amplitude (`A_d`): 5 kHz
*   Doppler Frequency (`f_d`): 0.5 Hz
*   Data Rate: 16 kbps
*   Samples per Bit: 8
*   Waveform SNR (`waveform_snr_db`): Tunable (e.g., 10.0 dB used in tuning)
*   KF Model: Constant Acceleration (`F` matrix for `[f, f_dot, f_ddot]`)
*   KF State: `x = [frequency, frequency_rate, frequency_acceleration]`
*   KF Measurement: Noisy frequency estimate from `estimate_frequency_phase_diff`.
*   KF Measurement Noise (`R`): Iteratively tuned (`R_variance_guess`).
*   KF Process Noise (`Q`): Iteratively tuned via `q_fddot_multiplier`, `q_f_base_mult`, `q_fdot_base_mult`.

**`akf_phase_freq_rate_track.py` (Phase/Freq Shift/Rate Tracking):**

*   Carrier Frequency (`fc`): 10 GHz
*   Doppler Amplitude (`A_d`): 5 kHz
*   Doppler Frequency (`f_d`): 0.5 Hz
*   Data Rate: 16 kbps
*   Samples per Bit: 8
*   Waveform SNR (`waveform_snr_db`): Tunable (e.g., 15.0 dB used in example)
*   KF Model: Constant Doppler Rate (`F` matrix for `[phase_diff, freq_shift, freq_rate]`)
*   KF State: `x = [phase_difference, frequency_shift, frequency_rate]`
*   KF Measurement: Noisy phase angle (`z = angle(noisy_signal_complex)`).
*   KF Measurement Noise (`R`): Fixed based on `waveform_snr_db` and heuristics.
*   KF Process Noise (`Q`): Fixed based on heuristics.

## Next Steps / Future Exploration

*   **Compare Performance:** Systematically compare the performance (e.g., RMS error vs. SNR) of the tuned Constant Acceleration frequency KF (`kf_realistic_noise_track.py`) and the Phase/Freq/Rate KF (`akf_phase_freq_rate_track.py`).
*   **Harmonic Model with Realistic Noise:** Implement the 2-state Harmonic Oscillator model (`kf_2state_frequency_track.py`) within the realistic noise framework (using `generate_bpsk_signal_with_doppler_complex` and `estimate_frequency_phase_diff`) and compare its performance to the Constant Acceleration model.
*   **Refine Phase Measurement:** Improve the phase measurement in `akf_phase_freq_rate_track.py` by attempting to remove the BPSK modulation before phase extraction, potentially reducing measurement noise (`R`).
*   **Adaptive Tuning:** Explore adaptive methods for `Q` and `R` within the filters (e.g., based on innovation statistics) instead of fixed or offline iterative tuning.
*   **Alternative Frequency Estimators:** Implement and evaluate more sophisticated frequency estimation algorithms to replace `estimate_frequency_phase_diff` and assess their impact on KF performance.
*   **Augmented State Model:** Revisit the concept of an augmented state model combining sinusoidal dynamics with random walk components.
