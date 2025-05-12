# BTP
# Kalman Filter Frequency Tracking

This repository contains implementations of various Kalman Filter (KF) approaches for tracking the frequency and phase characteristics of signals affected by Doppler shift.

## Project Overview

The project explores different state-space models and filtering techniques to track frequency variations in a BPSK signal affected by sinusoidal Doppler shift. The implementations progress from simple to more complex models, with increasingly realistic noise simulation and parameter tuning approaches.

## Implementation Workflow

The project's implementation progression is visualized in `kalman_filter_workflow.png`, showing how different Kalman Filter implementations evolved:

1. Started with a 2-State KF using a harmonic oscillator model
2. Transitioned to a constant velocity model with SNR-based noise configuration
3. Enhanced to a 3-State KF with realistic noise and complex signal processing
4. Added state constraints and BER analysis
5. Finally implemented adaptive methods with Sage-Husa and WAF

Each stage brought specific improvements and features, making the implementation progressively more sophisticated and realistic.

## Implementation Variants

### Harmonic Oscillator Models
- **KF Version**: `kf_2state_frequency_track.py`
- **Model**: 2-state model tracking frequency deviation and its rate
- **Features**: Accurately tracks sinusoidal frequency variations

### Constant Velocity Model
- **File**: `kf_const_vel_freq_track.py`
- **Model**: 2-state KF tracking frequency and frequency rate
- **Features**: Configurable SNR, improved tracking with tuned process noise

### Constant Acceleration Model with Realistic Noise
- **File**: `kf_random_walk_doppler_track.py` 
- **Model**: 3-state KF tracking frequency, frequency rate, and acceleration
- **Features**: Realistic noise simulation, iterative Q/R parameter tuning
- **Note**: Simulates complex signal waveform with AWGN based on SNR

### Constant Acceleration Model with Base Q Tuning
- **File**: `kf_realistic_noise_track.py`
- **Model**: 3-state KF with comprehensive parameter tuning
- **Features**: Extends tuning to base process noise components, most refined frequency tracking implementation

### Phase/Frequency/Rate Model
- **File**: `akf_phase_freq_rate_track.py`
- **Model**: 3-state KF tracking phase difference, frequency shift, and frequency rate
- **Features**: Uses phase angle measurements directly, handles phase wrapping
- **Note**: Potentially advantageous in low SNR scenarios

## Key Features

- **Realistic Signal Generation**: Complex BPSK signal with sinusoidal Doppler shift
- **Noise Modeling**: AWGN added to complex waveform with configurable SNR
- **Parameter Tuning**: Iterative optimization of process and measurement noise parameters
- **Performance Evaluation**: Visualization of tracking performance and estimation errors
- **Phase Handling**: Special considerations for phase wrapping in phase-based tracking

## Usage

Each implementation can be run independently. For example:

```bash
python kf_realistic_noise_track.py
```

This will:
1. Generate a simulated BPSK signal with Doppler shift
2. Run the parameter tuning process (if applicable)
3. Apply the Kalman filter with optimized parameters
4. Generate performance plots in the `plots/` directory

## Parameters

Key configurable parameters across implementations:

- **Carrier Frequency**: 10 GHz
- **Doppler Amplitude**: 5 kHz
- **Doppler Frequency**: 0.5 Hz
- **Data Rate**: 16 kbps
- **Samples per Bit**: 8
- **Waveform SNR**: Configurable (typically 10-15 dB)

## Project Structure

The repository contains:
- Individual implementation files for each model variant
- A comprehensive project report (`BTP.pdf`) detailing the theoretical background, implementation details, and results analysis
- A memory bank file (`project_memory_bank.md`) documenting the development process
- Generated plots in the `plots/` directory (not included in repository)
- A visual workflow diagram (`kalman_filter_workflow.png`) showing the evolution of implementations

## Future Work

Potential improvements and extensions:
- Systematic performance comparison across models at different SNR levels
- Implementation of adaptive tuning methods for Q and R parameters
- Refinement of phase measurement techniques to handle BPSK modulation
- Exploration of alternative frequency estimation algorithms

## Requirements

- Python 3.x
- NumPy
- Matplotlib
