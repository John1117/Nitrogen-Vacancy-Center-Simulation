# Nitrogen-Vacancy Center Simulation

This project simulates the behavior of the nitrogen-vacancy center (NVC) in a nano-diamond. By considering the tetrahedral micro-structure of the nano-diamond, we accounted for four possible orientations of the NVC’s quantization axis. This property is used to accurately simulate Optically Detected Magnetic Resonance (ODMR) and Optically Detected Relaxation Spectroscopy (ODRS). Additionally, the ODRS exhibits a stretched exponential decay curve, which I found it to be the sum of multiple exponential decays with different decay rates.

## Features

- **NVC distribution Simulation**: Considers the tetrahedral structure of nano-diamonds and Gaussian-like distribution.
- **ODMR Simulation**: Accurate simulation of optically detected magnetic resonance with or without degeneracies.
- **ODRS Simulation**: Simulation of optically detected relaxation spectroscopy, which behaves as a stretched exponential decay.
- **Decay Curve Decomposition**: The stretched exponential decay is demonstrated to be the sum of several distinct exponential decays.

## Project Structure

- `ODMR.py`: Script for simulating Optically Detected Magnetic Resonance (ODMR).
- `ODRS.py`: Script for simulating Optically Detected Relaxation Spectroscopy (ODRS).
- `stretched_exponential_simulation.py`: Simulation of the stretched exponential decay behavior of ODRS.
- `main.py`: Main script for running simulations.

## Prerequisites
- Python 3.x
- `numpy`
- `matplotlib`
- `scipy`
- `pandas`
