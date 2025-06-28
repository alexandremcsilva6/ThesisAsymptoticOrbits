# Differential-Equation Neural Solver

## Repository structure

Core Python modules
modules.py — Lightweight neural-network components (e.g. LinearModel, EquationsModel) and utility functions used across all simulations.

model.py — High-level training, evaluation and logging routines that tie a chosen equation class to its neural surrogate and optimisation loop.

equations.py — Abstract base classes and helpers for defining ordinary/partial differential equations, building trial solutions, and generating reference solutions with SciPy.

Example notebooks
The main_*.ipynb notebooks are ready-to-run demonstrations that implement:

Simple pendulum

Double-well potential

Inverted double-well potential

Fisher–Kolmogorov reaction-diffusion equation
