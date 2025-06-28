# PINN based framework for solving ODE's based on trial solutions for IVP and Dirichlet BC

## Repository structure

### Core Python modules

modules.py — Lightweight neural-network components (e.g. LinearModel, EquationsModel) and utility functions used across all simulations.

model.py — High-level training, evaluation and logging routines that tie a chosen equation class to its neural surrogate and optimisation loop.

equations.py — Abstract base classes and helpers for defining ordinary/partial differential equations, building trial solutions, and generating reference solutions with SciPy.


### Example notebooks

The main_*.ipynb notebooks are ready-to-run demonstrations that implement:

1) Simple pendulum - system and 2nd order for PVI, 2nd order for Dirichlet BCs.

2) Double-well potential - heteroclinics

3) Inverted double-well potential - heteroclinics

4) Fisher–Kolmogorov type of equation - heteroclinics with proposed approach and new approach.
