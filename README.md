# Neural Network Framework for Learning Constitutive Laws in 2D Granular Flow

> A research project exploring how feedforward neural networks can learn constitutive stress–strain relationships in 2D granular materials using Deep Learning and DEM simulation data.

---

## Overview

This repository contains the complete workflow developed during my undergraduate research project at **IIT Gandhinagar**.

The project investigates whether neural networks can learn the relationship between **velocity gradients** and **stress tensors** in granular materials directly from simulation data.

The workflow progresses from simple function approximation to a complete constitutive law prediction pipeline with architectural optimization and cross-dataset validation.

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_Mathematical_Foundations.ipynb
│   ├── 02_Neural_Network_Architecture.ipynb
│   ├── 03_Constitutive_Law_Training.ipynb
│   ├── 04_Cross_Dataset_Validation.ipynb
│   └── 05_Zero_Gravity_Validation.ipynb
│
├── report/
│   └── report.pdf
│
├── presentations/
│   ├── Midterm_Presentation.pdf
│   └── Final_Presentation.pdf
│
├── images/
│
├── models/
│
├── results/
│
├── requirements.txt
└── README.md
```

---

# Project Pipeline

## Phase 1 — Mathematical Foundations

**Notebook**

```
01_Mathematical_Foundations.ipynb
```

This notebook demonstrates that a feedforward neural network can approximate nonlinear functions before applying the same concepts to granular mechanics.

Topics covered:

- Feedforward neural networks
- Backpropagation
- Function approximation
- Training vs interpolation
- Relative error analysis

Output:

- Neural network approximation of sine functions
- Training region visualization
- Relative error evaluation

---

## Phase 2 — Neural Network Architecture

**Notebook**

```
02_Neural_Network_Architecture.ipynb
```

A systematic architectural benchmark comparing multiple combinations of activation functions and loss functions.

The notebook evaluates

- 5 activation functions
- 5 loss functions
- 3 mathematical benchmark functions

for a total of

**75 independent experiments**

Activation Functions

- ReLU
- Tanh
- Swish
- Mish
- GELU

Loss Functions

- MSE
- MAE
- Huber
- MSLE
- Relative Error

Benchmark Functions

- Exponential
- Logarithmic
- Fourier Series

Outputs include

- Ranked leaderboard
- Function approximation plots
- Convergence curves

---

## Phase 3 — Constitutive Law Training

**Notebook**

```
03_Constitutive_Law_Training.ipynb
```

This notebook trains neural networks using DEM-generated granular flow data.

Major steps include

- Data loading
- Horizontal averaging
- Quasi-linear masking
- Reference normalization
- Data augmentation
- Feature scaling
- Train/validation split
- Architectural tournament

The network predicts

- σxx
- σxy
- σyy

from

- Velocity gradient

---

## Phase 4 — Cross Dataset Validation

**Notebook**

```
04_Cross_Dataset_Validation.ipynb
```

The trained model is evaluated on an entirely different DEM dataset.

Objectives

- Test generalization
- Verify scale invariance
- Evaluate reference normalization

The notebook performs

- Frozen model inference
- Ensemble prediction
- Relative error calculation
- Comparison with simulation data

---

## Phase 5 — Zero Gravity Validation

**Notebook**

```
05_Zero_Gravity_Validation.ipynb
```

The final validation evaluates the trained model under a different physical regime where gravity is removed.

This investigates

- Model robustness
- Pressure-source decoupling
- Constitutive behavior under unseen conditions

Outputs include

- Stress comparison plots
- Relative error analysis
- Trend comparison between simulation and prediction

---

# Research Highlights

### Reference Normalization

Scaling inputs and outputs by dataset reference values significantly improved cross-dataset prediction accuracy and enabled scale-independent learning.

---

### Architectural Tournament

A complete benchmark of

- 5 activation functions
- 5 loss functions
- 3 benchmark domains

identified the best-performing neural network configurations before training on simulation data.

---

### Cross-Dataset Generalization

The trained model successfully generalized to unseen DEM simulations using the learned normalization strategy.

---

### Zero-Gravity Validation

Testing on bulk flow simulations without gravity revealed hidden dependencies in learned constitutive relationships, providing insight into model limitations.

---

# Dataset

The DEM simulation datasets are **not included** in this repository because of their size and research constraints.

The notebooks remain available to demonstrate the complete methodology and workflow.

---

# Requirements

Install dependencies

```bash
pip install -r requirements.txt
```

Main libraries

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Documentation

Additional documentation is available in

- `report/report.pdf`
- `presentations/Midterm_Presentation.pdf`
- `presentations/Final_Presentation.pdf`

---

# Future Improvements

Possible extensions include

- Physics-informed neural networks
- Multi-input constitutive models
- Three-dimensional granular flow
- Experimental dataset validation
- Uncertainty estimation
- Graph neural networks for particle interactions

---

# Author

**Abhishek Tagalpallewar**

Undergraduate Research Project

Indian Institute of Technology Gandhinagar

Supervisor: **Prof. Harkirat Singh**

---

If you found this repository useful, feel free to ⭐ the project.