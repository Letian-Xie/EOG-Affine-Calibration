# EOG-Affine-Calibration
EOG-based gaze direction classification using trajectory and displacement-based affine calibration. 

## Overview

Electrooculography (EOG) provides a simple and low-cost way to detect eye movements by measuring the corneo-retinal potential difference. Because EOG signals are highly susceptible to low-frequency noise and drift, accurately estimating gaze direction remains challenging.

This project evaluates the use of affine calibration techniques to estimate gaze displacement from EOG signals. Two calibration strategies were implemented and compared:

- Trajectory-based affine calibration
- Displacement-based affine calibration

The calibrated signals were used to classify gaze direction using multiple directional schemes, including 4-class, skewed 4-class, and 8-class configurations.

Results suggest that displacement-based affine calibration provides more reliable gaze displacement estimation and improves classification performance in multi-directional tasks.
