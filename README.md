# EOG-Affine-Calibration
EOG-based gaze direction classification using trajectory and displacement-based affine calibration. 

## Overview

Electrooculography (EOG) provides a simple and low-cost way to detect eye movements by measuring the corneo-retinal potential difference. Because EOG signals are highly susceptible to low-frequency noise and drift, accurately estimating gaze direction remains challenging.

This project evaluates the use of affine calibration techniques to estimate gaze displacement from EOG signals. Two calibration strategies were implemented and compared:

- Trajectory-based affine calibration
- Displacement-based affine calibration

The calibrated signals were used to classify gaze direction using multiple directional schemes, including 4-class, skewed 4-class, and 8-class configurations.

Results suggest that displacement-based affine calibration provides more reliable gaze displacement estimation and improves classification performance in multi-directional tasks.

## Key Contributions

This repository provides:

- Implementation of trajectory-based and displacement-based affine calibration methods for EOG signals
- Direction classification frameworks for:
  - 4-quadrant classification
  - skewed 4-quadrant classification
  - 8-direction classification
- Diagnostic analysis of classification errors across direction classes
- Reproducible pipeline for evaluating EOG gaze direction estimation

## Methods

The analysis pipeline consists of the following stages:

1. Affine calibration
   - trajectory-based calibration
   - displacement-based calibration
2. Gaze displacement estimation
3. Direction classification
4. Performance evaluation

## Results Summary

The classification framework was evaluated across multiple directional schemes.

Key observations include:

- Displacement-based affine calibration produced more consistent gaze displacement estimates.
- Classification performance improved when using displacement-based calibration.
- Two main error sources: low-amplitude saccades and near-boundary enpoints.

The highest performance was achieved in the 8-direction classification scheme using displacement-based calibration with amplitude(<1°), with an accuracy of approximately **71.85%**.

## Dataset

This project uses a publicly available electrooculography dataset - dataset 2 from Univeristy of Malta.

Dataset source:
[https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/]

The dataset is not included in this repository. Please download it from the original source and place it in the `data/` directory before running the analysis.

## Citation

If you use this code or build upon this method, please cite:

Xie, L. (2026).  
EOG-Based Gaze Direction Classification Using Affine Calibration.  
GitHub Repository.

Available at:
https://github.com/SterneSehen/EOG-Affine-Calibration

## License

This project is released under the MIT License. See the LICENSE file for details.

## Acknowledgements

This project was completed as part of a research project under the supervision of Associate Progfessor Sam John at the University of Melbourne.
