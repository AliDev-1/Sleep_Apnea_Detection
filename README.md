# ECG-Based Sleep Apnea Classification with Hybrid LSTM Network

A sophisticated deep learning framework developed during my research volunteering at the [Sleep, Cognition and Neuroimaging Laboratory (SCNLab)](https://scnlab.com). This project represents a novel approach to sleep apnea detection by combining electrocardiogram (ECG) signal analysis with demographic data through a hybrid neural network architecture. The system leverages advanced signal processing techniques to extract RR intervals and QRS complexes from ECG recordings, while simultaneously incorporating patient demographics to enhance detection accuracy. By utilizing a custom-designed neural network that combines LSTM layers for temporal pattern recognition with dense layers for demographic feature processing, the framework achieves robust sleep apnea detection without requiring full polysomnography studies.

## Project Overview

Sleep apnea, a serious sleep disorder characterized by repeated breathing interruptions, affects millions globally and can lead to severe health complications if left undiagnosed. Traditional diagnosis requires overnight polysomnography in sleep laboratories, which can be expensive, time-consuming, and not widely accessible.

This project offers an alternative approach by:
- Utilizing easily obtainable ECG signals for detection
- Incorporating demographic factors to improve accuracy
- Implementing advanced signal processing techniques
- Leveraging state-of-the-art deep learning methods

The system's hybrid architecture uniquely combines:
- Temporal analysis of ECG features through LSTM networks
- Integration of demographic data through dense neural networks
- Advanced preprocessing pipeline for robust feature extraction

This project implements an automated system for sleep apnea detection using:
- RR intervals (time between heartbeats)
- QRS amplitudes (heart signal strength)
- Demographic data (age and sex)

The system consists of two main components:
1. Signal preprocessing pipeline
2. Deep learning model combining LSTM and Dense neural networks

## Requirements

```
tensorflow
numpy
scipy
wfdb
scikit-learn
matplotlib
tqdm
```

## Project Structure

```
├── pre_proc.py         # Signal preprocessing and feature extraction
├── train.py           # Model architecture and training
└── README.md
```

## Data Processing Pipeline

### Signal Preprocessing (pre_proc.py)

The preprocessing script handles:

- ECG Signal Processing:
  - Reads raw ECG signals using the wfdb library
  - Computes QRS amplitudes and RR intervals
  - Applies median filtering
  - Ensures RR intervals are within physiological limits (20-300 BPM)

- Data Interpolation:
  - Uses cubic splines for RR intervals and QRS amplitudes
  - Generates evenly spaced time points
  - Sampling frequency: 100 Hz (raw ECG), 4 Hz (interpolated signals)

- Dataset Management:
  - Processes multiple datasets for training, validation, and testing
  - Divides ECG signals into 60-second windows
  - Handles participant metadata (age and gender)
  - Normalizes features using z-score normalization

### Model Architecture (train.py)

The neural network combines two input streams:

1. Sequential Data Processing:
   - Three LSTM layers (256 units each)
   - Recurrent dropout (0.5) for regularization
   - Processes RR intervals and QRS amplitudes

2. Demographic Data Processing:
   - Dense layers for age and sex information
   - Merged with LSTM output for final prediction

Training Features:
- Binary cross-entropy loss
- Adam optimizer
- Class weighting for imbalanced data
- Early stopping based on validation loss
- Batch size: 256
- Maximum epochs: 1000

## Output Files

The preprocessing generates six pickle files:
- train_input.pickle, train_label.pickle
- val_input.pickle, val_label.pickle
- test_input.pickle, test_label.pickle

## Applications

This system can be used for:
1. Healthcare monitoring systems for sleep apnea detection
2. General classification tasks involving physiological and demographic data
3. Research in sleep disorders and cardiac rhythms

## Usage

1. Preprocess the raw ECG data:
```bash
python pre_proc.py
```

2. Train the model:
```bash
python train.py
```

## Data Format

Input data should be in WFDB format with:
- ECG signals
- QRS annotations
- Apnea annotations ('N' for normal, 'A' for apnea)
- Demographic information (age, sex)

## Model Performance

The model evaluates performance using:
- Binary classification accuracy
- Training loss trends
- Validation loss monitoring

## Contributing

Feel free to contribute to this project by:
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request
   

## Acknowledgments

- [Dr. Thien Thanh Dang-Vu](https://scnlab.com/team/thien-thanh-dang-vu/), Director of the Sleep, Cognition and Neuroimaging Laboratory (SCNLab)
- WFDB library for ECG processing
- TensorFlow team for the deep learning framework

