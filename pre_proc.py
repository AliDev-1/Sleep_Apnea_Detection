# Import necessary libraries
import wfdb  # Library to read and process physiological data (e.g., ECG signals)
import matplotlib.pyplot as plt  # For visualizing data using plots
import numpy as np  # For numerical computations
from scipy.signal import medfilt  # For applying median filtering to smooth data
from scipy.signal import detrend  # For removing linear trends from data
from scipy import interpolate  # For performing data interpolation (e.g., cubic splines)
from tqdm import tqdm  # For displaying progress bars during loops
import pickle  # For saving and loading data objects in binary format
import os  # For handling file paths and directories

# Sampling frequency of the ECG signal (100 Hz)
FS = 100.0  

# Define a function to create cumulative time information for RR intervals (RRI)
def create_time_info(rri):
    """
    Converts a list of RR intervals into cumulative time information in seconds.

    Args:
        rri (list or array): RR intervals in milliseconds.

    Returns:
        array: Cumulative time in seconds starting from zero.
    """
    rri_time = np.cumsum(rri) / 1000.0  # Convert cumulative RRI from ms to seconds
    return rri_time - rri_time[0]  # Ensure the time starts at zero

# Define a function to create evenly spaced time points for interpolation
def create_interp_time(rri, fs):
    """
    Creates evenly spaced time points for interpolating the RR intervals.

    Args:
        rri (list or array): RR intervals in milliseconds.
        fs (float): Target sampling frequency for interpolation (Hz).

    Returns:
        array: Interpolated time points at the given sampling frequency.
    """
    time_rri = create_time_info(rri)  # Generate cumulative time from RR intervals
    return np.arange(0, time_rri[-1], 1 / float(fs))  # Generate evenly spaced time points

# Define a function for cubic spline interpolation of RR intervals
def interp_cubic_spline(rri, fs):
    """
    Interpolates RR intervals using cubic splines to create smooth time series.

    Args:
        rri (list or array): RR intervals in milliseconds.
        fs (float): Target sampling frequency for interpolation (Hz).

    Returns:
        tuple: Interpolated time points and interpolated RR intervals.
    """
    time_rri = create_time_info(rri)  # Generate cumulative time for RR intervals
    time_rri_interp = create_interp_time(rri, fs)  # Create time points for interpolation
    tck = interpolate.splrep(time_rri, rri, s=0)  # Fit a cubic spline to the data
    rri_interp = interpolate.splev(time_rri_interp, tck, der=0)  # Evaluate the spline
    return time_rri_interp, rri_interp  # Return interpolated time and RR intervals

# Define a function for cubic spline interpolation of QRS amplitudes
def interp_cubic_spline_qrs(qrs_index, qrs_amp, fs):
    """
    Interpolates QRS amplitudes using cubic splines.

    Args:
        qrs_index (array): Indices of QRS peaks in the signal.
        qrs_amp (array): Amplitudes of QRS peaks.
        fs (float): Target sampling frequency for interpolation (Hz).

    Returns:
        tuple: Interpolated time points and interpolated QRS amplitudes.
    """
    time_qrs = qrs_index / float(FS)  # Convert QRS indices to time in seconds
    time_qrs = time_qrs - time_qrs[0]  # Ensure the time starts at zero
    time_qrs_interp = np.arange(0, time_qrs[-1], 1 / float(fs))  # Generate interpolated time points
    tck = interpolate.splrep(time_qrs, qrs_amp, s=0)  # Fit a cubic spline to the QRS data
    qrs_interp = interpolate.splev(time_qrs_interp, tck, der=0)  # Evaluate the spline
    return time_qrs_interp, qrs_interp  # Return interpolated time and QRS amplitudes

# Path to the folder containing ECG dataset files
data_path = 'C:/Users/DELL/Desktop/sleep_apnea-master/apnea-ecg-database-1.0.0/'

# List of dataset names for training, validation, and testing
train_data_name = [
    'a02', 'a03', 'a04', 'a05',  # Training dataset identifiers
    'a06', 'a07', 'a08', 'a09', 'a10',
    'a11', 'a12', 'a13', 'a14', 'a15',
    'a16', 'a17', 'a18', 'a19',
    'b02', 'b03', 'b04',
    'c02', 'c03', 'c04', 'c05',
    'c06', 'c07', 'c08', 'c09',
]
val_data_name = ['a01', 'b01', 'c01']  # Validation dataset identifiers
test_data_name = ['a20', 'b05', 'c10']  # Test dataset identifiers

# List of ages for the individuals in the dataset
age = [
    51, 38, 54, 52, 58,  # Ages of individuals in training datasets
    63, 44, 51, 52, 58,
    58, 52, 51, 51, 60,
    44, 40, 52, 55, 58,
    44, 53, 53, 42, 52,
    31, 37, 39, 41, 28,
    28, 30, 42, 37, 27
]

# List of genders for the individuals in the dataset
# 1 represents male, 0 represents female
sex = [
    1, 1, 1, 1, 1,  # Genders of individuals in training datasets
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 1, 1, 1, 1,
    1, 1, 1, 0, 0,
    0, 0, 1, 1, 1
]

# Function to extract the QRS amplitudes from the ECG signal
def get_qrs_amp(ecg, qrs):
    """
    Extracts the maximum QRS amplitude within a 250ms window around each QRS peak.

    Args:
        ecg (array): The raw ECG signal data.
        qrs (array): Indices of detected QRS peaks in the ECG signal.

    Returns:
        qrs_amp (list): A list of maximum QRS amplitudes corresponding to each QRS peak.
    """
    interval = int(FS * 0.250)  # Define a 250ms window (sampling frequency * time in seconds)
    qrs_amp = []  # Initialize a list to store QRS amplitudes

    for index in range(len(qrs)):  # Loop through each detected QRS peak
        curr_qrs = qrs[index]  # Current QRS index
        # Find the maximum amplitude in a 250ms window centered on the QRS peak
        amp = np.max(ecg[curr_qrs - interval: curr_qrs + interval])
        qrs_amp.append(amp)  # Append the amplitude to the list

    return qrs_amp  # Return the list of QRS amplitudes

# Constants for preprocessing and signal constraints
MARGIN = 10  # Time margin (in seconds) added around each 60-second window
FS_INTP = 4  # Target sampling frequency for interpolated signals (4 Hz)
MAX_HR = 300.0  # Maximum heart rate (in beats per minute)
MIN_HR = 20.0  # Minimum heart rate (in beats per minute)

# Calculate the corresponding RR interval constraints based on heart rates
MIN_RRI = 1.0 / (MAX_HR / 60.0) * 1000  # Minimum RR interval (in ms)
MAX_RRI = 1.0 / (MIN_HR / 60.0) * 1000  # Maximum RR interval (in ms)

# Initialize lists to store training data and labels
train_input_array = []  # Stores the input features for training
train_label_array = []  # Stores the labels for training

# Loop through each training dataset
for data_index in range(len(train_data_name)):
    print(train_data_name[data_index])  # Print the current dataset being processed

    # Read the number of apnea annotations (e.g., apnea/non-apnea labels)
    win_num = len(wfdb.rdann(os.path.join(data_path, train_data_name[data_index]), 'apn').symbol)

    # Read the raw ECG signal and metadata
    signals, fields = wfdb.rdsamp(os.path.join(data_path, train_data_name[data_index]))

    # Process each 60-second window in the dataset
    for index in tqdm(range(1, win_num)):
        samp_from = index * 60 * FS  # Start sample of the 60-second window
        samp_to = samp_from + 60 * FS  # End sample of the 60-second window

        # Read QRS annotations with a margin for preprocessing
        qrs_ann = wfdb.rdann(
            data_path + train_data_name[data_index], 'qrs',
            sampfrom=samp_from - (MARGIN * 100),  # Include 10-second margin before the window
            sampto=samp_to + (MARGIN * 100)  # Include 10-second margin after the window
        ).sample

        # Read apnea annotations for the current window
        apn_ann = wfdb.rdann(
            data_path + train_data_name[data_index], 'apn',
            sampfrom=samp_from, sampto=samp_to - 1
        ).symbol

        # Extract QRS amplitudes using the get_qrs_amp function
        qrs_amp = get_qrs_amp(signals, qrs_ann)

        # Compute RR intervals (time differences between consecutive QRS peaks)
        rri = np.diff(qrs_ann)  # Calculate RR intervals in sample indices
        rri_ms = rri.astype('float') / FS * 1000.0  # Convert RR intervals to milliseconds

        try:
            # Apply a median filter to smooth the RR intervals
            rri_filt = medfilt(rri_ms)

            # Ensure the RR intervals are within physiological limits
            if len(rri_filt) > 5 and (np.min(rri_filt) >= MIN_RRI and np.max(rri_filt) <= MAX_RRI):

                # Interpolate the RR intervals using cubic splines
                time_intp, rri_intp = interp_cubic_spline(rri_filt, FS_INTP)

                # Interpolate the QRS amplitudes using cubic splines
                qrs_time_intp, qrs_intp = interp_cubic_spline_qrs(qrs_ann, qrs_amp, FS_INTP)

                # Crop the interpolated data to exclude the margin
                rri_intp = rri_intp[(time_intp >= MARGIN) & (time_intp < (60 + MARGIN))]
                qrs_intp = qrs_intp[(qrs_time_intp >= MARGIN) & (qrs_time_intp < (60 + MARGIN))]

                # Check if the interpolated RR intervals have the correct length
                if len(rri_intp) != (FS_INTP * 60):  # Expected length: 60 seconds at FS_INTP Hz
                    skip = 1  # Skip the window if the length is incorrect
                else:
                    skip = 0

                if skip == 0:  # If the window passes all checks
                    # Normalize the RR intervals and QRS amplitudes by subtracting their means
                    rri_intp = rri_intp - np.mean(rri_intp)
                    qrs_intp = qrs_intp - np.mean(qrs_intp)

                    # Assign a label based on the apnea annotation
                    if apn_ann[0] == 'N':  # 'N' indicates no apnea
                        label = 0.0  # Normal
                    elif apn_ann[0] == 'A':  # 'A' indicates apnea
                        label = 1.0  # Apnea
                    else:  # Any other label (e.g., unknown)
                        label = 2.0

                    # Append the processed data and label to the training arrays
                    train_input_array.append([rri_intp, qrs_intp, age[data_index], sex[data_index]])
                    train_label_array.append(label)

        except:  # Handle any exceptions during processing
            hrv_module_error = 1  # Placeholder for error handling (not used further)

# Save the processed training inputs and labels as pickle files
with open('train_input.pickle', 'wb') as f:
    pickle.dump(train_input_array, f)  # Save training inputs (features) to a file
with open('train_label.pickle', 'wb') as f:
    pickle.dump(train_label_array, f)  # Save training labels to a file

# Initialize lists to store validation data and labels
val_input_array = []  # List to store input features for validation data
val_label_array = []  # List to store labels for validation data

# Loop through each validation dataset
for data_index in range(len(val_data_name)):
    print(val_data_name[data_index])  # Print the current validation dataset being processed

    # Read the number of apnea annotations (e.g., apnea/non-apnea labels)
    win_num = len(wfdb.rdann(os.path.join(data_path, val_data_name[data_index]), 'apn').symbol)

    # Read the raw ECG signal and metadata
    signals, fields = wfdb.rdsamp(os.path.join(data_path, val_data_name[data_index]))

    # Process each 60-second window in the validation dataset
    for index in tqdm(range(1, win_num)):  # tqdm provides a progress bar
        samp_from = index * 60 * FS  # Starting sample for the current 60-second window
        samp_to = samp_from + 60 * FS  # Ending sample for the current 60-second window

        # Read QRS annotations with a margin for preprocessing
        qrs_ann = wfdb.rdann(
            data_path + val_data_name[data_index], 'qrs',
            sampfrom=samp_from - (MARGIN * 100),  # Include 10-second margin before the window
            sampto=samp_to + (MARGIN * 100)  # Include 10-second margin after the window
        ).sample

        # Read apnea annotations for the current window
        apn_ann = wfdb.rdann(
            data_path + val_data_name[data_index], 'apn',
            sampfrom=samp_from, sampto=samp_to - 1
        ).symbol

        # Extract QRS amplitudes using the get_qrs_amp function
        qrs_amp = get_qrs_amp(signals, qrs_ann)

        # Compute RR intervals (time differences between consecutive QRS peaks)
        rri = np.diff(qrs_ann)  # Calculate RR intervals in sample indices
        rri_ms = rri.astype('float') / FS * 1000.0  # Convert RR intervals to milliseconds

        try:
            # Apply a median filter to smooth the RR intervals
            rri_filt = medfilt(rri_ms)

            # Ensure the RR intervals are within physiological limits
            if len(rri_filt) > 5 and (np.min(rri_filt) >= MIN_RRI and np.max(rri_filt) <= MAX_RRI):

                # Interpolate the RR intervals using cubic splines
                time_intp, rri_intp = interp_cubic_spline(rri_filt, FS_INTP)

                # Interpolate the QRS amplitudes using cubic splines
                qrs_time_intp, qrs_intp = interp_cubic_spline_qrs(qrs_ann, qrs_amp, FS_INTP)

                # Crop the interpolated data to exclude the margin
                rri_intp = rri_intp[(time_intp >= MARGIN) & (time_intp < (60 + MARGIN))]
                qrs_intp = qrs_intp[(qrs_time_intp >= MARGIN) & (qrs_time_intp < (60 + MARGIN))]

                # Check if the interpolated RR intervals have the correct length
                if len(rri_intp) != (FS_INTP * 60):  # Expected length: 60 seconds at FS_INTP Hz
                    skip = 1  # Skip the window if the length is incorrect
                else:
                    skip = 0

                if skip == 0:  # If the window passes all checks
                    # Normalize the RR intervals and QRS amplitudes by subtracting their means
                    rri_intp = rri_intp - np.mean(rri_intp)
                    qrs_intp = qrs_intp - np.mean(qrs_intp)

                    # Assign a label based on the apnea annotation
                    if apn_ann[0] == 'N':  # 'N' indicates no apnea
                        label = 0.0  # Normal
                    elif apn_ann[0] == 'A':  # 'A' indicates apnea
                        label = 1.0  # Apnea
                    else:  # Any other label (e.g., unknown)
                        label = 2.0

                    # Append the processed data and label to the validation arrays
                    val_input_array.append([rri_intp, qrs_intp, age[data_index], sex[data_index]])
                    val_label_array.append(label)

        except:  # Handle any exceptions during processing
            hrv_module_error = 1  # Placeholder for error handling (not used further)

# Save the processed validation inputs and labels as pickle files
with open('val_input.pickle', 'wb') as f:
    pickle.dump(val_input_array, f)  # Save validation inputs (features) to a file
with open('val_label.pickle', 'wb') as f:
    pickle.dump(val_label_array, f)  # Save validation labels to a file

# Initialize lists to store test data and labels (to be processed later)
test_input_array = []  # List to store input features for test data
test_label_array = []  # List to store labels for test data
# Loop through each test dataset
for data_index in range(len(test_data_name)):
    print(test_data_name[data_index])  # Print the name of the current test dataset being processed

    # Read the number of apnea annotations (e.g., apnea/non-apnea labels)
    win_num = len(wfdb.rdann(os.path.join(data_path, test_data_name[data_index]), 'apn').symbol)

    # Read the raw ECG signal and metadata
    signals, fields = wfdb.rdsamp(os.path.join(data_path, test_data_name[data_index]))

    # Process each 60-second window in the test dataset
    for index in tqdm(range(1, win_num)):  # tqdm provides a progress bar for loop progress
        samp_from = index * 60 * FS  # Starting sample for the 60-second window
        samp_to = samp_from + 60 * FS  # Ending sample for the 60-second window

        # Read QRS annotations with a margin for preprocessing
        qrs_ann = wfdb.rdann(
            data_path + test_data_name[data_index], 'qrs',
            sampfrom=samp_from - (MARGIN * 100),  # Include 10-second margin before the window
            sampto=samp_to + (MARGIN * 100)  # Include 10-second margin after the window
        ).sample

        # Read apnea annotations for the current window
        apn_ann = wfdb.rdann(
            data_path + test_data_name[data_index], 'apn',
            sampfrom=samp_from, sampto=samp_to - 1
        ).symbol

        # Extract QRS amplitudes using the get_qrs_amp function
        qrs_amp = get_qrs_amp(signals, qrs_ann)

        # Compute RR intervals (time differences between consecutive QRS peaks)
        rri = np.diff(qrs_ann)  # Calculate RR intervals in sample indices
        rri_ms = rri.astype('float') / FS * 1000.0  # Convert RR intervals to milliseconds

        try:
            # Apply a median filter to smooth the RR intervals
            rri_filt = medfilt(rri_ms)

            # Ensure the RR intervals are within physiological limits
            if len(rri_filt) > 5 and (np.min(rri_filt) >= MIN_RRI and np.max(rri_filt) <= MAX_RRI):

                # Interpolate the RR intervals using cubic splines
                time_intp, rri_intp = interp_cubic_spline(rri_filt, FS_INTP)

                # Interpolate the QRS amplitudes using cubic splines
                qrs_time_intp, qrs_intp = interp_cubic_spline_qrs(qrs_ann, qrs_amp, FS_INTP)

                # Crop the interpolated data to exclude the margin
                rri_intp = rri_intp[(time_intp >= MARGIN) & (time_intp < (60 + MARGIN))]
                qrs_intp = qrs_intp[(qrs_time_intp >= MARGIN) & (qrs_time_intp < (60 + MARGIN))]

                # Check if the interpolated RR intervals have the correct length
                if len(rri_intp) != (FS_INTP * 60):  # Expected length: 60 seconds at FS_INTP Hz
                    skip = 1  # Skip the window if the length is incorrect
                else:
                    skip = 0

                if skip == 0:  # If the window passes all checks
                    # Normalize the RR intervals and QRS amplitudes by subtracting their means
                    rri_intp = rri_intp - np.mean(rri_intp)
                    qrs_intp = qrs_intp - np.mean(qrs_intp)

                    # Assign a label based on the apnea annotation
                    if apn_ann[0] == 'N':  # 'N' indicates no apnea
                        label = 0.0  # Normal
                    elif apn_ann[0] == 'A':  # 'A' indicates apnea
                        label = 1.0  # Apnea
                    else:  # Any other label (e.g., unknown)
                        label = 2.0

                    # Append the processed data and label to the test arrays
                    test_input_array.append([rri_intp, qrs_intp, age[data_index], sex[data_index]])
                    test_label_array.append(label)

        except:  # Handle any exceptions during processing
            hrv_module_error = 1  # Placeholder for error handling (not used further)

# Save the processed test inputs and labels as pickle files
with open('test_input.pickle', 'wb') as f:
    pickle.dump(test_input_array, f)  # Save test inputs (features) to a file
with open('test_label.pickle', 'wb') as f:
    pickle.dump(test_label_array, f)  # Save test labels to a file
