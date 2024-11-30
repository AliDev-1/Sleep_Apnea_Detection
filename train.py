# Import necessary libraries
import matplotlib.pyplot as plt  # For visualization (though not used in this script directly)
import numpy as np  # For numerical computations
import time  # For tracking time (not used in this script)
import tensorflow as tf  # For building and training neural network models
import pickle  # For loading and saving serialized data
import wfdb  # For working with physiological signal data
from sklearn.utils import class_weight  # For handling class imbalances
from sklearn.model_selection import train_test_split  # For splitting datasets into training and testing subsets

# Hyper-parameters
sequence_length = 240  # The expected length of each input sequence
epochs = 1000  # Number of epochs for training (can be adjusted by the user)
FS = 100.0  # Sampling frequency of the ECG data (in Hz)

# Function to apply z-normalization to data
def z_norm(result):
    """
    Performs z-score normalization on the input data.

    Args:
        result (array-like): The input data.

    Returns:
        array: The normalized data where each element is transformed as:
               (value - mean) / standard deviation
    """
    result_mean = np.mean(result)  # Calculate the mean of the data
    result_std = np.std(result)  # Calculate the standard deviation of the data
    result = (result - result_mean) / result_std  # Apply z-score normalization
    return result

# Function to split the input data into two parts
def split_data(X):
    """
    Splits the input data into two separate arrays.

    Args:
        X (array): The input data where each element is expected to have a nested structure,
                   e.g., X[index] = [signal1, signal2, feature1, feature2].

    Returns:
        tuple: Two numpy arrays:
               - The first array contains signal data (e.g., RR intervals, QRS amplitudes).
               - The second array contains demographic or static data (e.g., age, sex).
    """
    X1 = []  # Initialize a list to store the first part of the data (e.g., signals)
    X2 = []  # Initialize a list to store the second part of the data (e.g., demographic data)

    # Loop through the input data
    for index in range(len(X)):
        X1.append([X[index][0], X[index][1]])  # Extract signal data (RR intervals, QRS amplitudes)
        X2.append([X[index][2], X[index][3]])  # Extract demographic data (age, sex)

    return np.array(X1).astype('float64'), np.array(X2).astype('float64')  # Return the two parts as numpy arrays

# Function to load data and preprocess it
def get_data():
    """
    Loads the preprocessed data from pickle files and splits it into training, validation, and testing sets.
    Additionally, transposes and reshapes the signal data for compatibility with machine learning models.

    Returns:
        tuple: Processed data arrays ready for training and testing:
               - X_train1, X_train2: Features for training (signal data, demographic data)
               - y_train: Labels for training
               - X_val1, X_val2: Features for validation (signal data, demographic data)
               - y_val: Labels for validation
               - X_test1, X_test2: Features for testing (signal data, demographic data)
               - y_test: Labels for testing
    """
    # Load training input and labels from pickle files
    with open('train_input.pickle', 'rb') as f:
        X_train = np.asarray(pickle.load(f))  # Load training features
    with open('train_label.pickle', 'rb') as f:
        y_train = np.asarray(pickle.load(f))  # Load training labels

    # Load validation input and labels from pickle files
    with open('val_input.pickle', 'rb') as f:
        X_val = np.asarray(pickle.load(f))  # Load validation features
    with open('val_label.pickle', 'rb') as f:
        y_val = np.asarray(pickle.load(f))  # Load validation labels

    # Load testing input and labels from pickle files
    with open('test_input.pickle', 'rb') as f:
        X_test = np.asarray(pickle.load(f))  # Load testing features
    with open('test_label.pickle', 'rb') as f:
        y_test = np.asarray(pickle.load(f))  # Load testing labels

    # Split the data into two parts:
    # - Part 1: Signal data (e.g., RR intervals, QRS amplitudes)
    # - Part 2: Demographic or static data (e.g., age, sex)
    X_train1, X_train2 = split_data(X_train)  # Split training data
    X_val1, X_val2 = split_data(X_val)  # Split validation data
    X_test1, X_test2 = split_data(X_test)  # Split testing data

    # Print the shape of the first part of the training data before transposing
    print(f"Shape of X_train1 before transpose: {X_train1.shape}")

    # Transpose the signal data to match the expected input format for machine learning models
    # The new shape is (samples, features, time_steps)
    X_train1 = np.transpose(X_train1, (0, 2, 1))  # Transpose training signal data
    X_test1 = np.transpose(X_test1, (0, 2, 1))  # Transpose testing signal data
    X_val1 = np.transpose(X_val1, (0, 2, 1))  # Transpose validation signal data

    # Return the processed data arrays
    return X_train1, X_train2, y_train, X_val1, X_val2, y_val, X_test1, X_test2, y_test



def build_model():
    """
    Builds a neural network model for binary classification using a combination of LSTM layers 
    (for sequential data) and dense layers (for static demographic data).
    
    Returns:
        model: A compiled Keras model ready for training.
    """
    # Define the architecture's layers
    layers = {
        'input': 2,        # Input features for the LSTM (e.g., RR intervals, QRS amplitudes)
        'hidden1': 256,    # Number of units in the first LSTM layer
        'hidden2': 256,    # Number of units in the second LSTM layer
        'hidden3': 256,    # Number of units in the third LSTM layer
        'output': 1        # Output layer with a single neuron for binary classification
    }

    # Sequential Input: Time-series data (e.g., RR intervals, QRS amplitudes)
    x1 = tf.keras.layers.Input(shape=(sequence_length, layers['input']))
    
    # First LSTM Layer: Processes sequential data with recurrent dropout for regularization
    m1 = tf.keras.layers.LSTM(
        layers['hidden1'],  # Number of units in the LSTM layer
        recurrent_dropout=0.5,  # Dropout applied to recurrent connections
        return_sequences=True  # Return full sequence to pass to the next layer
    )(x1)
    
    # Second LSTM Layer: Further processing of sequential data
    m1 = tf.keras.layers.LSTM(
        layers['hidden2'],  # Number of units in the LSTM layer
        recurrent_dropout=0.5,  # Dropout applied to recurrent connections
        return_sequences=True  # Return full sequence to pass to the next layer
    )(m1)
    
    # Third LSTM Layer: Final processing of sequential data
    m1 = tf.keras.layers.LSTM(
        layers['hidden3'],  # Number of units in the LSTM layer
        recurrent_dropout=0.5,  # Dropout applied to recurrent connections
        return_sequences=False  # Only return the last output (not the entire sequence)
    )(m1)

    # Static Input: Demographic data (e.g., age, sex)
    x2 = tf.keras.layers.Input(shape=(2,))  # Input layer for 2 static features
    m2 = tf.keras.layers.Dense(32)(x2)  # Dense layer to process static inputs

    # Merge Sequential and Static Data
    # Concatenates the processed outputs of the sequential and static data branches
    merged = tf.keras.layers.Concatenate(axis=1)([m1, m2])

    # Dense Layer after merging
    out = tf.keras.layers.Dense(8)(merged)  # Dense layer with 8 units
    out = tf.keras.layers.Dense(
        layers['output'],  # Single output neuron for binary classification
        kernel_initializer='normal'  # Initializes weights using a normal distribution
    )(out)
    
    # Sigmoid Activation Function: Outputs probabilities for binary classification
    out = tf.keras.layers.Activation("sigmoid")(out)

    # Create the model with two inputs (sequential and static) and one output
    model = tf.keras.models.Model(inputs=[x1, x2], outputs=[out])

    # Compile the model
    start = time.time()  # Start timer to measure compilation time
    model.compile(
        loss="binary_crossentropy",  # Loss function for binary classification
        optimizer="adam",  # Optimizer for gradient descent
        metrics=['accuracy']  # Metric to evaluate during training
    )
    print("Compilation Time : ", time.time() - start)  # Print compilation time

    # Print the model summary to show the architecture
    model.summary()

    return model  # Return the compiled model


def run_network(model=None, data=None):
    """
    Trains and evaluates a neural network model for binary classification using time-series
    (LSTM-based) and demographic data.

    Args:
        model (Keras model, optional): A pre-built and compiled Keras model. If None, a new model is built.
        data (tuple, optional): Placeholder for custom data inputs. Defaults to loading data from the `get_data` function.

    Returns:
        model: The trained Keras model.
    """
    # Start the global timer to track total runtime
    global_start_time = time.time()

    print('\nData Loaded. Compiling...\n')
    print('Loading data... ')
    
    # Load preprocessed training, validation, and testing datasets
    X_train1, X_train2, y_train, X_val1, X_val2, y_val, X_test1, X_test2, y_test = get_data()

    # Compute class weights to handle class imbalance
    class_w = class_weight.compute_class_weight(
        class_weight='balanced',  # Balances weights based on class frequencies
        classes=np.unique(y_train),  # The unique classes in the training labels
        y=y_train  # The training labels
    )
    print(class_w)  # Print the computed class weights

    # Build the model if one isn't provided
    if model is None:
        model = build_model()  # Call the build_model function to create and compile a model

    try:
        print("Training")

        # Convert the computed class weights into a dictionary format for use in model training
        class_w = {i: class_w[i] for i in range(2)}  # Create a dictionary with keys for each class

        # Define an early stopping callback to stop training if the validation loss doesn't improve for 3 epochs
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=3  # Stop training if no improvement for 3 consecutive epochs
        )

        # Train the model
        history = model.fit(
            [X_train1, X_train2],  # Training inputs: Sequential (X_train1) and demographic (X_train2) data
            y_train,  # Training labels
            validation_data=([X_val1, X_val2], y_val),  # Validation data and labels
            callbacks=[callback],  # Include early stopping
            epochs=epochs,  # Number of training epochs
            batch_size=256,  # Batch size for gradient updates
            class_weight=class_w  # Apply class weights to handle imbalances
        )

        # Visualize the training loss over epochs (commented-out code)
        import matplotlib.pyplot as plt
    
        plt.plot(history.losses)  # Plot the recorded losses from the training history
        plt.ylabel('loss')  # Label the y-axis
        plt.xlabel('epoch')  # Label the x-axis
        plt.legend(['train'], loc='upper left')  # Add a legend
        plt.show()  # Display the plot
        

        # Make predictions on the test data
        y_pred = model.predict([X_test1, X_test2])  # Predict probabilities or labels for test data

        # Evaluate the model's performance on the test data
        scores = model.evaluate([X_test1, X_test2], y_test)  # Compute evaluation metrics (loss and accuracy)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))  # Print accuracy percentage

    except KeyboardInterrupt:
        # Handle manual interruption during training (e.g., pressing Ctrl+C)
        print("prediction exception")
        print('Training duration (s) : ', time.time() - global_start_time)  # Print elapsed time
        return model  # Return the partially trained model

    # Print the total training duration
    print('Training duration (s) : ', time.time() - global_start_time)

    return model  # Return the trained model

# Call the run_network function to train and evaluate the model
run_network()
