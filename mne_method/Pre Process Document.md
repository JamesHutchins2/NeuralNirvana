# Data Pre Processing, and Emedding process

## Data Pre Processing

### 1. Data Cleaning

Data cleaning methods are contained in the file `preprocess_eeg.py`. This file contains 2 class, ***`EEGDataLoader`***, and ***`EEGDataProcessor`***.

***`EEGDataLoader`***

This class preforms the aggregation of data into pytorch dataloaders. It also preforms a test train split with a ratio of 0.8, and aligns the data into batches of 32 epochs. 

***`EEGDataProcessor`***

This class preforms 3 main steps.

1. Loading raw data.
    This step loads the eeg data using the `mne` package. It also collectes data infromation such as sampling frequency.

2. Data filtering
    The next step in the process is an attenution of the input data into a range from 90 to 0Hz by the employment of a low pass filter. 

3. Epoch Extraction
    This function preforms two main tasks. Local normalization of data, and the splitting of data into epochs. The local normalization is preformed by taking 128ms prior to the event indication, and normalizing the 512ms after the event using such data as a reference. 

There are also a few helper functions in this file for running the entire system, and ittertating through all of the participants for data loading.


### 2. Data Embedding

The embedding of the data is preformed in numerous steps being:

1. Patching of trial periods into discrete time periods of 4 ms in length 32 of which make up a single trial of 128ms.
2. The convolution of each of these patches into 1D arrays of the length of the channel input.
3. The application of positional encodings through each of the trial length patches.

## Patching, Convolution and Masking
The patching methods are contained within the `EEG_patch.py` file. This file contains 2 classes, ***`EEGDataLoader`***, and ***`EEGDataProcessor`***. This process is preformed in parallel with the convolution, and masking process. The Patching function in class: `EEGDataPatcher` takes in a pytoch data loader object which is created by the pre processing pipline, along with a convolution indicator `conv` which is a boolean value and indicates if convolution should be preformed. The data is then by calling this class patched into the patch lengths as discussed, and then convolved if the `conv` indicator is set to true. The masking process is preformed by the `mask` function in the `EEGDataPatcher` class. It calls another class called `masker` that is contained in the same file. This class preforms a similar method to that of ***Dream Diffusion***. Each epoch is passed into the function, and returned is three data items. 
1. `masked_data` - This is the data that has been masked by the masking function.
2. `mask` - This is the mask that has been applied to the data.
3. `ids_resotre` - This provides the values, and indicies that were masked which allows for the restoration of the data to its original form.


## Positional Encoding

Positional Encoding is done using a new method that combines both relative, and absolute positional encodings know as `tAPE`. 

This uses the common application of sinusoidal positional encodings, but also adds a relative positional encoding to the input. This is done by taking the difference between the current time step, and the previous time step, and encoding this as a sinusoidal positional encoding. Code for this process is within the `positional_encoding` folder. For the primary development of this system we will only use absolute positional encoding, but will look to add relative positional encoding in the future.

The reason for the application of the `tAPE` positional enocder is first, the encoder scales the values to the length of the input sequence. As we are only passing in a sequence length of 128 time chunks, often absolute positional encoders fail to sufficently apply enough variation between time steps in small temportaly dimensioned data. The second application is the simplicity, and fast computation rate acheived by the `tAPE` positional encoder. 

<img src="assets/pipline.png">