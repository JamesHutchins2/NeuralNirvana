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