# Import the MNE library for working with EEG data
import mne
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# Define a class called EEGDataProcessor for processing EEG data
class EEGDataProcessor:
    def __init__(self, raw_file_path, event_description, resample_factor=3):
        # Constructor method to initialize the object with file path, event description, and resample factor
        self.raw_file_path = raw_file_path  # Store the raw EEG file path
        self.event_description = event_description  # Store the event description
        self.resample_factor = resample_factor  # Store the resample factor

    def load_raw_data(self):
        # Method to load the raw EEG data from the specified file
        self.raw = mne.io.read_raw_brainvision(self.raw_file_path, preload=True)  # Read and preload the data
        self.current_sfreq = self.raw.info["sfreq"]  # Get the current sampling frequency
        #print out all columns
        print(self.raw.info["ch_names"])
        print(self.raw.info)
        
    def normalize_channel(self, channel):
        if channel.size == 0:
            return channel  # Return the channel as-is if it's empty
        return (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
    
    def preprocess_raw_data(self):
        # Method to preprocess the loaded raw EEG data
        self.raw.set_eeg_reference('average', projection=True)  # Set EEG reference to average
        self.raw.filter(None, 90., fir_design='firwin')  # Apply a low-pass filter at 90 Hz
        
        
        # we will now normalize the data using the mean and standard deviation
        data = self.raw.get_data()
        
        # extract the annotations
        original_annotations = self.raw.annotations
        
        # normalize the data between 1 and 0
        normalized_data = np.array([self.normalize_channel(channel) for channel in data])
        
        info = self.raw.info
        #put back into mne form
        normalized_raw = mne.io.RawArray(normalized_data, info)
        #add back the annotations
        normalized_raw.set_annotations(original_annotations)
        
        #set back to the class
        self.raw = normalized_raw
        
        #print out the sample frequency
        print(self.raw.info["sfreq"])
        #print out the shape
        print(self.raw.get_data().shape)
        
            
    

        

    #Here we can see the window around the event
    def extract_epochs(self, tmin=-0.128, tmax=0.512):
        events, event_dict = mne.events_from_annotations(self.raw, event_id=None, regexp=None)
        print(events.shape)  # Check the shape of the events array
        # Method to extract epochs from the preprocessed data
        events, event_dict = mne.events_from_annotations(self.raw, event_id=None, regexp=self.event_description)
        self.epochs = mne.Epochs(self.raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True)

    def process_eeg_data(self):
        # Method to process EEG data by sequentially loading, preprocessing, and extracting epochs
        self.load_raw_data()
        self.preprocess_raw_data()
        
        self.extract_epochs()


class EEGDataLoader:
    def __init__(self, epochs, batch_size=32, train_split=0.8):
        # Assuming 'epochs' is your MNE Epochs object
        eeg_data = epochs.get_data()  # Shape (n_epochs, n_channels, n_times)
        print(eeg_data.shape)
        eeg_data = eeg_data[:, :, :512]

        self.n_channels, self.n_times = eeg_data.shape[1], eeg_data.shape[2]
        eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)

        # Create a TensorDataset to wrap the EEG data tensor
        self.dataset = TensorDataset(eeg_data_tensor)

        # Split the dataset into training and testing subsets
        train_size = int(train_split * len(self.dataset))
        test_size = len(self.dataset) - train_size

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        # Create DataLoaders for both training and testing datasets
        self.batch_size = batch_size
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
    


def load_and_preprocess_eeg_data(raw_file_path, event_description):
    eeg_processor = EEGDataProcessor(raw_file_path, event_description)
    eeg_processor.process_eeg_data()
    
    # Create an instance of EEGDataLoader using the processed data
    eeg_loader = EEGDataLoader(eeg_processor.epochs)
    
    # Access train and test DataLoaders
    train_loader = eeg_loader.train_data_loader
    test_loader = eeg_loader.test_data_loader
    n_channels = eeg_loader.n_channels
    n_times = eeg_loader.n_times
    
    return train_loader, test_loader, n_channels, n_times


def create_file_names():
        paths_EEG = []
        paths_IMG = []
        for i in range(0,50):
            root = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-"
            mid = "/eeg/sub-"
            end = "_task-rsvp_eeg.vhdr"
            if i < 9:
                path = root + "0" + str(i+1) + mid + "0" + str(i+1) + end
            else:
                path = root + str(i+1) + mid + str(i+1) + end

            paths_EEG.append(path)

            end_img = "_task-rsvp_events.csv"

            if i < 9:
                path = root + "0" + str(i+1) + mid + "0" + str(i+1) + end_img
            else:
                path = root + str(i+1) + mid + str(i+1) + end_img
            paths_IMG.append(path)

        return paths_EEG


def manage_loader(participant_number):
    
    eventDescription = 'Event/E  1'
    
    #load the data for that participant
    paths_EEG = create_file_names()
    raw_file_path = paths_EEG[0]
    train_loader, test_loader, n_channels, n_times = load_and_preprocess_eeg_data(raw_file_path, eventDescription)
    
    return train_loader, test_loader, n_channels, n_times
    
    
def main():
    
    manage_loader(1)
    
    

    
if __name__ == "__main__":
    main()