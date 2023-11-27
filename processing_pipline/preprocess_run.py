import pandas as pd
import numpy as np
import asyncio
import os
import mne
from add_events import AddEvents

class pre_process_run:
    def __init__(self, eeg_path, event_path):
        self.eeg_path = eeg_path
        self.event_path = event_path


    
    
    def epoch_data(self, data, info):
        #get the sampling rate
        sfreq = info["sfreq"]

        #get the values 500ms before and 1500ms after the stimulus
        tmin = -0.5
        tmax = 1.5

        #now we can epoch the data
        events = data["stimulus"]

        #now we will make index ranges based on the stimulus values

        #get the 500 rows before each 1 in events. and the 1000 rows after
        time_ranges = []
        for i in range(len(events)):
            if events[i] == 1:
                start = i - 125
                end = i + 250
                time_ranges.append([start, end])

        # return the time ranges to be epoched
        return time_ranges

        #now we can epoch the data
    def get_subsets(data, timeRange):
        subsets = []
        for time in timeRange:
            subset = data.iloc[time[0]:time[1]]
            subsets.append(subset)

        return subsets
    
        return
    def downsample_data(self, data):

        #here we will down sample by a factor of 4
        
        #we will do this by averaging every 4 rows into 1 row
        #this will reduce the sampling rate to 125Hz
        downsampled_data = []
        for i in range(0, len(data), 4):
            downsampled_data.append(data[i:i+4].mean())

        return downsampled_data
    
    def adjust_events_downsample(self, events, names, data):
        #here we will adjust the events to match the downsampled data
        
        # start by creating a column in data that is the index * 4
        data["index"] = data.index
        data["index"] = data["index"] * 4

        #now we can adjust the events

        #create a new column for the events
        data["events"] = 0
        data["event_names"] = ""

        #loop through each event
        for i in range(len(events)):
            #get the event time
            event_time = events[i]

            #get the nearest index to that event time
            index = data["index"].sub(event_time).abs().idxmin()

            #set the event value to 1
            data.at[index, "events"] = 1
            data.at[index, "event_names"] = names[i]

        #now we can drop the index column
        data = data.drop(columns=["index"])

        return data
    
    def get_data(self):
        #this function will get the data and info from the eeg and event files
        raw = mne.io.read_raw_brainvision(self.eeg_path, preload=True)
        data = raw.to_data_frame(index="time", time_format="ms")
        info = raw.info

        return data, info
    
    def get_events(self):
        #this function will get the events from the event file
        df = pd.read_csv(self.event_path)
        stm_on = df["time_stimon"].values
        stim_names = df['stimname']


        #convert to ms
        stm_on = stm_on * 1000

        #round to nearest integer
        stm_on = np.round(stm_on).astype(int)

        return stm_on, stim_names
    
    def pre_process(self):

        #this function will be the main function that preofrms the pre processing proceedures

        #get the eeg data
        data, info = self.get_data()

        events, names = self.get_events()

        #now we downsample the data
        data = self.downsample_data(data)

        #now we adjust the events to match the downsampled data
        data = self.adjust_events_downsample(events, names, data)

        #now we can epoch the data
        time_ranges = self.epoch_data(data, info)

        #now we can get the subsets
        subsets = self.get_subsets(data, time_ranges)

        return subsets, info


        

                






