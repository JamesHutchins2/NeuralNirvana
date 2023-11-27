import pandas as pd
import numpy as np
import asyncio
import os
import mne



def create_file_names():
        paths_EEG = []
        paths_IMG = []
        for i in range(50,60):
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

        return paths_EEG, paths_IMG


def get_data(eeg_path):
        #this function will get the data and info from the eeg and event files
        raw = mne.io.read_raw_brainvision(eeg_path, preload=True)
        data = raw.to_data_frame(index="time", time_format="ms")
        #print(data.head(20))
        info = raw.info

        return data, info

def get_events(event_path):
        #this function will get the events from the event file
        df = pd.read_csv(event_path)
        stm_on = df["time_stimon"].values
        stim_names = df['stimname']


        #convert to ms
        stm_on = stm_on * 1000

        #round to nearest integer
        stm_on = np.round(stm_on).astype(int)

        return stm_on, stim_names

def downsample_data(df):
    # Number of rows in the DataFrame
    n_rows = df.shape[0]
    
    # Determine the number of rows to include in the downsampled data
    # If the number of rows is not divisible by 4, we exclude the last few rows
    valid_rows = n_rows - (n_rows % 4)

    # Reshape the DataFrame and calculate the mean along the first axis
    # The resulting DataFrame will have the same column names
    downsampled_df = pd.DataFrame(df.iloc[:valid_rows].values.reshape(-1, 4, df.shape[1]).mean(axis=1),
                                  columns=df.columns)

    return downsampled_df

def adjust_events_downsample(events, names, data):
        #here we will adjust the events to match the downsampled data

        #convert data to a dataframe
        
        
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


def run_pre_preocess(save_folder):
     
     #first create file names
        paths_EEG, paths_IMG = create_file_names()

        #create a list to hold the data
        data_list = []

        #loop through each file
        for i in range(len(paths_EEG)):
            #get the data and info
            data, info = get_data(paths_EEG[i])

            #get the events
            events, names = get_events(paths_IMG[i])

            #downsample the data
            data = downsample_data(data)

            #adjust the events to match the downsampled data
            data = adjust_events_downsample(events, names, data)

            #save the file to the save_folder as the name of the file in csv format
            save_path = save_folder + "/" + str(i+50) + "_sub" + ".csv"
            print(save_path)
            data.to_parquet(save_path)

    

save_folder = "./data"
run_pre_preocess(save_folder)
