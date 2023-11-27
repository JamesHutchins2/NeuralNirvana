import pandas as pd
import numpy as np
import asyncio
import os
import mne

class AddEvents:
    def __init__(self, eeg_data_path, event_data_path):
        self.eeg_data_path = eeg_data_path
        self.event_data_path = event_data_path

   



    async def get_data_frame(self):
        df = await pd.read_csv(self.event_data_path)
        stm_on = await df["time_stimon"].values

        stm_on = np.array(stm_on, dtype=data.index.dtype)

        #convert to ms
        stm_on = stm_on * 1000

    #round to nearest integer
        stm_on = await np.round(stm_on).astype(int)

        return stm_on
        




    async def get_eeg_data(self):
        raw = await mne.io.read_raw_brainvision(self.eeg_data_path, preload=True)
        data = await raw.to_data_frame(index="time", time_format="ms")
        info = await raw.info
        
        return data, info

    async def add_events(self):

        stm_on = self.get_data_frame()
        data = await self.get_eeg_data()

        info = data[1]

        data = data[0]

        data["stm_on"] = 0
        for on_time in stm_on:
            if on_time in data.index:
                data.at[on_time, 'stimulus'] = 1
        return data, info
    

    async def save_data(self, path):
        data = await self.add_events()
        data.to_csv(path)
        return data
    
    
    




