import pandas as pd
import numpy as np
import asyncio
import os
import mne
from preprocess_run import pre_process_run as PreProcess  # Import the correct class

class EEGProcess:  # Renamed for clarity

    def __init__(self, path):
        self.path = path

    def create_file_names(self):
        paths_EEG = []
        paths_IMG = []
        for i in range(0, 2):
            root = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-"
            mid = "/eeg/sub-"
            end = "_task-rsvp_eeg.vhdr"
            path = root + f"{i+1:02d}" + mid + f"{i+1:02d}" + end
            paths_EEG.append(path)

            end_img = "_task-rsvp_events.csv"
            path = root + f"{i+1:02d}" + mid + f"{i+1:02d}" + end_img
            paths_IMG.append(path)

        return paths_EEG, paths_IMG
    
    async def process_runner(self, save_path):
        paths_EEG, paths_IMG = self.create_file_names()
        tasks = []

        for i in range(len(paths_EEG)):
            task = asyncio.create_task(self.process(paths_EEG[i], paths_IMG[i]))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Example of saving data
        for i, result in enumerate(results):
            # Replace this with your actual saving logic
            with open(f"{save_path}{i+1}.csv", 'w') as f:
                f.write(str(result))

        return
    
    async def process(self, eeg_path, event_path):
        pre_process = PreProcess(eeg_path, event_path)  # Instantiate the PreProcess class
        data, info = await pre_process.pre_process()  # Call the asynchronous method
        return data

if __name__ == "__main__":
    process_instance = EEGProcess(path="processed_data")
    asyncio.run(process_instance.process_runner(save_path="processed_data/"))
