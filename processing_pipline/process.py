import pandas as pd
import numpy as np
import asyncio
import os
import mne
import preprocess_run as preprocess_run

class process: 

    def __init__(self, path):
        self.path = path  # Corrected __init__ method

    def create_file_names(self):
        paths_EEG = []
        paths_IMG = []
        for i in range(0,5):
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
    
    async def process_runner(self, save_path):
        paths_EEG, paths_IMG = self.create_file_names()
        tasks = []
        number_of_tasks = len(paths_EEG)
        for i in range(len(paths_EEG)):
            task = asyncio.create_task(self.process(paths_EEG[i], paths_IMG[i]))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        for i in range(number_of_tasks):
            tasks[i].save_data(save_path + str(i+1) + ".csv")

        return
    
    async def process(self, eeg_path, event_path):
        pre_process = preprocess_run(eeg_path, event_path)
        data, info = await pre_process.pre_process()
        return data
    


if __name__ == "__main__":
    process_instance = process(path="processed_data")
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(process_instance.process_runner(save_path="processed_data/"))
    finally:
        loop.close()