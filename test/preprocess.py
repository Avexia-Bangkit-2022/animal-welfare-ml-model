# untuk model machine learning
from tensorflow import lite
import tensorflow as tf

import matplotlib.pyplot as plt

# untuk processing suara
import librosa
import librosa.display

# untuk download dataset dari github
import os

# untuk general use
import pandas as pd
import numpy as np
import json

# bikin file metadata
dataset_path = "/content/dataset"
json_path = "data.json"

# dictionary untuk menyimpan hasil mapping, label, MFCC, dan files
data = {
    "mapping": [],
    "labels": [],
    "MFCCs": [],
    "files": []
}

for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    print(i, (dirpath, dirnames, filenames)) # mau ngecek apa yang terjadi
    if dirpath is not dataset_path:

        # masukkan path directory saat ini (yang bukan dataset_path, tapi level bawahnya)
        # sebagai mapping dalam dictionary (label dalam bentuk numerik)
        label = dirpath.split("/")[-1]
        data["mapping"].append(label)
        print("\nProcessing: '{}'".format(label))

        # proses semua file audio di subdir dataset (happy, unhappy, greeting)
        for f in filenames:
            file_path = os.path.join(dirpath, f)
      
            # load file audio dan potong untuk memastikan panjang file audio konsisten
            signal, sample_rate = librosa.load(file_path)

            # sample_rate di dataset adalah 22050 Hz
            # jumlah kanal di semua file audio adalah mono (bukan stereo), tidak perlu pemrosesan lagi
            # print(sample_rate) # cek sample rate, untuk coba-coba (ternyata 22050 Hz semua)
            # print(signal.shape) # cek apakah audio-nya mono/stereo (ternyata mono semua)

            # supaya panjang file audio konsisten, semua file audio dioverlay dengan suara diam
            # selama durasi tertentu
            # durasi yang digunakan adalah 12 detik (12 kali sample_rate)
            # karena data dengan durasi terpanjang dalam dataset adalah 12 detik
            padding_array = np.zeros(12*sample_rate, dtype=float) # padding_array isi nol
            signal.resize(padding_array.shape, refcheck=False)
            signal = signal + padding_array

            # drop file audio dengan jumlah sampel lebih sedikit dari durasi tertentu
            # ambil durasi 4 detik (cek dulu)
            if len(signal) >= 4*sample_rate:

                # ensure consistency of the length of the signal
                signal = signal[:4*sample_rate]

                # extract MFCCs
                num_mfcc=13
                n_fft=2048
                hop_length=512
                MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)

                # store data for analysed track
                data["MFCCs"].append(MFCCs.T.tolist())
                data["labels"].append(i-1)
                data["files"].append(file_path)
                print("{}: {}".format(file_path, i-1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)