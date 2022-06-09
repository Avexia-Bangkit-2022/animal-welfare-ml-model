from distutils.command.upload import upload
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from google.cloud import storage

import tensorflow as tf
from tensorflow import keras
import librosa
import numpy as np

from flask import Flask, render_template, request, jsonify, redirect

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050
model = keras.models.load_model(SAVED_MODEL_PATH)

class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = None
    _mapping = [
        "greeting",
        "happy",
        "unhappy"
    ]
    _instance = None


    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        print(predictions)
        predicted_index = np.argmax(predictions)
        print(predicted_index)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
            
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

        # drop audio files with less than pre-decided number of samples
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
        return MFCCs.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance



app = Flask(__name__)

app_cwd = os.getcwd()

@app.route("/")
def index():
    return "Ok", 200

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        if request.files:
            audio = request.files["audio"]
            audio_path = os.path.join(app_cwd, 'audio', audio.filename)
            audio.save(audio_path)

            print("Audio saved") 

            try:
                kss = Keyword_Spotting_Service()
                kss1 = Keyword_Spotting_Service()

                assert kss is kss1
                prediction = kss.predict(audio_path)
            except Exception as e:
                print(str(e))
                return jsonify({"error": "Failed to predict"}), 500            
        
        return jsonify({"prediction": prediction}), 200

    return render_template("/public/upload_audio.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))