import pandas as pd
import librosa
import numpy as np
import json
import os


class Iluvatar:
    def __init__(self):
        pass

    @staticmethod
    def extract_features(file_path):
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features = np.hstack((np.mean(mfccs, axis=1),
                           np.mean(chroma, axis=1), 
                           np.mean(mel, axis=1),
                           np.mean(tempo, axis=0),
                           np.mean(zcr, axis=1)))
        return features

    @staticmethod
    def process_input(input_array):
        if len(input_array) != 155:
            raise ValueError("O array de entrada deve ter exatamente 155 posições.")
        int_output_1 = int(np.clip(np.mean(input_array), 0, 6))
        float_output_2 = np.clip(np.mean(input_array) / np.max(input_array), 0, 1)
        float_output_2 = round(float_output_2, 4)
        int_output_3 = int(np.clip(np.sum(input_array) % 3, 0, 2))
        int_output_4 = int(np.clip(np.sum(input_array) / (np.max(input_array) + 1) * 10, 1, 10))
        return np.array([int_output_1, float_output_2, int_output_3, int_output_4])

    @staticmethod
    def format_to_godot_noise(mesh):
        return {
                "noise_type": mesh[0],
                "frequency": mesh[1],
                "fractal": {
                    "type": mesh[2],
                    "octaves": mesh[3]
                }
            }

