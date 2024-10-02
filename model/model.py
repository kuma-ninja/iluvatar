import mlflow
import librosa
import numpy as np
import torch.nn as nn
import torch

LABELS = ['sadness', 'happiness', 'fear', 'anger', 'surprise', 'disgust']

class MusicSentimentModel(nn.Module):
    def __init__(self):
        super(MusicSentimentModel, self).__init__()
        self.fc1 = nn.Linear(155, 128)  # Aumentar a dimensão
        self.dropout1 = nn.Dropout(0.3)  # Adicionar dropout
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)  # Aplicar dropout
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)  # Aplicar dropout
        x = self.fc3(x)
        return x


class SentimentModel():
    def __init__(self):
        self.model = MusicSentimentModel()
        self.model.load_state_dict(torch.load('model/sentiment_model.pth'))

    def predict(self, features):
        predicted = self.model(torch.tensor(features, dtype=torch.float32))
        max_index = torch.argmax(predicted).item()
        return LABELS[max_index]


