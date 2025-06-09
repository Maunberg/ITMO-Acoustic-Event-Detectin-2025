from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from transformers import AutoProcessor, ASTModel, ASTFeatureExtractor, AutoFeatureExtractor
import pandas as pd
import numpy as np
import os
import librosa
import torch
from tqdm import tqdm
import pickle
from IPython.display import FileLink
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import torch.utils.data as data_utils
import torch.optim as optim
import random

from models.classifier import Classifier
from train_utils import model_register

with open('data/train.pkl', 'rb') as f:
    train_features = pickle.load(f)

with open('data/test.pkl', 'rb') as f:
    test_features = pickle.load(f)

with open('data/subm.pkl', 'rb') as f:
    subm_test = pickle.load(f)

with open('models/unique_labels.pkl', 'rb') as f:
    unique_labels = pickle.load(f)


model = model_register()
model.batch_size = 10
model.gen_datasets(train_features, test_features, 'spectr')
model.get_model()
model.train(epochs=15)

pred = []
for i in tqdm(subm_test['spectr']):
    with torch.no_grad():
        i = torch.tensor(i, dtype=torch.float32).to(model.device)
        i = i.unsqueeze(0).to(model.device)
        out = model.model(i).to('cpu')
    probabilities = torch.softmax(out[0], dim=0)
    predicted_classes = torch.argmax(probabilities, dim=0)
    pred.append(unique_labels[predicted_classes])

pd.DataFrame({'fname':subm['fname'].to_list(), 'label':pred}).to_csv('submissions/submission.csv', index=False)