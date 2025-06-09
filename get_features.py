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
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/lw1-acc/train.csv')
way = '/kaggle/input/lw1-acc/audio_train/train/'
df['way'] = way + df['fname']
df['label_id'] = pd.factorize(df['label'])[0]
unique_labels = df['label'].unique().tolist()


train, test = train_test_split(df, test_size=0.015, random_state=42, stratify=df['label'])
extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def get_features(data):
    features = {'spectr':[], 'mel':[], 'mfcc':[], 'labels':[]}
    for i in tqdm(range(len(data))):
        info = dict(data.iloc[i])
        x = librosa.load(info['way'], sr=16000)[0]
        x, _ = librosa.effects.trim(x)
        x = extractor(x, sampling_rate=16000, return_tensors="pt")["input_values"]
        features['spectr'].append(x)
        features['labels'].append(info['label_id'])
    return features

train_features = get_features(df)
test_features = get_features(test)

train_features['spectr'] = np.array([m[0] for m in train_features['spectr']], dtype=np.float32)
test_features['spectr'] = np.array([m[0] for m in test_features['spectr']], dtype=np.float32)

with open('data/train.pkl', 'wb') as f:
    pickle.dump(train_features, f)

with open('data/test.pkl', 'wb') as f:
    pickle.dump(test_features, f)

with opem('unique_labels.pkl', 'wb') as f:
    pickle.dump(unique_labels, f)

subm = pd.read_csv('/kaggle/input/lw1-acc/sample_submission.csv')
subm['way'] = '/kaggle/input/lw1-acc/audio_test/audio_test/test/' + subm['fname']
subm['label_id'] = 0
subm_test = get_features(subm)

subm_test['spectr'] = np.array([m[0] for m in subm_test['spectr']], dtype=np.float32)

with open('subm.pkl', 'wb') as f:
    pickle.dump(subm_test, f)