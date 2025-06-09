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

class model_register():
    def __init__(self, ):
        self.batch_size = 10 #128
        self.loss_function = nn.CrossEntropyLoss()
        self.lr = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.epoch = 0
        
    def gen_datasets(self, train, test, mode='spectr'):
        X_train, X_test, y_train, y_test = train[mode], test[mode], train['labels'], test['labels']
        inputs_train = torch.tensor(X_train, dtype=torch.float32)
        targets_train = torch.tensor([i for i in y_train], dtype=torch.long)
        inputs_test = torch.tensor(X_test, dtype=torch.float32)
        targets_test = torch.tensor([i for i in y_test], dtype=torch.long)
        self.input_dim = inputs_train.shape[1:]
        self.input_dim = torch.prod(torch.tensor(model.input_dim))
        inputs_train = inputs_train.view(inputs_train.shape[0], self.input_dim)
        inputs_test = inputs_test.view(inputs_test.shape[0], self.input_dim)
        train = data_utils.TensorDataset(inputs_train.to(self.device), targets_train.to(self.device))
        test = data_utils.TensorDataset(inputs_test.to(self.device), targets_test.to(self.device))
        self.trainset = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.testset = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False)

    def get_model(self):
        self.model = Classifier(self.input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def train(self, epochs=10):
        for epoch in range(epochs):
                with tqdm(self.trainset, desc=f"Epoch {epoch+1}/{epochs}", leave=True) as pbar:
                    for X, y in pbar:
                        self.optimizer.zero_grad()
                        # print(X.shape)
                        out = self.model(X.to(self.device))
                        out = out.view(-1, out.shape[-1])
                        loss = self.loss_function(out, y.to(self.device))
                        loss.backward()
                        self.optimizer.step()
                        pbar.set_postfix(loss=loss.item())
                self.scheduler.step()

    def test(self, to_print=True):
        self.model.eval()
        predictions = []
        targets = []
    
        with torch.no_grad():
            with tqdm(self.testset, desc="Testing", leave=True) as pbar:
                for X, y in pbar:
                    X, y = X.to(self.device), y.cpu()
    
                    output = self.model(X)
    
                    preds = torch.argmax(output, dim=-1).cpu().numpy()
    
                    targets.extend(y.numpy())
                    predictions.extend(preds)
    
        f1 = f1_score(targets, predictions, average="macro")

        if to_print:
            print(f"F1-score (macro): {f1:.4f}")
        return f1

