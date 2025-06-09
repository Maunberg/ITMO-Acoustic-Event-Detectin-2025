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

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionResidualBlock, self).__init__()
        out_channel = int(out_channels // 4)

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.se = SEBlock(out_channels)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.residual_conv(x)

        out = torch.cat([
            self.branch1x1(x),
            self.branch5x5(x),
            self.branch3x3(x),
            self.branch_pool(x)
        ], 1)

        out += identity
        out = self.se(out)
        out = self.final_bn(out)
        return self.relu(out)

class Classifier(nn.Module):
    def __init__(self, num_classes=41, dropout_rate=0.4):
        super(Classifier, self).__init__()
        self.model_extractor = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.cnn_input_dim = 768  # AST feature dimension

        self.inception_blocks = nn.Sequential(
            InceptionResidualBlock(1, 128),
            InceptionResidualBlock(128, 256),
            InceptionResidualBlock(256, 512),
            InceptionResidualBlock(512, 768)
        )

        self.feature_reduce = nn.Conv2d(768, 256, kernel_size=1) 
        self.norm = nn.LayerNorm(196608) 
        self.fc1 = nn.Linear(196608, 512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        batch = x.size(0)
        x = self.model_extractor(x.view(batch, 1024, 128))['pooler_output'] 

        x = x.view(batch, 1, 24, 32)
        x = self.inception_blocks(x)

        x = self.feature_reduce(x) 
        x = x.flatten(start_dim=1) 

        x = self.norm(x)
        x = F.relu(self.fc1(x)).unsqueeze(0) 

        x = self.transformer(x).squeeze(0)
        x = self.dropout(x)
        return self.fc_out(x)