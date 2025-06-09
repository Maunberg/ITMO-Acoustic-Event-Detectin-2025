# [Audio Classification Solution](https://www.kaggle.com/competitions/itmo-acoustic-event-detectin-2025) - 1st Place (Private F1: 0.89353)

## Repository Overview

This repository contains the winning solution for an audio classification competition, achieving a private F1 score of 0.89353. The solution combines AST (Audio Spectrogram Transformer) features with a custom neural network architecture featuring Inception-Residual blocks and Transformer layers.

## Solution Architecture

### Key Components

1. **Feature Extraction**:
   - Uses AST (Audio Spectrogram Transformer) pretrained on AudioSet
   - Extracts spectrogram features using mel-spectrogramms

2. **Custom Neural Network**:
   - Inception-Residual blocks with Squeeze-and-Excitation (SE) attention
   - Transformer encoder layers for sequence modeling
   - Feature reduction and classification head

### Model Details

```python
class Classifier(nn.Module):
    def __init__(self, num_classes=41, dropout_rate=0.4):
        super(Classifier, self).__init__()
        # AST model for feature extraction
        self.model_extractor = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        # Inception-Residual blocks
        self.inception_blocks = nn.Sequential(
            InceptionResidualBlock(1, 128),
            InceptionResidualBlock(128, 256),
            InceptionResidualBlock(256, 512),
            InceptionResidualBlock(512, 768)
        )
        
        # Feature processing
        self.feature_reduce = nn.Conv2d(768, 256, kernel_size=1)
        self.norm = nn.LayerNorm(196608)
        self.fc1 = nn.Linear(196608, 512)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(512, num_classes)
```

## Training Process

- **Optimization**: Adam optimizer with learning rate 1e-5
- **Scheduling**: StepLR with step_size=5 and gamma=0.1
- **Batch Size**: 10 (limited by GPU memory)
- **Epochs**: 15
- **Loss Function**: CrossEntropyLoss

## Key Innovations

1. **Inception-Residual Blocks with SE Attention**:
   - Combines multiple convolution paths (1x1, 3x3, 5x5, pooling)
   - Includes residual connections and channel attention

2. **Hybrid Architecture**:
   - Leverages pretrained AST features
   - Processes with CNN blocks
   - Refines with Transformer layers

3. **Feature Processing**:
   - Careful dimensionality reduction
   - Layer normalization before Transformer

## File Structure

```
/audio-classification/
│
├── data/
│   ├── train.pkl              # Processed training features
│   ├── test.pkl               # Processed test features
│   ├── subm.pkl               # Processed submission features
│   └── unique_labels.pkl      # Class labels mapping
│
├── models/
│   ├── classifier.py          # Model definition
│   └── train_utils.py         # Training utilities
│
├── notebooks/
│   └── sollutions.ipynb       # Main solution notebook
│
├── submissions/
│   └── submission.csv         # Final predictions
│
├── README.md                  # This file
├── get_features.py            # File to generate features to train
└── train.py                   # File to train model and get submission
```

## How to Reproduce
1. Get kaggle [dataset](https://www.kaggle.com/datasets/maunberg/lw1-acc)
2. Get features:
   ```bash
   python get_features.py
   ```

3. Run the training pipeline:
   ```bash
   python train.py
   ```
4. Get submission.csv

## Performance

- **Private F1 Score**: 0.89353 (1st place)
- **Training Time**: ~4 hours on a single GPU (P100 on Kaggle)

## Future Improvements

1. Experiment with larger batch sizes given more GPU memory
2. Try different learning rate schedules
3. Incorporate additional audio augmentations
4. Test other pretrained audio models as feature extractors

This solution demonstrates the effectiveness of combining pretrained audio transformers with carefully designed custom architectures for audio classification tasks.