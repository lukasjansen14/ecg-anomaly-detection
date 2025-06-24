# ECG Anomaly Detection with Deep Autoencoders

Deep learning approaches for ECG anomaly detection on ECG-5000 dataset with TensorFlow Lite quantization and MLflow tracking.

## Overview

This project implements multiple autoencoder architectures for ECG anomaly detection:
- **CNN Autoencoder**: Traditional convolutional approach
- **TCN Autoencoder**: Temporal Convolutional Network with superior performance
- Trains on normal ECG patterns only
- Detects anomalies based on reconstruction error
- Quantizes models to INT8 using TensorFlow Lite
- Tracks experiments with MLflow
- Benchmarks GPU latency

## Features

- **Multiple Architectures**: CNN and TCN autoencoders for comparison
- **Anomaly Detection**: Threshold-based detection using reconstruction MSE
- **Model Compression**: Post-training INT8 quantization with TFLite
- **Experiment Tracking**: MLflow integration for metrics and artifacts
- **Performance Metrics**: AUROC, AUPRC, F1, precision, recall
- **Latency Benchmarking**: GPU inference timing

## Dataset

Uses the ECG-5000 dataset from [UCR Time Series Archive](https://www.timeseriesclassification.com/):
- 5000 ECG recordings (140 timesteps each)
- Class 1: Normal heartbeats
- Classes 2-5: Various arrhythmias (treated as anomalies)

## Architectures

### CNN Autoencoder (`cnn_ae.py`)
```
Input (140, 1) 
    ↓
Encoder:
    Conv1D(32, 7) → MaxPool1D(2)
    Conv1D(16, 7) → MaxPool1D(2) 
    Conv1D(8, 7)  → MaxPool1D(2)
    ↓ (18 timesteps)
Decoder:
    Conv1D(8, 7)  → UpSampling1D(2)
    Conv1D(16, 7) → UpSampling1D(2)
    Conv1D(32, 7) → UpSampling1D(2)
    Conv1D(1, 7)  → Crop to (140, 1)
```

### TCN Autoencoder (`tcn_ae.py`)
```
Input (140, 1)
    ↓
Stem: Conv1D(32, 1)
    ↓
TCN Blocks (dilations 1,2,4,8):
    Conv1D(32, 7, dilation=d) → BatchNorm → ReLU
    Conv1D(32, 7, dilation=d) → BatchNorm
    + Residual connection → ReLU
    ↓
Bottleneck: Conv1D(16, 1) → GlobalAveragePooling1D
    ↓
Decoder: Dense(140) → Reshape(140, 1)
```

## Performance Comparison

| Model | AUROC | AUPRC | F1 Score | Precision | Recall |
|-------|-------|-------|----------|-----------|--------|
| **CNN AE** | **0.972** | **0.914** | **0.672** | **0.507** | **0.999** |
| TCN AE | 0.953 | 0.867 | 0.681 | 0.517 | 0.996 |

The CNN autoencoder achieves excellent performance with near-perfect recall. Both models perform very well for ECG anomaly detection.

## Usage

### CNN Autoencoder (Recommended)
```bash
python cnn_ae.py
```

### TCN Autoencoder (Alternative)
```bash
python tcn_ae.py
```

Both scripts will:
1. Load and preprocess ECG-5000 data
2. Train autoencoder on normal samples only
3. Evaluate anomaly detection performance
4. Create quantized TFLite model
5. Log everything to MLflow
6. Print final metrics

## Requirements

```
tensorflow>=2.13.0
numpy
pandas
scikit-learn
mlflow
```

## Key Parameters

### CNN Autoencoder
- `EPOCHS = 20`: Training epochs
- `BATCH_SIZE = 512`: Batch size  
- `THRESH_FRAC = 0.90`: 90th percentile threshold for anomaly detection
- `CALIB_SAMPLES = 400`: Samples for quantization calibration

### TCN Autoencoder
- `EPOCHS = 20`: Training epochs
- `BATCH_SIZE = 512`: Batch size
- `THRESH_FRAC = 0.30`: 30th percentile threshold (more sensitive)
- `CALIB_SAMPLES = 400`: Samples for quantization calibration

## Results

Both models achieve excellent anomaly detection performance by:
- Training only on normal ECG patterns (unsupervised anomaly detection)
- Using proper encoder-decoder architectures
- Optimized threshold selection based on training reconstruction error
- INT8 quantization for efficient deployment
- Comprehensive MLflow experiment tracking

## Files

- `cnn_ae.py`: CNN autoencoder training and evaluation (recommended)
- `tcn_ae.py`: TCN autoencoder training and evaluation (alternative)
- `data/`: ECG-5000 dataset (TSV format)
- `mlruns/`: MLflow experiment tracking
- `.gitignore`: Excludes large files and artifacts
- `requirements.txt`: Python dependencies

## Model Artifacts

MLflow saves:
- FP32 SavedModel for TFLite conversion
- INT8 quantized TFLite model
- Training metrics and parameters
- Model architecture summary
- Performance benchmarks
