# ECG Anomaly Detection with CNN Autoencoder

A 1D CNN autoencoder for anomaly detection on ECG-5000 dataset with TensorFlow Lite quantization and MLflow tracking.

## Overview

This project implements a convolutional autoencoder for ECG anomaly detection that:
- Trains on normal ECG patterns only
- Detects anomalies based on reconstruction error
- Quantizes the model to INT8 using TensorFlow Lite
- Tracks experiments with MLflow
- Benchmarks GPU latency

## Features

- **1D CNN Autoencoder**: Encoder-decoder architecture with conv1d layers and pooling
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

## Architecture

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

## Usage

```bash
python cnn_ae.py
```

The script will:
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

- `EPOCHS = 20`: Training epochs
- `BATCH_SIZE = 512`: Batch size
- `THRESH_FRAC = 0.90`: 90th percentile threshold for anomaly detection
- `CALIB_SAMPLES = 400`: Samples for quantization calibration

## Results

The model achieves improved anomaly detection performance by:
- Training only on normal ECG patterns
- Using proper encoder-decoder architecture (vs. global pooling)
- Optimized threshold selection (90th vs 95th percentile)
- Enhanced debugging and metric tracking

## Files

- `cnn_ae.py`: Main training and evaluation script
- `data/`: ECG-5000 dataset (TSV format)
- `mlruns/`: MLflow experiment tracking
- `.gitignore`: Excludes large files and artifacts

## Model Artifacts

MLflow saves:
- FP32 SavedModel for TFLite conversion
- INT8 quantized TFLite model
- Training metrics and parameters
- Model architecture summary
- Performance benchmarks
