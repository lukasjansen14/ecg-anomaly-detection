#!/usr/bin/env python
"""
Train a 1-D CNN Auto-encoder on ECG-5000, quantise to INT8 TFLite,
compute anomaly-detection metrics, benchmark GPU latency, and log
everything to MLflow.
"""
import os, time, tempfile, pathlib, json, contextlib
import numpy as np, pandas as pd, tensorflow as tf, mlflow, mlflow.tensorflow
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,\
                             precision_score, recall_score

# ---------- 0. Repro / misc ----------
SEED               = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_URL           = "https://www.timeseriesclassification.com/Downloads/ECG5000.zip"
BATCH_SIZE         = 512
EPOCHS             = 20
CALIB_SAMPLES      = 400                      # for representative dataset
THRESH_FRAC        = .90                      # 90th-percentile reconstruction error threshold
MLFLOW_EXPERIMENT  = "ECG5000-CNN-AE-Compression"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # silence TF verbosity

# ---------- 1. Load ECG-5000 ----------
# zip_path = tf.keras.utils.get_file("ECG5000.zip", origin=DATA_URL, extract=True)
# base_dir = pathlib.Path(zip_path).with_suffix('')  # strip .zip
# train_csv = "ECG5000_TRAIN.txt"
# test_csv  = base_dir / "ECG5000_TEST.txt"

train = pd.read_csv("data/ECG5000_TRAIN.tsv", sep="\t", header=None).values
test  = pd.read_csv("data/ECG5000_TEST.tsv", sep="\t", header=None).values
x_train, y_train = train[:,1:], train[:,0]
x_test,  y_test  = test[:,1:],  test[:,0]

# reshape, normalise
x_train = ((x_train - x_train.mean()) / x_train.std()).astype("float32")[...,None]
x_test  = ((x_test  - x_train.mean()) / x_train.std()).astype("float32")[...,None]
y_train_bin = (y_train != 1).astype("int8")           # 0=normal, 1=anomaly
y_test_bin  = (y_test  != 1).astype("int8")

# Filter training data to only normal samples (class 1) for autoencoder training
normal_mask = (y_train == 1)
x_train_normal = x_train[normal_mask]
print(f"Training on {len(x_train_normal)} normal samples out of {len(x_train)} total")

# ---------- 2. Build 1-D CNN Auto-encoder ----------
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(140,1)),
        # Encoder
        tf.keras.layers.Conv1D(32, 7, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling1D(2, padding="same"),
        tf.keras.layers.Conv1D(16, 7, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling1D(2, padding="same"),
        tf.keras.layers.Conv1D(8, 7, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling1D(2, padding="same"),  # 18 timesteps
        
        # Decoder
        tf.keras.layers.Conv1D(8, 7, padding="same", activation='relu'),
        tf.keras.layers.UpSampling1D(2),
        tf.keras.layers.Conv1D(16, 7, padding="same", activation='relu'),
        tf.keras.layers.UpSampling1D(2),
        tf.keras.layers.Conv1D(32, 7, padding="same", activation='relu'),
        tf.keras.layers.UpSampling1D(2),
        tf.keras.layers.Conv1D(1, 7, padding="same", activation=None),
        
        # Crop to original size if needed
        tf.keras.layers.Lambda(lambda x: x[:, :140, :])
    ], name="cnn_ae")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

model = build_model()

# ---------- 3. MLflow run ----------
mlflow.set_experiment(MLFLOW_EXPERIMENT)
with mlflow.start_run(run_name="fp32_train") as run:
    mlflow.log_params({"epochs":EPOCHS, "batch_size":BATCH_SIZE,
                       "lr": tf.keras.backend.get_value(model.optimizer.learning_rate)})

    mlflow.tensorflow.autolog(log_models=False)     # log metrics every epoch

    model.fit(x_train_normal, x_train_normal, epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(x_test,x_test),
              verbose=2)

    # ---------- 4. Threshold & metrics on FP32 ----------
    recon_train = model.predict(x_train_normal, verbose=0)
    recon_test  = model.predict(x_test,  verbose=0)
    mse_train   = np.mean(np.square(recon_train - x_train_normal), axis=(1,2))
    mse_test    = np.mean(np.square(recon_test  - x_test ), axis=(1,2))

    thresh = np.quantile(mse_train, THRESH_FRAC)
    y_pred = (mse_test > thresh).astype("int8")
    
    # Debug information
    print(f"\nAnomaly Detection Debug Info:")
    print(f"Normal samples in training: {len(x_train_normal)}")
    print(f"Test samples: {len(x_test)} (Normal: {np.sum(y_test_bin == 0)}, Anomaly: {np.sum(y_test_bin == 1)})")
    print(f"MSE stats - Train: min={mse_train.min():.4f}, max={mse_train.max():.4f}, mean={mse_train.mean():.4f}")
    print(f"MSE stats - Test: min={mse_test.min():.4f}, max={mse_test.max():.4f}, mean={mse_test.mean():.4f}")
    print(f"Threshold ({int(THRESH_FRAC*100)}th percentile): {thresh:.4f}")
    print(f"Predicted anomalies: {np.sum(y_pred)} out of {len(y_pred)}")
    print(f"Actual anomalies: {np.sum(y_test_bin)} out of {len(y_test_bin)}")

    metrics = {
        "threshold":            float(thresh),
        "AUROC":                roc_auc_score(y_test_bin, mse_test),
        "AUPRC":                average_precision_score(y_test_bin, mse_test),
        "F1":                   f1_score(y_test_bin, y_pred),
        "precision":            precision_score(y_test_bin, y_pred),
        "recall":               recall_score(y_test_bin, y_pred)
    }
    mlflow.log_metrics(metrics)

    # ---------- 5. Save FP32 SavedModel artifact ----------
    # save_dir = tempfile.mkdtemp()
    # model.save(save_dir, include_optimizer=False)
    # mlflow.log_artifacts(save_dir, artifact_path="saved_model_fp32")
    
    export_dir = tempfile.mkdtemp()
    model.export(export_dir)          # SavedModel for TFLite
    mlflow.log_artifacts(export_dir, artifact_path="saved_model_fp32")

    # ---------- 6. Post-training INT8 quantisation ----------
    def representative_gen():
        for i in range(min(CALIB_SAMPLES, len(x_train_normal))):
            yield [x_train_normal[i:i+1]]
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    tflite_path = pathlib.Path(export_dir) / "cnn_ae_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    mlflow.log_artifact(str(tflite_path), artifact_path="tflite_int8")

    # ---------- 7. File size & (GPU) latency benchmarks ----------
    size_kb = os.path.getsize(tflite_path) / 1024
    mlflow.log_metric("tflite_size_kb", size_kb)

    # quick GPU latency sample for curiosity (NPU bench happens on board)
    with tf.device('/GPU:0'):
        dummy = tf.convert_to_tensor(x_train_normal[:min(1024, len(x_train_normal))])
        _ = model(dummy, training=False)
        t0 = time.perf_counter()
        _ = model(dummy, training=False)
        gpu_lat_ms = (time.perf_counter() - t0) / len(dummy) * 1e3
    mlflow.log_metric("gpu_ms_per_trace", gpu_lat_ms)

    # ---------- 8. Persist metrics to plain JSON (optional) ----------
    (pathlib.Path(export_dir)/"metrics.json").write_text(json.dumps(metrics, indent=2))
    mlflow.log_artifact(str(pathlib.Path(export_dir)/"metrics.json"))
    
    # Print metrics for easy copying
    print("\n" + "="*50)
    print("FINAL METRICS:")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    print("="*50)
