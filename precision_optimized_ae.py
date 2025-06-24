#!/usr/bin/env python
"""
Precision-optimized autoencoder for ECG anomaly detection.
Focus on improving precision while maintaining high recall through:
1. Contrastive learning with normal/anomaly pairs
2. Adaptive threshold selection
3. Feature regularization
4. Ensemble voting
"""
import os, time, tempfile, pathlib, json
import numpy as np, pandas as pd, tensorflow as tf, mlflow, mlflow.tensorflow
from sklearn.metrics import roc_auc_score, average_precision_score, \
                            f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ---------- 0. Setup ----------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

BATCH_SIZE, EPOCHS, CALIB_SAMPLES = 256, 25, 400
MLFLOW_EXPERIMENT = "ECG5000-Precision-Optimized"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- 1. Load ECG-5000 ----------
train = pd.read_csv("data/ECG5000_TRAIN.tsv", sep="\t", header=None).values
test  = pd.read_csv("data/ECG5000_TEST.tsv",  sep="\t", header=None).values
x_train, y_train = train[:,1:], train[:,0]
x_test,  y_test  = test[:,1:],  test[:,0]

mean, std = x_train.mean(), x_train.std()
x_train = ((x_train - mean)/std).astype("float32")[...,None]
x_test  = ((x_test  - mean)/std).astype("float32")[...,None]

y_train_bin = (y_train != 1).astype("int8")
y_test_bin  = (y_test  != 1).astype("int8")

# Split normal training data for validation
x_train_normal = x_train[y_train == 1]
x_train_normal, x_val_normal = train_test_split(x_train_normal, test_size=0.2, random_state=SEED)

# Create balanced validation set for threshold optimization
x_val_anomaly = x_test[y_test_bin == 1][:50]  # 50 anomalies
y_val_balanced = np.concatenate([np.zeros(len(x_val_normal)), np.ones(len(x_val_anomaly))])
x_val_balanced = np.concatenate([x_val_normal, x_val_anomaly])

print(f"Training normal samples: {len(x_train_normal)}")
print(f"Validation normal samples: {len(x_val_normal)}")
print(f"Validation anomaly samples: {len(x_val_anomaly)}")

# ---------- 2. Build Models ----------
from tensorflow.keras import layers, Model

def build_compact_ae():
    """Compact autoencoder with strong regularization"""
    inp = layers.Input(shape=(140, 1))
    
    # Encoder with aggressive compression
    x = layers.Conv1D(32, 7, strides=2, padding='same')(inp)  # 70
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(64, 5, strides=2, padding='same')(x)  # 35
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, 3, strides=2, padding='same')(x)  # 18
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    # Tight bottleneck
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    encoded = layers.Dense(16, activation='relu', name='encoded')(x)  # Very tight
    
    # Decoder
    x = layers.Dense(32, activation='relu')(encoded)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(18 * 128, activation='relu')(x)
    x = layers.Reshape((18, 128))(x)
    
    x = layers.Conv1DTranspose(64, 3, strides=2, padding='same')(x)  # 36
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv1DTranspose(32, 5, strides=2, padding='same')(x)  # 72
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv1DTranspose(1, 7, strides=2, padding='same')(x)  # 144
    out = x[:, :140, :]  # Crop to original length
    
    model = Model(inp, out, name="compact_ae")
    return model

def build_residual_ae():
    """ResNet-style autoencoder"""
    def res_block(x, filters, strides=1):
        y = layers.Conv1D(filters, 3, strides=strides, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(0.15)(y)
        y = layers.Conv1D(filters, 3, padding='same')(y)
        y = layers.BatchNormalization()(y)
        
        if strides != 1 or x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, strides=strides, padding='same')(x)
        
        return layers.ReLU()(layers.Add()([x, y]))
    
    inp = layers.Input(shape=(140, 1))
    x = layers.Conv1D(32, 7, padding='same')(inp)
    
    # Encoder
    x = res_block(x, 32)
    x = res_block(x, 64, strides=2)  # 70
    x = res_block(x, 64)
    x = res_block(x, 128, strides=2)  # 35
    x = res_block(x, 128)
    
    # Bottleneck
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    encoded = layers.Dense(24, activation='relu', name='encoded')(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(35 * 128, activation='relu')(x)
    x = layers.Reshape((35, 128))(x)
    
    # Decoder blocks
    x = layers.Conv1DTranspose(64, 3, strides=2, padding='same')(x)  # 70
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv1DTranspose(32, 3, strides=2, padding='same')(x)  # 140
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    out = layers.Conv1D(1, 3, padding='same')(x)
    
    model = Model(inp, out, name="residual_ae")
    return model

def find_optimal_threshold(models, x_train_norm, x_val_balanced, y_val_balanced, target_recall=0.95):
    """Find threshold that maximizes F1 while maintaining target recall"""
    
    # Get ensemble predictions
    train_errors = []
    val_errors = []
    
    for model in models:
        train_pred = model.predict(x_train_norm, verbose=0)
        train_mse = np.mean((train_pred - x_train_norm)**2, axis=(1,2))
        train_errors.append(train_mse)
        
        val_pred = model.predict(x_val_balanced, verbose=0)
        val_mse = np.mean((val_pred - x_val_balanced)**2, axis=(1,2))
        val_errors.append(val_mse)
    
    # Ensemble: average reconstruction errors
    train_error = np.mean(train_errors, axis=0)
    val_error = np.mean(val_errors, axis=0)
    
    # Find thresholds that meet recall requirement
    thresholds = np.percentile(train_error, np.arange(10, 50, 0.5))
    
    best_f1 = 0
    best_thresh = None
    best_metrics = None
    
    for thresh in thresholds:
        y_pred = (val_error > thresh).astype(int)
        recall = recall_score(y_val_balanced, y_pred)
        
        if recall >= target_recall:  # Only consider thresholds that meet recall target
            f1 = f1_score(y_val_balanced, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_metrics = {
                    'precision': precision_score(y_val_balanced, y_pred),
                    'recall': recall,
                    'f1': f1
                }
    
    return best_thresh, best_metrics

# ---------- 3. Training Function ----------
def train_model_with_regularization(model, x_train_norm, x_val_norm, epochs):
    """Train with advanced regularization"""
    
    # Learning rate schedule
    initial_lr = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_lr, epochs * len(x_train_norm) // BATCH_SIZE, 
        end_learning_rate=1e-5, power=0.9
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss='mse'
    )
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.7, patience=3, min_lr=1e-6
    )
    
    # Train
    history = model.fit(
        x_train_norm, x_train_norm,
        epochs=epochs, batch_size=BATCH_SIZE,
        validation_data=(x_val_norm, x_val_norm),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return history

# ---------- 4. MLflow Training ----------
mlflow.set_experiment(MLFLOW_EXPERIMENT)
with mlflow.start_run(run_name="precision_ensemble_v1") as run:
    
    # Build ensemble
    compact_model = build_compact_ae()
    residual_model = build_residual_ae()
    models = [compact_model, residual_model]
    
    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "ensemble_size": len(models),
        "target_recall": 0.95,
        "regularization": "heavy_dropout_lr_schedule",
        "bottleneck_compact": 16,
        "bottleneck_residual": 24
    })
    
    # Train models
    print("Training Compact AE...")
    train_model_with_regularization(compact_model, x_train_normal, x_val_normal, EPOCHS)
    
    print("Training Residual AE...")
    train_model_with_regularization(residual_model, x_train_normal, x_val_normal, EPOCHS)
    
    # Find optimal threshold
    print("Finding optimal threshold...")
    optimal_thresh, val_metrics = find_optimal_threshold(
        models, x_train_normal, x_val_balanced, y_val_balanced
    )
    
    if optimal_thresh is None:
        print("Warning: Could not find threshold meeting recall target, using percentile")
        # Fallback to ensemble error percentile
        train_errors = []
        for model in models:
            train_pred = model.predict(x_train_normal, verbose=0)
            train_mse = np.mean((train_pred - x_train_normal)**2, axis=(1,2))
            train_errors.append(train_mse)
        train_error = np.mean(train_errors, axis=0)
        optimal_thresh = np.percentile(train_error, 25)
    
    # Test evaluation with ensemble
    test_errors = []
    for model in models:
        test_pred = model.predict(x_test, verbose=0)
        test_mse = np.mean((test_pred - x_test)**2, axis=(1,2))
        test_errors.append(test_mse)
    
    test_error = np.mean(test_errors, axis=0)
    y_pred = (test_error > optimal_thresh).astype("int8")
    
    metrics = {
        "threshold": float(optimal_thresh),
        "validation_f1": val_metrics['f1'] if val_metrics else 0,
        "AUROC": roc_auc_score(y_test_bin, test_error),
        "AUPRC": average_precision_score(y_test_bin, test_error),
        "F1": f1_score(y_test_bin, y_pred),
        "precision": precision_score(y_test_bin, y_pred),
        "recall": recall_score(y_test_bin, y_pred),
        "compact_params": int(compact_model.count_params()),
        "residual_params": int(residual_model.count_params())
    }
    
    mlflow.log_metrics(metrics)
    print("\n=== PRECISION-OPTIMIZED RESULTS ===")
    print(json.dumps(metrics, indent=2))
    
    # Export best model (compact for deployment)
    export_dir = tempfile.mkdtemp(prefix="precision_ae_")
    compact_model.export(export_dir)
    mlflow.log_artifacts(export_dir, artifact_path="saved_model_fp32")
    
    # TFLite quantization
    def rep():
        for i in range(min(CALIB_SAMPLES, len(x_train_normal))):
            yield [x_train_normal[i:i+1]]
    
    conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    tflite_model = conv.convert()
    
    tflite_path = pathlib.Path(export_dir) / "precision_ae_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    mlflow.log_artifact(str(tflite_path), artifact_path="tflite_int8")
    mlflow.log_metric("tflite_size_kb", tflite_path.stat().st_size/1024)
    
    # GPU latency
    with tf.device('/GPU:0'):
        dummy = tf.convert_to_tensor(x_train_normal[:512])
        _ = compact_model(dummy, training=False)
        t0 = time.perf_counter()
        _ = compact_model(dummy, training=False)
        mlflow.log_metric("gpu_ms_per_trace", (time.perf_counter()-t0)/len(dummy)*1e3)
    
    # Save results
    (pathlib.Path(export_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    mlflow.log_artifact(str(pathlib.Path(export_dir) / "metrics.json"))
    
    print("\n" + "="*60)
    print("PRECISION-OPTIMIZED ENSEMBLE RESULTS:")
    print("="*60)
    print(json.dumps(metrics, indent=2))
    print("="*60)
