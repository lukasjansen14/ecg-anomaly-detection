#!/usr/bin/env python
"""
Train a Temporal Convolutional-Network Auto-encoder (TCN-AE) on ECG-5000,
quantise to INT8 TFLite, compute anomaly-detection metrics, benchmark GPU
latency, and log everything to MLflow.
"""
import os, time, tempfile, pathlib, json
import numpy as np, pandas as pd, tensorflow as tf, mlflow, mlflow.tensorflow
from sklearn.metrics import roc_auc_score, average_precision_score, \
                            f1_score, precision_score, recall_score

# ---------- 0. Repro / misc ----------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

BATCH_SIZE, EPOCHS, CALIB_SAMPLES = 512, 30, 400  # More epochs only
THRESH_FRAC = .20                        # Try lower threshold for better recall
MLFLOW_EXPERIMENT = "ECG5000-TCN-AE-Conservative"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- 1. Load ECG-5000 ----------
train = pd.read_csv("data/ECG5000_TRAIN.tsv", sep="\t", header=None).values
test  = pd.read_csv("data/ECG5000_TEST.tsv",  sep="\t", header=None).values
x_train, y_train = train[:,1:], train[:,0]
x_test,  y_test  = test[:,1:],  test[:,0]

mean, std = x_train.mean(), x_train.std()
x_train = ((x_train - mean)/std).astype("float32")[...,None]
x_test  = ((x_test  - mean)/std).astype("float32")[...,None]

y_train_bin = (y_train != 1).astype("int8")        # 1 = normal
y_test_bin  = (y_test  != 1).astype("int8")
x_train_normal = x_train[y_train == 1]
print(f"Training on {len(x_train_normal)} normal samples")

# ---------- 2. Build TCN-AE ----------
from tensorflow.keras import layers
def _tcn_block(x, nf, d, dropout_rate=0.05):  # Minimal dropout
    # Get input filters for residual connection
    input_filters = x.shape[-1]
    
    y = layers.Conv1D(nf, 7, padding='same', dilation_rate=d)(x)
    y = layers.BatchNormalization()(y); y = layers.ReLU()(y)
    y = layers.Dropout(dropout_rate)(y)  # Light dropout
    y = layers.Conv1D(nf, 7, padding='same', dilation_rate=d)(y)
    y = layers.BatchNormalization()(y)
    
    # If input and output channels differ, project the residual
    if input_filters != nf:
        x = layers.Conv1D(nf, 1, padding='same')(x)
    
    y = layers.Add()([x, y])
    return layers.ReLU()(y)

def build_model():
    inp = layers.Input(shape=(140,1))
    x   = layers.Conv1D(32, 1, padding='same')(inp)  # Back to original size
    
    # Keep original TCN structure but add one more dilation
    for d in [1,2,4,8,16]:  # Just one extra dilation
        x = _tcn_block(x, 32, d, dropout_rate=0.05)  # Light dropout
    
    # Keep the original decoder structure that worked
    x = layers.Conv1D(16, 1, padding='same')(x)  # Light compression
    x = layers.GlobalAveragePooling1D()(x)  # Keep original approach
    
    # Original decoder
    x = layers.Dense(140)(x)
    out = layers.Reshape((140,1))(x)
    
    model = tf.keras.Model(inp, out, name="tcn_ae_conservative")
    model.compile(optimizer=tf.keras.optimizers.Adam(8e-4), loss="mse")  # Slightly lower LR
    return model

model = build_model()

# ---------- 3. MLflow run ----------
mlflow.set_experiment(MLFLOW_EXPERIMENT)
with mlflow.start_run(run_name="tcn_conservative") as run:
    mlflow.log_params({"epochs":EPOCHS,"batch_size":BATCH_SIZE,
                       "dilations":"1,2,4,8,16","filters":"32","dropout":"0.05",
                       "lr":"8e-4","thresh_frac":THRESH_FRAC})
    mlflow.tensorflow.autolog(log_models=False)

    model.fit(x_train_normal, x_train_normal,
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(x_test,x_test), verbose=2)

    # ---------- 4. Metrics ----------
    mse_train = np.mean((model.predict(x_train_normal,verbose=0)-x_train_normal)**2,axis=(1,2))
    mse_test  = np.mean((model.predict(x_test,verbose=0)-x_test)**2,axis=(1,2))
    thresh    = np.quantile(mse_train, THRESH_FRAC)
    y_pred    = (mse_test > thresh).astype("int8")

    metrics = {
        "threshold": thresh,
        "AUROC":     roc_auc_score(y_test_bin, mse_test),
        "AUPRC":     average_precision_score(y_test_bin, mse_test),
        "F1":        f1_score(y_test_bin, y_pred),
        "precision": precision_score(y_test_bin, y_pred),
        "recall":    recall_score(y_test_bin, y_pred)
    }
    mlflow.log_metrics(metrics)
    print("\n=== FINAL METRICS ===\n", json.dumps(metrics, indent=2))

    # ---------- 5. Export SavedModel ----------
    export_dir = tempfile.mkdtemp(prefix="tcn_ae_")
    model.export(export_dir)
    mlflow.log_artifacts(export_dir, artifact_path="saved_model_fp32")

    # ---------- 6. INT8 quantisation ----------
    def rep():
        for i in range(min(CALIB_SAMPLES, len(x_train_normal))):
            yield [x_train_normal[i:i+1]]
    conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    conv.optimizations=[tf.lite.Optimize.DEFAULT]
    conv.representative_dataset=rep
    conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type=conv.inference_output_type=tf.int8
    tflite_model = conv.convert()

    tflite_path = pathlib.Path(export_dir)/"tcn_ae_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    mlflow.log_artifact(str(tflite_path), artifact_path="tflite_int8")

    # ---------- 7. Size & GPU latency ----------
    mlflow.log_metric("tflite_size_kb", tflite_path.stat().st_size/1024)
    with tf.device('/GPU:0'):
        dummy = tf.convert_to_tensor(x_train_normal[:1024])
        _ = model(dummy, training=False)
        t0 = time.perf_counter(); _ = model(dummy, training=False)
        mlflow.log_metric("gpu_ms_per_trace",
                          (time.perf_counter()-t0)/len(dummy)*1e3)

    (pathlib.Path(export_dir) / "metrics.json").write_text(json.dumps(metrics,indent=2))
    mlflow.log_artifact(str(pathlib.Path(export_dir) / "metrics.json"))
    print("\n" + "="*50)
    print("FINAL METRICS:")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    print("="*50)
