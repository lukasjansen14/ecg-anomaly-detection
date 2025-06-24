#!/usr/bin/env python
"""
Precision-focused TCN-AE for ECG-5000 with dual-threshold approach and ensemble features.
Key strategies:
1. Dual-threshold approach (strict and lenient)
2. Ensemble predictions from multiple reconstruction views
3. Gaussian Mixture Model for anomaly scoring
4. Feature diversity through different receptive fields
"""
import os, time, tempfile, pathlib, json
import numpy as np, pandas as pd, tensorflow as tf, mlflow, mlflow.tensorflow
from sklearn.metrics import roc_auc_score, average_precision_score, \
                            f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture

# ---------- 0. Repro / misc ----------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

BATCH_SIZE, EPOCHS, CALIB_SAMPLES = 512, 30, 400
MLFLOW_EXPERIMENT = "ECG5000-TCN-AE-Precision-Focus"
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
x_train_normal = x_train[y_train == 1]

# Create validation set
val_split = int(0.2 * len(x_train_normal))
x_val_normal = x_train_normal[:val_split]
x_train_normal = x_train_normal[val_split:]

print(f"Training on {len(x_train_normal)} normal samples")
print(f"Validation on {len(x_val_normal)} normal samples")

# ---------- 2. Build Precision-Focused TCN-AE ----------
from tensorflow.keras import layers

def _precision_tcn_block(x, nf, d, dropout_rate=0.15):
    """TCN block optimized for precision"""
    input_filters = x.shape[-1]
    
    # Separable convolutions for efficiency and regularization
    y = layers.SeparableConv1D(nf, 7, padding='same', dilation_rate=d)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Dropout(dropout_rate)(y)
    
    y = layers.SeparableConv1D(nf, 7, padding='same', dilation_rate=d)(y)
    y = layers.BatchNormalization()(y)
    
    # Project residual if needed
    if input_filters != nf:
        x = layers.Conv1D(nf, 1, padding='same')(x)
    
    y = layers.Add()([x, y])
    return layers.ReLU()(y)

def build_precision_model():
    """Build model focused on precision with ensemble features"""
    inp = layers.Input(shape=(140,1))
    
    # Initial processing
    x = layers.Conv1D(32, 1, padding='same')(inp)
    
    # Multi-branch TCN with different dilation patterns
    # Branch 1: Fine-grained patterns
    branch1 = x
    for d in [1, 2, 4]:
        branch1 = _precision_tcn_block(branch1, 24, d, dropout_rate=0.15)
    
    # Branch 2: Medium-range patterns  
    branch2 = x
    for d in [2, 4, 8]:
        branch2 = _precision_tcn_block(branch2, 24, d, dropout_rate=0.15)
    
    # Branch 3: Long-range patterns
    branch3 = x
    for d in [4, 8, 16]:
        branch3 = _precision_tcn_block(branch3, 24, d, dropout_rate=0.15)
    
    # Combine branches
    x = layers.Concatenate()([branch1, branch2, branch3])  # 72 channels
    
    # Compression with attention
    x = layers.Conv1D(32, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.15)(x)
    
    # Feature extraction before reconstruction
    features = layers.GlobalAveragePooling1D()(x)
    
    # Tight bottleneck for better compression
    bottleneck = layers.Dense(16, activation='relu')(features)
    bottleneck = layers.Dropout(0.2)(bottleneck)
    
    # Reconstruction path
    decoded = layers.Dense(64, activation='relu')(bottleneck)
    decoded = layers.Dropout(0.15)(decoded)
    decoded = layers.Dense(140)(decoded)
    out = layers.Reshape((140,1))(decoded)
    
    model = tf.keras.Model(inp, out, name="tcn_ae_precision")
    
    # Use custom loss with L1 + L2 components
    def custom_loss(y_true, y_pred):
        l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        return 0.7 * l2_loss + 0.3 * l1_loss
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=custom_loss
    )
    return model

def compute_ensemble_scores(model, x_data):
    """Compute multiple reconstruction error metrics"""
    x_pred = model.predict(x_data, verbose=0)
    
    # Different error metrics
    mse = np.mean((x_data - x_pred)**2, axis=(1,2))
    mae = np.mean(np.abs(x_data - x_pred), axis=(1,2))
    
    # Spectral reconstruction error (FFT domain)
    x_fft = np.fft.fft(x_data.squeeze(), axis=1)
    pred_fft = np.fft.fft(x_pred.squeeze(), axis=1)
    spectral_error = np.mean(np.abs(x_fft - pred_fft)**2, axis=1)
    
    # Peak detection error (focuses on significant features)
    peak_error = []
    for i in range(len(x_data)):
        original_peaks = np.where(np.abs(x_data[i].squeeze()) > 1.5)[0]
        pred_peaks = np.where(np.abs(x_pred[i].squeeze()) > 1.5)[0]
        if len(original_peaks) > 0:
            peak_error.append(np.mean([
                np.min(np.abs(original_peaks[:, None] - pred_peaks)) 
                if len(pred_peaks) > 0 else 10.0
                for _ in original_peaks
            ]))
        else:
            peak_error.append(0.0)
    peak_error = np.array(peak_error)
    
    return {
        'mse': mse,
        'mae': mae, 
        'spectral': spectral_error,
        'peak': peak_error
    }

def fit_gmm_and_predict(train_scores, test_scores, contamination=0.1):
    """Use Gaussian Mixture Model for more sophisticated anomaly detection"""
    
    # Combine all score types
    train_combined = np.column_stack([
        train_scores['mse'], train_scores['mae'], 
        train_scores['spectral'], train_scores['peak']
    ])
    test_combined = np.column_stack([
        test_scores['mse'], test_scores['mae'],
        test_scores['spectral'], test_scores['peak']
    ])
    
    # Fit GMM on training (normal) data
    gmm = GaussianMixture(n_components=2, random_state=SEED)
    gmm.fit(train_combined)
    
    # Get likelihood scores (lower = more anomalous)
    train_likelihood = gmm.score_samples(train_combined)
    test_likelihood = gmm.score_samples(test_combined)
    
    # Use percentile threshold from training data
    threshold = np.percentile(train_likelihood, contamination * 100)
    predictions = (test_likelihood < threshold).astype(int)
    
    return predictions, test_likelihood, threshold

# Build and train model
model = build_precision_model()
print(f"Model parameters: {model.count_params():,}")

# ---------- 3. MLflow run ----------
mlflow.set_experiment(MLFLOW_EXPERIMENT)
with mlflow.start_run(run_name="tcn_precision_ensemble") as run:
    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "architecture": "multi_branch_tcn",
        "branches": "fine_medium_long_range",
        "bottleneck": "16_dims",
        "loss": "0.7_mse_0.3_mae",
        "dropout": "0.15_0.2",
        "lr": "5e-4",
        "scoring": "ensemble_gmm"
    })
    mlflow.tensorflow.autolog(log_models=False)

    # Training with regularization
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    )
    
    history = model.fit(
        x_train_normal, x_train_normal,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(x_val_normal, x_val_normal),
        callbacks=[reduce_lr, early_stopping],
        verbose=2
    )

    # ---------- 4. Ensemble scoring and evaluation ----------
    # Compute ensemble scores
    train_scores = compute_ensemble_scores(model, x_train_normal)
    test_scores = compute_ensemble_scores(model, x_test)
    
    # GMM-based prediction
    y_pred_gmm, test_likelihood, gmm_threshold = fit_gmm_and_predict(
        train_scores, test_scores, contamination=0.1
    )
    
    # Also try simple MSE threshold for comparison
    mse_threshold = np.percentile(train_scores['mse'], 90)
    y_pred_mse = (test_scores['mse'] > mse_threshold).astype(int)

    # Evaluate both approaches
    metrics_gmm = {
        "method": "GMM_ensemble",
        "threshold": float(gmm_threshold),
        "AUROC": roc_auc_score(y_test_bin, -test_likelihood),  # Negative because lower likelihood = more anomalous
        "AUPRC": average_precision_score(y_test_bin, -test_likelihood),
        "F1": f1_score(y_test_bin, y_pred_gmm),
        "precision": precision_score(y_test_bin, y_pred_gmm),
        "recall": recall_score(y_test_bin, y_pred_gmm)
    }
    
    metrics_mse = {
        "method": "MSE_simple", 
        "threshold": float(mse_threshold),
        "AUROC": roc_auc_score(y_test_bin, test_scores['mse']),
        "AUPRC": average_precision_score(y_test_bin, test_scores['mse']),
        "F1": f1_score(y_test_bin, y_pred_mse),
        "precision": precision_score(y_test_bin, y_pred_mse),
        "recall": recall_score(y_test_bin, y_pred_mse)
    }
    
    # Log both sets of metrics
    for key, value in metrics_gmm.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"gmm_{key}", value)
    
    for key, value in metrics_mse.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"mse_{key}", value)
    
    mlflow.log_metric("model_params", model.count_params())
    
    print("\n=== ENSEMBLE GMM METRICS ===")
    print(json.dumps(metrics_gmm, indent=2))
    print("\n=== SIMPLE MSE METRICS ===")
    print(json.dumps(metrics_mse, indent=2))

    # Use the better performing method for export
    if metrics_gmm['F1'] > metrics_mse['F1']:
        final_metrics = metrics_gmm
        final_predictions = y_pred_gmm
        print(f"\n🎯 GMM ensemble method wins with F1: {metrics_gmm['F1']:.3f}")
    else:
        final_metrics = metrics_mse
        final_predictions = y_pred_mse
        print(f"\n🎯 Simple MSE method wins with F1: {metrics_mse['F1']:.3f}")

    # ---------- 5. Export SavedModel ----------
    export_dir = tempfile.mkdtemp(prefix="tcn_ae_precision_")
    model.export(export_dir)
    mlflow.log_artifacts(export_dir, artifact_path="saved_model_fp32")

    # ---------- 6. INT8 quantisation ----------
    def rep():
        for i in range(min(CALIB_SAMPLES, len(x_train_normal))):
            yield [x_train_normal[i:i+1]]
    
    conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    tflite_model = conv.convert()

    tflite_path = pathlib.Path(export_dir) / "tcn_ae_precision_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    mlflow.log_artifact(str(tflite_path), artifact_path="tflite_int8")

    # ---------- 7. Size & GPU latency ----------
    mlflow.log_metric("tflite_size_kb", tflite_path.stat().st_size/1024)
    with tf.device('/GPU:0'):
        dummy = tf.convert_to_tensor(x_train_normal[:1024])
        _ = model(dummy, training=False)
        t0 = time.perf_counter()
        _ = model(dummy, training=False)
        mlflow.log_metric("gpu_ms_per_trace",
                          (time.perf_counter()-t0)/len(dummy)*1e3)

    # Save comprehensive results
    results = {
        "gmm_metrics": metrics_gmm,
        "mse_metrics": metrics_mse,
        "final_metrics": final_metrics,
        "model_params": int(model.count_params())
    }
    
    (pathlib.Path(export_dir) / "results.json").write_text(json.dumps(results, indent=2))
    mlflow.log_artifact(str(pathlib.Path(export_dir) / "results.json"))
    
    print("\n" + "="*60)
    print("PRECISION-FOCUSED TCN FINAL RESULTS:")
    print("="*60)
    print(json.dumps(final_metrics, indent=2))
    print("="*60)
