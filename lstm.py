#!/usr/bin/env python
"""
Train an LSTM Auto-encoder on ECG-5000, quantise to INT8 TFLite,
compute anomaly metrics, benchmark GPU latency, and log to MLflow.
"""
import os, time, tempfile, pathlib, json
import numpy as np, pandas as pd, tensorflow as tf
import mlflow, mlflow.tensorflow
from sklearn.metrics import roc_auc_score, average_precision_score, \
                             f1_score, precision_score, recall_score

# 0. Repro & constants -------------------------------------------------------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
BATCH, EPOCHS, CAL_SAMPLES, THRESH_FRAC = 256, 30, 400, 0.30
EXPNAME = "ECG5000-LSTM-AE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 1. Load ECG-5000 TSV -------------------------------------------------------
train = pd.read_csv("data/ECG5000_TRAIN.tsv", sep="\t", header=None).values
test  = pd.read_csv("data/ECG5000_TEST.tsv",  sep="\t", header=None).values
x_tr, y_tr = train[:,1:], train[:,0]
x_te, y_te = test[:,1:],  test[:,0]

mean, std = x_tr.mean(), x_tr.std()
x_tr = ((x_tr-mean)/std).astype("float32")[...,None]
x_te = ((x_te-mean)/std).astype("float32")[...,None]
y_tr_b = (y_tr!=1).astype("int8"); y_te_b = (y_te!=1).astype("int8")

normal_mask  = (y_tr==1)
x_tr_norm    = x_tr[normal_mask]

# 2. Build 2-layer LSTM-AE ----------------------------------------------------
from tensorflow.keras import layers, Model

def build_lstm_ae():
    inp = layers.Input((140,1))
    x   = layers.LSTM(64, return_sequences=True)(inp)
    x   = layers.LSTM(32, return_sequences=False)(x)
    x   = layers.RepeatVector(140)(x)
    x   = layers.LSTM(32, return_sequences=True)(x)
    x   = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(1))(x)
    model = Model(inp,out,name="lstm_ae")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model

model = build_lstm_ae()

# 3. MLflow run --------------------------------------------------------------
mlflow.set_experiment(EXPNAME)
with mlflow.start_run(run_name="fp32_train"):
    mlflow.log_params({"epochs":EPOCHS,"batch":BATCH,"layers":"64-32-32-64"})

    mlflow.tensorflow.autolog(log_models=False)
    model.fit(x_tr_norm, x_tr_norm, epochs=EPOCHS, batch_size=BATCH,
              validation_data=(x_te,x_te), verbose=2)

    # 4. Metrics -------------------------------------------------------------
    mse_tr = np.mean((model.predict(x_tr_norm)-x_tr_norm)**2,axis=(1,2))
    mse_te = np.mean((model.predict(x_te)-x_te)**2,axis=(1,2))
    thr    = np.quantile(mse_tr, THRESH_FRAC)
    y_pr   = (mse_te>thr).astype("int8")

    mets = {"threshold":float(thr),
            "AUROC":roc_auc_score(y_te_b,mse_te),
            "AUPRC":average_precision_score(y_te_b,mse_te),
            "F1":f1_score(y_te_b,y_pr),
            "precision":precision_score(y_te_b,y_pr),
            "recall":recall_score(y_te_b,y_pr)}
    mlflow.log_metrics(mets)
    print(json.dumps(mets,indent=2))

    # 5. Export SavedModel ---------------------------------------------------
    save_dir = tempfile.mkdtemp(prefix="lstm_ae_")
    model.export(save_dir)
    mlflow.log_artifacts(save_dir,"saved_model_fp32")

    # 6. INT8 Post-training quantisation ------------------------------------
    def rep_ds():
        for i in range(min(CAL_SAMPLES,len(x_tr_norm))):
            yield [x_tr_norm[i:i+1]]
    conv = tf.lite.TFLiteConverter.from_saved_model(save_dir)
    conv.optimizations=[tf.lite.Optimize.DEFAULT]
    conv.representative_dataset=rep_ds
    conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type=conv.inference_output_type=tf.int8
    tflite = conv.convert()
    tflite_path = pathlib.Path(save_dir)/"lstm_ae_int8.tflite"
    tflite_path.write_bytes(tflite)
    mlflow.log_artifact(str(tflite_path),"tflite_int8")
    mlflow.log_metric("tflite_kB", tflite_path.stat().st_size/1024)

    # 7. Quick GPU latency check (for reference) ----------------------------
    with tf.device('/GPU:0'):
        dummy=x_tr_norm[:256]; model(dummy,training=False)
        t0=time.perf_counter(); model(dummy,training=False)
        mlflow.log_metric("gpu_ms_per_trace",(time.perf_counter()-t0)/256*1e3)

    # 8. Save metrics as JSON ----------------------------------------------
    (pathlib.Path(save_dir)/"metrics.json").write_text(json.dumps(mets,indent=2))
    mlflow.log_artifact(str(pathlib.Path(save_dir)/"metrics.json"))
    # Print metrics for easy copying
    print("\n" + "="*50)
    print("FINAL METRICS:")
    print("="*50)
    print(json.dumps(mets, indent=2))
    print("="*50)
# End of script --------------------------------------------------------------
# Note: This script assumes the ECG-5000 dataset is available in the specified paths.
# Make sure to adjust the paths and parameters as needed for your environment.
# The script uses TensorFlow and MLflow for model training, evaluation, and logging.        