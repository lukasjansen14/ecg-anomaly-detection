#!/usr/bin/env python
"""
Train a lightweight ResNet-1D Auto-encoder on ECG-5000, quantise to INT8
TFLite, compute anomaly metrics, benchmark GPU latency, and log to MLflow.
"""
import os, time, tempfile, pathlib, json, numpy as np, pandas as pd, tensorflow as tf
import mlflow, mlflow.tensorflow
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

# ---------- 0. Repro / misc ----------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
BATCH, EPOCHS, CAL_SAMPLES, THRESH_FRAC = 512, 20, 400, 0.30
EXPNAME = "ECG5000-ResNet1D-AE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- 1. Load ECG-5000 (tsv) ----------
train = pd.read_csv("data/ECG5000_TRAIN.tsv", sep="\t", header=None).values
test  = pd.read_csv("data/ECG5000_TEST.tsv",  sep="\t", header=None).values
x_tr, y_tr = train[:,1:], train[:,0]
x_te, y_te = test[:,1:],  test[:,0]

mean, std = x_tr.mean(), x_tr.std()
x_tr = ((x_tr-mean)/std).astype("float32")[...,None]
x_te = ((x_te-mean)/std).astype("float32")[...,None]
y_tr_b = (y_tr!=1).astype("int8"); y_te_b = (y_te!=1).astype("int8")

normal_mask = (y_tr==1)
x_tr_norm = x_tr[normal_mask]

# ---------- 2. Build ResNet-1D AE ----------
from tensorflow.keras import layers, Model
def res_block(x, f):
    input_filters = x.shape[-1]
    y = layers.Conv1D(f,3,padding='same')(x); y = layers.BatchNormalization()(y); y = layers.ReLU()(y)
    y = layers.Conv1D(f,3,padding='same')(y); y = layers.BatchNormalization()(y)
    
    # If input and output channels differ, project the residual
    if input_filters != f:
        x = layers.Conv1D(f, 1, padding='same')(x)
    
    return layers.ReLU()(layers.Add()([x,y]))
def build_resnet1d():
    inp = layers.Input((140,1))
    x   = layers.Conv1D(32,7,padding='same')(inp)
    for f in [32,32,64,64,64,64]:         # 6 blocks ≈250 k params
        x = res_block(x,f)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(140)(x)
    out = layers.Reshape((140,1))(x)
    model = Model(inp,out,name="resnet1d_ae")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model
model = build_resnet1d()

# ---------- 3. MLflow run ----------
mlflow.set_experiment(EXPNAME)
with mlflow.start_run(run_name="fp32_train"):
    mlflow.log_params({"epochs":EPOCHS,"batch":BATCH,"blocks":6,"filters":"32-64"})
    mlflow.tensorflow.autolog(log_models=False)

    model.fit(x_tr_norm,x_tr_norm,epochs=EPOCHS,batch_size=BATCH,
              validation_data=(x_te,x_te),verbose=2)

    # ----- 4. Metrics -----
    m_tr = np.mean((model.predict(x_tr_norm)-x_tr_norm)**2,axis=(1,2))
    m_te = np.mean((model.predict(x_te)-x_te)**2,axis=(1,2))
    thr  = np.quantile(m_tr, THRESH_FRAC)
    y_pr = (m_te>thr).astype("int8")
    mets = {"threshold":float(thr),
            "AUROC":roc_auc_score(y_te_b,m_te),
            "AUPRC":average_precision_score(y_te_b,m_te),
            "F1":f1_score(y_te_b,y_pr),
            "precision":precision_score(y_te_b,y_pr),
            "recall":recall_score(y_te_b,y_pr)}
    mlflow.log_metrics(mets); print(json.dumps(mets,indent=2))

    # ----- 5. Export SavedModel -----
    save_dir = tempfile.mkdtemp(prefix="resnet1d_")
    model.export(save_dir)
    mlflow.log_artifacts(save_dir,"saved_model_fp32")

    # ----- 6. INT8 PTQ -----
    def rep_ds():                    # ≤400 normals
        for i in range(min(CAL_SAMPLES,len(x_tr_norm))):
            yield [x_tr_norm[i:i+1]]
    conv = tf.lite.TFLiteConverter.from_saved_model(save_dir)
    conv.optimizations=[tf.lite.Optimize.DEFAULT]
    conv.representative_dataset=rep_ds
    conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type=conv.inference_output_type=tf.int8
    tflite = conv.convert()
    tflite_path = pathlib.Path(save_dir)/"resnet1d_int8.tflite"
    tflite_path.write_bytes(tflite)
    mlflow.log_artifact(str(tflite_path),"tflite_int8")
    mlflow.log_metric("tflite_kB",tflite_path.stat().st_size/1024)

    # ----- 7. GPU latency quick-test -----
    with tf.device('/GPU:0'):
        dummy = tf.convert_to_tensor(x_tr_norm[:1024])
        model(dummy,training=False)   # warm-up
        t0=time.perf_counter(); model(dummy,training=False)
        mlflow.log_metric("gpu_ms_per_trace", (time.perf_counter()-t0)/1024*1e3)

    # ----- 8. Save JSON for clipboard -----
    (pathlib.Path(save_dir)/"metrics.json").write_text(json.dumps(mets,indent=2))
    mlflow.log_artifact(str(pathlib.Path(save_dir)/"metrics.json"))
    print("\n" + "="*50)
    print("FINAL METRICS:")
    print("="*50)
    print(json.dumps(mets, indent=2))
    print("="*50)