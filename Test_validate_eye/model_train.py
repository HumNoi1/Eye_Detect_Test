# model_train.py
import os, glob, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from eye_crop import crop_eyes_both

IMG_SIZE = 112
EMB_DIM  = 128
BATCH    = 32
EPOCHS   = 10

def build_embedding_model():
    # ใส่ grayscale เป็น 3 แชนแนลเพื่อใช้ backbone pretrained ได้
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x3  = layers.Concatenate()([inp, inp, inp])  # (H,W,3)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None  # ตั้ง weights=None ถ้าข้อมูลเฉพาะทาง
    )
    x = base(x3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    emb = layers.Dense(EMB_DIM, use_bias=False)(x)         # 128-D
    emb = tf.math.l2_normalize(emb, axis=-1, name="l2norm") # L2 normalize
    return models.Model(inp, emb, name="eye_embedder")

def build_classifier(num_classes):
    backbone = build_embedding_model()
    out = layers.Dense(num_classes, activation='softmax', name='softmax')(backbone.output)
    model = models.Model(backbone.input, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_dataset(root):
    """
    โครงสร้างโฟลเดอร์:
      dataset/
        personA/ *.jpg, *.png ...
        personB/ ...
    จะใช้ "ตามคน" เป็นคลาส
    """
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
    class2id = {c:i for i,c in enumerate(classes)}
    X, y = [], []
    for c in classes:
        for p in glob.glob(os.path.join(root, c, "*")):
            img = cv2.imread(p)
            if img is None: 
                continue
            eyes = crop_eyes_both(img)   # ได้ [L, R]
            for e in eyes:
                X.append(e)
                y.append(class2id[c])
    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)
    return X, y, class2id

# --- main train ---
if __name__ == "__main__":
    import cv2
    DATA_DIR = "dataset"         # ชี้ไปยังชุดข้อมูลของคุณ
    X, y, class2id = load_dataset(DATA_DIR)

    # แบ่งเทรน/วาลิเดต
    idx = np.arange(len(X)); np.random.shuffle(idx)
    X = X[idx]; y = y[idx]
    n = int(len(X)*0.85)
    Xtr, Xval = X[:n], X[n:]
    ytr, yval = y[:n], y[n:]

    # เพิ่ม normalization และขยายมิติ
    def prep(a): 
        a = a.astype("float32")/255.0
        return np.expand_dims(a, -1)

    Xtr = prep(Xtr); Xval = prep(Xval)

    model = build_classifier(num_classes=len(class2id))
    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=BATCH)

    # เซฟคลาสสิฟายเออร์ และโมเดลฝั่ง embedding (backbone)
    model.save("eye_classifier.h5")
    embedder = models.Model(model.input, model.get_layer("l2norm").output)
    embedder.save("eye_embedder.h5")

    # เผื่อใช้ภายหลัง
    import json
    with open("class_map.json","w",encoding="utf-8") as f:
        json.dump(class2id, f, ensure_ascii=False, indent=2)
