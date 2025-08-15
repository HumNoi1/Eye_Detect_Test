import os
import argparse
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

# -------------------------------------------------------
# ตั้งค่า seed เพื่อให้ผลลัพธ์ reproducible มากขึ้น
# -------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -------------------------------------------------------
# ตรวจโครงสร้างข้อมูล + สร้าง tf.data.Dataset
# รองรับทั้ง directory structure และ CSV labels
# -------------------------------------------------------
def build_datasets(
    data_dir,
    img_size=(160, 160),
    batch_size=32,
    labels_csv=None,
    val_split=0.15,
    test_split=0.0,
    seed=42,
):
    AUTOTUNE = tf.data.AUTOTUNE
    data_dir = Path(data_dir)

    def _augment():
        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ],
            name="data_augmentation",
        )

    data_augmentation = _augment()

    # กรณีใช้ CSV: คาดว่าไฟล์มีคอลัมน์ filepath,label (label เป็น string ชื่อคลาส)
    if labels_csv:
        import pandas as pd
        df = pd.read_csv(data_dir / labels_csv)

        # map label -> index
        class_names = sorted(df["label"].unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        # split train/val/test
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test

        df_train = df.iloc[:n_train]
        df_val = df.iloc[n_train:n_train+n_val]
        df_test = df.iloc[n_train+n_val:] if n_test > 0 else None

        def path_to_img(path):
            return str(data_dir / path)

        def make_ds(df_sub, training=False):
            paths = df_sub["filepath"].map(path_to_img).tolist()
            labels = df_sub["label"].map(class_to_idx).astype(int).tolist()
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))

            def load_image(path, label):
                img = tf.io.read_file(path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, img_size)
                img = tf.cast(img, tf.float32) / 255.0
                return img, tf.cast(label, tf.int32)

            ds = ds.shuffle(len(df_sub), seed=seed) if training else ds
            ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
            if training:
                ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=AUTOTUNE)
            ds = ds.batch(batch_size).prefetch(AUTOTUNE)
            return ds

        train_ds = make_ds(df_train, training=True)
        val_ds = make_ds(df_val, training=False)
        test_ds = make_ds(df_test, training=False) if df_test is not None else None
        return train_ds, val_ds, test_ds, class_names

    # กรณีเป็นโฟลเดอร์: รองรับได้ทั้ง
    # - data_dir/train, data_dir/val, data_dir/test
    # - หรือ data_dir/* เป็นโฟลเดอร์คลาสทั้งหมด (จะ split ให้)
    has_split_folders = (data_dir / "train").exists() and (data_dir / "val").exists()
    if has_split_folders:
        train_dir = str(data_dir / "train")
        val_dir = str(data_dir / "val")
        test_dir = str(data_dir / "test") if (data_dir / "test").exists() else None

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir, image_size=img_size, batch_size=batch_size, shuffle=True
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir, image_size=img_size, batch_size=batch_size, shuffle=False
        )
        test_ds = (
            tf.keras.preprocessing.image_dataset_from_directory(
                test_dir, image_size=img_size, batch_size=batch_size, shuffle=False
            )
            if test_dir
            else None
        )
        class_names = train_ds.class_names

    else:
        # ไม่มีโฟลเดอร์ย่อย train/val -> สร้าง split จากโฟลเดอร์คลาสใน root
        full_ds = tf.keras.preprocessing.image_dataset_from_directory(
            str(data_dir),
            image_size=img_size,
            batch_size=batch_size,
            validation_split=val_split + test_split if (val_split + test_split) > 0 else None,
            subset="training" if (val_split + test_split) > 0 else None,
            seed=seed,
            shuffle=True,
        )
        class_names = full_ds.class_names

        val_test_ds = (
            tf.keras.preprocessing.image_dataset_from_directory(
                str(data_dir),
                image_size=img_size,
                batch_size=batch_size,
                validation_split=val_split + test_split,
                subset="validation",
                seed=seed,
                shuffle=True,
            )
            if (val_split + test_split) > 0
            else None
        )

        # แยก val กับ test ออกจากกัน ถ้ามี test_split > 0
        if val_test_ds and test_split > 0:
            # นับจำนวน batch แล้วแยกคร่าว ๆ ตามสัดส่วน
            val_batches = int(len(val_test_ds) * (val_split / (val_split + test_split)))
            val_ds = val_test_ds.take(val_batches)
            test_ds = val_test_ds.skip(val_batches)
            train_ds = full_ds
        else:
            train_ds = full_ds
            val_ds = val_test_ds
            test_ds = None

    # เพิ่ม normalize + augmentation ลงใน pipeline
    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)
    AUTOTUNE = tf.data.AUTOTUNE

    def add_map(ds, training=False):
        def norm_map(x, y):
            return normalization(x), y

        ds = ds.map(norm_map, num_parallel_calls=AUTOTUNE)
        if training:
            aug = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.05),
                    tf.keras.layers.RandomZoom(0.1),
                    tf.keras.layers.RandomContrast(0.1),
                ]
            )
            ds = ds.map(lambda x, y: (aug(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
        return ds.cache().prefetch(AUTOTUNE)

    train_ds = add_map(train_ds, training=True)
    val_ds = add_map(val_ds, training=False) if val_ds is not None else None
    test_ds = add_map(test_ds, training=False) if test_ds is not None else None

    return train_ds, val_ds, test_ds, class_names

# -------------------------------------------------------
# สร้างโมเดล: MobileNetV2 (Transfer Learning) หรือ CNN ง่าย ๆ
# -------------------------------------------------------
def build_model(img_size, num_classes, base="mobilenetv2", dropout=0.2):
    inputs = tf.keras.Input(shape=(*img_size, 3))
    if base.lower() == "mobilenetv2":
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(*img_size, 3), include_top=False, weights="imagenet"
        )
        backbone.trainable = False  # เริ่มจาก freeze ก่อน
        x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        x = backbone(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(
            num_classes, activation="sigmoid" if num_classes == 1 else "softmax"
        )(x)
        model = tf.keras.Model(inputs, outputs)
    else:
        # โมเดลง่าย ๆ
        x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(
            num_classes, activation="sigmoid" if num_classes == 1 else "softmax"
        )(x)
        model = tf.keras.Model(inputs, outputs)

    return model

# -------------------------------------------------------
# คำนวณ class weights (กัน bias ถ้า dataset ไม่สมดุล)
# -------------------------------------------------------
def compute_class_weights(train_ds, num_classes):
    import math
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in train_ds.unbatch():
        if num_classes == 1:
            # binary: y is scalar (0/1)
            counts[int(y.numpy())] += 1
        else:
            counts[int(y.numpy())] += 1
    total = counts.sum()
    weights = {i: total / (num_classes * count) for i, count in enumerate(counts) if count > 0}
    return weights

# -------------------------------------------------------
# เทรน + ประเมินผล + บันทึกโมเดล
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Eye detection training script (Keras)")
    parser.add_argument("--data_dir", type=str, required=True, help="โฟลเดอร์ข้อมูลที่ unzip แล้ว")
    parser.add_argument("--labels_csv", type=str, default=None, help="ไฟล์ CSV (filepath,label)")
    parser.add_argument("--img_size", type=int, default=160, help="ขนาดภาพสี่เหลี่ยม (เช่น 160)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["mobilenetv2", "cnn"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine_tune", action="store_true", help="เปิด fine-tune backbone หลังจาก warmup")
    parser.add_argument("--fine_tune_at", type=int, default=100, help="unfreeze ตั้งแต่ชั้นที่เท่านี้ (เฉพาะ MobileNetV2)")
    parser.add_argument("--use_class_weights", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    img_size = (args.img_size, args.img_size)
    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        labels_csv=args.labels_csv,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    num_classes = 1 if len(class_names) == 2 else len(class_names)
    model = build_model(img_size, num_classes, base=args.model)

    loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")] if num_classes == 1 else ["accuracy"]
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=loss, metrics=metrics)

    # callbacks
    ckpt_path = os.path.join(args.output_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(args.output_dir, "training_log.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.output_dir, "tb_logs")),
    ]

    class_weight = None
    if args.use_class_weights:
        class_weight = compute_class_weights(train_ds, num_classes)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # บันทึก final model (นอกจาก best checkpoint)
    final_path = os.path.join(args.output_dir, "final_model.keras")
    model.save(final_path)

    # บันทึก class_names
    with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # ประเมินผล
    if val_ds is not None:
        val_metrics = model.evaluate(val_ds, verbose=0)
        print("[VAL] metrics:", dict(zip(model.metrics_names, val_metrics)))

    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, verbose=0)
        print("[TEST] metrics:", dict(zip(model.metrics_names, test_metrics)))

    # Confusion Matrix + รายงาน
    try:
        import sklearn.metrics as skm
        y_true, y_pred = [], []
        eval_ds = test_ds if test_ds is not None else val_ds
        for x, y in eval_ds:
            preds = model.predict(x, verbose=0)
            if num_classes == 1:
                preds = (preds.ravel() > 0.5).astype(int)
            else:
                preds = preds.argmax(axis=1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.tolist())

        cm = skm.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(skm.classification_report(y_true, y_pred, target_names=class_names))
    except Exception as e:
        print("Skipping confusion matrix/report:", e)

    print("\nTraining done.")
    print(f"- Best model: {ckpt_path}")
    print(f"- Final model: {final_path}")
    print(f"- Class names: {class_names}")

    # ตัวอย่างการใช้งานทำนายรูปเดียว
    # python train_eye_detection.py ... แล้วเรียกฟังก์ชัน predict_single ในโค้ดอื่น ๆ ก็ได้
    # หรือ copy ส่วนด้านล่างไปใช้ต่อ
    # ---------------------------------------------------
    # from tensorflow.keras.utils import load_img, img_to_array
    # img = load_img('path/to/image.jpg', target_size=img_size)
    # arr = img_to_array(img) / 255.0
    # arr = np.expand_dims(arr, 0)
    # pred = model.predict(arr)
    # if num_classes == 1:
    #     print("Eye Open" if pred[0][0] > 0.5 else "Eye Closed")
    # else:
    #     print(class_names[int(np.argmax(pred[0]))])

if __name__ == "__main__":
    main()
