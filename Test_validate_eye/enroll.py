# enroll.py
import os, glob, numpy as np, cv2, tensorflow as tf
from eye_crop import crop_eyes_both

IMG_SIZE = 112
embedder = tf.keras.models.load_model("eye_embedder.h5", compile=False)

def img_to_emb(img):
    eyes = crop_eyes_both(img)
    embs = []
    for e in eyes:
        x = e.astype("float32")/255.0
        x = np.expand_dims(x, (0,-1))   # (1,112,112,1)
        emb = embedder.predict(x, verbose=0)[0]
        embs.append(emb)
    return embs

def enroll_from_folder(person_name, folder="enroll_samples"):
    all_embs = []
    for p in glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")):
        img = cv2.imread(p)
        if img is None: continue
        embs = img_to_emb(img)
        all_embs.extend(embs)
    if not all_embs:
        raise RuntimeError("ไม่พบ embedding จากภาพที่ให้มา")
    profile = np.mean(np.stack(all_embs, axis=0), axis=0)   # เฉลี่ย
    profile = profile / (np.linalg.norm(profile) + 1e-9)    # L2 normalize อีกรอบ
    np.save(f"profile_{person_name}.npy", profile)
    print(f"Saved profile_{person_name}.npy, {len(all_embs)} embeddings")

if __name__ == "__main__":
    # สร้างโฟลเดอร์ enroll_samples แล้วใส่รูปเจ้าของไว้ 10–20 ภาพ (หลายสภาพแสง/มุม)
    enroll_from_folder(person_name="owner", folder="enroll_samples")
