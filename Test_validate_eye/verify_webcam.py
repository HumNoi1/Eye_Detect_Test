# verify_webcam.py
import cv2, numpy as np, tensorflow as tf
from eye_crop import crop_eyes_both

THRESH = 0.92
PROFILE_PATHS = ["profile_owner.npy"]   # จะมีหลายคนก็เพิ่มไฟล์ได้

embedder = tf.keras.models.load_model("eye_embedder.h5", compile=False)
profiles = [(p.split("profile_")[1].split(".npy")[0], np.load(p)) for p in PROFILE_PATHS]

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break

    eyes = crop_eyes_both(frame, margin=0.25, target=112)

    info = "No eye"
    color = (0,0,255)

    if eyes:
        sims = []
        for e in eyes:
            x = e.astype("float32")/255.0
            x = np.expand_dims(x, (0,-1))
            emb = embedder.predict(x, verbose=0)[0]
            # เทียบกับทุกโปรไฟล์
            for name, prof in profiles:
                s = cosine_sim(emb, prof)
                sims.append((name, s))
        if sims:
            best_name, best_s = max(sims, key=lambda z:z[1])
            if best_s >= THRESH:
                info = f"✅ {best_name}  sim={best_s:.3f}"
                color = (0,255,0)
            else:
                info = f"❌ Unknown  best_sim={best_s:.3f}"
                color = (0,0,255)

    cv2.putText(frame, info, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.imshow("Eye-based Validation (DL)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()