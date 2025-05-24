# app.py  ★不要部分を削ってシンプルに
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFile
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename

import numpy as np
import os, logging

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ディレクトリ
REGISTER_FOLDER = "registered_images"
os.makedirs(REGISTER_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# ---------- 共通前処理 ----------
def preprocess_pil(img: Image.Image, size=100) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("L").resize((size, size))
    return img


# ---------- ヘルスチェック ----------
@app.route("/ping")
def ping():
    return "ok", 200


# ---------- 画像登録 ----------
@app.route("/register_image", methods=["POST"])
def register_image():
    name  = request.form.get("name")
    file  = request.files.get("image")
    if not name or not file:
        return "invalid request", 400

    try:
        img = preprocess_pil(Image.open(file.stream), size=100)
        path = os.path.join(REGISTER_FOLDER, secure_filename(f"{name}.jpg"))
        img.save(path, format="JPEG", quality=85)
        return "OK", 200

    except Exception as e:
        app.logger.exception(e)
        return "error", 500


# ---------- 画像予測 ----------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify(error="画像がありません"), 400

    query = preprocess_pil(Image.open(file.stream), size=100)
    q_arr = np.asarray(query)

    if not os.listdir(REGISTER_FOLDER):
        return jsonify(error="登録済み画像なし"), 500

    best, best_score = None, -1
    for fname in os.listdir(REGISTER_FOLDER):
        if not fname.lower().endswith((".jpg", ".jpeg")):
            continue
        r_arr = np.asarray(Image.open(os.path.join(REGISTER_FOLDER, fname)))
        score, _ = ssim(q_arr, r_arr, full=True)
        if score > best_score:
            best_score, best = score, os.path.splitext(fname)[0]

    if best and best_score >= 0.22:
        return jsonify(name=best, score=round(best_score, 4))
    return jsonify(error="一致なし", score=round(best_score, 4)), 404


# ---------- 登録済み一覧 ----------
@app.route("/list_registered")
def list_registered():
    return jsonify(files=os.listdir(REGISTER_FOLDER))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run("0.0.0.0", port, debug=False)
