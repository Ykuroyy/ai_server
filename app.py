from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFile
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename
import numpy as np
import os, logging

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 保存ディレクトリ
REGISTER_FOLDER = "registered_images"
os.makedirs(REGISTER_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)


def preprocess_pil(img: Image.Image, size=100) -> Image.Image:
    # SSIM 用にグレースケール＆リサイズ
    return ImageOps.exif_transpose(img).convert("L").resize((size, size))


@app.route("/ping")
def ping():
    return "ok", 200


@app.route("/register_image", methods=["POST"])
def register_image():
    name = request.form.get("name")
    file = request.files.get("image")
    if not name or not file:
        return "invalid request", 400

    try:
        # → JPEG に変換＆大きすぎる場合は最大 640px に縮小
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
       
        # Pillow 10.x 以降では ANTIALIAS は Resampling.LANCZOS に置き換わりました
      
        img.thumbnail((640, 640), Image.Resampling.LANCZOS)



        # フルサイズを保存（比較用登録）
        save_path = os.path.join(REGISTER_FOLDER, secure_filename(f"{name}.jpg"))
        img.save(save_path, format="JPEG", quality=80, optimize=True)

        return "OK", 200
    except Exception as e:
        app.logger.exception(e)
        return "error", 500


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify(error="画像がありません"), 400

    try:
        # → まずはメモリ上で JPEG として開き、リサイズ＆グレースケール
        raw = Image.open(file.stream)
        query = preprocess_pil(raw, size=100)
        q_arr = np.asarray(query)

        if not os.listdir(REGISTER_FOLDER):
            return jsonify(error="登録済み画像なし"), 500

        best, best_score = None, -1
        for fn in os.listdir(REGISTER_FOLDER):
            if not fn.lower().endswith((".jpg", ".jpeg")):
                continue
            ref = Image.open(os.path.join(REGISTER_FOLDER, fn)).convert("L").resize((100,100))
            r_arr = np.asarray(ref)
            score, _ = ssim(q_arr, r_arr, full=True)
            if score > best_score:
                best_score, best = score, os.path.splitext(fn)[0]

        if best and best_score >= 0.22:
            return jsonify(name=best, score=round(best_score,4))
        return jsonify(error="一致なし", score=round(best_score,4)), 404

    except Exception as e:
        app.logger.exception(e)
        return jsonify(error="処理エラー"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run("0.0.0.0", port, debug=False)
