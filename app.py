from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFile
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename
import numpy as np
import os, logging
import uuid
import json
import requests
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 保存ディレクトリ
REGISTER_FOLDER = "registered_images"
os.makedirs(REGISTER_FOLDER, exist_ok=True)

# 商品名マッピングファイル
MAPPING_FILE = "name_mapping.json"
try:
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        name_mapping = json.load(f)
except:
    name_mapping = {}

# Flask アプリ設定
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# 前処理（グレースケール＋リサイズ）
def preprocess_pil(img: Image.Image, size=200) -> Image.Image:
    img = img.convert("L")                      # グレースケール化
    img = ImageOps.exif_transpose(img)         # 回転を正しく
    img = ImageOps.fit(img, (size, size))      # サイズ統一 & クロップ
    img = ImageOps.autocontrast(img)           # 明るさ補正
    return img


@app.route("/ping")
def ping():
    return "ok", 200


@app.route("/register_image", methods=["POST"])
def register_image():
    name = request.form.get("name")
    file = request.files.get("image")
    app.logger.info(f"📌 received name: {name}, image.filename: {file.filename if file else 'None'}")

    if not name or not file:
        return "invalid request", 400

    try:
        # 画像変換＆リサイズ
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((640, 640), Image.Resampling.LANCZOS)

        # UUIDファイル名で保存
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(REGISTER_FOLDER, filename)
        img.save(save_path, format="JPEG", quality=80, optimize=True)

        # 🔽 商品名マッピングを保存＋ファイル書き込み
        name_mapping[filename] = name
        with open(MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(name_mapping, f, ensure_ascii=False, indent=2)
        app.logger.info(f"✅ name_mapping 登録: {filename} → {name}")
        app.logger.info(f"✅ saved to: {save_path} (商品名: {name})")

        return "OK", 200

    except Exception as e:
        app.logger.exception(e)
        return "error", 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        app.logger.info("📥 /predict にリクエスト受信")
        # ✅ 1. 本番環境（S3のURLが送られてくる）
        if "image_url" in request.form:
            image_url = request.form["image_url"]
            response = requests.get(image_url)
            response.raise_for_status()
            raw = Image.open(BytesIO(response.content))

        # ✅ 2. 開発環境（ローカル画像が multipart で送られてくる）
        elif "image" in request.files:
            file = request.files["image"]
            raw = Image.open(file.stream)

        else:
            return jsonify(error="画像がありません"), 400

        # 画像の前処理
        query = preprocess_pil(raw, size=100)
        q_arr = np.asarray(query)

        if not os.listdir(REGISTER_FOLDER):
            return jsonify(error="登録済み画像なし"), 500

        # 一番近い画像を探す
        best, best_score = None, -1
        for fn in os.listdir(REGISTER_FOLDER):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # 登録画像の読み込み＋前処理
            ref = Image.open(os.path.join(REGISTER_FOLDER, fn)).convert("L").resize((100, 100))
            r_arr = np.asarray(ref)

            # 類似度計算
            score, _ = ssim(q_arr, r_arr, full=True)

            # ←ここで全件ログ出力する
            app.logger.info(f"比較: {fn} - 類似度スコア: {score:.4f}")

            # 最良スコアの更新
            if score > best_score:
                best_score = score
                best = fn
     

        if best and best_score >= 0.3:
            filename_with_ext = best if best.endswith(".jpg") else best + ".jpg"
            predicted_name = name_mapping.get(filename_with_ext, os.path.splitext(best)[0])
            app.logger.info(f"🎯 matched: {filename_with_ext} → {predicted_name}")
            return jsonify(name=predicted_name, score=round(best_score, 4))

        return jsonify(error="一致なし", score=round(best_score, 4)), 404

    except Exception as e:
        app.logger.exception(e)
        return jsonify(error="処理エラー"), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run("0.0.0.0", port, debug=False)
