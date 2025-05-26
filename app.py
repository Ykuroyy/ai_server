import os
import logging
import json
import uuid
import requests
import boto3
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFile
from skimage.metrics import structural_similarity as ssim
import numpy as np

# トランケートされた画像も読み込めるように
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 環境変数から S3 バケット名を取得
S3_BUCKET = os.environ["S3_BUCKET"]

# ローカル登録用ディレクトリ＆マッピングファイル
REGISTER_FOLDER = "registered_images"
os.makedirs(REGISTER_FOLDER, exist_ok=True)
MAPPING_FILE = "name_mapping.json"

# 商品名マッピングの読み込み
try:
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        name_mapping = json.load(f)
except FileNotFoundError:
    name_mapping = {}

# Flask アプリ設定
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# 画像の前処理：グレースケール＋リサイズ＋コントラスト調整
def preprocess_pil(img: Image.Image, size=200) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.exif_transpose(img)
    img = ImageOps.fit(img, (size, size))
    img = ImageOps.autocontrast(img)
    return img

# S3 クライアント（環境変数の認証情報を利用）
s3 = boto3.client("s3")

@app.route("/ping")
def ping():
    return "ok", 200

@app.route("/register_image", methods=["POST"])
def register_image():
    """
    ローカルディレクトリに画像保存し、name_mapping.json に商品名を記録
    """
    name = request.form.get("name")
    file = request.files.get("image")
    app.logger.info(f"📌 received name: {name}, image.filename: {file.filename if file else 'None'}")

    if not name or not file:
        return "invalid request", 400

    try:
        # リサイズ＆保存
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((640, 640), Image.Resampling.LANCZOS)
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(REGISTER_FOLDER, filename)
        img.save(save_path, format="JPEG", quality=80, optimize=True)

        # マッピング更新
        name_mapping[filename] = name
        with open(MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(name_mapping, f, ensure_ascii=False, indent=2)

        app.logger.info(f"✅ saved to: {save_path} (商品名: {name})")
        return "OK", 200

    except Exception as e:
        app.logger.exception(e)
        return "error", 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    S3 バケット上の全画像と比較して最も類似度の高い商品を返却
    """
    try:
        app.logger.info("📥 /predict リクエスト受信")

        # 本番：S3 上の URL が送られてくる場合
        if "image_url" in request.form:
            image_url = request.form["image_url"]
            resp = requests.get(image_url)
            resp.raise_for_status()
            raw = Image.open(BytesIO(resp.content))

        # 開発：multipart で送られてきた画像
        elif "image" in request.files:
            raw = Image.open(request.files["image"].stream)

        else:
            return jsonify(error="画像がありません"), 400

        # クエリ画像の前処理
        query = preprocess_pil(raw, size=100)
        q_arr = np.asarray(query)

        # S3 バケット内のオブジェクト一覧をページングで取得
        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=S3_BUCKET)

        best_score = -1.0
        best_key   = None

        for page in page_iterator:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # 対象ファイル拡張子のみ
                if not key.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                # S3 から画像をダウンロード
                resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                body = resp["Body"].read()
                img = Image.open(BytesIO(body))

                # 前処理＋配列化
                ref = preprocess_pil(img, size=100)
                r_arr = np.asarray(ref)

                # SSIM 類似度計算＆ログ出力
                score, _ = ssim(q_arr, r_arr, full=True)
                app.logger.info(f"比較: {key} – 類似度スコア: {score:.4f}")

                # ベスト更新
                if score > best_score:
                    best_score = score
                    best_key   = key

        # マッチなし
        if best_key is None:
            return jsonify(error="一致なし", score=0), 404

        # 商品名マッピング or ファイル名ベース
        predicted = name_mapping.get(best_key, os.path.splitext(best_key)[0])
        app.logger.info(f"🎯 matched: {best_key} → {predicted} (score={best_score:.4f})")

        return jsonify(name=predicted, score=round(best_score, 4)), 200

    except Exception as e:
        app.logger.exception(e)
        return jsonify(error="処理エラー"), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run("0.0.0.0", port, debug=False)
