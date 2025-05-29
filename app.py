# app.py

import os
import json
import uuid
import argparse
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import faiss    # pip install faiss-cpu
import cv2
from PIL import Image, ImageOps, ImageFile, ImageFilter

from flask import Flask, request, jsonify
from flask_cors import CORS

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# ── 共通設定 ─────────────────────────────────────────

# DB
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./local_dev.db"
)
engine  = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base    = declarative_base()

class ProductMapping(Base):
    __tablename__ = "products"
    id     = Column(Integer, primary_key=True)
    name   = Column(String)
    s3_key = Column(String)

# S3 クライアント
ImageFile.LOAD_TRUNCATED_IMAGES = True
S3_BUCKET = os.environ.get("S3_BUCKET", "registered_images")
s3        = boto3.client("s3")

# Flask
app = Flask(__name__)
CORS(app)
app.logger.setLevel("INFO")

# キャッシュ置き場
CACHE_DIR  = "cache"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
KEYS_PATH  = os.path.join(CACHE_DIR, "keys.json")

# ── 前処理ヘルパー ────────────────────────────────────

def crop_to_object(pil_img, thresh=200):
    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binimg = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    cnts, _  = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pil_img
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return pil_img.crop((x, y, x+w, y+h))

def preprocess_pil(img, size=100):
    img = img.convert("L")
    img = ImageOps.exif_transpose(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img = ImageOps.fit(img, (size, size))
    return ImageOps.autocontrast(img, cutoff=1)

# ── キャッシュ構築機能 ─────────────────────────────────

def build_cache(cache_dir=CACHE_DIR, index_path=INDEX_PATH, dim=128):
    os.makedirs(cache_dir, exist_ok=True)

    # 1) S3 からキー一覧
    keys = [
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET)
        for obj in page.get("Contents", [])
        if obj["Key"].lower().endswith((".jpg","jpeg","png"))
    ]
    descriptors = []

    # 2) 各画像をダウンロード → ORB → 固定長ベクトル
    orb = cv2.ORB_create()
    for key in keys:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        img  = Image.open(BytesIO(resp["Body"].read()))
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, des = orb.detectAndCompute(gray, None)
        vec = np.zeros(dim, dtype="float32")
        if des is not None:
            flat = des.flatten()
            vec[: min(dim, flat.shape[0])] = flat[:dim]
        descriptors.append(vec)
        np.save(os.path.join(cache_dir, f"{key}.npy"), vec)

    # 3) keys.json を保存
    with open(KEYS_PATH, "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False, indent=2)

    # 4) Faiss インデックス構築＋保存
    xb    = np.stack(descriptors)
    index = faiss.IndexFlatL2(dim)
    index.add(xb)
    faiss.write_index(index, index_path)

    app.logger.info(f"✅ キャッシュ({len(keys)}件) & インデックスを生成しました → {cache_dir}/ , {index_path}")

# ── 画像登録エンドポイント ───────────────────────────────

@app.route("/register_image", methods=["POST"])
def register_image():
    name = request.form.get("name")
    if not name:
        return "invalid request (no name)", 400

    if "image" in request.files:
        stream = request.files["image"].stream
    elif "image_url" in request.form:
        import requests
        try:
            r = requests.get(request.form["image_url"])
            r.raise_for_status()
            stream = BytesIO(r.content)
        except Exception as e:
            app.logger.error(f"Failed download image_url: {e}")
            return "invalid image_url", 400
    else:
        return "invalid request (no image or image_url)", 400

    try:
        img = Image.open(stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((640, 640), Image.Resampling.LANCZOS)
        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join("registered_images", filename)
        os.makedirs("registered_images", exist_ok=True)
        img.save(path, format="JPEG", quality=80, optimize=True)

        s3.upload_file(path, S3_BUCKET, filename, ExtraArgs={"ContentType":"image/jpeg"})
        app.logger.info(f"☁️ uploaded to S3://{S3_BUCKET}/{filename}")
        return "OK", 200
    except Exception as e:
        app.logger.exception(e)
        return "error", 500

# ── 画像認識エンドポイント ─────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" in request.files:
            raw = Image.open(request.files["image"].stream)
        elif "image_url" in request.form:
            import requests
            r = requests.get(request.form["image_url"]); r.raise_for_status()
            raw = Image.open(BytesIO(r.content))
        else:
            return jsonify(error="画像がありません"), 400


        # 2) ORB で特徴量を取り出して 128 次元ベクトルに
        raw   = crop_to_object(raw)
        gray  = cv2.cvtColor(np.array(raw.convert("RGB")), cv2.COLOR_RGB2GRAY)
        orb   = cv2.ORB_create()
        _, des = orb.detectAndCompute(gray, None)
        q_arr = np.zeros(128, dtype="float32")
        if des is not None:
            flat = des.flatten()
            q_arr[: min(128, flat.shape[0])] = flat[:128]
    

        index = faiss.read_index(INDEX_PATH)
        D, I  = index.search(np.expand_dims(q_arr, 0), k=10)

        with open(KEYS_PATH, "r", encoding="utf-8") as f:
            keys = json.load(f)

        session    = Session()
        all_scores = []
        for dist, idx in zip(D[0], I[0]):
            key   = keys[idx]
            score = float(1.0 / (1 + dist))
            prod  = session.query(ProductMapping).filter_by(s3_key=key).first()
            name  = prod.name if prod else key.rsplit(".",1)[0]
            all_scores.append({"name": name, "score": round(score,4)})
        session.close()

        return jsonify(all_similarity_scores=all_scores), 200

    except Exception as e:
        app.logger.exception(e)
        return jsonify(error="処理エラー"), 500

# ── モジュール読み込み時にキャッシュチェック ───────────────────────
if not Path(INDEX_PATH).exists() or not Path(KEYS_PATH).exists():
    # ローカル起動でも、Gunicorn 起動でもここが実行される
    app.logger.info("キャッシュ／インデックスが見つからないので自動生成します (モジュール読み込み時)")
    build_cache()
# ────────────────────────────────────────────────────────────────

# ── エントリポイント ─────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-cache", action="store_true",
        help="S3 から特徴量キャッシュ＆Faissインデックスを作成"
    )
    args = parser.parse_args()

    if args.build_cache:
        build_cache()
    else:
        port = int(os.environ.get("PORT", 10000))
        app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()