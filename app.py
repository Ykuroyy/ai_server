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

def build_cache(cache_dir=CACHE_DIR, index_path=INDEX_PATH, dim=256):
    os.makedirs(cache_dir, exist_ok=True)

    # 1) DB に登録されている s3_key のみ取得
    session = Session()
    keys = [pm.s3_key for pm in session.query(ProductMapping).all()]
    session.close()

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


def calculate_orb_matches(img1: Image.Image, img2: Image.Image) -> int:
    orb = cv2.ORB_create()
    gray1 = cv2.cvtColor(np.array(img1.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2.convert("RGB")), cv2.COLOR_RGB2GRAY)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)



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
        # 1) 画像取得
        if "image" in request.files:
            raw = Image.open(request.files["image"].stream)
        elif "image_url" in request.form:
            import requests
            r = requests.get(request.form["image_url"]); r.raise_for_status()
            raw = Image.open(BytesIO(r.content))
        else:
            return jsonify(error:="画像がありません"), 400
       
        gray = cv2.cvtColor(np.array(raw.convert("RGB")), cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        _, des = sift.detectAndCompute(gray, None)
        q_arr = np.zeros(256, dtype="float32")  # SIFT の場合は特徴長を合わせる
        if des is not None:
            flat = des.flatten()
            q_arr[: min(256, flat.shape[0])] = flat[:256]



        # 3) インデックス読み込み
        index = faiss.read_index(INDEX_PATH)

        # 4) keys.json を読み込み
        with open(KEYS_PATH, "r", encoding="utf-8") as f:
            keys = json.load(f)

        # 5) 検索：k はキー数に合わせる
        k = len(keys)
        D, I = index.search(np.expand_dims(q_arr, 0), k=k)

        # 6) 結果整形（重複をスキップ）
        session    = Session()
        seen_names = set()
        all_scores = []
        for dist, idx in zip(D[0], I[0]):
            key  = keys[idx]
            # DB から商品名を取ってくる
            prod = session.query(ProductMapping).filter_by(s3_key=key).first()
            name = prod.name if prod else key.rsplit(".",1)[0]
            if name in seen_names:
                continue
            seen_names.add(name)
            # ここを指数関数に置き換える
            
            # ORBマッチ数による補正を入れる
            match_count = 0
            try:
                # S3 から対象画像を一時取得
                obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                db_img = Image.open(BytesIO(obj["Body"].read()))
                match_count = calculate_orb_matches(raw, db_img)
            except Exception as e:
                app.logger.warning(f"⚠️ ORBマッチ失敗: {key} - {e}")

            # スコアの補正：Faissスコア + ORB特徴マッチ
            sigma = 1000.0
            score_faiss = float(np.exp(-dist / sigma))
            score_total = round(score_faiss + match_count * 0.01, 4)
    
            all_scores.append({
                "name": name,
                "score": score_total
            })
                    
        session.close()

        return jsonify(all_similarity_scores=all_scores), 200

    except Exception as e:
        app.logger.exception(e)
        return jsonify(error="処理エラー"), 500




# ── モジュール読み込み時にキャッシュチェック ───────────────────────
if not Path(INDEX_PATH).exists() or not Path(KEYS_PATH).exists():
    app.logger.info("キャッシュ／インデックスが見つからないので自動生成します (モジュール読み込み時)")
    build_cache(dim=256)

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