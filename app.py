# app.py
import os, json, uuid, argparse
from io import BytesIO
from pathlib import Path
import numpy as np
import cv2
import faiss
import boto3

from PIL import Image, ImageOps, ImageFile
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import DateTime


# --- 共通設定 ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./local_dev.db")
S3_BUCKET    = os.environ.get("S3_BUCKET", "registered_images")

engine  = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base    = declarative_base()
s3      = boto3.client("s3")

CACHE_DIR  = "cache"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
KEYS_PATH  = os.path.join(CACHE_DIR, "keys.json")

class ProductMapping(Base):
    __tablename__ = "products"
    id     = Column(Integer, primary_key=True)
    name   = Column(String)
    s3_key = Column(String)
    created_at = Column(DateTime)  # ← 追加
    updated_at = Column(DateTime)  # ← 追加

# --- Flask 初期化 ---
app = Flask(__name__)
CORS(app)
Base.metadata.create_all(bind=engine)

# --- 画像登録エンドポイント ---
@app.route("/register_image", methods=["POST"])
def register_image():
    name = request.form.get("name")
    if not name:
        return "no name", 400

    if "image" not in request.files:
        return "no image", 400

    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
        img = ImageOps.exif_transpose(img)
        img.thumbnail((640, 640))
        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join("registered_images", filename)
        os.makedirs("registered_images", exist_ok=True)
        img.save(path, format="JPEG", quality=80)

        s3.upload_file(path, S3_BUCKET, filename, ExtraArgs={"ContentType": "image/jpeg"})
        session = Session()
        new_product = ProductMapping(
            name=name,
            s3_key=filename,
            created_at=datetime.utcnow(),  # ← 追加
            updated_at=datetime.utcnow()   # ← 追加
        )
        session.add(new_product)
        session.commit()
        session.close()
   
        return "OK", 200
    except Exception as e:
        app.logger.exception("登録失敗")
        return str(e), 500

# --- キャッシュ構築エンドポイント ---
@app.route("/build_cache", methods=["POST"])
def trigger_build_cache():
    try:
        build_cache(dim=256)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        app.logger.exception("キャッシュ作成失敗")
        return jsonify({"error": str(e)}), 500

# --- キャッシュ作成処理 ---
def build_cache(cache_dir=CACHE_DIR, index_path=INDEX_PATH, dim=256):
    os.makedirs(cache_dir, exist_ok=True)
    session = Session()
    keys = [p.s3_key for p in session.query(ProductMapping).all()]
    session.close()

    descriptors = []
    for key in keys:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        img  = Image.open(BytesIO(resp["Body"].read()))
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        _, des = sift.detectAndCompute(gray, None)

        vec = np.zeros(dim, dtype="float32")
        if des is not None:
            flat = des.flatten()
            vec[:min(dim, len(flat))] = flat[:dim]
            descriptors.append(vec)
        else:
            app.logger.warning(f"❌ 特徴量抽出失敗: {key}")

    if not descriptors:
        raise RuntimeError("🚫 特徴量ゼロ件。登録画像を確認してください")

    xb = np.stack(descriptors)
    faiss.write_index(faiss.IndexFlatL2(dim).add(xb), index_path)
    with open(KEYS_PATH, "w", encoding="utf-8") as f:
        json.dump(keys, f)

    app.logger.info(f"✅ キャッシュ生成: {len(keys)}件")

# --- 画像認識エンドポイント ---
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "画像がありません"}), 400

    raw = Image.open(request.files["image"].stream).convert("RGB")
    gray = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(gray, None)

    if des is None:
        return jsonify({"error": "画像の特徴量が抽出できません"}), 400

    vec = des.flatten()[:256]
    if np.linalg.norm(vec) != 0:
        vec = vec / np.linalg.norm(vec)

    q_arr = np.zeros(256, dtype="float32")
    q_arr[:len(vec)] = vec

    index = faiss.read_index(INDEX_PATH)
    with open(KEYS_PATH, encoding="utf-8") as f:
        keys = json.load(f)

    k = len(keys)
    D, I = index.search(np.expand_dims(q_arr, 0), k=k)

    session = Session()
    results = []
    seen = set()
    for dist, idx in zip(D[0], I[0]):
        key = keys[idx]
        prod = session.query(ProductMapping).filter_by(s3_key=key).first()
        name = prod.name if prod else key.rsplit(".", 1)[0]
        if name in seen:
            continue
        seen.add(name)
        score = max(0.0, 1 - dist / 10000000)
        results.append({"name": name, "score": round(score, 4)})
    session.close()

    return jsonify(all_similarity_scores=results), 200

# --- エントリポイント ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cache", action="store_true")
    args = parser.parse_args()

    if args.build_cache:
        build_cache()
    else:
        if not Path(INDEX_PATH).exists():
            app.logger.info("キャッシュが無いので自動作成")
            build_cache(dim=256)
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

if __name__ == "__main__":
    main()
