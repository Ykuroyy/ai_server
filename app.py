import os
import logging
import json
import uuid
import requests
import boto3
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFile, ImageFilter
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 設定 ──────────────────────────────────────────────
DATABASE_URL = os.environ["DATABASE_URL"]  # 例: postgres://user:pass@host/db
engine = create_engine(DATABASE_URL)
Session  = sessionmaker(bind=engine)
Base     = declarative_base()

class ProductMapping(Base):
    __tablename__ = "products"
    id     = Column(Integer, primary_key=True)
    name   = Column(String)
    s3_key = Column(String)


ImageFile.LOAD_TRUNCATED_IMAGES = True
S3_BUCKET = os.environ["S3_BUCKET"]
s3 = boto3.client("s3")

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

REGISTER_FOLDER = "registered_images"
os.makedirs(REGISTER_FOLDER, exist_ok=True)
MAPPING_FILE = "name_mapping.json"

try:
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        name_mapping = json.load(f)
except FileNotFoundError:
    name_mapping = {}

# S3 上にキーがないマッピングを削除
valid_keys = {
    obj["Key"]
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET)
    for obj in page.get("Contents", [])
}
name_mapping = {k: v for k, v in name_mapping.items() if k in valid_keys}
app.logger.info(f"マッピング件数: {len(name_mapping)}")

# ORB/SIFT 初期化
orb   = cv2.ORB_create()
bf    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
sift  = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher(
    {"algorithm": 1, "trees": 5},
    {"checks": 50}
)

# ── ヘルパー ────────────────────────────────────────────

def compute_score_for_key(key, q_arr, raw_img):
    # S3 から取得
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    img = Image.open(BytesIO(resp["Body"].read()))
    # 前処理
    img = crop_to_object(img)
    ref  = preprocess_pil(img, size=100)
    r_arr = np.asarray(ref)
    # スコア計算
    score_ssim = ssim(q_arr, r_arr, full=True)[0]
    score_hist = calc_color_hist_score(raw_img, img)
    score_sift = calc_sift_score(raw_img, img)
    return key, 0.2*score_ssim + 0.1*score_hist + 0.2*score_sift






def crop_to_object(pil_img, thresh=200):
    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binimg = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    cnts, _  = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pil_img
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return pil_img.crop((x, y, x+w, y+h))

def preprocess_pil(img, size=200):
    img = img.convert("L")
    img = ImageOps.exif_transpose(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img = ImageOps.fit(img, (size, size))
    img = ImageOps.autocontrast(img, cutoff=1)
    return img

def calc_color_hist_score(raw_img, ref_img, size=100):
    raw = np.array(raw_img.convert("RGB").resize((size, size)))
    ref = np.array(ref_img.convert("RGB").resize((size, size)))
    raw_hsv = cv2.cvtColor(raw, cv2.COLOR_RGB2HSV)
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_RGB2HSV)
    raw_hist = cv2.calcHist([raw_hsv], [0], None, [50], [0, 180])
    ref_hist = cv2.calcHist([ref_hsv], [0], None, [50], [0, 180])
    cv2.normalize(raw_hist, raw_hist)
    cv2.normalize(ref_hist, ref_hist)
    return float(cv2.compareHist(raw_hist, ref_hist, cv2.HISTCMP_CORREL))

def calc_orb_score(raw_img, ref_img, size=200):
    raw = np.array(raw_img.convert("L").resize((size, size)))
    ref = np.array(ref_img.convert("L").resize((size, size)))
    kp1, des1 = orb.detectAndCompute(raw, None)
    kp2, des2 = orb.detectAndCompute(ref, None)
    if des1 is None or des2 is None:
        return 0.0
    matches = bf.match(des1, des2)
    return float(len(matches) / max(len(kp1), 1))

def calc_sift_score(raw_img, ref_img, size=200):
    raw = np.array(raw_img.convert("L").resize((size, size)))
    ref = np.array(ref_img.convert("L").resize((size, size)))
    kp1, des1 = sift.detectAndCompute(raw, None)
    kp2, des2 = sift.detectAndCompute(ref, None)
    if des1 is None or des2 is None:
        return 0.0
    matches = flann.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return float(len(good) / max(len(kp1), 1))

# ── 画像登録エンドポイント ───────────────────────────────
# app.py の /register_image 部分を以下のように置き換え

@app.route("/register_image", methods=["POST"])
def register_image():
    name = request.form.get("name")
    if not name:
        return "invalid request (no name)", 400

    # ① imageファイル or ② image_url のどちらかで画像を取得
    if "image" in request.files:
        file_stream = request.files["image"].stream
    elif "image_url" in request.form:
        try:
            resp = requests.get(request.form["image_url"])
            resp.raise_for_status()
            file_stream = BytesIO(resp.content)
        except Exception as e:
            app.logger.error(f"Failed download image_url: {e}")
            return "invalid image_url", 400
    else:
        return "invalid request (no image or image_url)", 400

    try:
        img = Image.open(file_stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((640, 640), Image.Resampling.LANCZOS)
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(REGISTER_FOLDER, filename)
        img.save(save_path, format="JPEG", quality=80, optimize=True)

        # name_mapping の更新
        name_mapping[filename] = name
        with open(MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(name_mapping, f, ensure_ascii=False, indent=2)

        # S3 アップロード
        s3.upload_file(
            Filename=save_path,
            Bucket=S3_BUCKET,
            Key=filename,
            ExtraArgs={"ContentType": "image/jpeg"}
        )
        app.logger.info(f"☁️ uploaded to S3: s3://{S3_BUCKET}/{filename}")
        return "OK", 200

    except Exception as e:
        app.logger.exception(e)
        return "error", 500
      

# ── 画像認識 & 結果返却（predict_result は廃止） ────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 画像取得
        if "image_url" in request.form:
            resp = requests.get(request.form["image_url"])
            resp.raise_for_status()
            raw = Image.open(BytesIO(resp.content))
        elif "image" in request.files:
            raw = Image.open(request.files["image"].stream)
        else:
            return jsonify(error="画像がありません"), 400

        raw = crop_to_object(raw)
        query = preprocess_pil(raw, size=100)
        q_arr = np.asarray(query)

        # S3 上のキーを一度に取り出し
        keys = [
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET)
        for obj in page.get("Contents", [])
        if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        scores = []
        # スレッドで並列実行
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(compute_score_for_key, key, q_arr, raw): key for key in keys}
            for fut in as_completed(futures):
                try:
                    scores.append(fut.result())
                except Exception as e:
                    app.logger.error(f"比較エラー {futures[fut]}: {e}")

        # あとは降順ソートして JSON 化
        scores.sort(key=lambda x: x[1], reverse=True)



        # ベスト・候補３件
        best_key, best_score = scores[0]
        candidates = [
            {"name": name_mapping.get(k, os.path.splitext(k)[0]), "score": round(s,4)}
            for k, s in scores[1:4]
        ]

        # 全件スコア（DBマッピング版）
        session = Session()
        all_scores = []
        for key, score in scores:
            # DB に登録された名前を探す
            prod = session.query(ProductMapping).filter_by(s3_key=key).first()
            display_name = prod.name if prod else os.path.splitext(key)[0]
            all_scores.append({
                "name":  display_name,
                "score": round(score, 4)
            })
        session.close()
         

        return jsonify(
            best              = {"name": name_mapping.get(best_key, best_key), "score": round(best_score,4)},
            candidates        = candidates,
            all_similarity_scores = all_scores
        ), 200

    except Exception as e:
        app.logger.exception(e)
        return jsonify(error="処理エラー"), 500

# ── アプリ起動 ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)