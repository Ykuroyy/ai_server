import os
import logging
import json
import uuid
import requests
import boto3
from io import BytesIO
from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFile, ImageFilter
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

# ── 設定 ──────────────────────────────────────────────
# トランケートされた画像も読み込めるように
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 環境変数から S3 バケット名を取得（boto3.client の前に）
S3_BUCKET = os.environ["S3_BUCKET"]

# S3 クライアント（環境変数の認証情報を利用）
s3 = boto3.client("s3")

# Flask アプリと Blueprint
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)
api = Blueprint("api", __name__)

# ローカル登録用ディレクトリ＆マッピングファイル
REGISTER_FOLDER = "registered_images"
os.makedirs(REGISTER_FOLDER, exist_ok=True)
MAPPING_FILE = "name_mapping.json"

# 商品名マッピングの読み込み（S3 上に存在しないキーは除外）
try:
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        name_mapping = json.load(f)
except FileNotFoundError:
    name_mapping = {}

valid_keys = {
    obj["Key"]
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET)
    for obj in page.get("Contents", [])
}
orig_count = len(name_mapping)
name_mapping = {k: v for k, v in name_mapping.items() if k in valid_keys}
app.logger.info(f"マッピングフィルタリング: 元{orig_count}件 → 現在S3上にあるのは{len(name_mapping)}件")


# ── 特徴量マッチャー初期化 ─────────────────────────────
orb   = cv2.ORB_create()
bf    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
sift  = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher(
    {"algorithm": 1, "trees": 5},
    {"checks": 50}
)

# ── ヘルパー関数 ────────────────────────────────────
def crop_to_object(pil_img: Image.Image, thresh=200) -> Image.Image:
    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binimg = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    cnts, _  = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pil_img
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return pil_img.crop((x, y, x+w, y+h))

def preprocess_pil(img: Image.Image, size=200) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.exif_transpose(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img = ImageOps.fit(img, (size, size))
    img = ImageOps.autocontrast(img, cutoff=1)
    return img

def calc_color_hist_score(raw_img: Image.Image, ref_img: Image.Image, size=100) -> float:
    raw = np.array(raw_img.convert("RGB").resize((size, size)))
    ref = np.array(ref_img.convert("RGB").resize((size, size)))
    raw_hsv = cv2.cvtColor(raw, cv2.COLOR_RGB2HSV)
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_RGB2HSV)
    h_bins = 50
    raw_hist = cv2.calcHist([raw_hsv], [0], None, [h_bins], [0, 180])
    ref_hist = cv2.calcHist([ref_hsv], [0], None, [h_bins], [0, 180])
    cv2.normalize(raw_hist, raw_hist)
    cv2.normalize(ref_hist, ref_hist)
    return float(cv2.compareHist(raw_hist, ref_hist, cv2.HISTCMP_CORREL))

def calc_orb_score(raw_img: Image.Image, ref_img: Image.Image, size=200) -> float:
    raw = np.array(raw_img.convert("L").resize((size, size)))
    ref = np.array(ref_img.convert("L").resize((size, size)))
    kp1, des1 = orb.detectAndCompute(raw, None)
    kp2, des2 = orb.detectAndCompute(ref, None)
    if des1 is None or des2 is None:
        return 0.0
    matches = bf.match(des1, des2)
    return float(len(matches) / max(len(kp1), 1))

def calc_sift_score(raw_img: Image.Image, ref_img: Image.Image, size=200) -> float:
    raw = np.array(raw_img.convert("L").resize((size, size)))
    ref = np.array(ref_img.convert("L").resize((size, size)))
    kp1, des1 = sift.detectAndCompute(raw, None)
    kp2, des2 = sift.detectAndCompute(ref, None)
    if des1 is None or des2 is None:
        return 0.0
    matches = flann.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return float(len(good) / max(len(kp1), 1))


# ── エンドポイント：画像登録 ───────────────────────────
@api.route("/register_image", methods=["POST"])
def register_image():
    name = request.form.get("name")
    file = request.files.get("image")
    if not name or not file:
        return "invalid request", 400

    try:
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


# ── エンドポイント：画像認識 ───────────────────────────
@api.route("/predict", methods=["POST"])
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

        # ROI 切り出し＋前処理
        raw = crop_to_object(raw)
        query = preprocess_pil(raw, size=100)
        q_arr  = np.asarray(query)

        # S3 上の全キー取得
        pages = list(s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET))
        best_score, best_key = -1.0, None

        # 比較ループ
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                # 参照画像取得
                resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                img  = Image.open(BytesIO(resp["Body"].read()))
                img  = crop_to_object(img)
                ref  = preprocess_pil(img, size=100)
                r_arr = np.asarray(ref)

                # 各種スコア算出
                score_ssim = ssim(q_arr, r_arr, full=True)[0]
                score_hist = calc_color_hist_score(raw, img)
                score_sift = calc_sift_score(raw, img)
                # ORB も試したい場合は calc_orb_score(raw, img)

                # 合成スコア（重みはチューニング可能）
                final_score = 0.6 * score_ssim + 0.1 * score_hist + 0.3 * score_sift

                if final_score > best_score:
                    best_score, best_key = final_score, key

        if best_key is None:
            return jsonify(error="一致なし", score=0), 404

        # しきい値以下は認識失敗
        if best_score < 0.5:
            return jsonify(error="認識精度不足", score=round(best_score, 3)), 404

        predicted = name_mapping.get(best_key, os.path.splitext(best_key)[0])
        return jsonify(name=predicted, score=round(best_score, 4)), 200

    except Exception as e:
        app.logger.exception(e)
        return jsonify(error="処理エラー"), 500


# Blueprint 登録と起動
app.register_blueprint(api, url_prefix="/api")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)