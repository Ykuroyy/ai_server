import os
import json
import uuid
import argparse
import requests

from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import faiss    # pip install faiss-cpu
import cv2
from PIL import Image, ImageOps, ImageFile, ImageFilter

from flask import Flask, request, jsonify
from flask_cors import CORS

from sqlalchemy import create_engine, Column, Integer, String, DateTime # DateTime をインポート
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func # func をインポート (デフォルトタイムスタンプ用)


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
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), server_onupdate=func.now())



# S3 クライアント
ImageFile.LOAD_TRUNCATED_IMAGES = True
S3_BUCKET = os.environ.get("S3_BUCKET", "registered_images")
s3        = boto3.client("s3") # グローバルS3クライアント


# 特徴量設定 (グローバル定数)
FEATURE_DIM = 256
SIFT_SIGMA = 1.6


# キャッシュ置き場
CACHE_DIR  = "cache"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss_v2.index") # バージョン管理のため名前変更も検討
KEYS_PATH  = os.path.join(CACHE_DIR, "keys.json")


# Flask
app = Flask(__name__)
CORS(app)
app.logger.setLevel("INFO")


# ✅ ここに追記（テーブルを作成）
Base.metadata.create_all(bind=engine)


# 🔽 ここに追記！ 🔽
@app.route("/build_cache", methods=["POST"])
def trigger_build_cache():
    try:
        build_cache() # グローバル FEATURE_DIM を使用
        return jsonify({"status": "ok", "message": "キャッシュを再構築しました"}), 200
    except Exception as e:
        app.logger.exception("キャッシュ再構築エラー")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/register_image_v2", methods=["POST"])
def register_image_v2():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        name = data.get("name")

        if not image_url or not name:
            return jsonify({"message": "image_url or name missing", "status": "error"}), 400

        # S3 から画像をダウンロード
        response = requests.get(image_url)
        response.raise_for_status() # HTTPエラーチェック
        img_pil = Image.open(BytesIO(response.content))
        
        # 一貫した前処理
        processed_img_pil = preprocess_pil(img_pil)

        desc = extract_sift(processed_img_pil) # dim はグローバル FEATURE_DIM を使用
        if desc is None:
            return jsonify({"message": "特徴量が見つかりませんでした", "status": "error"}), 400

        # 保存処理（例: S3キーとDB登録）
        key = f"registered_images/{uuid.uuid4().hex}.jpg"
        # グローバルなs3クライアントとS3_BUCKET変数を使用
        s3.upload_fileobj(BytesIO(response.content), S3_BUCKET, key)

        # DBへの保存はRails側で行うため、ここではs3_keyを返すだけにする
        # with Session() as session: # コンテキストマネージャを使用
        #     session.add(ProductMapping(name=name, s3_key=key))
        #     session.commit()

        return jsonify({"message": "登録成功", "status": "ok", "s3_key": key}), 200
    
    except requests.exceptions.RequestException as e:
        app.logger.error(f"❌ register_image_v2 画像ダウンロード失敗: {e}")
        return jsonify({"message": f"画像URLからのダウンロード失敗: {e}", "status": "error"}), 400
    except Exception as e:
        app.logger.exception("❌ register_image_v2 失敗") # トレースバックをログに出力
        return jsonify({"message": "内部エラー", "status": "error"}), 500


def extract_sift(pil_img_gray, dim=FEATURE_DIM):
    """preprocess_pilで処理済みのPILグレー画像からSIFT特徴量ベクトルを抽出"""
    if pil_img_gray.mode != "L":
        app.logger.warning(f"extract_siftは'L'モードのPIL画像を期待しましたが、{pil_img_gray.mode}を受け取りました。グレースケールに変換します。")
        pil_img_gray = pil_img_gray.convert("L")

    cv_gray_image = np.array(pil_img_gray)
    # SIFTパラメータを調整可能にする
    sift = cv2.SIFT_create(
        sigma=SIFT_SIGMA,
        # nfeatures=0,  # デフォルトのままか、調整する場合はコメントを外す (例: 500)
        contrastThreshold=0.03, # デフォルト(0.04)から調整する場合はコメントを外す
        # edgeThreshold=12 # デフォルト(10)から調整する場合はコメントを外す
    )
    _, des = sift.detectAndCompute(cv_gray_image, None)
    
    if des is None:
        return None
    
    vec = des.flatten()
    if vec.size == 0: # desがNoneでなくてもflatten結果が空になるケース対策
        return None

    # 固定長ベクトルにするためのパディング/トランケート
    if vec.shape[0] < dim: # &lt; を < に修正
        padded_vec = np.zeros(dim, dtype="float32")
        padded_vec[:vec.shape[0]] = vec
        vec = padded_vec
    else:
        vec = vec[:dim]

    norm = np.linalg.norm(vec)
    if norm == 0: # ゼロ除算を避ける
        return None
    return vec / norm


# ── 前処理ヘルパー ────────────────────────────────────

def crop_to_object(pil_img, thresh=200): # この関数は現在使用されていません
    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binimg = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    cnts, _  = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pil_img
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return pil_img.crop((x, y, x+w, y+h))


# def preprocess_pil(img, size=100): # 例: 現在の値から大きくしてみる (例: 100 -> 200)
def preprocess_pil(img, size=200):
    img = ImageOps.exif_transpose(img) # EXIF情報に基づく回転を先に行う
    img = img.convert("L")
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img = ImageOps.fit(img, (size, size))
    return ImageOps.autocontrast(img, cutoff=1)

# ── キャッシュ構築機能 ─────────────────────────────────

def build_cache(cache_dir=CACHE_DIR, index_path=INDEX_PATH): # dim引数は不要、グローバルFEATURE_DIMを使用
    os.makedirs(cache_dir, exist_ok=True)

    # 1) DB に登録されている s3_key のみ取得
    with Session() as session:
        products = session.query(ProductMapping).all()

    s3_keys_for_index = [] # 実際にインデックスに追加されたキーのリスト
    descriptors = []

    # 2) 各画像をダウンロード → 前処理 → SIFT特徴量抽出
    for product in products:
        key = product.s3_key
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
            img_pil = Image.open(BytesIO(resp["Body"].read()))
            
            processed_img_pil = preprocess_pil(img_pil) # 前処理を適用
            desc = extract_sift(processed_img_pil)      # SIFT特徴量を抽出

            if desc is not None:
                descriptors.append(desc)
                s3_keys_for_index.append(key)
                # 個別npyファイルの保存はFaissインデックスが主なら不要かも
                # sanitized_key = key.replace('/', '_') # パス区切り文字を置換
                # np.save(os.path.join(cache_dir, f"{sanitized_key}.npy"), desc)
            else:
                app.logger.warning(f"❌ 特徴量が取れませんでした (build_cache): {key}")
        except Exception as e:
            app.logger.error(f"❌ キャッシュ構築中にエラー (画像処理: {key}): {e}")
            continue # エラーが発生した画像はスキップ
     
    if not descriptors:
        app.logger.error("🚫 有効な特徴量が抽出された画像が 0 件です。キャッシュ作成中止")
        # 既存のインデックスファイルがあれば削除する
        if Path(index_path).exists():
            try:
                Path(index_path).unlink()
                app.logger.info(f"Removed existing index: {index_path}")
            except OSError as e_unlink:
                app.logger.error(f"Error removing index file {index_path}: {e_unlink}")
        if Path(KEYS_PATH).exists():
            try:
                Path(KEYS_PATH).unlink()
                app.logger.info(f"Removed existing keys file: {KEYS_PATH}")
            except OSError as e_unlink:
                app.logger.error(f"Error removing keys file {KEYS_PATH}: {e_unlink}")
        return

    xb = np.stack(descriptors).astype('float32') # Faissのためにfloat32型に変換

    # 3) keys.json を保存
    with open(KEYS_PATH, "w", encoding="utf-8") as f: # s3_keys_for_index を保存
        json.dump(s3_keys_for_index, f, ensure_ascii=False, indent=2)

    # 4) Faiss インデックス構築＋保存
    index = faiss.IndexFlatL2(FEATURE_DIM) # グローバル FEATURE_DIM を使用
    index.add(xb)
    faiss.write_index(index, index_path)

    app.logger.info(f"✅ キャッシュ({len(s3_keys_for_index)}件) & インデックスを生成しました → {cache_dir}/ , {index_path}")

# ── 画像登録エンドポイント ───────────────────────────────
@app.route("/register_image", methods=["POST"])
def register_image():
    name = request.form.get("name")
    if not name:
        return "invalid request (no name)", 400

    if "image" in request.files:
        stream = request.files["image"].stream
    elif "image_url" in request.form:
        # import requests # グローバルにあるので不要
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
        
        # S3キーの形式を v2 と合わせる (ディレクトリ構造)
        # filename = f"{uuid.uuid4().hex}.jpg" # 元の形式
        filename = f"registered_images/{uuid.uuid4().hex}.jpg" # v2に合わせた形式
        
        # ローカル保存はS3アップロードが主なら不要かも
        # path = os.path.join("registered_images", filename)
        # os.makedirs("registered_images", exist_ok=True)
        # img.save(path, format="JPEG", quality=80, optimize=True)
        # s3.upload_file(path, S3_BUCKET, filename, ExtraArgs={"ContentType":"image/jpeg"})
        
        # S3へ直接アップロード
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG", quality=80, optimize=True)
        img_byte_arr.seek(0)
        s3.upload_fileobj(img_byte_arr, S3_BUCKET, filename, ExtraArgs={"ContentType":"image/jpeg"})

        app.logger.info(f"☁️ uploaded to S3://{S3_BUCKET}/{filename}")

        # DBへの保存はRails側で行うため、ここではs3_keyを返すだけにする
        # with Session() as session: # コンテキストマネージャを使用
        #     product = ProductMapping(name=name, s3_key=filename)
        #     session.add(product)
        #     session.commit()

        return jsonify({"message": "登録成功", "status": "ok", "s3_key": filename}), 200 # s3_keyを返すように変更

    except Exception as e:
        app.logger.exception(e) # トレースバックをログに出力
        return "error", 500


# ── 画像認識エンドポイント ─────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1) 画像取得
        img_stream = None
        if "image" in request.files:
            img_stream = request.files["image"].stream
        elif "image_url" in request.form:
            # import requests as req_local # グローバルにあるので不要、エイリアスも不要
            try:
                r = requests.get(request.form["image_url"])
                r.raise_for_status()
                img_stream = BytesIO(r.content)
            except requests.exceptions.RequestException as e:
                app.logger.error(f"predictでの画像ダウンロード失敗: {e}")
                return jsonify(error=f"画像URLからのダウンロード失敗: {e}"), 400
        else:
            return jsonify(error="画像がありません"), 400

        raw_pil_img = Image.open(img_stream)
        
        # 一貫した前処理
        processed_pil_img = preprocess_pil(raw_pil_img)
        
        # 2) 特徴量抽出 (統一された関数を使用)
        q_vec = extract_sift(processed_pil_img) # dim はグローバル FEATURE_DIM を使用

        if q_vec is None:
            app.logger.warning("❌ クエリ画像の特徴量が抽出できませんでした")
            return jsonify(error="画像が不明瞭で特徴量を抽出できませんでした"), 400
        
        # Faissは (n_samples, dim) の形式の配列を期待
        q_arr_expanded = np.expand_dims(q_vec.astype('float32'), 0)

        # 3) インデックスとキー読み込み
        if not Path(INDEX_PATH).exists() or not Path(KEYS_PATH).exists():
            app.logger.error("🚫 Faissインデックスまたはキーファイルが見つかりません。先に /build_cache を実行するか、アプリを再起動してください。")
            return jsonify(error="検索インデックスが準備されていません。キャッシュを構築してください。"), 503 # Service Unavailable

        index = faiss.read_index(INDEX_PATH)
        with open(KEYS_PATH, "r", encoding="utf-8") as f:
            indexed_s3_keys = json.load(f) # インデックス内のベクトルに対応するキー

        if index.ntotal == 0 or not indexed_s3_keys:
             app.logger.warning("🤷 検索対象のインデックスが空です。")
             return jsonify(all_similarity_scores=[]), 200

        # 4) 検索
        # インデックス内のアイテム数までを検索対象とする
        k_search = min(len(indexed_s3_keys), index.ntotal) 
        if k_search == 0:
            return jsonify(all_similarity_scores=[]), 200
            
        D, I = index.search(q_arr_expanded, k=k_search)

        # 5) 結果整形（重複名除外）
        all_scores = []
        with Session() as session: # コンテキストマネージャを使用
            seen_names = set()
            for dist, idx_in_index in zip(D[0], I[0]):
                if idx_in_index < 0: # Faissが返す-1は無効なインデックス. &lt; を < に修正
                    continue
                
                s3_key = indexed_s3_keys[idx_in_index]
                prod = session.query(ProductMapping).filter_by(s3_key=s3_key).first()
                
                if not prod:
                    app.logger.warning(f"DBに商品が見つかりません: s3_key={s3_key} (インデックスより)。スキップします。")
                    continue

                name = prod.name
                if name in seen_names:
                    continue
                seen_names.add(name)

                score = round(1.0 / (1.0 + dist), 4) if dist >= 0 else 0.0 # &gt;= を >= に修正
                app.logger.info(f"📊 dist={dist:.2f}, score={score:.4f}, name={name}, s3_key={s3_key}")

                all_scores.append({
                    "name": name,
                    "score": float(score) # JSONシリアライズのためにfloat型を保証
                })

        # スコアで降順ソート
        all_scores_sorted = sorted(all_scores, key=lambda x: x["score"], reverse=True)

        return jsonify(all_similarity_scores=all_scores_sorted), 200

    except Exception as e:
        app.logger.exception("❌ predict エンドポイントでエラーが発生しました")
        return jsonify(error="内部サーバーエラーが発生しました。"), 500
        

# ── エントリポイント ─────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-cache", action="store_true",
        help="S3 から特徴量キャッシュ＆Faissインデックスを作成"
    )
    args = parser.parse_args()

    if args.build_cache:
        build_cache() # グローバル FEATURE_DIM を内部で使用
    else:
        try:
            if not Path(INDEX_PATH).exists() or not Path(KEYS_PATH).exists():
                app.logger.info("キャッシュ／インデックスが見つからないので自動生成します (モジュール読み込み時)")
                build_cache() # グローバル FEATURE_DIM を内部で使用
        except Exception as e:
            app.logger.error(f"❌ 起動時のキャッシュ生成失敗: {e}")

        port = int(os.environ.get("PORT", 10000))
        app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()
