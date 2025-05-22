from flask import Flask, request, jsonify
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from flask_cors import CORS
import numpy as np
import os
import logging

# 保存ディレクトリ（存在しなければ作成）
REGISTER_FOLDER = "registered_images"
UPLOAD_FOLDER = "uploaded_images"
TEMP_IMAGE_PATH = "temp_image.png"

# Flask起動時に必要なフォルダを作成
os.makedirs(REGISTER_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
CORS(app)

# 画像比較用の関数（SSIM）
def compare_images(img1, img2):
    img1 = img1.resize((100, 100)).convert("L")
    img2 = img2.resize((100, 100)).convert("L")
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    score, _ = ssim(arr1, arr2, full=True)
    return score


@app.route("/ping", methods=["GET"])
def ping():
    return "ok", 200




# 商品画像を登録（画像保存のみ）
@app.route("/register_image", methods=["POST"])
def register_image():
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "image または name がありません"}), 400

    image = request.files["image"]
    name = request.form["name"]

    image.save(os.path.join(REGISTER_FOLDER, f"{name}.png"))
    return jsonify({"message": f"{name} を保存しました"})

# 商品名を予測（SSIMによる類似度比較）
@app.route("/predict", methods=["POST"])
def predict_image():
    try:
        app.logger.info("✅ /predict にアクセス")

        if "image" not in request.files:
            return jsonify({"error": "画像が見つかりません"}), 400

        image = request.files["image"]
        image.save(TEMP_IMAGE_PATH)
        temp_img = Image.open(TEMP_IMAGE_PATH)

        if not os.path.exists(REGISTER_FOLDER) or len(os.listdir(REGISTER_FOLDER)) == 0:
            app.logger.warning("⚠️ 登録画像がありません！")
            return jsonify({"error": "登録済み商品がありません"}), 500

        max_score = -1
        best_match = None

        for filename in os.listdir(REGISTER_FOLDER):
            reg_path = os.path.join(REGISTER_FOLDER, filename)
            if not filename.lower().endswith(".png"):
                continue
            reg_img = Image.open(reg_path)
            score = compare_images(temp_img, reg_img)

            # ✅ ログ出力で確認
            app.logger.info(f"比較: {filename} - 類似度スコア: {score:.4f}")


            if score > max_score:
                max_score = score
                best_match = filename.rsplit(".", 1)[0]

        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)

        # if best_match and max_score >= 0.6:
        # 試しに下げる
        if best_match and max_score >= 0.22:
            return jsonify({"name": best_match, "score": round(max_score, 4)})
        else:
            return jsonify({"error": "一致する商品が見つかりません", "score": round(max_score, 4)}), 404

    except Exception as e:
        app.logger.error(f"🔥 /predict内で予期しないエラー: {str(e)}")
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
        return jsonify({"error": "Flaskサーバー内でエラーが発生しました"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
