from flask import Flask, request, jsonify
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

app = Flask(__name__)

# 保存ディレクトリ（存在しなければ作成）
REGISTER_FOLDER = "registered_images"
TEMP_IMAGE_PATH = "temp_image.png"

os.makedirs(REGISTER_FOLDER, exist_ok=True)

# 画像比較用の関数（SSIM）
def compare_images(img1, img2):
    img1 = img1.resize((100, 100)).convert("L")
    img2 = img2.resize((100, 100)).convert("L")
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    score, _ = ssim(arr1, arr2, full=True)
    return score

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
    if "image" not in request.files:
        return jsonify({"error": "画像が見つかりません"}), 400

    image = request.files["image"]
    image.save(TEMP_IMAGE_PATH)
    temp_img = Image.open(TEMP_IMAGE_PATH)

    if not os.path.exists(REGISTER_FOLDER):
        return jsonify({"error": "登録済み商品がありません"}), 500

    max_score = -1
    best_match = None

    try:
        for filename in os.listdir(REGISTER_FOLDER):
            reg_path = os.path.join(REGISTER_FOLDER, filename)
            if not filename.lower().endswith(".png"):
                continue  # PNG以外無視（拡張性を意識）
            reg_img = Image.open(reg_path)
            score = compare_images(temp_img, reg_img)
            if score > max_score:
                max_score = score
                best_match = filename.rsplit(".", 1)[0]
    except Exception as e:
        return jsonify({"error": f"画像比較中にエラー: {str(e)}"}), 500
    finally:
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)

    if best_match:
        return jsonify({"name": best_match, "score": round(max_score, 4)})
    else:
        return jsonify({"error": "一致する商品が見つかりません"}), 404

if __name__ == "__main__":
    app.run(debug=True)
