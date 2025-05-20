from flask import Flask, request, jsonify
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

app = Flask(__name__)

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

    save_dir = "registered_images"
    os.makedirs(save_dir, exist_ok=True)
    image.save(os.path.join(save_dir, f"{name}.png"))
    return jsonify({"message": f"{name} を保存しました"})

# 商品名を予測（SSIMによる類似度比較）
@app.route("/predict", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "画像が見つかりません"}), 400

    image = request.files["image"]
    temp_path = "temp_image.png"
    image.save(temp_path)
    temp_img = Image.open(temp_path)

    max_score = -1
    best_match = None

    for filename in os.listdir("registered_images"):
        reg_img = Image.open(os.path.join("registered_images", filename))
        score = compare_images(temp_img, reg_img)
        if score > max_score:
            max_score = score
            best_match = filename.rsplit(".", 1)[0]

    os.remove(temp_path)
    
    if best_match:
        return jsonify({"name": best_match, "score": round(max_score, 4)})
    else:
        return jsonify({"error": "一致する商品が見つかりません"}), 404

if __name__ == "__main__":
    app.run(debug=True)
