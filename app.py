from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)

# 商品画像を保存するフォルダ（登録用）
REGISTER_FOLDER = "registered_images"
os.makedirs(REGISTER_FOLDER, exist_ok=True)

# アップロード画像保存用（オプション）
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 登録済み商品画像の読み込み（1枚ずつグレースケールで）
def load_registered_images():
    images = {}
    for filename in os.listdir(REGISTER_FOLDER):
        path = os.path.join(REGISTER_FOLDER, filename)
        try:
            img = Image.open(path).resize((64, 64)).convert('L')
            images[filename] = np.array(img).astype("float32") / 255
        except:
            continue
    return images

# 比較して一番近い画像の名前を返す
def predict_product(upload_image):
    upload_image = upload_image.resize((64, 64)).convert('L')
    upload_array = np.array(upload_image).astype("float32") / 255

    min_diff = float('inf')
    best_match = None

    registered_images = load_registered_images()
    for filename, img_array in registered_images.items():
        diff = np.mean((upload_array - img_array) ** 2)
        if diff < min_diff:
            min_diff = diff
            best_match = filename

    if best_match:
        name = os.path.splitext(best_match)[0]
        return name
    return None

# 🔁 商品登録時の画像保存（Rails → Flask）
@app.route('/register_image', methods=['POST'])
def register_image():
    if 'image' not in request.files:
        return jsonify({"error": "画像がありません"}), 400

    image_file = request.files['image']

    # タイムスタンプ付きで保存
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}.png"
    save_path = os.path.join(REGISTER_FOLDER, filename)

    image_file.save(save_path)

    return jsonify({"message": f"画像を保存しました: {filename}"})

# 📷 商品認識API（Rails → Flask）
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "画像がありません"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    result = predict_product(image)

    if result:
        return jsonify({"name": result})
    else:
        return jsonify({"error": "商品が見つかりませんでした"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

