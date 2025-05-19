from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)

REGISTER_FOLDER = "registered_images"
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(REGISTER_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@app.route('/register_image', methods=['POST'])
def register_image():
    if 'image' not in request.files:
        return jsonify({"error": "画像がありません"}), 400

    image_file = request.files['image']
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}.png"
    save_path = os.path.join(REGISTER_FOLDER, filename)
    image_file.save(save_path)

    return jsonify({"message": f"画像を保存しました: {filename}"})


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

@app.route('/')
def index():
    return "AI Server is running", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
