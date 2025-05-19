from flask import Flask, request, jsonify
from PIL import Image
import io
import random

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "画像がありません"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    sample_products = ["メロンパン", "クロワッサン"]
    predicted = random.choice(sample_products)

    return jsonify({"name": predicted})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
