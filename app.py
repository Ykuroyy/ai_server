from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


app = Flask(__name__)

# モデルとベクトライザの読み込み
model_path = "product_recognition_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model, vectorizer = pickle.load(f)
else:
    model, vectorizer = None, None

# 画像を特徴量ベクトルに変換する関数
def image_to_feature_vector(image_path):
    img = Image.open(image_path).resize((100, 100)).convert("L")  # グレースケールに変換
    return np.array(img).flatten()





# 商品名を登録
@app.route("/register_image", methods=["POST"])
def register_image():
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "image または name がありません"}), 400

    image = request.files["image"]
    name = request.form["name"]

    save_dir = "registered_images"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{name}.png"
    path = os.path.join(save_dir, filename)
    image.save(path)

    print(f"✅ {filename} を保存")

    # 再学習
    try:
        train_model()
        return jsonify({"message": "登録完了・再学習済み"})
    except Exception as e:
        print(f"❌ 再学習失敗: {e}")  # ← この行を追加
        return jsonify({"error": f"再学習失敗: {e}"}), 500




# train
def train_model():
    global model, vectorizer
    images = []
    labels = []

    for filename in os.listdir("registered_images"):
        label = filename.rsplit(".", 1)[0]
        path = os.path.join("registered_images", filename)
        vec = image_to_feature_vector(path)
        images.append(vec)
        labels.append(label)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([" ".join(map(str, v)) for v in images])
    y = labels

    model = SVC(kernel="linear")
    model.fit(X, y)

    with open("product_recognition_model.pkl", "wb") as f:
        pickle.dump((model, vectorizer), f)

    print(f"✅ モデル再学習完了：{len(y)} 件")









# 商品名を予測するAPI
@app.route("/predict", methods=["POST"])
def predict_image():
    print("✅ /predict に画像が届きました")

    if "image" not in request.files:
        print("❌ image が request.files にありません")
        return jsonify({"error": "画像が見つかりません"}), 400

    image = request.files["image"]
    if image:
        temp_path = "temp_image.png"
        image.save(temp_path)
        print("📷 画像保存済み: ", temp_path)

        try:
            vec = image_to_feature_vector(temp_path)
            print("🧠 特徴ベクトル作成済み")

            if model is not None and vectorizer is not None:
                X = vectorizer.transform([" ".join(map(str, vec))])
                prediction = model.predict(X)[0]
                print(f"🎯 予測結果: {prediction}")
                return jsonify({"name": prediction})
            else:
                print("⚠️ モデル未学習")
                return jsonify({"error": "モデルが未学習です"}), 503
        except Exception as e:
            print("❌ 予測中にエラー: ", e)
            return jsonify({"error": f"予測エラー: {e}"}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return jsonify({"error": "画像が無効です"}), 400




if __name__ == "__main__":
    app.run(debug=True)


