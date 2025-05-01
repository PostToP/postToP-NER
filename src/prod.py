import dill
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from features import FeatureExtraction
import numpy as np
from model import decode_prediction
from text_cleaner import preprocess_tokens
from tensormonad import TensorMonad


def main():
    with open("out/model.pkl", "rb") as f:
        model = dill.load(f)

        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        def predict():
            data = request.get_json(force=True)
            title = data["title"]
            channel_name = data["channel_name"]
            description = data["description"]
            grouped_results = model.predict(title, channel_name, description)
            return jsonify({"prediction": grouped_results, "status": "success"})

        if __name__ == "__main__":
            app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
