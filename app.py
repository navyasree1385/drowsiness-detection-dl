from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/drowsiness_model.h5", compile=False)

def prepare_image(image):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            processed = prepare_image(img)
            pred = model.predict(processed)[0][0]

            if pred < 0.5:
                prediction = "Closed Eyes (Drowsy)"
            else:
                prediction = "Open Eyes"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)