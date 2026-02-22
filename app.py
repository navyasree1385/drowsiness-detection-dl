from flask import Flask, render_template, request
from flask_cors import CORS
import cv2
import tensorflow as tf
import numpy as np
import base64
import time
import winsound

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("model/drowsiness_model.h5")

closed_start=None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=["POST"])
def predict():
    global closed_start

    data=request.data
    data=data.split(b',')[1]
    img=base64.b64decode(data)

    npimg=np.frombuffer(img,np.uint8)
    frame=cv2.imdecode(npimg,cv2.IMREAD_COLOR)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    eye_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_eye.xml")

    eyes=eye_cascade.detectMultiScale(gray,1.2,5)

    status="Open"

    for (x,y,w,h) in eyes:

        eye=frame[y:y+h,x:x+w]
        eye=cv2.resize(eye,(64,64))
        eye=eye/255.0
        eye=np.expand_dims(eye,axis=0)

        pred=model.predict(eye,verbose=0)[0][0]

        if pred<0.5:
            status="Closed"
        else:
            status="Open"

        break

    if status=="Closed":
        if closed_start is None:
            closed_start=time.time()

        if time.time()-closed_start>2:
            winsound.Beep(1000,500)
    else:
        closed_start=None

    return "done"

if __name__=="__main__":
    port=int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)