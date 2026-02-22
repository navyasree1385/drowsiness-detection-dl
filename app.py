from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np
import time
import os

app = Flask(__name__)

# Load Model
model = tf.keras.models.load_model("model/drowsiness_model.h5", compile=False,
safe_mode=False)

# Haar Cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

closed_start = None
alarm_status = False

def generate_frames():

    global closed_start, alarm_status

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        status = "Open"
        color = (0,255,0)
        alarm = False

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (fx,fy,fw,fh) in faces:

            cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(255,0,0),2)

            face_gray = gray[fy:fy+fh, fx:fx+fw]
            face_color = frame[fy:fy+fh, fx:fx+fw]

            eyes = eye_cascade.detectMultiScale(face_gray,1.1,3)

            for (x,y,w,h) in eyes:

                cv2.rectangle(face_color,(x,y),(x+w,y+h),(0,255,0),2)

                eye = face_color[y:y+h, x:x+w]
                eye = cv2.resize(eye,(64,64))
                eye = eye/255.0
                eye = np.expand_dims(eye,axis=0)

                pred = model.predict(eye,verbose=0)[0][0]

                if pred < 0.5:
                    status="Closed"
                    color=(0,0,255)
                else:
                    status="Open"
                    color=(0,255,0)

                break

        if status=="Closed":

            if closed_start is None:
                closed_start=time.time()

            if time.time()-closed_start>2:
                alarm = True
        else:
            closed_start=None

        alarm_status = alarm

        cv2.putText(frame,"Eyes: "+status,(40,50),
        cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

        if alarm:
            cv2.putText(frame,"DROWSY!",(40,100),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alarm')
def alarm():
    global alarm_status
    return str(alarm_status)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)