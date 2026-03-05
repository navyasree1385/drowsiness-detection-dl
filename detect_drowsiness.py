import cv2
import tensorflow as tf
import numpy as np
import time
import winsound

# Load Model
model = tf.keras.models.load_model("model/drowsiness_model.h5")

# Haar Cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

print("Face Cascade Loaded:", not face_cascade.empty())
print("Eye Cascade Loaded:", not eye_cascade.empty())

# Start Camera
cap = cv2.VideoCapture(0)

closed_start = None

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Camera Error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    status = "Detecting..."
    color = (255,255,0)

    try:

        faces = face_cascade.detectMultiScale(
            gray,
            1.3,
            5
        )

        for (fx,fy,fw,fh) in faces:

            cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(255,0,0),2)

            face_gray = gray[fy:fy+fh, fx:fx+fw]
            face_color = frame[fy:fy+fh, fx:fx+fw]

            eyes = eye_cascade.detectMultiScale(
                face_gray,
                1.1,
                3
            )

            for (x,y,w,h) in eyes:

                cv2.rectangle(face_color,(x,y),(x+w,y+h),(0,255,0),2)

                eye = face_color[y:y+h, x:x+w]

                eye = cv2.resize(eye,(64,64))
                eye = eye/255.0

                eye = np.expand_dims(eye,axis=0)

                pred = model.predict(eye,verbose=0)[0][0]

                print("Prediction:",pred)

                # Better threshold
                if pred < 0.5:
                    status="Closed"
                    color=(0,0,255)
                else:
                    status="Open"
                    color=(0,255,0)

                break

    except:
        print("Detection skipped")

    # Alarm
    if status=="Closed":

        if closed_start is None:
            closed_start=time.time()

        if time.time()-closed_start>2:
            winsound.Beep(1000,500)

    else:
        closed_start=None

    cv2.putText(
        frame,
        "Eyes: "+status,
        (40,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Drowsiness Detection",frame)

    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()