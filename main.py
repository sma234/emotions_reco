import os
import cv2 #opencv library for live video processing
import time
import numpy as np
import tensorflow as tf
import traceback
#emotions that the model predits(standard emotions)
EMOTIONS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

"""
load_emotion_model function explaining
Loading the emotion recognision model i trained. In this way the main stays more easy
to understand and more simple. 

parameter: model_path:str --> path to the Keras model (.h5 file)
result: model:tf.Keras.Model --> loads Keras model ready to predict emotions
"""
def load_emotion_model(model_path="emotion_model.h5"):
    print(f"[INFO] Loading model from {model_path} ...")
    #sends an understandable error if the file is missing
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file '{model_path}' not found.")
        raise FileNotFoundError(model_path)
    #uses Keras build in loader for .h5 models
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully.")
    return model

"""
!!!!!! PRE_PROCESS_FACE FUNCTION EXPLAINING
-Processes a cropped grayscale image before giving it to the model
-My Model trained on 48x48 grayscalled images from Kaggle which were normalized in [0,1]
so I followed the same steps for image procssing to avoid conflict.
Steps I followed:
1.Resize to (48,48)
2.Convert to float32 and normalise pixel values to [0,1]
3.add channel dimension (48,48,1)
4.add batch dimension (1,48,48,1) bcs keras model expect batches

Parameter: gray_face-->np.ndarray
grayscale face image (a cropped region that interest us from the full camera frame)
Result: face_norm-->np.ndarray
processes face image ready to be passed to model.predict()

"""
def preprocess_face(gray_face):
    face_resized=cv2.resize(gray_face,(48, 48))
    face_norm=face_resized.astype("float32")/255.0
    face_norm=np.expand_dims(face_norm,axis=-1)
    face_norm=np.expand_dims(face_norm,axis=0)
    return face_norm

"""
!!!!! MAIN EXPLANATION
The pipeline
1.Sets folder to store saved face images
2.Open the default camera with index 0 using opencv
3.Tests if camera works requesting some frames
4.Loads the pre-trained emotion recognition model
5.Load Haar cascade for face detection
6.For each frame:
-Detects faces from camera input
-For each face:preprocess,run through the model,get predicted emotion
-Draws bounding boxes and labels on the frame
-Shows FPS and number of detected faces on top left corner of screen
-If the user wants it saves detected faces when 's' is pressed
7.Clean up(close camera,close windows)
"""
def main():
    print("main.py starts normally")

    try:
        faces_dir = "faces"
        os.makedirs(faces_dir, exist_ok=True)
        print(f"Faces directory: {os.path.abspath(faces_dir)}")

        #Opens the camera
        print("[INFO] Opening camera 0 ...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[ERROR] Could not open camera in main.py")
            return

        print("[INFO] Camera opened. Warming up...")

        #Tests a few frames to see if video works correctly
        warm_ok = False
        for i in range(30):
            ret, frame = cap.read()
            if ret:
                warm_ok = True
                break
            time.sleep(0.05)

        if not warm_ok:
            print("Couldnt use camera during the testing.")
            cap.release()
            return

        print("Camera works good!")

        #Uploading the model i made for emotions recognition
        emotion_model = load_emotion_model("emotion_model.h5")

        #Uploading Haar cascade
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        print(f"Using Haar cascade: {face_cascade_path}")
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        if face_cascade.empty():
            print("Haar cascade couldnt open.")
            cap.release()
            return

        print("The program is working!! Press 'q' to quit,'s' to save faces.")
        saved_count = 0
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame in main loop.")
                break

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                face_gray = gray[y:y + h, x:x + w]
                face_input = preprocess_face(face_gray)

                preds = emotion_model.predict(face_input, verbose=0)[0]
                idx = int(np.argmax(preds))
                emotion = EMOTIONS[idx]
                conf = float(preds[idx]) * 100

                label = f"{emotion} ({conf:.1f}%)"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

            cv2.putText(frame, f"Faces: {len(faces)}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Emotion Recognition - q quit, s save", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("q pressed, stoping the program")
                break
            elif key == ord('s'):
                for i, (x, y, w, h) in enumerate(faces):
                    face_img = frame[y:y + h, x:x + w]
                    save_path = os.path.join(faces_dir, f"face_{saved_count}_{i}.jpg")
                    cv2.imwrite(save_path, face_img)
                    print(f"Saved face to {save_path}")
                saved_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("DEBUG: main() finished")

    except Exception:
        print("Something doesnt wrong in main:/")
        traceback.print_exc()

if __name__ == "__main__":
    print("DEBUG: __main__")
    main()
