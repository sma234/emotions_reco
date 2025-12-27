A Computer Vision project for real-time facial emotion recognision using Python, OpenCV and a custom trained CNN.

The face recognision happend through the computers webcam using OpenCV for face detection and frame processing, Haar Cascades for detecting the front of the face, a custom CNN(trained on 48x48 grayscale images from Kaggle) for classifying facial expressions and Keras to load and run the trained model. The emotions that are been predicted through webcam frames are: Angry,Disgust,Fear,Happy,Sad,Surprise,Neutral. Once the program is running you can press 'q' for quiting the program and 's' to save all detected faces (like a screenshot in your storage). There is also a requirements file attached which includes all the libraries that the project needs to run.

Model file:
emotion_model.h5

Project Structure
|-- main.py
|--emotion_model.h5
|--faces/ 
|--README.md

Installation:

pip install opencv-python opencv-contrib-python tensorflow numpy


