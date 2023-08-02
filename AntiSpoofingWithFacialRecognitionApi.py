import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json
from flask import Flask, request, jsonify
import pymongo
import pickle
import face_recognition
import logging

app = Flask(__name__)

def classify_single_frame(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected in the input frame.")
        return None

    for (x, y, w, h) in faces:
        face = frame[y-5:y+h+5, x-5:x+w+5]
        resized_face = cv2.resize(face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)

      
        preds = model.predict(resized_face)[0]
        print(preds)

        if preds > 0.5:
            return True  

    return False  

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model graph
json_file = open('/Users/madhavpatel/Desktop/FC3/Model2/antispoofing_models/antispoofing_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load antispoofing model weights
model.load_weights('/Users/madhavpatel/Desktop/FC3/Model2/antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://madhvptel:wabalabadubdub@cluster0.ma647k2.mongodb.net/")
db = client["face_recognition_db"]
user_collection = db["users"]

def get_user_embeddings(user_id):
    # Function to fetch user embeddings from the MongoDB database
    user_data = user_collection.find_one({"user_id": user_id})
    if user_data:
        return user_data["embeddings"]
    return None

def recognize_user(query_embeddings, user_id):
    # Function to recognize the user based on embeddings from MongoDB
    embeddings = get_user_embeddings(user_id)
    if embeddings is not None:
        if face_recognition.compare_faces([embeddings], query_embeddings)[0]:
            return True
    return False

@app.route('/classify_video', methods=['POST'])
def classify_video():
    if 'file' not in request.files or 'user_id' not in request.form:
        return jsonify({'error': 'No file part or user ID in the request'}), 400

    video_file = request.files['file']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    user_id = request.form['user_id']

    video_path = '/Users/madhavpatel/Desktop/FC3/FinalFolder/TempVideo.mp4'
    video_file.save(video_path)

    video = cv2.VideoCapture(video_path)

    frames_count = 0
    fake_count = 0

    frame_for_login = None

    while frames_count < 10: 
        ret, frame = video.read()

        if not ret:
            break

        result = classify_single_frame(frame, model)
        if result:
            fake_count += 1
        else:
            frame_for_login = frame  # Store one frame for login verification

        frames_count += 1

    video.release()

    if fake_count > 0:
        logging.warning("Video classified as spoofed for user ID: {}".format(user_id))
        return jsonify({'result': '01'})
    else:
        if frame_for_login is not None:
            # If frame count < 20, perform login verification using the stored frame
            face_locations = face_recognition.face_locations(frame_for_login)
            if len(face_locations) > 0:
                face_encodings = face_recognition.face_encodings(frame_for_login, face_locations)
                if len(face_encodings) > 0:
                    query_embeddings = face_encodings[0]
                    if recognize_user(query_embeddings, user_id):
                        logging.info("Person verified:{}".format(user_id))
                        return jsonify({'result': '00'})
                    else:
                        logging.warning("Login verification failed for user ID: {}".format(user_id))
                        return jsonify({'result': 'Login verification failed'})
                else:
                    return jsonify({'result': 'No face detected in the stored frame'})
            else:
                return jsonify({'result': 'No face detected in the stored frame'})
        else:
            return jsonify({'result': 'Try Again and ensure nobody else is in the background'})


if __name__ == "__main__":
    logging.basicConfig(filename='api.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
    app.run(host='0.0.0.0', port=5005)
