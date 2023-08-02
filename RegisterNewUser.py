import os.path
import datetime
import pickle
import cv2
import face_recognition
import pymongo


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        # Connect to the MongoDB database
        self.client = pymongo.MongoClient("mongodb+srv://madhvptel:wabalabadubdub@cluster0.ma647k2.mongodb.net/")
        self.db = self.client["face_recognition_db"]
        self.collection = self.db["users"]

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def process_webcam():
        v=cv2.VideoCapture(0)
        p=v.frame(0)
        
        
    def register_new_user(self):
        print('Please, input user id:')
        user_id = input()

        self.process_webcam()
        embeddings = face_recognition.face_encodings(self.most_recent_capture_arr)

        if not embeddings:
            print('No face detected in the captured image.')
            return

        file_path = os.path.join(self.db_dir, '{}.pickle'.format(user_id))
        with open(file_path, 'wb') as file:
            pickle.dump(embeddings[0], file)

       
        user_data = {"user_id": user_id, "embeddings": embeddings[0].tolist()}
        self.collection.insert_one(user_data)

        print('Success! User with ID {} was registered successfully.'.format(user_id))

    def recognize_user(self, query_embeddings, user_id):
        user_data = self.collection.find_one({"user_id": user_id})
        if user_data:
            embeddings = user_data["embeddings"]
            if face_recognition.compare_faces([embeddings], query_embeddings)[0]:
                return user_data["user_id"]
            else:
                print("User not recognized.")
        else:
            print("User not found.")
        return None



if __name__ == "__main__":
    app = App()
    app.register_new_user()
