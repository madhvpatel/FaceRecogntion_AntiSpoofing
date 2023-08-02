import cv2
import time
import requests
import os

def record_video():
    video_capture = cv2.VideoCapture(0)
    frames = []
    for _ in range(100):  # Record 10 seconds (assuming 10 frames per second)
        ret, frame = video_capture.read()
        if ret:
            frames.append(frame)
            time.sleep(0.1)  # Adjust this value to control the video recording frame rate

    video_capture.release()
    return frames

def send_video_to_api(video_frames, user_id):
    temp_video_path = 'temp_video.mp4'
    height, width, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for frame in video_frames:
        video_writer.write(frame)

    video_writer.release()

    with open(temp_video_path, 'rb') as video_file:
        files = {'file': ('video.mp4', video_file, 'video/mp4')}
        api_url = 'http://127.0.0.1:5005/classify_video'  # Replace with the actual API URL
        data = {'user_id': user_id}  # Add the user_id to the request data
        response = requests.post(api_url, files=files, data=data)

    # Delete the temporary video file
    video_file.close()
    cv2.destroyAllWindows()
    os.remove(temp_video_path)

    return response

if __name__ == "__main__":
    # Get user input for user id
    user_id = input("Enter user id: ")

    print("Recording video...")
    video_frames = record_video()

    print("Sending video to API...")
    api_response = send_video_to_api(video_frames, user_id)

    print(api_response.text)
