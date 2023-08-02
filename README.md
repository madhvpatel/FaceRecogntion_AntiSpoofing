# FaceRecogntion_AntiSpoofing
The anti_spoofing part works by recieveing a 10 second video as a post request which it analyses frame by frame to detect any spoofing attempts, it is imperative that the video must contain only the face of the registered user and no other faces in the background.

The facial recognition model works by storing 128 embeddings of the face in an array against the USER ID of the user and comparing those embeddings everytime a login attempt is made.
