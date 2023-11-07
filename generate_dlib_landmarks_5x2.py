import dlib
import cv2
import os
import numpy as np

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")  # You need to download this file

# Define the folder containing images
images_folder = r"datasets\examples" # filelocation may need to be changed
detections_folder = r"datasets\examples\detections" # filelocation may need to be changed

for filename in os.listdir(images_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)

        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Assuming that you want landmarks for the eyes and mouth only, which are typically
            # the first and second points (0-based index) of each eye and the center of the mouth
            selected_landmarks_indices = [36, 39, 42, 45, 48, 54]  # Indices of landmarks of interest
            selected_landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in selected_landmarks_indices]

            # Save landmarks to .txt file with the same basename as the image
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(detections_folder, txt_filename)
            with open(txt_path, 'w') as f:
                for (x, y) in selected_landmarks:
                    f.write(f"{x} {y}\n")

print("Landmark detection is done!")
