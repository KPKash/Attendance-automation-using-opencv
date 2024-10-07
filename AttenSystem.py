import cv2
import os
import numpy as np
from datetime import datetime
import time

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to get the names of people in the dataset
def get_people_names(dataset_path):
    return [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

# Function to preprocess the face image
def preprocess_face(face, size=(100, 100)):
    face = cv2.resize(face, size)
    face = cv2.equalizeHist(face)
    return face

# Function to augment data
def augment_image(image):
    # Flip horizontally
    yield cv2.flip(image, 1)
    
    # Rotate slightly
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), 10, 1)
    yield cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), -10, 1)
    yield cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# Function to load images from the dataset
def load_dataset(dataset_path):
    people = get_people_names(dataset_path)
    faces = []
    labels = []
    for i, person in enumerate(people):
        person_dir = os.path.join(dataset_path, person)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                face = preprocess_face(image)
                faces.append(face)
                labels.append(i)
                
                # Add augmented images
                for aug_face in augment_image(face):
                    faces.append(aug_face)
                    labels.append(i)
            else:
                print(f"Failed to load image: {image_path}")
    
    print(f"Total images loaded (including augmented): {len(faces)}")
    print(f"Number of people: {len(people)}")
    for person in people:
        print(f"  - {person}: {labels.count(people.index(person))} images")
    
    return faces, labels, people

# Function to recognize faces
def recognize_face(face, model, people):
    face = preprocess_face(face)
    label, confidence = model.predict(face)
    print(f"Predicted label: {label}, Person: {people[label]}, Confidence: {confidence}")
    if confidence < 140:  # Increased threshold
        return people[label], confidence
    return None, confidence

# Updated mark_attendance function
def mark_attendance(name):
    date = datetime.now().strftime('%Y-%m-%d')
    attendance_file = 'attendance.csv'
    
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write("Name,Status,Date\n")
    
    with open(attendance_file, 'r+') as f:
        lines = f.readlines()
        names_present = [line.split(',')[0] for line in lines]
        if name not in names_present:
            f.write(f'{name},P,{date}\n')
            print(f"Marked attendance for {name}")
        else:
            print(f"{name} already marked present")
    
    # Debug: Print the contents of the CSV file
    print("Current attendance record:")
    with open(attendance_file, 'r') as f:
        print(f.read())

# Main function
def main():
    dataset_path = 'dataset'
    faces, labels, people = load_dataset(dataset_path)
    
    if len(faces) == 0:
        print("No faces loaded from the dataset. Please check your dataset directory.")
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))

    cap = cv2.VideoCapture(1)
    start_time = time.time()
    duration = 30

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            name, confidence = recognize_face(face, model, people)
            if name:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                mark_attendance(name)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f"Unknown ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()