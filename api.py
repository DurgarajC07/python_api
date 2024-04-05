import mysql.connector
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from io import BytesIO
import pickle
from collections import Counter
import face_recognition
from pathlib import Path

app = Flask(__name__)

# MySQL database connection
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="detection_db"
    )
    mycursor = mydb.cursor()
except mysql.connector.Error as err:
    print("Error in database connection:", err)

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("yolov3.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load face encodings
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

def load_face_encodings(encodings_location):
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)
    return loaded_encodings

face_encodings = load_face_encodings(DEFAULT_ENCODINGS_PATH)

# Function to detect objects
def detect_objects(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            x, y, w, h = boxes[i]
            detected_objects.append({
                "label": label,
                "confidence": confidence,
                "x": x,
                "y": y,
                "width": w,
                "height": h
            })
            try:
                sql = "INSERT INTO object (label, confidence, width, height, x, y) VALUES (%s, %s, %s, %s, %s, %s)"
                val = (label, confidence, w, h, x, y)
                mycursor.execute(sql, val)
                mydb.commit()
            except mysql.connector.Error as err:
                print("Error:", err)

    return detected_objects

# Recognize faces in the image
def recognize_faces(image, face_encodings):
    input_image = face_recognition.load_image_file(BytesIO(image))
    input_face_locations = face_recognition.face_locations(input_image)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    recognition_results = []

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name, accuracy = _recognize_face(unknown_encoding, face_encodings)
        if not name:
            name = "Unknown"
        recognition_results.append({"bounding_box": bounding_box, "name": name, "accuracy": accuracy})

    return recognition_results

# Recognize the face
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    if votes:
        max_votes = max(votes.values())
        total_votes = sum(votes.values())
        accuracy = max_votes / total_votes
        recognized_name = votes.most_common(1)[0][0]
        return recognized_name, accuracy
    return None, 0.0

# API endpoint for object detection and face recognition
@app.route('/detect', methods=['POST'])
def detect():
    url = request.json.get('image_url')
    image_data = requests.get(url).content

    # Detect objects
    detected_objects = detect_objects(image_data)

    # Recognize faces
    recognition_results = recognize_faces(image_data, face_encodings)

    # Process recognition results
    for result in recognition_results:
        bounding_box = result["bounding_box"]
        name = result["name"]
        accuracy = result["accuracy"]
        # Extract height, width, x, and y coordinates from bounding box
        top, right, bottom, left = bounding_box
        height = bottom - top
        width = right - left
        x = left
        y = top
        detected_objects.append({"label": name, "confidence": accuracy, "x": x, "y": y, "width": width, "height": height})
        try:
            sql = "INSERT INTO object (label, confidence, width, height, x, y) VALUES (%s, %s, %s, %s, %s, %s)"
            val = (name, accuracy, width, height, x, y)
            mycursor.execute(sql, val)
            mydb.commit()
        except mysql.connector.Error as err:
            print("Error:", err)
    return jsonify(detected_objects)

if __name__ == '__main__':
    app.run(debug=True)
