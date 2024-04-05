import pickle
from pathlib import Path
from flask import Flask, request, jsonify
import face_recognition

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

app = Flask(__name__)

# Load face encodings
def load_face_encodings(encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> dict:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    return loaded_encodings

# Encode known faces
def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

@app.route('/train', methods=['POST'])
def train_model():
    encode_known_faces()  # Train the model
    return jsonify({"message": "Model trained successfully!"})

if __name__ == '__main__':
    app.run(debug=True)
