# fertestcustom.py

from keras.models import model_from_json
import numpy as np
import cv2

def load_emotion_model(json_path, weights_path):
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model

def predict_emotion(model, image_path, labels):
    image_path = str(image_path)  # Convert image_path to a string if it's not already a string
    # print(image_path)
    full_size_image = cv2.imread(image_path)
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3, 10)

    emotions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        yhat = model.predict(cropped_img)
        emotion_label = labels[int(np.argmax(yhat))]
        emotions.append(emotion_label)
        print(emotions)
    return emotions if emotions else []
