from flask import Flask, request
import numpy as np
import os
import cv2
from keras.models import load_model

# Configuration
model_file = "models/cat_dog_classifier.hdf5"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model
model = load_model(model_file)


@app.route('/', methods=['POST'])
def index():
    try:
        if request.method == 'POST':
            image = request.files['file']
            if image:
                # Save the image to a temporary location
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                image.save(path_to_save)
                frame = cv2.imread(path_to_save)

                # Preprocess the image
                frame = cv2.resize(frame, dsize=(150, 150))
                # Convert to tensor
                frame = np.expand_dims(frame, axis=0)   # Used for single image
                # Predict the class
                prediction_prob = model.predict(frame)[0][0]

                # Convert to class label
                if prediction_prob > 0.5: # 0-0.5 = cat, 0.5-1 = dog
                    prediction = "dog"
                else:
                    prediction = "cat"

                return prediction
        else:
            return "We only accept POST with image files."
        
    except Exception as ex:
        print(ex)
        return "ERROR: " + str(ex)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    