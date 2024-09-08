from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import os

app = Flask(__name__)
model = MobileNetV2(weights='imagenet')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Upload</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                color: #ffffff;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh; /* Full viewport height */
                overflow: hidden; /* Ensure stars don't create scrollbars */
                position: relative; /* For absolute positioning of stars */
                background: 
                    url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6fOsL8rfAWZ8Z_kIkKBtjzyodNoBU9GLBajOKXYEss_IHkptwad7LoDuAqps4y7fTD-8&usqp=CAU') no-repeat center center; /* Background image */
                background-size: contain; /* Resize the image to fit the container while preserving aspect ratio */
                background-color: #000000; /* Background color to fill any remaining space */
                background-blend-mode: normal; /* Remove any blending effects */
            }
            .container {
                text-align: center;
                background: rgba(0, 0, 0, 0.5); /* Semi-transparent background for contrast */
                padding: 20px;
                border-radius: 10px;
                max-width: 90%; /* Responsive max width */
                margin: auto;
                box-sizing: border-box; /* Ensure padding is included in the width */
                position: relative; /* Ensure container is positioned correctly */
                z-index: 1; /* Ensure container is above stars */
            }
            h1 {
                margin-bottom: 20px;
            }
            input[type="file"] {
                margin-bottom: 10px;
            }
            pre {
                white-space: pre-wrap; /* Preserve formatting in the results display */
                text-align: left;
            }

            /* Falling Stars Animation */
            .star {
                position: absolute;
                width: 5px;
                height: 5px;
                background: white;
                border-radius: 50%;
                animation: fall linear infinite;
            }
            @keyframes fall {
                0% {
                    transform: translateY(-100px);
                    opacity: 1;
                }
                100% {
                    transform: translateY(100vh);
                    opacity: 0;
                }
            }

            /* Randomize star positions and animations */
            .star:nth-child(1) { left: 10%; animation-duration: 2s; }
            .star:nth-child(2) { left: 30%; animation-duration: 3s; }
            .star:nth-child(3) { left: 50%; animation-duration: 2.5s; }
            .star:nth-child(4) { left: 70%; animation-duration: 3.5s; }
            .star:nth-child(5) { left: 90%; animation-duration: 4s; }

            /* Media Query for Smaller Devices */
            @media (max-width: 600px) {
                .container {
                    padding: 10px; /* Adjust padding for smaller screens */
                    max-width: 95%; /* Ensure the container fits on smaller screens */
                }
            }
        </style>
    </head>
    <body>
        <!-- Container for Stars -->
        <div class="star" style="animation-duration: 2s; left: 15%;"></div>
        <div class="star" style="animation-duration: 2.5s; left: 35%;"></div>
        <div class="star" style="animation-duration: 3s; left: 55%;"></div>
        <div class="star" style="animation-duration: 3.5s; left: 75%;"></div>
        <div class="star" style="animation-duration: 4s; left: 85%;"></div>
        
        <div class="container">
            <h1>Upload an Image for Processing</h1>
            <form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Upload Image">
            </form>
            <h2>Results:</h2>
            <pre id="results"></pre>
        </div>
        <script>
            document.getElementById('uploadForm').addEventListener('submit', function(event) {
                event.preventDefault(); // Prevent the form from submitting the traditional way

                const formData = new FormData(this);
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    const results = document.getElementById('results');
                    results.textContent = JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        img, img_array = load_and_preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_classes = decode_predictions(predictions, top=3)[0]

        result = [{'label': label, 'score': float(score)} for (_, label, score) in predicted_classes]
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
