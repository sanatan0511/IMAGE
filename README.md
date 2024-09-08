import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from io import BytesIO

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array

def process_image(img_path):
    """Process and display the image with background."""
    # Load and preprocess the image
    img, img_array = load_and_preprocess_image(img_path)
    
    # Make predictions
    predictions = model.predict(img_array)
    predicted_classes = decode_predictions(predictions, top=3)[0]
    
    # Load the background image
    background_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6fOsL8rfAWZ8Z_kIkKBtjzyodNoBU9GLBajOKXYEss_IHkptwad7LoDuAqps4y7fTD-8&usqp=CAU'
    response = requests.get(background_url)
    background_img = Image.open(BytesIO(response.content))
    
    # Create a new figure with the background image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(background_img, aspect='auto')  # Display the background image
    ax.axis('off')  # Hide axes
    
    # Overlay the processed image
    ax_img = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Define axes for the processed image
    ax_img.imshow(img)
    ax_img.axis('off')  # Hide axes for processed image
    
    # Display the predictions on top of the background
    for i, (imagenet_id, label, score) in enumerate(predicted_classes):
        ax.text(10, 10 + i * 30, f"{label}: {score:.2f}", color='white', backgroundcolor='black', fontsize=12)
    
    plt.show()

def upload_image():
    """Handle the image upload and processing."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    
    # Process the uploaded image
    process_image(file_path)

# Create the main window
root = tk.Tk()
root.title("Image Upload and Processing")

# Create and place widgets
upload_button = tk.Button(root, text="Upload and Process Image", command=upload_image)
upload_button.pack(pady=20)

# Run the application
root.mainloop()
