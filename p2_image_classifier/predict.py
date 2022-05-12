"""Module to make predictions on what flowers are most likely present in images."""

import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from preprocess import process_image

# Define Constants
IMAGE_SIZE = 224
MAX_IMAGE_VALUE = 255

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help="Path to the image whose class is to be predicted")
parser.add_argument('model', help="Path to the saved Keras model")
parser.add_argument('--top_k', help="The number of most likely classes to return", type=int, default=3)
parser.add_argument('--category_names', help="Path to JSON file mapping labels to flower names")
args = parser.parse_args()

# Load Image 
im = Image.open(args.image_path)
image = np.asarray(im)
image = process_image(image, IMAGE_SIZE, MAX_IMAGE_VALUE)
image = np.expand_dims(image, 0)

# Load Model
loaded_model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})

# Make Predictions
predictions = loaded_model.predict(image)
sorted_indices = np.argsort(-predictions)
probs = list(predictions[0][sorted_indices[0][:args.top_k]])
classes = list((sorted_indices[0][:args.top_k] + 1).astype(str))

if args.category_names:
  with open(args.category_names, 'r') as f:
    class_names = json.load(f)
  classes = [class_names[key] for key in classes]

# Display Results
print(f"probs = {probs}")
print(f"classes = {classes}")