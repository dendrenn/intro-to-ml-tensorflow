"""Module to preprocess images."""

import tensorflow as tf

def process_image(image, image_size, max_image_value):
  """Preprocess image in preparation for input to the model.
  
  Args:
  image (ndarray): image to be processed
  image_size (int): height/width of image in pixels
  max_image_value (int): maximum intensity of any pixel in image

  Returns:
  image (ndarray): processed image
  """
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (image_size, image_size))
  image /= max_image_value
  image = image.numpy()
  return image