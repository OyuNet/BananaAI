import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(image_path, model_path='banana_classifier_model.keras'):
    model = tf.keras.models.load_model(model_path)
    
    img = image.load_img(image_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Tek bir resim için batch boyutu ekleyin

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['Banana', 'Not Banana']
    print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence.")

predict_image('/Users/arda/Downloads/images-2.jpeg')