import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image

dataset_dir = os.path.join('coins', 'data')
cat_to_name_path = os.path.join(dataset_dir, 'cat_to_name.json')

with open(cat_to_name_path, 'r') as f:
    cat_to_name = json.load(f)

model = tf.keras.models.load_model('coin_classifier_final.h5')

def predict_coin(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))
    
    prediction = model.predict(img_array)
    predicted_class = prediction.argmax()
    predicted_label = list(cat_to_name.values())[predicted_class]
    
    return predicted_label

if __name__ == '__main__':
    image_path = input("Enter the path to the coin image: ")
    try:
        label = predict_coin(image_path)
        print(f"The predicted label is: {label}")
    except Exception as e:
        print(f"Error: {e}")