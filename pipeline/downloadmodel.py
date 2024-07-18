# download_model.py
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import os

def save_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-1].output)
    current_directory = os.getcwd()
    model_path = os.path.join(current_directory, 'vgg16_model.keras')
    model.save(model_path)

if __name__ == "__main__":
    save_vgg16_model()
