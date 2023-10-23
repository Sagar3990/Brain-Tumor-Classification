import numpy as np
from keras.models import model_from_json
import cv2
from keras.utils import load_img,img_to_array


def preprocessing(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    image = image/255
    return image


model_path = open("myModel.json", "r")
loaded_model = model_path.read()
loaded_model = model_from_json(loaded_model)
loaded_model.load_weights("myModel_Weights.h5")
test_image_path = r"../augmented_dataset/no/image_0_4.jpeg"
test_image = load_img(test_image_path, target_size=(224, 224, 1))
test_image = preprocessing(img_to_array(test_image))
test_image = np.expand_dims(test_image, axis=0)
predictions = loaded_model.predict(test_image)
print(predictions.argmax(axis=1))
