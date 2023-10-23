import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam


def preprocessing(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    image = image/255
    return image


features = []
targets = []
for i in ["no", "yes"]:
    collectionImageNames = os.listdir("../augmented_dataset" + "/" + str(i))
    for j in collectionImageNames:
        img = cv2.imread("../augmented_dataset" + "/" + str(i) + "/" + j)
        img = cv2.resize(img, (224, 224))
        features.append(img)
        if i == "no":
            targets.append(0)
        elif i == "yes":
            targets.append(1)
    print("Loading in folder", i)

features = np.array(features)
targets = np.array(targets)
train_features, test_features, train_target, test_target = train_test_split(features, targets, test_size=0.2)

train_features = np.array(list(map(preprocessing, train_features)))
test_features = np.array(list(map(preprocessing, test_features)))

train_features = train_features.reshape(1174, 224, 224, 1)
test_features = test_features.reshape(294, 224, 224, 1)

train_target = to_categorical(train_target)

# step 1------ Specify the architecture
model = Sequential()

# Convolution Layers
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Fully connected Layers
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(2, activation="softmax"))

# Step 2----- Compile the model
model.compile(Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Step 3-----Train the model
fitting = model.fit(train_features, train_target, batch_size=5, epochs=2)
MyModel = model.to_json()
myModel = open("myModel.json",'w')
myModel.write((MyModel))
myModel.close()
myModel_Weights = model.save_weights("myModel_Weights.h5")
predictions = np.argmax(model.predict(test_features), axis=1)

count = 0
for i in range(len(test_target)):
    if predictions[i] == test_target[i]:
        count += 1
print(count)
print("The accuracy is ", count/len(test_target)*100)
