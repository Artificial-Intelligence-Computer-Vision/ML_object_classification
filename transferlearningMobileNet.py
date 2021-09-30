import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import glob
import cv2
import time


testImageDir = r'aeroplane_val'
trainingValImageDir = r'aeroplane_train'
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(testImageDir, image_size=(224,224), label_mode='categorical')
train_val_dataset = tf.keras.preprocessing.image_dataset_from_directory(trainingValImageDir, image_size=(224,224), label_mode='categorical')
train_dataset = train_val_dataset.take(143)
val_dataset = train_val_dataset.skip(143)
print("TRAIN DATASET:",len(train_dataset))
print("VAL DATASET:",len(val_dataset))

class_names = np.array(sorted([item for item in os.listdir(trainingValImageDir)]))
print(class_names)

#test_dataset.cache()
val_dataset.cache()
train_dataset.cache()

feature_extractor = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
#feature_extractor = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

feature_extractor_layer = hub.KerasLayer(feature_extractor, input_shape=(224,224,3))
feature_extractor_layer.trainable = False

#0.03
model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(20, activation='softmax')
])

model.summary()

model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

result=model.evaluate(test_dataset)

resultsFile = time.strftime("%Y%m%d-%H%M%S-AeroplaneResults.txt")
with open(resultsFile, 'a+') as results:
    results.write("History: {}\n".format(history))
    results.write("Results: [Loss, Acc]: {}\n".format(result))
aeroplane_index = np.where(class_names == 'aeroplane')[0][0]
print(aeroplane_index)


#I think there is an issue with this, but we aren't using these results at the moment
predictionsFile = time.strftime("%Y%m%d-%H%M%S-AeroplanePredictions.txt")
with open(predictionsFile, 'a+') as results:
    allImagePath = os.path.join(testImageDir,'*/*.png')
    for filename in glob.glob(allImagePath):
        strip_filename = os.path.splitext(os.path.basename(filename))[0]
        temp_filePath = os.path.dirname(filename)
        temp_className = os.path.basename(temp_filePath)
        #print(temp_className)

        img = cv2.imread(filename)
        img = cv2.resize(img, (224,224))
        img = np.reshape(img, [1,224,224,3])
        pred = model.predict(img)

        print("Aeroplane Confidence: ", pred[0][aeroplane_index])
        print("Actual Label: %s" % temp_className)
        print("Predicted Label: %s" % class_names[np.argmax(pred)])
        results.write("{} {} {} {}\n".format(strip_filename,pred[0][aeroplane_index], temp_className, class_names[np.argmax(pred)]))

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'r', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation Loss Graph')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend()
plt.show()
