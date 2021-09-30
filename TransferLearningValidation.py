 #!/usr/bin/env python
# coding: utf-8

# # Image Recognition Using Tensorflow

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
import queue
import threading
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys
tf.compat.v1.disable_eager_execution()
import glob
import time

class image_recognition_using_label_data(object):
    
    def __init__(self):
        # Initialize
        self.label = None
        self.graph = None
        self.load_label("/tmp/retrain_tmp/output_labels.txt")
        
        self.graph_def = None
        self.image_reader = None
        self.caster = None
        self.dimention = None
        self.recognition = None
        self.image_height = 299
        self.image_width = 299
        self.image_mean = 0
        self.image_standard_deviation = 255
        self.detection = None
    
    def image_graph(self, graph):
        self.graph = graph
        
    @property
    def training_graph(self):
        return self.graph
    
    @property
    def training_label(self):
        return self.label
    
    # Loading the Training Machine Learning Model
    def load_label(self, file_label):
        self.label = []
        ascii_lines = tf.io.gfile.GFile(file_label).readlines()
        for size in ascii_lines:
            self.label.append(size.rstrip())
            
    def load_graph(self, file_model):
        self.graph = tf.Graph()
        self.graph_def = tf.compat.v1.GraphDef()
        
        with open(file_model, "rb") as opened:
            self.graph_def.ParseFromString(opened.read())
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def)
        return self.graph
    
    # Image detection
    def image_detection(self, queue, session, bytes_image, image_path, image_input, image_output):
        self.detect_image_bytes(bytes_image)
        result = session.run(image_output.outputs[0], {image_input.outputs[0]: self.detection})
        result = np.squeeze(result)
        
        prediction = result.argsort()[-5:][::-1][0]
        queue.put( {'image_path':image_path, 'prediction':self.label[prediction].title(), 'percent':result[prediction]} )
        
        
    def detect_image_bytes(self, bytes_image):
        image_reader = tf.image.decode_png(bytes_image, channels=3, name="png_reader")
        caster = tf.cast(image_reader, tf.float32)
        dimention = tf.expand_dims(caster, 0)
        
        image_resize = tf.image.resize(dimention,[self.image_height, self.image_width])
        normalize = tf.divide(tf.subtract(image_resize, [self.image_mean]), [self.image_standard_deviation])
        session = tf.compat.v1.Session()
        self.detection = session.run(normalize)
    

def detecting_images(imageList, valueList):
    image_recognition = image_recognition_using_label_data()
    graph = image_recognition.load_graph('/tmp/retrain_tmp/output_graph.pb')
    image_recognition.image_graph(graph)
    
    image_input = image_recognition.training_graph.get_operation_by_name("import/Placeholder")
    image_output = image_recognition.training_graph.get_operation_by_name("import/final_result")
    session = tf.compat.v1.Session(graph=graph)
    
    # detection_images is the directory
    #detect_image = os.listdir('detection_images')
    detect_image = imageList
    queue_image = queue.Queue()
    
    for image in detect_image:
        image_path = image
        #image_path = '{}/{}'.format('detection_images', image)
        print('Image Processing {}'.format(image_path))
        
        # Time laps before processing another image
        while len(threading.enumerate()) > 10:
            time.sleep(0.0001)
        
        # Reading images as byte objects as it is expecting png file
        bytes_image = open(image_path, "rb").read()
        threading.Thread(target = image_recognition.image_detection, args = (queue_image, session, bytes_image, image_path, image_input, image_output)).start()
        
    print('Waiting For Threads to Finish...')
    while queue_image.qsize() < len(detect_image):
        time.sleep(0.001)
        
    prediction_accuracy = [queue_image.get() for i in range(queue_image.qsize())]
        
    correct = 0
    total_predictions = len(prediction_accuracy)
    for prediction in prediction_accuracy:
        print("Predicted {image_path} is a {prediction} with {percent:.2%} Accuracy".format(**prediction))
        filename = "{image_path}".format(**prediction)
        index = detect_image.index(filename)
        print("Index: ", index)

        filePath = os.path.dirname(filename)
        className = os.path.basename(filePath)

        print(valueList[index])
        print(type(valueList[index]))
        if(valueList[index] == "-1"):
            if(className.casefold() != "{image_path}".format(**prediction)):
                print("Success Non-Match!")
                correct = correct + 1
        else:
            if(className.casefold() == "{image_path}".format(**prediction)):
                print("Success Match!")
                correct = correct + 1

    precision = (correct / total_predictions) * 100
    print("Precision: {}".format(precision))
    return precision

def getImagesInSet(imageSetPath):
    imageDirectory = "ResizedPNGImages"
    #Open ImageSet File
    imageSetFile = open(imageSetPath, 'r')
    imgsInSet = []
    valuesInSet = []

    print("Starting to load subset of images in: ",imageSetPath)

    #Read in all Images in the ImageSet
    while (True):
        line = imageSetFile.readline().splitlines()
        #If end line - exit loop
        if not line:
            break
        #Convert the file name to a clean path to the associated file
        cleanLine = str(line)[1:-1].replace('\'', '')
        splitLine = cleanLine.split(None, 1)
        file = splitLine[0]
        val = splitLine[1]

        cleanPath = os.path.join(imageDirectory,'*/{}.png'.format(file))
        fullPath = glob.glob(cleanPath)
        cleanFullPath = str(fullPath)[1:-1].replace('\'', '')
        imgsInSet.append(cleanFullPath)

        valuesInSet.append(val)
    return imgsInSet, valuesInSet

if __name__ == "__main__":
    #detecting_images()

    resultsFile = time.strftime("%Y%m%d-%H%M%S.txt")
    with open(resultsFile, 'a+') as results:
        imageSetsDir = "ImageSets/Main"
        for file in os.listdir(imageSetsDir):
            if file.endswith("_val.txt"):
                filePath = os.path.join(imageSetsDir, file)
                print("Next ImageSet: ", os.path.join(imageSetsDir, file))
                imagePaths, values = getImagesInSet(filePath)
                print("Length of set: ",len(imagePaths),len(values))
                precision = detecting_images(imagePaths, values)
                results.write("{}: Precision:{}\n".format(file, precision))
                #break



