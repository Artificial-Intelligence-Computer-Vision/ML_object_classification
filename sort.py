import os
from PIL import Image
from csv import DictReader

def sortImages(directory):

    with open('trainLabels.csv', 'r') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        for row in csv_dict_reader:
            filename = row['id'] + ".png"
            className = row['label']
            print(filename, className)
            classDir = os.path.join(directory, className)
            if not os.path.exists(classDir):
                os.makedirs(classDir)
            newImageName = os.path.join(classDir, filename)
            filePath = os.path.join(directory, filename)
            os.rename(filePath, newImageName)
            print("NEW FILENAME: ", newImageName)

sortImages("training_images")
