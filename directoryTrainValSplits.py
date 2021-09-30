import os
from PIL import Image
import shutil
import glob

def fillOutDir(imageSetPath, newDir):
    imageDirectory = "ResizedPNGImages"
    #Open ImageSet File
    imageSetFile = open(imageSetPath, 'r')
    #imgsInSet = []
    #valuesInSet = []

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

        temp_filePath = os.path.dirname(cleanFullPath)
        temp_className = os.path.basename(temp_filePath)

        newClassPath = os.path.join(newDir,'{}'.format(temp_className))
        if not os.path.exists(newClassPath):
            os.makedirs(newClassPath)
        newPath = os.path.join(newDir,'{}/{}.png'.format(temp_className,file))
        print("NewPath: ", newPath)
        shutil.copy(cleanFullPath, newPath)
        #imgsInSet.append(cleanFullPath)

        #valuesInSet.append(val)
    #return imgsInSet, valuesInSet


imageSetsDir = "ImageSets/Main"
for file in os.listdir(imageSetsDir):
    if file.endswith("aeroplane_val.txt"):
        #Make directory
        newDir = "aeroplane_val"
        if not os.path.exists(newDir):
            os.makedirs(newDir)

        filePath = os.path.join(imageSetsDir, file)
        print("Loading: ", filePath)
        fillOutDir(filePath, newDir)

    if file.endswith("aeroplane_train.txt"):
        #Make directory
        newDir = "aeroplane_train"
        if not os.path.exists(newDir):
            os.makedirs(newDir)

        filePath = os.path.join(imageSetsDir, file)
        print("Loading: ", filePath)
        fillOutDir(filePath, newDir)
