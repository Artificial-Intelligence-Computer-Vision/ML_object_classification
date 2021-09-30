import os
import csv
import xml.etree.ElementTree as ET

def parseDir(directory):
    annotationDir = os.path.join(directory, 'Annotations')
    imagesDir = os.path.join(directory, 'JPEGImages')
    csvFile = os.path.join(directory, 'simpleAnnotations.csv')
    with open(csvFile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['FileName','Class'])
        for filename in os.listdir(annotationDir):
            if filename.endswith(".xml"):
                currentFile = os.path.join(annotationDir, filename)
                print(currentFile)
                tree = ET.parse(currentFile)
                root = tree.getroot()
                imageName = root.find('filename').text
                obj = root.find('object')
                className = obj.find('name').text
                currentImage = os.path.join(imagesDir, imageName)
                newDir = os.path.join(imagesDir, className)
                newImageName = os.path.join(newDir, imageName)
                if not os.path.exists(newDir):
                    os.makedirs(newDir)
                os.rename(currentImage, newImageName)
                writer.writerow([imageName,className])
            else:
               continue

directory1 = r'PascalVOC2012/voc2012/VOC2012'
directory2 = r'PascalVOC2012/VOC2012'
parseDir(directory1)
parseDir(directory2)
