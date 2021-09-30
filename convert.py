from PIL import Image
import os
import cv2

def convertImages2PNG(directory):
    imagesDir = os.path.join(directory, 'JPEGImages')

    pngDir = os.path.join(directory, 'PNGImages')
    if not os.path.exists(pngDir):
        os.makedirs(pngDir)

    for directory in os.listdir(os.path.join(imagesDir)):
        currentDir = os.path.join(imagesDir, directory)
        for filename in os.listdir(os.path.join(imagesDir, directory)):
            jpgFileName = os.path.join(currentDir, filename)
            filenameNoType = os.path.splitext(filename)[0]
            print(jpgFileName)
            image = Image.open(jpgFileName)

            classDir = os.path.join(pngDir, directory)
            if not os.path.exists(classDir):
                os.makedirs(classDir)

            pngFilePath = os.path.join(classDir, (filenameNoType + ".png"))
            print(pngFilePath)
            image.save(pngFilePath)

def resizeImages(directory):
    pngDir = os.path.join(directory, 'PNGImages')

    resizeDir = os.path.join(directory, 'ResizedPNGImagesSmall')

    if not os.path.exists(resizeDir):
        os.makedirs(resizeDir)

    for directory in os.listdir(os.path.join(pngDir)):
        currentDir = os.path.join(pngDir, directory)
        for filename in os.listdir(currentDir):
            pngFileName = os.path.join(currentDir, filename)
            print(pngFileName)

            image = cv2.imread(pngFileName)
            res = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

            classDir = os.path.join(resizeDir, directory)
            if not os.path.exists(classDir):
                os.makedirs(classDir)

            pngFilePath = os.path.join(classDir, filename)
            print(pngFilePath)
            cv2.imwrite(pngFilePath, res)



directory1 = r'PascalVOC2012/voc2012/VOC2012'
directory2 = r'PascalVOC2012/VOC2012'

#convertImages2PNG(directory1)
#convertImages2PNG(directory2)

resizeImages(directory1)
resizeImages(directory2)

