from PIL import Image
import numpy as np

#"Importing" image to Python
def defImage(imageString):
    im=Image.open(imageString)
    return im

#Making an image usable in black and white
def binarize(im, thresh, startx, starty, endx, endy):
    (width, height) = im.size
    for x in range(startx, endx):
        for y in range(starty, endy):
            (red, green, blue) = im.getpixel((x, y))
            luminance=int(0.21*red) + int(0.72*green) + int(0.07*blue)
            if luminance < thresh:
                im.putpixel((x,y), (0,0,0))
            if luminance > thresh:
                im.putpixel((x,y), (255,255,255))
  
#Adds an image to a numpy array
def add_image(file) :
    img = Image.open(file)
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

'''
#Reading an image to a numpy array
import cv2
im=cv2.imread("6.jpg")
print(type(im))
'''
