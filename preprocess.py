import io
import cv2
import base64 
import numpy as np
from PIL import Image

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    strImage = base64_string.replace("data:image/jpeg;base64,", "")
    imgdata = base64.b64decode(strImage)
    return np.fromstring(imgdata, dtype=np.uint8)

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.imdecode(image, flags=cv2.IMREAD_COLOR)

def decodeBase64Image(base64_string):
    image = stringToImage(base64_string)
    cvImage = toRGB(image)
    return cvImage