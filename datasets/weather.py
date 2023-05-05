import numpy as np
import cv2
from PIL import Image

### RAIN DATA AUGMENTATION ###

def generate_random_lines(imshape, slant, drop_length):
    drops = []
    num_drops = 800
    x = np.random.randint(0, imshape[1], size=(num_drops,))
    y = np.random.randint(0, imshape[0]-drop_length, size=(num_drops,))
    if slant < 0:
        x += slant
    else:
        x -= slant
    drops = np.column_stack((x, y)).tolist()
    return drops

def add_rain(pil_image):
    # Convert PIL image to OpenCV image
    open_cv_image = np.array(pil_image)
    image = open_cv_image[:, :, ::-1].copy()
    
    # Generate random lines using numpy functions
    imshape = image.shape
    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 20
    drop_width = 2
    drop_color = (200, 200, 200)
    x = np.arange(0, imshape[1], 10)
    y = np.random.randint(0, imshape[0], size=len(x))
    rain_drops = np.column_stack((x, y))
    
    # Draw rain drops using OpenCV function
    for rain_drop in rain_drops:
        cv2.line(image, (rain_drop[0], rain_drop[1]), 
                 (rain_drop[0]+slant, rain_drop[1]+drop_length), 
                 drop_color, drop_width)
    
    # Blur the image using OpenCV function
    image = cv2.blur(image, (7, 7))
    
    # Adjust the brightness of the image
    brightness_coefficient = 0.7
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image[:, :, 1] = image[:, :, 1] * brightness_coefficient
    image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    
    # Convert OpenCV image to PIL image and return
    im_pil = Image.fromarray(image)
    return im_pil

## OTHER WEATHER AUGMENTATION ## 
