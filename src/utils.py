import cv2
import numpy as np


def pre_processing(image, width, height):
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    cv2.imshow('image', image)
    return image[None, :, :].astype(np.float32)


def noise(image, noise_level):
    image2 = image.copy()
    for i in range(noise_level):
        x = np.random.randint(0, image.shape[1])
        y = np.random.randint(0, image.shape[2])
        image2[0, x, y] = 255 - image2[0, x, y]
    cv2.imshow(f'noise_{noise_level}', image2[0])
    return image2
