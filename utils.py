import numpy as np
import cv2
import requests
import albumentations as A

from config import SIZE


def get_image(file_url, save_path):
    arr = np.asarray(bytearray(requests.get(file_url).content), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    cv2.imwrite(save_path, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


transforms = A.Compose([
    A.Resize(SIZE, SIZE),
    A.Normalize()
])


def preprocess_img(img):
    img = transforms(image=img)['image']
    img = np.transpose(img, (2, 0, 1))
    return img
