import cv2
import numpy as np

cv2.cuda.setDevice(0)

image_path = 'D:\Documentos\PythonProjects\media-filter\\ferris.jpg'
cv2.cuda.setDevice(0)

img = cv2.imread(image_path)
kernel = np.ones((3, 3), np.float32) / 9

filtered_img = cv2.cuda.filter2D(img, -1, kernel)

cv2.imwrite("filtered_img.jpg", filtered_img)

