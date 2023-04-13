import numpy as np
import cv2

# define a function to convert an image to grayscale
def convert_to_grayscale(img):
    # convert the input image to grayscale using cv2.cvtColor
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # normalize the pixel values to be between 0 and 1 using numpy
    grayscale = grayscale.astype(np.float32) / 255.0
    return grayscale

# define the modified Gaussian filter function
def filter(img, sigma, gamma):
    # create the filter kernel
    size = int(2*np.ceil(3*sigma)+1)
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2*np.pi*sigma*2)) * (1 - np.exp(-(x**2 + y**2) / (2*sigma**2))) * np.exp(-(x**2 + y**2) / (2*gamma**2))
    kernel /= np.sum(kernel)

    # convert the input image to grayscale using the new function
    grayscale = convert_to_grayscale(img)

    # apply the filter to the grayscale image
    blurred = cv2.filter2D(grayscale, -1, kernel)

    return blurred

# load an example image
img = cv2.imread('example.jpg')

# apply the modified Gaussian filter with sigma = 5 and gamma = 10
blurred = filter(img, sigma=5, gamma=10)

# display the original and blurred images side by side
cv2.imshow('Original', img)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
