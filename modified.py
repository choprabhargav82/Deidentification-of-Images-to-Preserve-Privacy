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
def modified_gaussian_filter(img, sigma, gamma, a):
    # create the filter kernel
    size = int(2*np.ceil(3*2*sigma)+1)
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2*np.pi*(2*sigma)**2)) * (1 - np.exp(-(x**2 + y**2) / (2*(2*sigma)**2))) * np.exp(-(x**2 + y**2) / (2*gamma**2))
    kernel /= np.sum(kernel)

    # convert the input image to grayscale using the new function
    grayscale = convert_to_grayscale(img)

    # apply the filter to the grayscale image
    blurred = cv2.filter2D(grayscale, -1, kernel)

    # generate custom noise
    noise = a * (np.sin(np.arange(grayscale.shape[0])[:, None] * 0.5) + np.cos(np.arange(grayscale.shape[1])[None, :] * 0.5))
    noise /= np.max(noise)

    # add noise to the blurred image
    blurred_noisy = blurred + noise

    # normalize the pixel values to be between 0 and 1
    blurred_noisy = np.clip(blurred_noisy, 0, 1)

    return blurred_noisy

# load an example image
img = cv2.imread('example.jpg')

# apply the modified Gaussian filter with sigma = 5, gamma = 10, and a = 0.05
blurred_noisy = modified_gaussian_filter(img, sigma=5, gamma=10, a=0.05)

# display the original and blurred images side by side
cv2.imshow('Original', img)
cv2.imshow('Blurred and Noisy', blurred_noisy)
cv2.waitKey(0)
cv2.destroyAllWindows()
