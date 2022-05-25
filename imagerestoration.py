from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import *
plt.style.use('seaborn')
 
image = cv2.imread('/content/drive/MyDrive/colored noise 2 (1).jpg')
dst = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)
 
row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Elephant')
axs[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
axs[1].set_title('Fast Means Denoising')
plt.show()

import cv2 
import numpy as np

img = cv2.imread("/content/drive/MyDrive/colored noise 2 (1).jpg")

denoise_1 = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21) 
denoise_2 = cv2.fastNlMeansDenoisingColored(img,None,5,5,7,21) 
denoise_3 = cv2.fastNlMeansDenoisingColored(img,None,15,15,7,21)

cv2.imwrite('image_1.png', denoise_1) 
cv2.imwrite('image_2.png', denoise_2) 
cv2.imwrite('image_3.png', denoise_3)

from google.colab.patches import cv2_imshow

cv2_imshow(img)

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/content/drive/MyDrive/colored noise 2 (1).jpg')

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()

def show_image_contour(image, contours):
    plt.figure()
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
    plt.imshow(image, interpolation='nearest', cmap='gray_r')
    plt.title('Contours')
    plt.axis('off')

def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis('off')

from skimage.restoration import denoise_bilateral
from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color
from matplotlib import *

landscape_image = plt.imread('/content/drive/MyDrive/colored noise 2 (1).jpg')

# Apply bilateral filter denoising
denoised_image = denoise_bilateral(landscape_image, multichannel=True)

# Show original and resulting images
plot_comparison(landscape_image, denoised_image, 'Denoised Image')

from skimage.restoration import denoise_tv_chambolle

noisy_image = plt.imread('/content/drive/MyDrive/colored noise 2 (1).jpg')

# Apply total variation filter denoising
denoised_image = denoise_tv_chambolle(noisy_image, multichannel=True)

# Show the noisy and denoised image
plot_comparison(noisy_image, denoised_image, 'Denoised Image')


