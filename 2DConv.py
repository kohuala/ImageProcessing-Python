
# Guan, Hui Hua

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, freqz
from skimage.exposure import rescale_intensity


## 2D convolution 

def conv2D(image,kernel):
    '''
    Inputs:
    image - takes an image array
    kernel - takes an odd length kernel array, assume kernel is square
    '''
    
    # We will first pad the image around with an appropriate amount of zeros
    # We note that for a 3x3 kernel, in order to work properly, we need to pad the image
    # each side by 2 zeros. This way, when sliding the kernel through the image, the ninth 
    # cell of the kernel will convolve with the first pixel of the image.
    # From there, we can deduce that the appropriate amount of padding is one less than
    # the kernel length.
    padding = kernel.shape[0]-1
    img_pad= np.pad(img,padding,mode='constant',constant_values=0)
    
    # Next, we will create a array to hold the convolved image.
    # Note that the size is M+N-1, where M is the image size length, N is the kernel size length
    filtered_img = np.zeros((img.shape[0]+kernel.shape[0]-1,img.shape[1]+kernel.shape[1]-1))
    
    # To perform the 'convolve' operation, we would 'flip' and 'switch' the kernel
    flipped_kernel = np.flipud(np.fliplr(kernel))
    
    
    scl= int((kernel.shape[0]-1)/2) # 3->1, 5->2, 7->3
    
    for row in range(scl,img_pad.shape[0]-scl):
        for col in range(scl,img_pad.shape[1]-scl):
            # As we slide through the image, we will do elementwise multiplication and then add
            # each region up
            region = img_pad[row-scl:row+scl+1,col-scl:col+scl+1]
            filtered_img[row-scl,col-scl] =  np.sum(np.multiply(flipped_kernel,region))
    return filtered_img


# Let us do a quick check on our convolution 2D function against scipy's convolve2d.
# We can see from the results that the error is neglible. This means my conv2D is comparable with the scipy's convolve2d function.


# We can use an averaging filter
n=9
sev=np.ones((n,n))*(1/(n*n))

# Read in image of Lena
img = cv2.imread('lena512gray.png',0)
img=img.astype('uint8')

# scipy's convolve
s=convolve2d(img, sev, mode='full')
# my convolve
ss=conv2D(img, sev)

# Show the results
plt.subplot(141)
plt.title('Original Lena')
plt.imshow(img,cmap=plt.cm.gray )

plt.subplot(142)
plt.title('Scipy blur')
plt.imshow(s, cmap=plt.cm.gray)

plt.subplot(143)          
plt.imshow(ss, cmap=plt.cm.gray)
plt.title('My blur')

plt.subplot(144)
plt.imshow(s-ss, cmap=plt.cm.gray)
plt.title('Error')
plt.show()

print('The error between scipy.signal.convolve with full mode and my conv2D with full mode is '+ str(np.max(np.abs(s-ss))))


# Below is a function we can use to normalize the pixel values from 0 to 255


def histo_norm(img):
    
    # Calculate the histogram and corresponding bins 
   
    hist,_ = np.histogram(img.flatten(),256,[0,256])
    # Calculate the cdf and normalize the values to 0-255 
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255/ cdf[-1]
    
    # Replace the vales with normalized cdf values 
    img = cdf_normalized[img.astype('uint8')]
    return img


# We will read in an image into a greyscale image and then cast into uint8 type before doing any image processing


img = cv2.imread('lena512gray.png',0)


# Below is a function we can use to display the images nicely in row and column format


def grid_display(list_of_images, list_of_titles=[], no_of_columns=3, figsize=(20,20)):

    fig = plt.figure(figsize=figsize)
    column = 0
    for i in range(len(list_of_images)):
        column += 1
        #  check for end of column and create a new figure
        if column == no_of_columns+1:
            fig = plt.figure(figsize=figsize)
            column = 1
        fig.add_subplot(1, no_of_columns, column)
        if 'Mask' in list_of_titles[i]:
            plt.imshow(list_of_images[i], cmap='gray')
            
        else:
            plt.imshow(list_of_images[i], cmap='gray')
            
        plt.axis('off')
        if len(list_of_titles) >= len(list_of_images):
            plt.title(list_of_titles[i])
    plt.show()


def main(img_path, filt_size, filt_coeff):
    
    # Read in image
    img = cv2.imread(img_path,0)
    #img=img.astype('uint8')
    
    # Make kernel from the filter coefficients and filter size
    filt_coeff=np.array(filt_coeff)
    kernel=np.zeros((1,filt_size*filt_size))
    kernel[:len(filt_coeff)]=filt_coeff
    kernel = kernel.reshape((filt_size, filt_size))
    print(kernel)
    
    # Filter the image with conv2D
    filt_img = conv2D(img, kernel)
   
    
    # Normalised pixel value from 0 to 255
    filt_img=histo_norm(filt_img)
    
    # Save filtered image into file
    cv2.imwrite('filtered_image.jpg', filt_img)
    
    # Calculate and display magnitude of FT of original image
    FT = np.abs(np.fft.fft2(img))
    SFT=np.log(np.fft.fftshift(FT)+1)

    
    # Calculate and display magnitude of FT of filtered image
    filtimage_FT = np.abs(np.fft.fft2(filt_img))
    filtimage_SFT=np.log(np.fft.fftshift(filtimage_FT)+1)
    
    # Frequency response of filter
    filt_FT = np.abs(np.fft.fft2(kernel))
    filt_SFT=np.log(np.fft.fftshift(filt_FT)+1)
    
    
    grid_display([img,filt_img,SFT,filtimage_SFT , filt_SFT], ['Original', 'Filtered', 'Original FT', 'Filtered FT', 'Filter Frequency Response'],                 5, (20,20))
    return FT, filtimage_FT, filt_FT


# Below are the three kernels we will use: a 3x3 Gaussian filter, Laplacian operator( an edge detection) filter, and a sharpening filter. We list the coefficients here.


filter_H1 =[1/16, 2/16, 1/16,2/16,4/16,2/16,1/16,2/16,1/16]
filter_H2 = [-1, -1, -1,-1,8,-1,-1,-1,-1]
filter_H3 = [0, -1, 0,-1,5,-1,0,-1,0]

ft1,filtimgft1,filtft1=main('lena512gray.png', 3,filter_H1)
ft2,filtimgft2,filtft2=main('lena512gray.png', 3,filter_H2)
ft3,filtimgft3,filtft3 = main('lena512gray.png', 3,filter_H3)




mean = 0
sigma = 0.1
noise = np.random.normal(mean, sigma, img.shape)

noisy_img = img + np.clip(noise,0,255)
noisy_img = np.clip(noisy_img, 0, 255)


plt.figure()
plt.subplot(131)
plt.imshow(img,cmap=plt.cm.gray)
plt.title('Original Image')
plt.subplot(132)
plt.title('Noisy Image')
plt.imshow(noisy_img,cmap=plt.cm.gray)
plt.subplot(133)
plt.title('Salt and Pepper Noise')
plt.imshow(noisy_img-img,cmap=plt.cm.gray)
plt.show()


# We apply the previous program to filter the noise-added image using an average filter
# of size n x n, where n=odd. In this case, n=21. We choose a rather large size so it is more easy to see the blurring effect.



n = 21
avg_filt = np.ones((n,n))/(n*n)

plt.figure()
plt.subplot(121)
plt.title('Noisy Image')
plt.imshow(noisy_img, cmap=plt.cm.gray)   

output=conv2D(noisy_img, avg_filt)
plt.subplot(122)
plt.title('Filtered Image')
plt.imshow(output, cmap=plt.cm.gray)

plt.show()


# Now, we apply your previous program to filter the noise-added image using a Gaussian filter
# of size n x n. First, let us define a helper function to find the Gaussian kernel of a nxn filter.




def gkern(n, sig=1):
    ''' Adapted from last semester machine learning class '''
    ax = np.arange(-n // 2 + 1., n // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)


n = 21
gau_filt_21 = gkern(21)
output=conv2D(noisy_img, gau_filt_21)
plt.imshow(output, cmap=plt.cm.gray)
plt.title('Denoised Lena with 21x21 Gaussian Filter')
plt.show()


# Try two different noise levels (0.01 and 0.1) and for each noise level different filter sizes (ex: 5x5 to 9x9 in step size of 2).
# 


img = cv2.imread('lena512gray.png',0)
img=img.astype('uint8')




# Make a noisy image with noise level 0.01
mean = 0
sigma001 = 0.01
noise001 = np.random.normal(mean, sigma001, img.shape)
noisy_img001 = img + np.clip(noise001,0,255)
noisy_img_001 = np.clip(noisy_img001, 0, 255)

plt.figure()
plt.subplot(131)
plt.imshow(img,cmap=plt.cm.gray)
plt.title('Original Image')
plt.subplot(132)
plt.title('Noisy Image')
plt.imshow(noisy_img,cmap=plt.cm.gray)
plt.subplot(133)
plt.title('Salt and Pepper Noise')
plt.imshow(noisy_img_001-img,cmap=plt.cm.gray)


plt.show()



n5 = 5

# Denoise the noisy image
gaus_filter_5=float(1)/256*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,26,16,4],[1,4,6,4,1]])
out5_001=conv2D(noisy_img_001, gaus_filter_5)


n7 = 7

# Denoise the noisy image
gaus_filter_7=gkern(n7,1)
out7_001=conv2D(noisy_img_001, gaus_filter_7)



n9 = 9

# Denoise the noisy image
gaus_filter_9=gkern(n9,1)
out9_001=conv2D(noisy_img_001, gaus_filter_9)


# Now, let us change the noise level to 0.1.



# Make a noisy image with noise level 0.1
mean = 0
sigma01 = 0.1
noise01 = np.random.normal(mean, sigma01, img.shape)
noisy_img01 = img + np.clip(noise01,0,255)
noisy_img_01 = np.clip(noisy_img01, 0, 255)



n5 = 5

# Denoise the noisy image
gaus_filter_5=float(1)/256*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,26,16,4],[1,4,6,4,1]])
out5_01=conv2D(noisy_img_01, gaus_filter_5)


n7 = 7

# Denoise the noisy image
gaus_filter_7=gkern(n7,1)
out7_01=conv2D(noisy_img_01, gaus_filter_7)


n9 = 9

# Denoise the noisy image
gaus_filter_9=gkern(n9,1)
out9_01=conv2D(noisy_img_01, gaus_filter_9)


# Below is the original, noise-added, and filtered images for each combination of noise level and filter size. We then comment on for each noise level, which filter size is best for each filter and how does the two filters compare in their noise removal capability.

# For the noisy level of 0.01, below is the results for a 5x5, 7x7, and 9x9 Gaussian filter.

print('5x5')
padded5=np.pad(img,2,mode='constant',constant_values=0 )
print(np.average(np.abs(out5_001-padded5)))
grid_display([img,noisy_img_001,gaus_filter_5,out5_001,out5_001-padded5 ], ['Original', 'Noisy','Filter', 'Result', 'Error'],5)

print('7x7')
padded7=np.pad(img,3,mode='constant',constant_values=0 )
print(np.average(np.abs(out7_001-padded7)))
grid_display([img,noisy_img_001,gaus_filter_7,out7_001,out7_001-padded7], ['Original', 'Noisy','Filter', 'Result','Error'],5)

print('9x9')
padded9=np.pad(img,4,mode='constant',constant_values=0 )
print(np.average(np.abs(out9_001-padded9)))
grid_display([img,noisy_img_001,gaus_filter_9,out9_001,out9_001-padded9], ['Original', 'Noisy', 'Filter', 'Result','Error'],5)


# For the noisy level of 0.1, below is the results for a 5x5, 7x7, and 9x9 Gaussian filter.


print('5x5')
print(np.average(np.abs(out5_01-padded5)))
grid_display([img,noisy_img_01,gaus_filter_5,out5_01,out5_01-padded5], ['Original', 'Noisy', 'Filter', 'Result', 'Error'],                5)

print('7x7')
print(np.average(np.abs(out7_01-padded7)))
grid_display([img,noisy_img_01,gaus_filter_7,out7_01, out7_01-padded7], ['Original', 'Noisy','Filter', 'Result','Error'],                5)

print('9x9')
print(np.average(np.abs(out9_01-padded9)))
grid_display([img,noisy_img_01,gaus_filter_9,out9_01,out9_01-padded9], ['Original', 'Noisy','Filter', 'Result','Error'],                5)

