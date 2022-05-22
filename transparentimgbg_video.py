from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
# Apply the transformations needed
import torchvision.transforms as T


#BACKGROUND IMAGE PATH 
background_img_path = "images/usa1.png"  #path here <==========================================================================

cap = cv2.VideoCapture(0) #Webcam

readVideoFromfile = False #Set False if you want to use webcam, true if you want to use local file
videoFilename = "inputvideo.mp4"

#Redefine cap to read from this ^ file. Defaults to webcam if something goes wrong with that input (can happen a lot with video) 
if readVideoFromfile: 
    try: 
        cap = cv2.VideoCapture( videoFilename ) #video from videoFilename
    except:
        cap = cv2.VideoCapture(0) #Webcam as exception


'''If you instead want to use greenscreen run:
background_img_path = ''
cv2.imwrite( 'greenscreen.png' , makeGreenScreen(width=1000, height=1000) )
background_img_path = 'greenscreen.png'

after, of course, where you define the greenscreen function 
'''

#Create GreenScreen Image
#Chroma Key Green has RGB value 0, 177, 64  , so has a BGR value of (64,177,0) which we use for OpenCV
def makeGreenScreen(width=1000, height=1000, bgr_color=(64,177,0) ): #Color defaults to Chroma Key Green although technically a 'green' screen can be any color, this one just rare enough 
  width = int(width)
  height = int(height)

  greenScreen = np.zeros((height,width,3), np.uint8)
  greenScreen[:] = bgr_color      # (B, G, R) so (255,0,0) would be blue. Default is chroma key green in this function. 
  return greenScreen

#cv2.imshow("my green screen", makeGreenScreen() )

#MODIFY decode_segmap to take img as input, ie doesn't need to reed it from path 

# Define the helper function
def decode_segmap(image, source, bgimg, sourceFromPath=False, bgimgFromPath=True, nc=21):
    '''
    MODIFICATION:
    FromPath inputs asking if its from a local file, if true theyll instantiate imread which reads from path.
    If FromPath inputs are false then source will be an image input 
    e.g/  decode_segmap(image, source=YOUR_IMAGE, bgimg='./images/dog.png', sourceFromPath=False, bgimgFromPath=True, nc=21)
    '''


    label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]


    rgb = np.stack([r, g, b], axis=2)

    # Load the foreground input image
    if sourceFromPath:  
        foreground = cv2.imread(source)
    else: 
        foreground = source 

    # Load the background input image 
    if bgimgFromPath:  
        background = cv2.imread(bgimg)
    else: 
        background = bgimg

    # Change the color of foreground image to RGB 
    # and resize images to match shape of R-band in RGB output map
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))
    background = cv2.resize(background,(r.shape[1],r.shape[0]))


    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)

    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7,7),0)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)  

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)  

    # Add the masked foreground and background
    outImage = cv2.add(foreground, background)

    # Return a normalized output image for display
    return outImage/255

def grey_background_decode_segmap(image, source, bgimg, sourceFromPath=False, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]


    rgb = np.stack([r, g, b], axis=2)

    # Load the foreground input image 
    if sourceFromPath:  
        foreground = cv2.imread(source)
    else: 
        foreground = source 

    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))

    bgimg = cv2.imread(bgimg)
    #bgimg = cv2.cvtColor(bgimg, cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg ,(r.shape[1],r.shape[0]))

    #------------------------------------Transparent img code-----------------------------------------------------
    transparency_alpha = 0.4 # https://www.programcreek.com/python/example/89436/cv2.addWeighted
    transparency_beta = 0.25 

    # get dimensions of image
    img_dimensions = foreground.shape
    # height, width, number of channels in image
    img_height = foreground.shape[0]
    img_width = foreground.shape[1]
    img_channels = foreground.shape[2]
    
    #print('Image Dimension    : ', img_dimensions)
    #print('Image Height       : ', img_height)
    #print('Image Width        : ', img_width)
    #print('Number of Channels : ', img_channels)
    img_dim = ( img_width, img_height )

    #resize both to same size in order to merge them
    #background = cv2.resize(foreground, img_dim, interpolation = cv2.INTER_AREA)
    transparent_overlay = cv2.resize(bgimg, img_dim, interpolation = cv2.INTER_AREA)
    #transparent_overlay = cv2.resize(transparent_overlay ,(r.shape[1],r.shape[0]))

    background = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    #background = cv2.resize(background,(r.shape[1],r.shape[0]))
    background = cv2.addWeighted(background , transparency_alpha , transparent_overlay, transparency_beta, 0 )

    background = cv2.resize(background ,(r.shape[1],r.shape[0]))
    
    #Convert background to 3-channel RGB so we can perform certain operations
    #--------------------------------------------------------------------------------------------------------------
    # Change the color of foreground image to RGB 
    # and resize image to match shape of R-band in RGB output map  
    #foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    #foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))

    # Create a background image by copying foreground and converting into grayscale
    #background = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

    # convert single channel grayscale image to 3-channel grayscale image
    #background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)

    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7,7),0)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)  

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)  

    # Add the masked foreground and background
    outImage = cv2.add(foreground, background)

    # Return a normalized output image for display
    return outImage/255



def segment(net, seg_source, seg_bgimg, seg_sourceFromPath=False, seg_bgimgFromPath=True, show_orig=True, dev='cuda'):
    if seg_sourceFromPath:  
        img = Image.open(seg_source)
    else: 
        img = Image.fromarray(seg_source)  #Use fromarray to convert cv2 image into PIL image in order to perform torchvision operations on it 

    # Load the background input image 
    if seg_bgimgFromPath:  
        background = cv2.imread(seg_bgimg)
    else: 
        background = Image.fromarray(seg_bgimg)  #Use fromarray to convert cv2 image into PIL image in order to perform torchvision operations on it 

    #If statement to show original foreground image in matplot popup 
    if show_orig: plt.imshow(img); plt.axis('off'); plt.show()

    # Comment the Resize and CenterCrop for better inference results
    trf = T.Compose([T.Resize(400), 
                    #T.CenterCrop(224), 
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    rgb = decode_segmap(om, source=seg_source, bgimg=seg_bgimg, sourceFromPath=seg_sourceFromPath, bgimgFromPath= seg_bgimgFromPath)

    #print( "time 2 = ", time.time() , " RunTime = " , time.time() - t1  )

    #plt.imshow(rgb); plt.axis('off'); plt.show() #<==== ORIGINALLY SHOWED IMG IN MATPLOT. 
    return rgb 
  
def grey_background_segment(net, seg_source, bgimg, show_orig=False, dev='cuda'):
  img = Image.fromarray(seg_source)  #img = Image.open(seg_source)
  
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  # Comment the Resize and CenterCrop for better inference results
  trf = T.Compose([T.Resize(450), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  
  rgb = grey_background_decode_segmap(om, seg_source, bgimg)
    
  #plt.imshow(rgb); plt.axis('off'); plt.show()

  return rgb 
  


dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

#Original
#segment(dlab, './images/change/girl-with-hat.png','./images/change/background-building.png', show_orig=False)
#segment(dlab, './images/change/girl.png','./images/change/forest.png', show_orig=False)


while 1:
    ret, img2segment = cap.read()
    img2segment = cv2.flip(img2segment,1)
    img2segment = cv2.cvtColor(img2segment, cv2.COLOR_BGR2RGB)

    #img2segment_path = 'img2segment.png'
    #cv2.imwrite( img2segment_path , img2segment )

    #Change background to a given image:
    #changed_img = segment(net=dlab, seg_source=img2segment, seg_bgimg=background_img_path, seg_sourceFromPath=False, seg_bgimgFromPath=True, show_orig=False, dev='cuda')
    
    #Change your background to gray:
    changed_img = grey_background_segment(net=dlab, seg_source=img2segment, bgimg=background_img_path , show_orig=False, dev='cuda')
    
    #Change your background to rainbow or american flag 

    #change your background to an skimg algorithm input like rainbow one here https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_marked_watershed.html#sphx-glr-auto-examples-segmentation-plot-marked-watershed-py

    
    cv2.imshow('AutoGreenScreen' , changed_img )
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
cv2.destroyAllWindows()

