from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
# Apply the transformations needed
import torchvision.transforms as T

#import time 
#t1 = time.time()
#print( "time 1 = ", t1 )

#Create GreenScreen Image
#Chroma Key Green has RGB value 0, 177, 64  , so has a BGR value of (64,177,0) which we use for OpenCV
def makeGreenScreen(width=1000, height=1000, bgr_color=(64,177,0) ): #Color defaults to Chroma Key Green although technically a 'green' screen can be any color, this one just rare enough 
  width = int(width)
  height = int(height)

  greenScreen = np.zeros((height,width,3), np.uint8)
  greenScreen[:] = bgr_color      # (B, G, R) so (255,0,0) would be blue. Default is chroma key green in this function. 
  return greenScreen

#cv2.imshow("my green screen", makeGreenScreen() )

# Define the helper function
def decode_segmap(image, source, bgimg, nc=21):
  
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
  foreground = cv2.imread(source)

  # Load the background input image 
  background = cv2.imread(bgimg)

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

def segment(net, path, bgimagepath, show_orig=True, dev='cuda'):
  img = Image.open(path)
  
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
  
  rgb = decode_segmap(om, path, bgimagepath)
  
  #print( "time 2 = ", time.time() , " RunTime = " , time.time() - t1  )

  #plt.imshow(rgb); plt.axis('off'); plt.show() #<==== ORIGINALLY SHOWED IMG IN MATPLOT. 
  return rgb 
  

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

#Original
segment(dlab, './images/change/girl-with-hat.png','./images/change/background-building.png', show_orig=False)
segment(dlab, './images/change/girl.png','./images/change/forest.png', show_orig=False)

cap = cv2.VideoCapture(0)

while 1:
    ret, img2segment = cap.read()
    img2segment = cv2.flip(img2segment,1)
    img2segment = cv2.cvtColor(img2segment, cv2.COLOR_BGR2RGB)

    img2segment_path = 'img2segment.png'
    cv2.imwrite( img2segment_path , img2segment )

    background_img_path = './images/change/forest.png' #path here

    '''Note: Currently use img path call, ie we first save the video capture to a local png then call it
    Obviously inefficient and can slow program. But original segment function calls path so 
    we use path to test out the function. Should loop in image straight from source for actual software 
    '''


    changed_img = segment(dlab, img2segment_path, background_img_path, show_orig=False)
    cv2.imshow('AutoGreenScreen' , changed_img )
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
cv2.destroyAllWindows()
