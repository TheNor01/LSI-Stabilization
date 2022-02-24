import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import os
import time

from scripts.classes import CustomStabilization,VideoStabilization

def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  # Define the filter
  f = np.ones(window_size)/window_size
  # Add padding to the boundaries
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
  # Apply convolution
  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  # Remove padding
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed

def Smooth(trajectory):
  smoothed_trajectory = np.copy(trajectory)
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=50)
  return smoothed_trajectory

def PlotCurves(trajectory,smoothedTra):
    plt.subplot(1, 2, 1)
    plt.xlabel('frames')
    plt.ylabel('values')
    
    plt.plot(trajectory)
    plt.gca().legend(('deltaX','deltaY','deltaÆ'))

    plt.subplot(1, 2, 2)
    plt.plot(smoothedTra)
    plt.gca().legend(('deltaX','deltaY','deltaÆ'))
    plt.show()


def Stabilization1(sourceVideo,corners,blockSize,outPath="./output/video_out.mp4"):
    stableObject = CustomStabilization(sourceVideo)

    #cap = cv2.VideoCapture(sourceVideo)
    print("Getting info video \n")
    stableObject.InfoVideo()
    print(outPath)

    # Parameters for lucas kanade optical flow
    feature_params = dict(maxCorners=corners,qualityLevel=0.3,minDistance=7,blockSize=blockSize)
    lk_params = dict(winSize  = (10, 10),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    stableObject.Setup(outPath,feature_params,lk_params)
    
    trajectory,transforms = stableObject.processFrames()

    smoothedTra = Smooth(trajectory)

    PlotCurves(trajectory,smoothedTra)
    
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothedTra - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    stableObject.WritingStable(transforms_smooth)

    cv2.destroyAllWindows()


def Stabilization2(sourceVideo,option,boolObject,outPath="./output/video_out.mp4"):
    
    videoStab = VideoStabilization(option)
    if not boolObject:
      videoStab.Process(sourceVideo,outPath)
    else:
      videoStab.ObjectProcessing(outPath)
      
  
def Main(sourcePath,name,option="Fast",outPath="./output/video_out.mp4"):
  print("main")
  print(option)
  
  """
  if os.path.exists(outPath):
        os.remove(outPath)

  if name == "Custom":
    Stabilization1(sourcePath)
  elif name == "VidStab":
    Stabilization2(sourcePath,option)
  else:
    print("none")
  """
  cv2.destroyAllWindows()

if __name__ == '__main__':

    sourcePath = "./gallery/test2.mp4"
    outPath = "./output/video_out.mp4"

    if os.path.exists(outPath):
        os.remove(outPath)
    #Main(sourcePath,name="Custom")
  
    #Using custom logic
    #Stabilization1(sourcePath)

    #Usign videostab 
    #Stabilization2(sourcePath)
    
    
