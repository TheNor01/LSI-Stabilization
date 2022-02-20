import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import os
import time

from scripts.classes import CustomStabilization
from vidstab import VidStab

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

def Stabilization1(sourceVideo,outPath):
    stableObject = CustomStabilization(sourceVideo)

    #cap = cv2.VideoCapture(sourceVideo)
    print("Getting info video \n")
    stableObject.InfoVideo()
    stableObject.Setup(outPath)
    
    feature_params = dict(maxCorners=100,qualityLevel=0.10,minDistance=10,blockSize=3) 
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (15, 15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    trajectory,transforms = stableObject.processFrames(feature_params,lk_params)

    smoothedTra = Smooth(trajectory)

    PlotCurves(trajectory,smoothedTra)
    
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothedTra - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    stableObject.WritingStable(transforms_smooth)


def Stabilization2(sourceVideo,outPath):
    stabilizer = VidStab(kp_method="FAST")
    stabilizer.stabilize(input_path=sourceVideo, output_path=outPath,border_size=100)
    stabilizer.plot_trajectory()
    plt.show()
    stabilizer.plot_transforms()
    plt.show()


if __name__ == '__main__':

    sourceVideo = "./gallery/test2.mp4"
    #output codec
    outPath = "./output/video_out.mp4"
    if os.path.exists(outPath):
        os.remove(outPath)


    #Using custom logic
    #Stabilization1(sourceVideo,outPath)

    #Usign videostab 
    Stabilization2(sourceVideo,outPath)
    cv2.destroyAllWindows()
    
