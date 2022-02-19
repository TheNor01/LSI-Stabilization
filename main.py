from typing import Any
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


def infoVideo(cap:Any):
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(" number of frames is",n_frames,"\n","height :",height,"\n","width :",width)
    #output codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('./output/video_out.mp4', fourcc, n_frames, (width, height))
    
    return n_frames

def ComputeMatrix(good_old, good_new):
    m = cv2.estimateAffine2D(good_old, good_new)[0]
    #cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    #print("matrix",m)
    dx = m[0,2] #deltaX
    dy = m[1,2] #deltay 
    da = np.arctan2(m[1,0], m[0,0]) #deltaAngle

    #print("delta a",da)

    return [dx,dy,da]

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




#Lucas-Kanade Optical Flow
def processFrames(cap:Any, frames:int,feature_params :dict,lk_parames:dict):
    #process the first frame for starting the sequence
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    #mask previous frame to draw lines
    mask = np.zeros_like(prev_frame)
    # Create some random colors
    color = np.random.randint(0, 255, (150, 3))
    transforms = np.zeros((frames-1, 3), np.float32)
    
    for i in range(0,frames-2):
        print("frame i",i)
        prevPtsFeat = cv2.goodFeaturesToTrack(prev_gray,**feature_params)
        #print("goodFT",prevPtsFeat)
        success, frame = cap.read() #true,prev
        
        if (not success):
            print("not success")
            break
        
        #print("i#frame",i,"\n")
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(curr_gray)
        #cv2.imshow("image gray",curr_gray)
        #cv2.waitKey(0)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prevPtsFeat, None,**lk_parames)

        if curr_pts is not None:
            good_new = curr_pts[status==1]
            good_old = prevPtsFeat[status==1]

        # Sanity check
        assert prevPtsFeat.shape == curr_pts.shape

        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 3, color[j].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)


        #Find transformation matrix
        transforms[i]=ComputeMatrix(good_old,good_new)
        #print("delta frame i ",i)  
        #print(transforms[i])
    

        #space for next iter, esc to quit
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break

        prev_gray = curr_gray.copy()
        prevPtsFeat = good_new.reshape(-1, 1, 2)
    cv2.destroyAllWindows()

    trajectory = np.cumsum(transforms, axis=0)

    print(type(trajectory))
    print(trajectory)
    plt.xlabel('frames')
    plt.ylabel('values')
    plt.plot(trajectory)
    plt.gca().legend(('deltaX','deltaY','deltaÆ'))
    plt.show()

  

if __name__ == '__main__':
    cap = cv2.VideoCapture('./gallery/test1.mp4')
    print(cv2.__version__)
    print("Getting info video \n")
    frames = infoVideo(cap)
    feature_params = dict(maxCorners=100,qualityLevel=0.10,minDistance=40,blockSize=3)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (15, 15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    processFrames(cap,frames,feature_params,lk_params)
