from typing import Any
import numpy as np
from cv2 import cv2


def infoVideo(cap:Any):
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(" number of frames is",n_frames,"\n","height :",height,"\n","width :",width)
    #output codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('./output/video_out.mp4', fourcc, n_frames, (width, height))
    
    return n_frames

#Lucas-Kanade Optical Flow
def processFrames(cap:Any, frames:int,feature_params :dict,lk_parames:dict):
    #process the first frame for starting the sequence
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    #mask previous frame to draw lines
    mask = np.zeros_like(prev_frame)

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    
    for i in range(0,frames-2):
        prevPtsFeat = cv2.goodFeaturesToTrack(prev_gray,**feature_params)
        print("goodFT",prevPtsFeat)
        success, frame = cap.read() #true,prev
        
        if ( not success):
            break
        
        print("i#frame",i,"\n")
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(curr_gray)
        cv2.imshow("image gray",curr_gray)
        cv2.waitKey(0)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prevPtsFeat, None,**lk_parames)

        if curr_pts is not None:
            good_new = curr_pts[status==1]
            good_old = prevPtsFeat[status==1]
        # Sanity check
        assert prevPtsFeat.shape == curr_pts.shape

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        exit()
 




if __name__ == '__main__':
    cap = cv2.VideoCapture('./gallery/test1.mp4')
    frames = infoVideo(cap)

    feature_params = dict(maxCorners=200,qualityLevel=0.01,minDistance=30,blockSize=3)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (15, 15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    processFrames(cap,frames,feature_params,lk_params)
