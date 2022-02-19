import numpy as np
from cv2 import cv2


def setup():
    cap = cv2.VideoCapture('./gallery/test1.mp4')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("number of frames is",n_frames,"\n","height :",height,"\n","width :",width)

    #output codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('./output/video_out.mp4', fourcc, n_frames, (width, height))
    _, prev = cap.read() #true,prev
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    print(prev_gray)
    cv2.imshow("image gray",prev_gray)
    cv2.waitKey(0)

    

if __name__ == '__main__':
    setup()