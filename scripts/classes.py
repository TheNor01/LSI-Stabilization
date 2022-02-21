import numpy as np
from cv2 import cv2


class CustomStabilization:

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def Setup(self,outPath):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(outPath, fourcc, self.fps , (self.width,self.height))  

    def InfoVideo(self):
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(" number of frames is",self.n_frames,"\n","height :",self.height,"\n","width :",self.width,"FPS",self.fps)
    
    def fixBorder(self,frame):
        s = frame.shape
        # Scale the image 25% without moving the center
        T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.25)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

    def processFrames(self,feature_params :dict,lk_parames:dict):
        #process the first frame for starting the sequence
        _, prev_frame =self. cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #mask previous frame to draw lines
        mask = np.zeros_like(prev_frame)
        # Create some random colors
        color = np.random.randint(0, 255, (150, 3))
        transforms = np.zeros((self.n_frames-1, 3), np.float32)
        
        for i in range(0,self.n_frames-2):
            print("frame i",i)
            prevPtsFeat = cv2.goodFeaturesToTrack(prev_gray,**feature_params)
            #print("goodFT",prevPtsFeat)
            success, frame = self.cap.read() #true,prev
            
            if (not success):
                print("not success")
                break
            
            #print("i#frame",i,"\n")
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(curr_gray)
            #cv2.imshow("image gray",curr_gray)
            #cv2.waitKey(0)

            #Lucas-Kanade Optical Flow
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prevPtsFeat, None,**lk_parames)

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
            transforms[i]=self.ComputeMatrix(good_old,good_new)
            #print("delta frame i ",i)  
            #print(transforms[i])
        
            #space for next iter, esc to quit
            k = cv2.waitKey(0) & 0xff
            if k == 27:
                break

            prev_gray = curr_gray.copy()
            prevPtsFeat = good_new.reshape(-1, 1, 2)
        trajectory = np.cumsum(transforms, axis=0)
        return trajectory,transforms

    def ComputeMatrix(self,good_old, good_new):
        m = cv2.estimateAffine2D(good_old, good_new)[0]
        #cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        #print("matrix",m)
        dx = m[0,2] #deltaX
        dy = m[1,2] #deltay 
        da = np.arctan2(m[1,0], m[0,0]) #deltaAngle

        #print("delta a",da)

        return [dx,dy,da]

    def WritingStable(self,transforms_smooth):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        print("STABLING video")

        for i in range(self.n_frames-2):
            success, frame = self.cap.read()

            print("Stable i",i )
            if not success:
                self.out.release()
                print("not success")  
                break
            # Extract transformations from the new transformation array
            dx = transforms_smooth[i,0]
            dy = transforms_smooth[i,1]
            da = transforms_smooth[i,2]

            # Reconstruct transformation matrix accordingly to new values
            m = np.zeros((2,3), np.float32)
            m[0,0] = np.cos(da)
            m[0,1] = -np.sin(da)
            m[1,0] = np.sin(da)
            m[1,1] = np.cos(da)
            m[0,2] = dx
            m[1,2] = dy

            frame_stabilized = cv2.warpAffine(frame, m, (self.width,self.height))

            frame_stabilized = self.fixBorder(frame_stabilized)
            frame_compare = cv2.hconcat([frame, frame_stabilized])

            cv2.imshow("After", frame_compare)

            k = cv2.waitKey(int(self.fps)) & 0xff
            if k == 27:
                break
            self.out.write(frame_stabilized)
        self.cap.release()
        self.out.release()
