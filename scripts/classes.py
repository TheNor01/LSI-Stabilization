import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import convolve


class CustomStabilization:

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def Setup(self,outPath,feature_params,lk_params):
        self.feature_params=feature_params
        self.lk_parames=lk_params
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
        T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.05)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

    def processFrames(self):
        #process the first frame for starting the sequence
        _, prev_frame =self. cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #mask previous frame to draw lines
        mask = np.zeros_like(prev_frame)
        # Create some random colors
        #color = np.random.randint(0, 255, (150, 3))
        transforms = np.zeros((self.n_frames-1, 3), np.float32)
        
        for i in range(0,self.n_frames-2):
            print("frame i",i)
            prevPtsFeat = cv2.goodFeaturesToTrack(prev_gray,**self.feature_params)
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
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prevPtsFeat, None,**self.lk_parames)

            if curr_pts is not None:
                good_new = curr_pts[status==1]
                good_old = prevPtsFeat[status==1]

            # Sanity check
            assert prevPtsFeat.shape == curr_pts.shape

            for j,(new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 3, (0,255,0), -1)
            
        
            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)
            
            mask = np.zeros_like(frame)

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


class UpgradedCustom:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def load_images(self):
        again = True
        i = 0
        imgs = []
        while again:
            again, img = self.cap.read()
            if again:
                img_r = cv2.resize(img, None, fx=0.25, fy=0.25)
                imgs += [img_r]
                if not self.outPath is None:
                    filename = self.outPath + "".join([str(0)]*(3-len(str(i)))) + str(i) +'.png'
                    cv2.imwrite(filename, img_r)
                i += 1
            else:
                break
        return imgs
    
    def InfoVideo(self):
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
    def create_warp_stack(self,imgs):
        warp_stack = []
        for i, img in enumerate(imgs[:-1]):
            warp_stack += [self.get_homography(img, imgs[i+1])]
        return np.array(warp_stack)

    def Setup(self,outPath):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(outPath, fourcc, self.fps , (self.width,self.height)) 

    def Plotting(self,ws,name):
        i,j=0,2
        plt.scatter(np.arange(len(ws)), ws[:,i,j], label='X Velocity')
        plt.plot(np.arange(len(ws)), ws[:,i,j])
        plt.scatter(np.arange(len(ws)), np.cumsum(ws[:,i,j], axis=0), label='X Trajectory')
        plt.plot(np.arange(len(ws)), np.cumsum(ws[:,i,j], axis=0))
        plt.legend()
        plt.xlabel('Frame')
        plt.savefig(name+'_trajectory.png')
    
    def moving_average(self,warp_stack, sigma_mat):
        x,y = warp_stack.shape[1:]
        original_trajectory = np.cumsum(warp_stack, axis=0)
        smoothed_trajectory = np.zeros(original_trajectory.shape)
        for i in range(x):
            for j in range(y):
                kernel = signal.gaussian(1000, sigma_mat[i,j])
                kernel = kernel/np.sum(kernel)
                smoothed_trajectory[:,i,j] = convolve(original_trajectory[:,i,j], kernel, mode='reflect')
        smoothed_warp = np.apply_along_axis(lambda m: convolve(m, [0,1,-1], mode='reflect'), axis=0, arr=smoothed_trajectory)
        smoothed_warp[:,0,0] = 0
        smoothed_warp[:,1,1] = 0
        return smoothed_warp, smoothed_trajectory, original_trajectory

    def get_border_pads(self,img_shape, warp_stack):
        maxmin = []
        corners = np.array([[0,0,1], [img_shape[1], 0, 1], [0, img_shape[0],1], [img_shape[1], img_shape[0], 1]]).T
        warp_prev = np.eye(3)
        for warp in warp_stack:
            warp = np.concatenate([warp, [[0,0,1]]])
            warp = np.matmul(warp, warp_prev)
            warp_invs = np.linalg.inv(warp)
            new_corners = np.matmul(warp_invs, corners)
            xmax,xmin = new_corners[0].max(), new_corners[0].min()
            ymax,ymin = new_corners[1].max(), new_corners[1].min()
            maxmin += [[ymax,xmax], [ymin,xmin]]
            warp_prev = warp.copy()
        maxmin = np.array(maxmin)
        bottom = maxmin[:,0].max()
        print('bottom', maxmin[:,0].argmax()//2)
        top = maxmin[:,0].min()
        print('top', maxmin[:,0].argmin()//2)
        left = maxmin[:,1].min()
        print('right', maxmin[:,1].argmax()//2)
        right = maxmin[:,1].max()
        print('left', maxmin[:,1].argmin()//2)
        return int(-top), int(bottom-img_shape[0]), int(-left), int(right-img_shape[1])

    def homography_gen(self,warp_stack):
        H_tot = np.eye(3)
        wsp = np.dstack([warp_stack[:,0,:], warp_stack[:,1,:], np.array([[0,0,1]]*warp_stack.shape[0])])
        for i in range(len(warp_stack)):
            H_tot = np.matmul(wsp[i].T, H_tot)
            yield np.linalg.inv(H_tot)#[:2]

    ## APPLYING THE SMOOTHED TRAJECTORY TO THE IMAGES
    def apply_warping_fullview(self,images, warp_stack):
        top, bottom, left, right = self.get_border_pads(img_shape=images[0].shape, warp_stack=warp_stack)
        H = self.homography_gen(warp_stack)
        imgs = []
        for i, img in enumerate(images[1:]):
            H_tot = next(H)+np.array([[0,0,left],[0,0,top],[0,0,0]])
            img_warp = cv2.warpPerspective(img, H_tot, (img.shape[1]+left+right, img.shape[0]+top+bottom))
            if not self.outPath is None:
                filename = self.outPath + "".join([str(0)]*(3-len(str(i)))) + str(i) +'.png'
                cv2.imwrite(filename, img_warp)
            imgs += [img_warp]
        return imgs

    ## FINDING THE TRAJECTORY
    def get_homography(self,img1, img2, motion = cv2.MOTION_EUCLIDEAN):
        imga = img1.copy().astype(np.float32)
        imgb = img2.copy().astype(np.float32)
        if len(imga.shape) == 3:
            imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
        if len(imgb.shape) == 3:
            imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
        if motion == cv2.MOTION_HOMOGRAPHY:
            warpMatrix=np.eye(3, 3, dtype=np.float32)
        else:
            warpMatrix=np.eye(2, 3, dtype=np.float32)
        warp_matrix = cv2.findTransformECC(templateImage=imga,inputImage=imgb,warpMatrix=warpMatrix, motionType=motion)[1]
        return warp_matrix 

    ## HELP WITH VISUALIZING 
    def imshow_with_trajectory(self,images, warp_stack, PATH, ij):
        traj_dict = {(0,0):'Width', (0,1):'sin(Theta)', (1,0):'-sin(Theta)', (1,1):'Height', (0,2):'X', (1,2):'Y'}
        i,j = ij
        filenames = []
        for k in range(1,len(warp_stack)):
            f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})

            a0.axis('off')
            a0.imshow(images[k])

            a1.plot(np.arange(len(warp_stack)), np.cumsum(warp_stack[:,i,j]))
            a1.scatter(k, np.cumsum(warp_stack[:,i,j])[k], c='r', s=100)
            a1.set_xlabel('Frame')
            a1.set_ylabel(traj_dict[ij]+' Trajectory')
            
            if not self.outPath is None:
                filename = self.outPath + "".join([str(0)]*(3-len(str(k)))) + str(k) +'.png'
                plt.savefig(filename)
                filenames += [filename]
            plt.close()
        return filenames
    
    def Process(self,outPath):
        self.InfoVideo()
        self.outPath=outPath
        self.Setup(outPath)
        imgs = self.load_images()
        name = "result1"
        ws = self.create_warp_stack(imgs)
        self.Plotting(ws,name)
        #calculate the smoothed trajectory and output the zeroed images
        smoothed_warp, smoothed_trajectory, original_trajectory = self.moving_average(ws, sigma_mat= np.array([[1000,15, 10],[15,1000, 10]]))
        new_imgs = self.apply_warping_fullview(images=imgs, warp_stack=ws-smoothed_warp)
        #plot the original and smoothed trajectory
        f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1, 1]})
        i,j = 0,2
        a0.scatter(np.arange(len(original_trajectory)), np.array(original_trajectory)[:,i,j], label='Original')
        a0.plot(np.arange(len(original_trajectory)), np.array(original_trajectory)[:,i,j])
        a0.scatter(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:,i,j], label='Smoothed')
        a0.plot(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:,i,j])
        a0.legend()
        a0.set_ylabel('X trajectory')
        a0.xaxis.set_ticklabels([])
        i,j = 0,1
        a1.scatter(np.arange(len(original_trajectory)), np.array(original_trajectory)[:,i,j], label='Original')
        a1.plot(np.arange(len(original_trajectory)), np.array(original_trajectory)[:,i,j])
        a1.scatter(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:,i,j], label='Smoothed')
        a1.plot(np.arange(len(smoothed_trajectory)), np.array(smoothed_trajectory)[:,i,j])
        a1.legend()
        a1.set_xlabel('Frame')
        a1.set_ylabel('Sin(Theta) trajectory')
        plt.savefig(name+'_smoothed.png')

        #create a images that show both the trajectory and video frames
        filenames = self.imshow_with_trajectory(images=new_imgs, warp_stack=ws-smoothed_warp, PATH='./out_'+name+'/', ij=(0,2))