import numpy as np
import cv2 as cv

feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.98,
                       minDistance = 100,
                       blockSize = 8 )

lk_params = dict( winSize  = (10,10),
                  maxLevel = 5,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03), flags = cv.OPTFLOW_LK_GET_MIN_EIGENVALS )

class point:
    def __init__(self):
        self.data = []

    def add(self, value):
        self.data.append(value)
        if len(self.data) >= 6:
            self.data.pop(0)

    def getSum(self):
        sumX = 0; sumY = 0
        for i in self.data:
            sumX += i[0]
            sumY += i[1]
        return [sumX, sumY]

class SLAM:
    def __init__(self, lk, fp):
        self.lk_params = lk;
        self.feature_params = fp
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_EXPOSURE, -3)
        self.h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.mask = np.zeros((self.h,self.w,1), np.uint8); self.mask[:] = 255
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        self.diff_frame = np.zeros((self.h,self.w,3), np.uint8)

        self.frame = None
        self.getFrame()
        self.p0 = cv.goodFeaturesToTrack(self.frame, mask = self.mask, **self.feature_params)

        self.st = np.array([[0] for i in range(len(self.p0))])
        self.trackMemory = np.array([point() for i in range(len(self.p0))])

    def getFrame(self):
        ret, frame = self.cap.read()
        self.frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.frame = cv.bilateralFilter(self.frame, 9, 75, 75)
        #self.frame = self.clahe.apply(self.frame)

    def getFeatures(self):
        new = cv.goodFeaturesToTrack(self.frame, mask = self.mask, **self.feature_params)
        if np.any(new != None):
            self.p0 = np.concatenate((self.p0, new), axis=0)
            self.st = np.concatenate((self.st, [[0] for i in range(len(new))]), axis=0)
            self.trackMemory = np.concatenate((self.trackMemory, [point() for i in range(len(new))]), axis=0)

            return len(new)
        return 0

    def setMask(self):
        self.mask[:] = 255
        for x in self.p0:
            a,b = x.ravel()
            self.mask = cv.circle(self.mask,(a,b), self.feature_params["minDistance"], (0,0,0), -1)

    def track(self, st, new):
        old = self.p0[st != 0]

        diff = np.subtract(old, new)#.reshape(-1,1,2)
        for i in range(len(diff)):
            self.trackMemory[i].add([diff[i][0], diff[i][1]])

        delta = []
        for i in range(len(self.st)):
            if self.st[i][0] >= 5:
                sx, sy = self.trackMemory[i].getSum()
                delta.append([new[i][0], new[i][1], sx, sy])
        return delta
            
    

    def iterate(self):
        old_frame = self.frame.copy()
        self.getFrame()
        if np.any(old_frame == None):
            return
        p1, st, err = cv.calcOpticalFlowPyrLK(old_frame, self.frame, self.p0, None, **self.lk_params)
        self.st = np.add(self.st, st)
        self.st = self.st[st != 0].reshape(-1,1)
        self.trackMemory = self.trackMemory[st.reshape(-1) != 0]
        
        good_new = p1[st != 0]
        diff = self.track(st, good_new)

        self.p0 = good_new.reshape(-1,1,2)
        self.setMask()
        features_number = self.getFeatures()

        vframe = cv.cvtColor(self.frame, cv.COLOR_GRAY2BGR)
        for a,b in good_new:
            vframe = cv.circle(vframe,(a,b),5,(255,0,0),-1)
        cv.imshow('frame',vframe)

        self.diff_frame[:] = (0,0,0)

        for x,y,dx,dy in diff:
            delta = np.sqrt( dx**2 + dy**2 )
            d_color = int(delta/4 * 255/2)
            self.diff_frame = cv.circle(self.diff_frame,(x,y),3,(d_color,0,0),-1)
        cv.imshow('color',self.diff_frame)


S = SLAM(lk_params, feature_params)

while True:
    S.iterate()
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
