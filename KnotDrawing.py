import cv2
import scipy.ndimage as ndimage
import sympy
import numpy as np

class KnotDrawing:
    
    def __init__(self, filename):
        knotImage = cv2.imread(filename)
        self.knotData = knotImage[:,:,0]
        
    def initPath(self):
        knotPath = np.where(self.knotData==255)
        
        self.start = np.array([knotPath[0][0], knotPath[1][0]])
        self.pos = np.copy(self.start)
        self.step = np.array([1,0])
        self.turn = np.array([[0,1],[-1,0]])
        
        return

    def initMatrix(self):
        # Identify regions
        self.regions = ndimage.label(self.knotData==0)[0] - 1
        self.numRegions = np.max(self.regions) + 1
        self.numCrossings = self.numRegions-2
        
        # Initialize records for the Alexander matrix
        self.t = sympy.symbols("t")
        self.alexMat = sympy.zeros(self.numCrossings, self.numRegions)
        self.crossingCount = 0
        
        return
    
    def givePos(self, direction="none"):
        
        position = np.copy(self.pos)
        forward = self.step
        left = self.step @ self.turn
        
        if "front" in direction: position += forward
        if "back"  in direction: position -= forward
        if "left"  in direction: position += left
        if "right" in direction: position -= left
            
        return tuple(position)
        
        
    def atUndercrossing(self):
        checkF = self.knotData[self.givePos("front")]
        checkB = self.knotData[self.givePos("back" )]
        return (checkF==116 and checkB==116)
        
    def updateStepDirection(self):
        
        checkF = self.knotData[self.givePos("front")]
        checkL = self.knotData[self.givePos("left" )]
        checkR = self.knotData[self.givePos("right")]

        if checkL>0 and checkF==0:
            self.step = self.step @ self.turn
        elif checkR>0 and checkF==0:
            self.step = self.step @ -self.turn
        
        return
    
    def updateAlexMatrix(self):
        
        regionBL = self.regions[ self.givePos("back-left"  ) ]
        self.alexMat[self.crossingCount, regionBL] = -self.t
        
        regionBR = self.regions[ self.givePos("back-right" ) ]
        self.alexMat[self.crossingCount, regionBR] =  1
        
        regionFL = self.regions[ self.givePos("front-left" ) ]
        self.alexMat[self.crossingCount, regionFL] =  self.t
        
        regionFR = self.regions[ self.givePos("front-right") ]
        self.alexMat[self.crossingCount, regionFR] = -1
         
        self.crossingCount += 1
        
        return
        
    def createAlexMatrix(self):

        self.initPath()
        self.initMatrix()
        
        for stepIndex in range(5000):

            if self.atUndercrossing():
                self.updateAlexMatrix()

            self.updateStepDirection()
            self.pos += self.step

            
            distanceToStart = np.linalg.norm(self.pos - self.start)
            if distanceToStart < 0.1:
                break
                
    def alexPoly(self):
        alexanderDet = self.alexMat[:,2:].det()
        alexanderPolynomial = sympy.Poly(alexanderDet,self.t)
        coeffs = alexanderPolynomial.coeffs()
        
        if coeffs[0] < 0:
            coeffs = [-s for s in coeffs]
        
        return coeffs



index = 1
knot = KnotDrawing(f"images/9_{index}.png")
knot.createAlexMatrix()
print(knot.alexPoly())