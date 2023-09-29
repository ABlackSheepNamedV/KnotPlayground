import cv2
import scipy.ndimage as ndimage
import sympy
import numpy as np

class KnotDrawing:
    """ A class for reading in images of knots and determining their
        corresponding Alexander polynomial coefficients. 

        The Alexander polynomial is an invariant of knots that helps identify
        between similar but distinct knots. The polynomial is calculated by
        setting up an (n,n+2) matrix with the rows corresponding to the n
        crossings and the n+2 regions separated by the knot. We then trace along
        the knot, where at each undercrossing we populate the matrix with:
            +t to the region after the crossing to the left
            -1 to the region after the crossing to the right
            -t to the region before the crossing to the left
            +1 to the region before the crossing to the right.
            +0 to regions not adjacent to the crossing. 

        For more detailed information, see the Wikipedia page:
        https://en.wikipedia.org/wiki/Alexander_polynomial
    """
    
    def __init__(self, filename):
        """ Initialize the KnotDrawing object from a file. 

            We assume that the knot consists of a png file with white, black,
            and gray pixels. White and gray pixels outline the knot, with gray
            pixels indicating that a portion of the knot is next to an
            undercrossing. 
        """
        knotImage = cv2.imread(filename)
        self.knotData = knotImage[:,:,0]



    ### Knot tracing methods:

    def initPath(self):
        """ Initialize a path used to help trace along the boundary of a knot. 
        """
        knotPath = np.where(self.knotData==255)
        
        self.start = np.array([knotPath[0][0], knotPath[1][0]])
        self.pos = np.copy(self.start)
        self.step = np.array([1,0])
        self.turn = np.array([[0,1],[-1,0]])
        
        return

    def givePos(self, direction="none"):
        """ Provides the indices for the pixel in the image one step towards
            the specified direction.
        """
        
        position = np.copy(self.pos)
        forward = self.step
        left = self.step @ self.turn
        
        if "front" in direction: position += forward
        if "back"  in direction: position -= forward
        if "left"  in direction: position += left
        if "right" in direction: position -= left
            
        return tuple(position)
        
        
    def atUndercrossing(self):
        """ Provides a boolean specifying if the tracing has arrived at an
            undercrossing. 
        """
        checkF = self.knotData[self.givePos("front")]
        checkB = self.knotData[self.givePos("back" )]

        undercrossingF = (checkF!=0 and checkF!=255)
        undercrossingB = (checkB!=0 and checkB!=255)

        return (undercrossingF and undercrossingB)
        
    def updateStepDirection(self):
        """ Updates the forward facing direction for the current knot tracing.
        """
        
        checkF = self.knotData[self.givePos("front")]
        checkL = self.knotData[self.givePos("left" )]
        checkR = self.knotData[self.givePos("right")]

        if checkL>0 and checkF==0:
            self.step = self.step @ self.turn
        elif checkR>0 and checkF==0:
            self.step = self.step @ -self.turn
        
        return



    ### Alexander matrix methods:

    def initMatrix(self):
        """ Initialize the matrix used to calculate the Alexander polynomial.

            The polynomial utilizes the determinant of a (n,n+2) matrix that
            corresponds to the n crossings as rows and the n+2 regions separated
            by the knot as columns.

            See https://en.wikipedia.org/wiki/Alexander_polynomial for more
            details.
        """

        # Identify regions
        self.regions = ndimage.label(self.knotData==0)[0] - 1
        self.numRegions = np.max(self.regions) + 1
        self.numCrossings = self.numRegions-2
        
        # Initialize records for the Alexander matrix
        self.t = sympy.symbols("t")
        self.alexMat = sympy.zeros(self.numCrossings, self.numRegions)
        self.crossingCount = 0
        
        return

    def updateMatrix(self):
        """ Based on the current position and facing of the knot tracing,
            update the matrix used to calculate the Alexander polynomial.

            At undercrossings, we provide a -t, +1, +t, -t to the regions
            immediately to the backleft, backright, frontleft, and frontright
            respectively. 

            See https://en.wikipedia.org/wiki/Alexander_polynomial for more
            details.
        """
        
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
        """ Create the matrix used to calculate the Alexander polynomial knot
            invariant. See https://en.wikipedia.org/wiki/Alexander_polynomial
            for more details.
        """

        self.initPath()
        self.initMatrix()
        
        for stepIndex in range(5000):

            if self.atUndercrossing():
                self.updateMatrix()

            self.updateStepDirection()
            self.pos += self.step
            
            distanceToStart = np.linalg.norm(self.pos - self.start)
            if distanceToStart < 0.1:
                break
                
    def alexPoly(self):
        """ Calculate the Alexander polynomial for the knot of interest.
        """
        self.createAlexMatrix()
        alexanderDet = self.alexMat[:,2:].det()
        alexanderPolynomial = sympy.Poly(alexanderDet,self.t)
        coeffs = alexanderPolynomial.coeffs()
        
        if coeffs[0] < 0:
            coeffs = [-s for s in coeffs]
        
        return coeffs



index = 1
knot = KnotDrawing(f"images/9_{index}.png")
print(knot.alexPoly())