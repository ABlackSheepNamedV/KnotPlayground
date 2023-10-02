from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



### Helper functions: ##########################################################

def rotMatrix(axis, theta):
    """ Rotate about the specified axis by theta radians, using the right
        hand rule to establish the direction. Assumes axis is a (3,1) array.
    """

    c = np.cos(theta)
    s = np.sin(theta)
    x = axis[0,0]
    y = axis[1,0]
    z = axis[2,0]
    
    M = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    
    R = np.zeros( (4,4) )
    R[:3,:3] = (1-c) * axis[:3] @ axis[:3].T + c * np.eye(3) + s * M
    R[3,3] = 1
    
    return R

def slerp(axis1, axis2, t):
    """ Computes the slerp (Spherical Linear intERPolation) between the two 
        axes. That is, we find the constant speed path f(t) between the two axes
        such that f(0)=axis1, f(1)=axis2, ||f(t)|| = 1.
        
        See https://en.wikipedia.org/wiki/Slerp for more details.
    """
    
    omega = np.arccos(np.sum(axis1 * axis2, axis=0))
    
    path  = np.sin( (1-t)*omega ) / np.sin(omega) * axis1
    path += np.sin(    t *omega ) / np.sin(omega) * axis2
    
    return path

def sphereBezier(path, t):
    """ Computes a Bezier curve-esque path along the unit sphere using slerp
        (Spherical Linear intERPolation).
    """
    p1 = path[:, :1]
    p2 = path[:,1:2]
    p3 = path[:,2:3]
    p4 = path[:,3: ]
    
    p12 = slerp(p1, p2, t)
    p23 = slerp(p2, p3, t)
    p34 = slerp(p3, p4, t)
    
    p123 = slerp(p12, p23, t)
    p234 = slerp(p23, p34, t)
    
    path = slerp(p123, p234, t)
    
    return path



### Tangle Implementation: #####################################################

class Tangle3D:
    
    def __init__(self, rotationStep, alpha=1, crossingAngle=np.pi/2 ):
        
        self.rotationStep = rotationStep
        self.crossingIndex = 0
        self.alpha = alpha
        self.crossingAngle = crossingAngle
        self.paths = []
        
        return
    
    def addTwist(self, negate=False):
        
        t = (self.crossingIndex)*self.rotationStep
        SWNE_cross = np.array([[np.cos(t), np.sin(t), 0,  1]]).T
        NWSE_cross = np.array([[np.cos(t), np.sin(t), 0, -1]]).T
        
        # Obtain the positions of the control points:
        t2 = (self.crossingIndex+self.alpha/2)* self.rotationStep
        controlDist_upper = np.array([[np.cos(t2), np.sin(t2), 0,  1]]).T
        controlDist_lower = np.array([[np.cos(t2), np.sin(t2), 0, -1]]).T

        rot180         = rotMatrix(SWNE_cross,  np.pi               )
        rotCrossing    = rotMatrix(SWNE_cross,  self.crossingAngle/2)
        rotCrossingNeg = rotMatrix(SWNE_cross, -self.crossingAngle/2)
        
        NE_cont = rotCrossing @ controlDist_upper
        SW_cont = rot180 @ NE_cont
        
        SE_cont = rotCrossingNeg @ controlDist_lower
        NW_cont = rot180 @ SE_cont        
        
        if negate:
            flipCrossing = np.diag([1,1,1,-1])
            SWNE_cross = flipCrossing @ SWNE_cross
            NWSE_cross = flipCrossing @ NWSE_cross
            NE_cont = flipCrossing @ NE_cont
            NW_cont = flipCrossing @ NW_cont
            SE_cont = flipCrossing @ SE_cont
            SW_cont = flipCrossing @ SW_cont
        
        # Modify the paths used within the knot
        if self.crossingIndex == 0:
            self.cornerSW = np.hstack((NWSE_cross, SW_cont))
            self.cornerNW = np.hstack((SWNE_cross, NW_cont))
        else:            
            self.paths += [ np.hstack( (self.cornerNE, NW_cont, SWNE_cross) ) ]
            self.paths += [ np.hstack( (self.cornerSE, SW_cont, NWSE_cross) ) ]
        
        self.cornerNE = np.hstack( (NWSE_cross, NE_cont) )
        self.cornerSE = np.hstack( (SWNE_cross, SE_cont) )

        self.crossingIndex += 1
        
        return
    
    def NW_SE_reflection(self):
        """ A helper function that reflects the current knot across the NW-SE
            diagonal axis.
        """
        
        # Swap NE and SW corners:
        self.cornerNE, self.cornerSW = self.cornerSW, self.cornerNE
        
        # Obtain matrix that flips points along the "main" (NW-SE) diagonal.
        theta = (self.crossingIndex-1)/2 * self.rotationStep
        centerAxis = np.array([[np.cos(theta), np.sin(theta), 0, 0]]).T
        reflectionAxis = rotMatrix(centerAxis, -np.pi/4) @ np.array([[0,0,1,0]]).T
        reflectionMatrix = np.eye(4) - 2 * reflectionAxis @ reflectionAxis.T
        
        # Modify the incomplete corner curves and the list of completed paths.
        self.cornerNW = reflectionMatrix @ self.cornerNW
        self.cornerNE = reflectionMatrix @ self.cornerNE
        self.cornerSW = reflectionMatrix @ self.cornerSW
        self.cornerSE = reflectionMatrix @ self.cornerSE
        self.paths = [reflectionMatrix @ path for path in self.paths]
        
        return
    
    def addTangle(self, otherTangle):
        
        if self.rotationStep != otherTangle.rotationStep:
            raise Exception("Rotation steps between the two tangles disagree.")
        
        # Rotate the second tangle over to match ends appropriately:
        theta = self.crossingIndex * self.rotationStep
        rotationMatrix = rotMatrix(np.array([[0,0,1,0]]).T, theta)
        
        otherTangle.cornerNW = rotationMatrix @ otherTangle.cornerNW
        otherTangle.cornerNE = rotationMatrix @ otherTangle.cornerNE
        otherTangle.cornerSW = rotationMatrix @ otherTangle.cornerSW
        otherTangle.cornerSE = rotationMatrix @ otherTangle.cornerSE
        otherTangle.paths = [rotationMatrix @ path for path in otherTangle.paths]
        
        
        # Compose the two tangles together:
        self.paths += [ np.hstack( (self.cornerNE, otherTangle.cornerNW[:,::-1]) ) ]
        self.paths += [ np.hstack( (self.cornerSE, otherTangle.cornerSW[:,::-1]) ) ]
        
        self.cornerNE = otherTangle.cornerNE
        self.cornerSE = otherTangle.cornerSE
        self.crossingIndex += otherTangle.crossingIndex
        self.paths += otherTangle.paths
                
        return self
    
    def combineEndings(self):
        
        self.paths += [ np.hstack( (self.cornerNE, self.cornerNW[:,::-1]) ) ]
        self.paths += [ np.hstack( (self.cornerSE, self.cornerSW[:,::-1]) ) ]
        
        return self
    
    def plusOperator(self):
        self.NW_SE_reflection()
        self.addTwist()
        self.NW_SE_reflection()
        return self
    
    def minusOperator(self):
        self.NW_SE_reflection()
        self.addTwist(negate=True)
        self.NW_SE_reflection()
        return self
    
    
    def __add__(self, n):
        
        if isinstance(n, int) and n >= 0:
            for i in range(n):
                self.addTwist()
        
        elif isinstance(n, int) and n < 0:
            for i in range(-n):
                self.addTwist(negate=True)
        
        elif isinstance(n, Tangle3D):
            self.addTangle(n)
            
        else:
            raise Exception("Only integers and tangles can be added to a tangle.")
            
        return self
    
    def __mul__(self, n):
        
        if not isinstance(n, int):
            raise Exception("Cannot multiply tangles by non-integer values.")
    
        if self.crossingIndex != 0:
            self.NW_SE_reflection()
        return self + n



### Knot Implementation: #######################################################

class Knot3D(Tangle3D):
    """ A class that transforms the Conway notation of knots into curves embeded
        on the unit sphere.
        
        The class produces a list of 4x4 matrices representing each curve
        connecting the different crossings together. For each matrix, the
        columns represent:
            - the first crossing
            - a curve control point for the first crossing
            - a curve control point for the second crossing
            - the second crossing
        in order.
        
        The first three rows correspond to the (x,y,z) coordinates of these
        points. The last row is an indicator of whether the crossing corresponds
        to an over crossing (+1) or an undercrossing (-1).

        Currently, the code only works for Conway codes that correspond to
        knots formed by completing rational tangles. 
    """
    
    
    def __init__(self, code, alpha=1.0, crossingAngle=np.pi/2):
        """ Initialize the Knot3D class.
        
            Inputs:
                - code: A list of non-negative integers representing the conway
                    code of the knot of interest.
                - alpha: A float describing the spherical distance between the
                    crossing and the control points compared to the spherical
                    distance between adjacent crossings.
                - crossingAngle: A float describing the angle of separation
                    at each crossing. 
        """
        
        self.code = code
        numCrossings = sum(code)
        rotationStep = 2 * np.pi / numCrossings
        super().__init__(rotationStep, alpha=alpha, crossingAngle=crossingAngle)
        
        for t in self.code:
            self *= t
        
        self.combineEndings()
        
        return
    
    

### Example Usage: #############################################################

%matplotlib notebook

# Forming the knot [211,21,2]

theta = 2*np.pi/9
k  = (Tangle3D(theta) * 2 * 1) 
k += (Tangle3D(theta) * 2 * 1) 
k += (Tangle3D(theta) * 2 * 1).minusOperator()
k.combineEndings()

fig = plt.figure()
ax = plt.axes(projection='3d')

t = np.linspace(0,1, 50)
deltaH = 0.1
for i in range(len(k.paths)):
    path = k.paths[i]
    altPath = sphereBezier(path[:3], t)
    
    if path[3,0] == 1 and path[3,3] == 1:
        altPath *= 1+deltaH
    elif path[3,0] == 1 and path[3,3] == -1:
        altPath *= 1 + deltaH * np.cos(t * np.pi)
    elif path[3,0] == -1 and path[3,3] == 1:
        altPath *= 1 + deltaH * np.cos((1-t) * np.pi)
    else:
        altPath *= 1-deltaH
        

    ax.plot(altPath[0], altPath[1], altPath[2], linewidth=3) #, color="black")

    
ax.set_xlim3d([-1.3, 1.3])
ax.set_ylim3d([-1.3, 1.3])
ax.set_zlim3d([-1  , 1  ])

plt.title("Plot of knot with Conway code [211,21,2]")
plt.show()


# t1 = Tangle3D(2 * np.pi / 9) * 1 * 2
# t2 = Tangle3D(2 * np.pi / 9) * 4 * 1

# t1 + t2