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



### Knot Implementation: #######################################################

class Knot3D:
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
        self.alpha = alpha
        self.crossingAngle = crossingAngle

        self.paths = []
        self.NW_SE_flip = np.diag([1,1,1,-1])
        
        self.crossingIndex = 0
        self.rotationStep = (2*np.pi/sum(code))
        
        self.constructPoints()
    
    def addOne(self):
        """ A helper function that adds an additional twist (or crossing) to the
            knot.

            Each new crossing is placed at (cos(i * 2pi/n), sin(i*2pi/n), 0)
            where i is the number of crossings made so far and n is the total
            number of crossings to be added. 
        """
        
        # Obtain the position of the crossing:
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
    


    def timesZero(self):
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
    
    def constructPoints(self):
        """ Constructs the knot based off of the recorded Conway code.
        """
        
        # Construct the knot:
        for t in self.code:
            for i in range(t):
                self.addOne()
            self.timesZero()
        self.timesZero()

        # Connect the ends of the paths together:
        self.paths += [np.hstack((self.cornerNE, self.cornerNW[:,::-1]))]
        self.paths += [np.hstack((self.cornerSE, self.cornerSW[:,::-1]))]
    
        return



### Example Usage: #############################################################

# code = [9]
# code = [5,4]
code = [2,1,4,2]
# code = [2,2,1,2,2]
# code = [2,3,2,2]

k = Knot3D(code, crossingAngle=1.0, alpha=1)


fig = plt.figure()
ax = plt.axes(projection='3d')

t = np.linspace(0,1, 50)
for i in range(len(k.paths)):
    path = k.paths[i]
    altPath = sphereBezier(path[:3], t) 
    altPath *= 1 + (0.03 * np.cos(t * np.pi) * path[3,0])


    ax.plot(altPath[0], altPath[1], altPath[2], linewidth=3, color="black")
    
ax.set_xlim3d([-1.3, 1.3])
ax.set_ylim3d([-1.3, 1.3])
ax.set_zlim3d([-1  , 1  ])


modifiedCode = str(code).replace(" ","").replace(",","")
plt.title("Plot of knot with Conway code "+modifiedCode)
plt.show()