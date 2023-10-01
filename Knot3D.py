from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

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
    
    R = (1-c) * axis @ axis.T + c * np.eye(3) + s * M
    
    
    return R

def slerp(axis1, axis2, t):
    """ Computes the slerp (Spherical Linear intERPolation) between the two axes.
        That is, we find the geodesic path f(t) between the two axes such that
        f(0)=axis1, f(1)=axis2, and ||f'(t)|| is constant.
        
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


class Knot3D:
    
    def __init__(self, code, alpha=1.0, crossingAngle=np.pi/2):
        self.cornerNW = None
        self.cornerNE = None
        self.cornerSW = None
        self.cornerSE = None
        self.paths = []
        
        self.heights = []
        self.h = 1
        
        self.alpha=alpha
        self.crossingAngle = crossingAngle
        
        self.completedCrossings = 0
        self.code = code
        
        numCrossings = sum(code)
        self.rotationStep = (2*np.pi/numCrossings)
        
        self.constructPoints()
        
    def addTwist(self):
        
        t = (self.completedCrossings)*self.rotationStep
        crossing = np.array([[np.cos(t),np.sin(t),0]]).T

        t2 = (self.completedCrossings+self.alpha/2)* self.rotationStep
        diff = np.array([[np.cos(t2),np.sin(t2),0]]).T

        NE = rotMatrix(crossing,         self.crossingAngle/2) @ diff
        NW = rotMatrix(crossing, np.pi - self.crossingAngle/2) @ diff
        SE = rotMatrix(crossing,       - self.crossingAngle/2) @ diff
        SW = rotMatrix(crossing, np.pi + self.crossingAngle/2) @ diff
        
        NE_height =  self.h
        NW_height = -self.h
        SW_height =  self.h
        SE_height = -self.h
        
        if self.completedCrossings == 0:
            self.cornerSW = np.hstack((crossing, SW))
            self.cornerNW = np.hstack((crossing, NW))
            
            self.cornerSW_height = SW_height
            self.cornerNW_height = NW_height
            
        else:            
            self.paths += [ np.hstack( (self.cornerNE, NW, crossing) ) ]
            self.paths += [ np.hstack( (self.cornerSE, SW, crossing) ) ]
        
            self.heights += [ [self.cornerNE_height, NW_height] ]
            self.heights += [ [self.cornerSE_height, SW_height] ]
        
        self.cornerNE = np.hstack( (crossing, NE) )
        self.cornerSE = np.hstack( (crossing, SE) )
        
        self.cornerNE_height = NE_height
        self.cornerSE_height = SE_height
            
        self.completedCrossings += 1
        
        return
    
    def addRotate(self):
        
        #Cycle through the corners:
        temp = self.cornerNE
        self.cornerNE = self.cornerSE
        self.cornerSE = self.cornerSW
        self.cornerSW = self.cornerNW
        self.cornerNW = temp
        
        temp = self.cornerNE_height
        self.cornerNE_height = self.cornerSE_height
        self.cornerSE_height = self.cornerSW_height
        self.cornerSW_height = self.cornerNW_height
        self.cornerNW_height = temp
        
        
        theta = (self.completedCrossings/2-0.5) * self.rotationStep
        rotPoint = np.array([[np.cos(theta), np.sin(theta), 0]]).T
        R = rotMatrix(rotPoint, np.pi/2)
        
        self.cornerNW = R @ self.cornerNW
        self.cornerNE = R @ self.cornerNE
        self.cornerSW = R @ self.cornerSW
        self.cornerSE = R @ self.cornerSE
        
        self.paths = [R @ path for path in self.paths]
        
        self.h *= -1
        
        
        return
    
    def constructPoints(self):
        
        for t in self.code:
            for i in range(t):
                self.addTwist()
            self.addRotate()
        
        self.paths += [np.hstack((self.cornerNE, self.cornerSE[:,::-1]))]
        self.paths += [np.hstack((self.cornerNW, self.cornerSW[:,::-1]))]
        
        self.heights += [ [self.cornerNE_height, self.cornerSE_height] ]
        self.heights += [ [self.cornerSE_height, self.cornerSW_height] ]
    
        return


# k = Knot3D([9])
# k = Knot3D([5,4])
# k = Knot3D([2,1,3,1,2])
k = Knot3D([2,2,1,2,2])
# k = Knot3D([2,3,2,2])

fig = plt.figure()
ax = plt.axes(projection='3d')

t = np.linspace(0,1, 50)

for i in range(len(k.paths)):
    path = k.paths[i]
    height = k.heights[i]
    altPath = sphereBezier(path, t) 
    altPath *= 1 + (0.1 * np.cos(t * np.pi) * height[0])


    ax.plot(altPath[0], altPath[1], altPath[2], color="black")
    
h = 1.2
ax.set_xlim3d([-h,h])
ax.set_ylim3d([-h,h])
ax.set_zlim3d([-1,1])


modifiedCode = str(k.code).replace(" ","").replace(",","")
plt.title("Plot of knot with Conway code "+modifiedCode)
plt.show()