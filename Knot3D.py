from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import math


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
    
    R = (1-c) * axis[:3] @ axis[:3].T + c * np.eye(3) + s*M
    
    return R

### Tangle Implementation: #####################################################

class Tangle3D:
    
    def __init__(self, rotationStep):
        
        self.rotationStep = rotationStep
        self.crossingIndex = 0
        self.paths = []
        self.crossingPoints = []
    
        self.NE = None
        self.NW = None
        self.SE = None
        self.SW = None
    
        return
    
    
    def addTwist(self):
        
        t = (self.crossingIndex)*self.rotationStep
        crossing = np.array([[np.cos(t), np.sin(t), 0]]).T
        
        self.crossingPoints += [crossing]
        
        if self.crossingIndex == 0:
            self.SW =  self.crossingIndex+1
            self.NW = -self.crossingIndex-1
        else:            
            self.paths += [ [self.NE, -self.crossingIndex-1] ]
            self.paths += [ [self.SE,  self.crossingIndex+1] ]
            
        
        self.NE =  self.crossingIndex+1
        self.SE = -self.crossingIndex-1

        self.crossingIndex += 1
        
        return
    
    def NW_SE_reflection(self):
        """ A helper function that reflects the current knot across the NW-SE
            diagonal axis.
        """
        if self.crossingIndex == 0:
            return
        
        # Swap NE and SW corners:
        self.NE, self.SW = self.SW, self.NE
        
        # Obtain matrix that flips points along the "main" (NW-SE) diagonal.
        theta = (self.crossingIndex-1)/2 * self.rotationStep
        centerAxis = np.array([[np.cos(theta), np.sin(theta), 0]]).T
        reflectionAxis = rotMatrix(centerAxis, -np.pi/4) @ np.array([[0,0,1]]).T
        reflectionMatrix = np.eye(3) - 2 * reflectionAxis @ reflectionAxis.T
        
        # Modify the incomplete corner curves and the list of completed paths.
        self.crossingPoints = [reflectionMatrix @ point for point in self.crossingPoints]
        
        return
    
    def addTangle(self, otherTangle):
        
        if self.rotationStep != otherTangle.rotationStep:
            raise Exception("Rotation steps between the two tangles disagree.")
                
        # Rotate the second tangle over to match ends appropriately:
        theta = self.crossingIndex * self.rotationStep
        rotationMatrix = rotMatrix(np.array([[0,0,1]]).T, theta)
        otherTangle.paths = [rotationMatrix @ path for path in otherTangle.paths]
        self.crossingPoints += otherTangle.crossingPoints
        
        # Compose the two tangles together:
        if self.crossingIndex == 0:
            self.NW = otherTangle.NW
            self.SW = otherTangle.SW
        else:
            self.paths += [ [self.NE, otherTangle.NW] ]
            self.paths += [ [self.SE, otherTangle.SW] ]
        
        
        self.NE = otherTangle.NE
        self.SE = otherTangle.SE
        self.crossingIndex += otherTangle.crossingIndex
        
        
        for segment in otherTangle.paths:
            i1 = segment[0]
            i2 = segment[1]
            if i1 > 0:
                i1 += self.crossingIndex
            else:
                i1 -= self.crossingIndex
            if i2 > 0:
                i2 += self.crossingIndex
            else:
                i2 -= self.crossingIndex
            self.paths += [ [i1,i2] ]
        
        self.paths += otherTangle.paths
        
        
        return self
    
    def combineEndings(self):
        self.paths += [ [self.NE, self.NW] ]
        self.paths += [ [self.SE, self.SW] ]
        return self
    
    def plusOperator(self):
        self.NW_SE_reflection()
        self.addTwist()
        self.NW_SE_reflection()
        return self
    
    def minusOperator(self):
        self.NW_SE_reflection()
        self.addTwist()
        self.NW_SE_reflection()
        return self
    
    def flipCrossings(self):
        self.paths = [[-p[0], -p[1]] for p in self.paths]
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
        
        if isinstance(n, int) or isinstance(n, Tangle3D):
            if self.crossingIndex != 0:
                self.NW_SE_reflection()
            return self + n
                
        raise Exception("Tangle3D can only be multiplied by integers or tangles.")
    
        



### Knot Implementation: #######################################################

class Knot3D:
    """ A class that transforms the Conway notation of knots into curves embeded
        on the unit sphere.

        Currently, the code only works for Conway codes that correspond to
        knots formed by algebraic tangles.
    """
    
    
    def __init__(self, code, alpha=1.0):
        
        self.code = code
        digits = re.findall(r"\d", code)
        self.numCrossings = sum([int(i) for i in digits])
        self.numCrossings += len(re.findall(r"[\+\-][,\]]", code))
        
        self.rotationStep = 2*np.pi / self.numCrossings
        
        if code[0] != "[" or code[-1] != "]":
            raise Exception("Conway code not in appropriate form.")
        
        self.knot = self.formKnot(code[1:-1])
        self.knot.combineEndings()
        
        return
    
    def separateFirstGroup(self, txt):    
        # A valid separation here are strings of the form (A)B
        # where A and B have balanced parentheses and B does not
        # contain any commas. 
        count = 0
        for i,s in enumerate(txt):
            if   s == "(": 
                count += 1
            elif s == ")": 
                count -= 1

            if count == 0:
                break

        return txt[1:i], txt[i+1:]

    def separateByCommas(self, txt):    
        # Find the positions of valid commas:
        count = 0
        validCommas = []
        for i,s in enumerate(txt):
            if   s == "(": 
                count += 1
            elif s == ")": 
                count -= 1
            elif s == "," and count == 0:
                validCommas += [i]

        validCommas = [-1] + validCommas + [len(txt)]

        # Separate out the string by these positions;
        chunks = []
        numValidCommas = len(validCommas)
        for i in range(numValidCommas-1):
            chunks += [txt[validCommas[i]+1:validCommas[i+1]]]
        return chunks

    def newTangle(self):
        tangle = Tangle3D(self.rotationStep)
        return tangle
    
    def formKnot(self,code):
        """ Form a knot from the provided code. Uses a recursive descendant parser strategy.
        """

        if ("." in code) or ("*" in code) or (":" in code):
            raise Exception("Polynomial knots are not accepted as of yet.")

        if re.match(r"[\d\(\)\,\-\+]+", code) is None:
            raise Exception("Conway code contains errant characters.")


        # If we have multiple components, ramify them together:
        components = self.separateByCommas(code)
        if len(components) > 1:
            tangles = [self.formKnot(comp) for comp in components]
            returnTangle = self.newTangle()
            for tangle in tangles:
                returnTangle += tangle*0
            return returnTangle

        # If the code is just a sequence of digits, form the tangle:
        if code.isdigit():
            returnTangle = self.newTangle()
            for i in code:
                returnTangle *= int(i)
            return returnTangle

        # If the code terminates in a +:
        if code[-1] == "+":
            returnTangle = self.formKnot(code[:-1])
            returnTangle.plusOperator()
            return returnTangle

        # If the code terminates in a -:
        if code[-1] == "-":
            returnTangle = self.formKnot(code[:-1])
            returnTangle.minusOperator()
            return returnTangle

        if code[0] == "-":
            returnTangle = self.formKnot(code[1:])
            returnTangle.flipCrossings()
            return returnTangle

        if code[0] == "(":
            firstCode, secondCode = self.separateFirstGroup(code)
            returnTangle = self.formKnot(firstCode)
            if secondCode != "":
                returnTangle *= self.formKnot(secondCode)

            return returnTangle

        raise Exception("Could not parse the Conway code for this knot.")

        return code
    
    

### Example Usage: #############################################################

code = "[32212]"
k = Knot3D(code,alpha=0.001)

fig = plt.figure()
ax = plt.axes(projection='3d')


points = k.knot.crossingPoints
for point in points:
    ax.scatter3D(point[0,0], point[1,0], point[2,0], color="black")

    
for path in k.knot.paths:
    point1 = points[abs(path[0])-1]
    point2 = points[abs(path[1])-1]
    s1 = 1 + 0.2 * (path[0]>0)
    s2 = 1 + 0.2 * (path[1]>0)
    
    ax.plot([point1[0,0]*s1, point2[0,0]*s2], 
            [point1[1,0]*s1, point2[1,0]*s2], 
            [point1[2,0]*s1, point2[2,0]*s2], 
                  color="red")
    
ax.set_xlim3d([-1.3, 1.3])
ax.set_ylim3d([-1.3, 1.3])
ax.set_zlim3d([-1  , 1  ])

plt.title(f"Plot of knot with Conway code {k.code}")
plt.show()