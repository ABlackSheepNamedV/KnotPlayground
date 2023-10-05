from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import regex as re


### Helper functions: ##########################################################

def getRotationMatrix(axis, theta):
    """ Computes a (4,4) rotation matrix about the specified axis.

        Inputs:
            - axis: a (4,1) array of specifying (x,y,z,c) coordinates where
                (x,y,z) corresponds to a unit vector to be rotated about and c 
                is arbitrary
            - theta: a float specifying how far rotation is performed.

        Outputs:
            A (4,4) diagonal matrix where the upper 3x3 entries are a rotation
            matrix about the corresponding (x,y,z) of theta radians, and the
            lower right entry is 1.
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
    """ Computes the slerp (Spherical Linear intERPolation) between two axes.
        
        See https://en.wikipedia.org/wiki/Slerp for more details.

        Inputs:
            - axis1: a (3,1) array containing the (x,y,z) coordinates of a unit
                vector.
            - axis2: a (3,1) array containing the (x,y,z) coordinates of a
                second unit vector
            - t: a 1-dimension vector (or float) of values in the interval [0,1]

        Outputs:
            A (3,n) array of the form [F(t[0]), F(t[1]), ..., F(t[n])] where
            F(t) is a continuous function of unit vectors where F(0) = axis1,
            F(1) = axis2, and ||F'(t)|| is constant.

            If t is a float, then only F(t) is returned.
    """
    
    omega = np.arccos(np.sum(axis1 * axis2, axis=0))
    
    path  = np.sin( (1-t)*omega ) / np.sin(omega) * axis1
    path += np.sin(    t *omega ) / np.sin(omega) * axis2
    
    return path

def sphereBezier(path, t):
    """ Computes a Bezier curve on a unit sphere based on the specified points.

        Inputs: 
            - path: a (3,4) array specifying four points on the unit sphere to
                be used for the Bezier curve.
            - t: a float or 1-dimensional vector of values in the interval [0,1]
        
        Outputs:
            A (3,n) array of the form [F(t[0]), F(t[1]), ..., F(t[n])] where
            F(t) is the Bezier curve between path[:,0] and path[:,3] as the
            endpoints and path[:,1] and path[:,2] as the respective control
            points.

            If t is a float, then only F(t) is returned.

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
    """ A class that creates three-dimensional embeddings of curves used to
        represent tangles in knot theory.
    """
    
    def __init__(self, rotationStep, alpha=1, crossingAngle=np.pi/2 ):
        """ Initializes a trivial tangle that is embedded on the unit sphere.

            Inputs:
                - rotationStep: A float representing how far successive
                    crossings are from one another on the unit sphere as
                    measured in radians.
                - alpha: A float representing the ratio of the distance between
                    control points to their respective crossings to the
                    rotationStep. (Default is 1)
                - crossingAngle: A float between representing the angle between
                    curves at crossings. The value should be in the interval 
                    [0,pi]. Larger values correspond to larger gaps in the paths
                    between successive crossings. (Default is pi/2)
        """
        
        self.rotationStep = rotationStep
        self.crossingCount = 0
        self.alpha = alpha
        self.crossingAngle = crossingAngle
        self.paths = []
        
        return
    
    def addTwist(self, negate=False):
        """ Adds a twist to the current tangle.

            Inputs:
                - negate: A boolean specifying which direction is the over
                    crossing. If false, the SW-NE direction is made the over
                    crossing. If true, the NW-SE direction is made the over
                    crossing. (Default is False).

            Output:
                The current tangle, with an extra crossing formed by pulling the
                SE endpoint of the tangle over the NE endpoint. If negate is set
                to True, then the crossing is made by pulling the NE endpoint
                over the SE endpoint instead.
        """
        
        # Obtain position of the new crossing
        t = (self.crossingCount)*self.rotationStep
        SWNE_cross = np.array([[np.cos(t), np.sin(t), 0,  1]]).T
        NWSE_cross = np.array([[np.cos(t), np.sin(t), 0, -1]]).T
        
        # Obtain positions of the new control points
        t2 = (self.crossingCount+self.alpha/2)* self.rotationStep
        controlDist_upper = np.array([[np.cos(t2), np.sin(t2), 0,  1]]).T
        controlDist_lower = np.array([[np.cos(t2), np.sin(t2), 0, -1]]).T

        rot180         = getRotationMatrix(SWNE_cross,  np.pi               )
        rotCrossing    = getRotationMatrix(SWNE_cross,  self.crossingAngle/2)
        rotCrossingNeg = getRotationMatrix(SWNE_cross, -self.crossingAngle/2)
        
        NE_cont = rotCrossing @ controlDist_upper
        SW_cont = rot180 @ NE_cont
        
        SE_cont = rotCrossingNeg @ controlDist_lower
        NW_cont = rot180 @ SE_cont        
        
        # If negating, we swap crossing info for crossing and control points
        if negate:
            flipCrossing = np.diag([1,1,1,-1])
            SWNE_cross = flipCrossing @ SWNE_cross
            NWSE_cross = flipCrossing @ NWSE_cross
            NE_cont = flipCrossing @ NE_cont
            NW_cont = flipCrossing @ NW_cont
            SE_cont = flipCrossing @ SE_cont
            SW_cont = flipCrossing @ SW_cont
        
        # Modify the paths used within the knot
        if self.crossingCount == 0:
            self.cornerSW = np.hstack((NWSE_cross, SW_cont))
            self.cornerNW = np.hstack((SWNE_cross, NW_cont))
        else:            
            self.paths += [ np.hstack((self.cornerNE, NW_cont, SWNE_cross)) ]
            self.paths += [ np.hstack((self.cornerSE, SW_cont, NWSE_cross)) ]
        
        self.cornerNE = np.hstack( (NWSE_cross, NE_cont) )
        self.cornerSE = np.hstack( (SWNE_cross, SE_cont) )

        self.crossingCount += 1
        
        return
    
    def NW_SE_reflection(self):
        """ A helper function that reflects the current knot across the NW-SE
            diagonal axis.

            Output:
                The current tangle, where all crossings and curves are reflected
                such that the NE and SW corners are swapped with one another.
        """

        # If there are no crossings, do nothing
        if self.crossingCount == 0:
            return
        
        # Swap NE and SW corners:
        self.cornerNE, self.cornerSW = self.cornerSW, self.cornerNE
        
        # Obtain the reflection matrix for the NW-SE diagonal.
        theta = (self.crossingCount-1)/2 * self.rotationStep
        centerAxis = np.array([[np.cos(theta), np.sin(theta), 0, 0]]).T
        zAxis = np.array([[0,0,1,0]]).T
        reflAxis = getRotationMatrix(centerAxis, -np.pi/4) @ zAxis
        reflMatrix = np.eye(4) - 2 * reflAxis @ reflAxis.T
        
        # Modify the incomplete corner curves and the list of completed paths.
        self.cornerNW = reflMatrix @ self.cornerNW
        self.cornerNE = reflMatrix @ self.cornerNE
        self.cornerSW = reflMatrix @ self.cornerSW
        self.cornerSE = reflMatrix @ self.cornerSE
        self.paths = [reflMatrix @ path for path in self.paths]
        
        return
    
    def addTangle(self, otherTangle):
        """ Computes the sum between two tangles.

            Inputs:
                - otherTangle: a Tangle3D to be added to the current tangle.

            Outputs:
                The current tangle with the given tangle added on. In tangle
                addition, the other tangle is shifted over, and the eastern ends
                of the current tangle are combined with the western ends of the
                provided other tangle to be added.
        """
        
        if self.rotationStep != otherTangle.rotationStep:
            raise Exception("Rotation steps between the two tangles disagree.")
                
        # Rotate the second tangle over to match ends appropriately:
        theta = self.crossingCount * self.rotationStep
        rotMatrix = getRotationMatrix(np.array([[0,0,1,0]]).T, theta)
        
        otherTangle.cornerNW = rotMatrix @ otherTangle.cornerNW
        otherTangle.cornerNE = rotMatrix @ otherTangle.cornerNE
        otherTangle.cornerSW = rotMatrix @ otherTangle.cornerSW
        otherTangle.cornerSE = rotMatrix @ otherTangle.cornerSE
        otherTangle.paths = [rotMatrix @ path for path in otherTangle.paths]
        
        # Compose the two tangles together:
        if self.crossingCount == 0:
            self.cornerNW = otherTangle.cornerNW
            self.cornerSW = otherTangle.cornerSW
        else:
            path1 = np.hstack( (self.cornerNE, otherTangle.cornerNW[:,::-1]) )
            path2 = np.hstack( (self.cornerSE, otherTangle.cornerSW[:,::-1]) )
            self.paths += [path1, path2]
        
        self.cornerNE = otherTangle.cornerNE
        self.cornerSE = otherTangle.cornerSE
        self.crossingCount += otherTangle.crossingCount
        self.paths += otherTangle.paths
                
        return self
    
    def combineEndings(self):
        """ Merges together the endings of the four corners of the tangle.

            Outputs:
                The current tangle where the NW and NE ends have been combined
                together and the SW and SE ends have been combined together.
        """
        self.paths += [ np.hstack( (self.cornerNE, self.cornerNW[:,::-1]) ) ]
        self.paths += [ np.hstack( (self.cornerSE, self.cornerSW[:,::-1]) ) ]
        return self
    
    def plusOperator(self):
        """ Applies the + operation to the tangle. That is, an additional
            crossing is made by pulling the SE ending of the tangle over the
            SW ending.

            Output:
                The current tangle, but with an additional crossing.
        """
        self.NW_SE_reflection()
        self.addTwist()
        self.NW_SE_reflection()
        return self
    
    def minusOperator(self):
        """ Applies the - operation to the tangle. That is, an additional
            crossing is made by pulling the SW ending of the tangle over the
            SE ending.

            Output:
                The current tangle, but with an additional crossing.
        """
        self.NW_SE_reflection()
        self.addTwist(negate=True)
        self.NW_SE_reflection()
        return self
    
    def flipCrossings(self):
        """ Inverts all crossings in the current tangle. All overcrossings
            become undercrossings and vice versa.

            Outputs:
                The current tangle, but with all crossings inverted.
        """

        flipCrossings = np.diag([1,1,1,-1])
        self.cornerNW = flipCrossings @ self.cornerNW
        self.cornerNE = flipCrossings @ self.cornerNE
        self.cornerSW = flipCrossings @ self.cornerSW
        self.cornerSE = flipCrossings @ self.cornerSE
        self.paths = [flipCrossings @ path for path in self.paths]
        return self
        


    def __add__(self, val):
        """ Computes the sum of the current tangle and a specified integer or
            tangle.

            Inputs:
                - val: either an integer or a Tangle3D to be added

            Outputs:
                The current tangle, but after the addition. If val is an int,
                the specified number of twists are appended to the NE/SE
                endpoints. If val is a tangle, then the crossings of val are
                appended to the eastern side of the current tangle.
        """
        
        if isinstance(val, int) and val >= 0:
            for i in range(val):
                self.addTwist()
        
        elif isinstance(val, int) and val < 0:
            for i in range(-val):
                self.addTwist(negate=True)
        
        elif isinstance(val, Tangle3D):
            self.addTangle(val)
            
        else:
            outputStr = "Only integers or tangles can be added to a tangle."
            raise Exception(outputStr)
            
        return self
    
    def __mul__(self, val):
        """ Computes the product of the current tangle and a specified integer
            or tangle.

            Inputs:
                - val: either an integer or a Tangle3D to be multiplied.

            Outputs:
                The current tangle after the product has been performed.
                Note that tangle products are computed as a*b = a0 + b where
                a0 is the reflection of a along the NW-SE axis.
        """
        
        if isinstance(val, int) or isinstance(val, Tangle3D):
            if self.crossingCount != 0:
                self.NW_SE_reflection()
            return self + val

        outputStr = "Tangle3D can only be multiplied by integers or tangles."
        raise Exception(outputStr)
    
        



### Knot Implementation: #######################################################

class Knot3D:
    """ A class that transforms the Conway notation of knots into curves embeded
        on the unit sphere.

        Currently, the code only works for Conway codes that correspond to
        knots formed by algebraic operations tangles. Polynomial tangles are not
        yet supported.
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
        
        if re.match(r"\[[\d\(\)\,\-\+]+\]", code) is None:
            raise Exception("Conway code contains errant characters.")

        if ("." in code) or ("*" in code) or (":" in code):
            raise Exception("Polynomial knots are not accepted as of yet.")

        # Store inputs:
        self.alpha = alpha
        self.crossingAngle = crossingAngle
        self.code = code

        # Calculate the number of crossings needed and the rotationStep
        digits = re.findall(r"\d", code)
        self.numCrossings = sum([int(i) for i in digits])
        self.numCrossings += len(re.findall(r"[\+\-][,\]]", code))
        self.rotationStep = 2*np.pi / self.numCrossings
        
        self.knot = self.formKnot(code[1:-1])
        self.knot.combineEndings()
        
        return
    
    def separateFirstGroup(self, txt):
        """ Separates strings of the form (strA)strB into the tuple strA, strB

            Inputs:
                - txt: A string representing a collection of tangle operations

            Outputs:
                Strings strA, strB where strA has balanced parentheses (same
                number of opening parentheses as closing parentheses).
        """

        # Use a count of the parentheses to determine the position of the
        # closing parentheses corresponding to the initial opening parentheses
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
        """ Separates strings of the form "str1,str2,...,strN" into individual
            components. Commas between matching parentheses are ignored.

            Inputs:
                - txt: A string representing a collection of tangle operations

            Outputs:
                A list of strings [str1, str2, ..., strN] where each component
                has balanced parentheses.
        """

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
        """ Produce a new tangle given the parameters that have been saved.

            Outputs:
                A Tangle3D object with the alpha and crossing angles provided to
                the class on initialization.
        """
        tangle = Tangle3D(self.rotationStep, alpha=self.alpha, 
            crossingAngle=self.crossingAngle)
        return tangle
    
    def formKnot(self,code):
        """ Given a code for a tangle, compute the corresponding tangle.

            Inputs:
                - code: a string consisting of digits, and "(" or ")" or ","
                    or "+" or "-" operators.

            Outputs:
                A Tangle3D object corresponding to the given code.
        """

        # We use a recursive strategy to parse out the code in order to develop
        # the desired tangle.
        


        # If we have multiple components separated by commas, ramify them:
        components = self.separateByCommas(code)
        if len(components) > 1:
            tangles = [self.formKnot(comp) for comp in components]
            returnTangle = self.newTangle()
            for tangle in tangles:
                returnTangle += tangle*0
            return returnTangle

        # If the code is just digits, form the tangle directly:
        if code.isdigit():
            returnTangle = self.newTangle()
            for i in code:
                returnTangle *= int(i)
            return returnTangle

        # If the code terminates in a +, perform the + operation
        # (A SW-NE crossing south of the current tangle)
        if code[-1] == "+":
            returnTangle = self.formKnot(code[:-1])
            returnTangle.plusOperator()
            return returnTangle

        # If the code terminates in a -, perform the - operation
        # (A NW-SE crossing south of the current tangle)
        if code[-1] == "-":
            returnTangle = self.formKnot(code[:-1])
            returnTangle.minusOperator()
            return returnTangle

        # If the code starts with a -, flip all crossings
        if code[0] == "-":
            returnTangle = self.formKnot(code[1:])
            returnTangle.flipCrossings()
            return returnTangle

        # If we start with a "(", calculate the tangle in the first set of
        # parentheses and the tangle afterwards and combine them together.
        if code[0] == "(":
            firstCode, secondCode = self.separateFirstGroup(code)
            returnTangle = self.formKnot(firstCode)
            if secondCode != "":
                returnTangle *= self.formKnot(secondCode)

            return returnTangle

        # If none of the rules apply, then we made a mistake
        raise Exception("Could not parse the Conway code for this knot.")

        return
    
    

### Example Usage: #############################################################


code = "[(3,2)-(21,2)]"
k = Knot3D(code)

fig = plt.figure()
ax = plt.axes(projection='3d')

t = np.linspace(0,1, 50)
deltaH = 0.1
for path in k.knot.paths: #range(len(k.paths)):
    altPath = sphereBezier(path[:3], t)
    
    if path[3,0] == 1 and path[3,3] == 1:
        altPath *= 1+deltaH
    elif path[3,0] == 1 and path[3,3] == -1:
        altPath *= 1 + deltaH * np.cos(t * np.pi)
    elif path[3,0] == -1 and path[3,3] == 1:
        altPath *= 1 + deltaH * np.cos((1-t) * np.pi)
    else:
        altPath *= 1-deltaH
        

    ax.plot(altPath[0], altPath[1], altPath[2], linewidth=3, color="black")

    
ax.set_xlim3d([-1.3, 1.3])
ax.set_ylim3d([-1.3, 1.3])
ax.set_zlim3d([-1  , 1  ])

plt.title(f"Plot of knot with Conway code {k.code}")
plt.show()