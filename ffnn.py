class NN():
    # A simple feedforward neural network class.
    # It can be initialized with an architecture (list of layer sizes) and optionally weights and biases.
    # The architecture is a list where each element represents the number of neurons in that layer.
    # The weights and biases can be provided as lists of matrices and vectors respectively.
    def __init__(self, architecture, weights=None, biases=None):
        self.architecture = architecture
        if weights is None:
            self.weights = [matrix(QQ, self.architecture[i+1], self.architecture[i]) for i in range(len(self.architecture)-1)]
        else:
            self.weights = [matrix(QQ, w) for w in weights]
        
        if biases is None:
            self.biases = [vector(QQ, self.architecture[i+1]) for i in range(len(self.architecture)-1)]
        else:
            self.biases = [vector(QQ, b) for b in biases]
        
    def forward(self, input_vector):
        # Forward pass through the network.
        # input_vector is a list or vector of inputs that matches the size of the first layer.
        if isinstance(input_vector, list):
            input_vector = vector(QQ, input_vector)
        elif not hasattr(input_vector, 'parent'):
            raise TypeError("Input must be a list or a vector.")
        assert len(input_vector) == self.architecture[0], "Input vector size must match the first layer size."
        output = input_vector
        for i in range(len(self.architecture) - 1):
            output = self.weights[i] * output + self.biases[i]
            output = output.apply_map(relu)
        return output
    def __repr__(self):
        return f"Feedforward neural network with architecture {self.architecture}."
    
    def show(self):
        #todo
        return

    def plot3d(self, x_range=(-5, 5), y_range=(-5, 5)):
        # Create a 3D plot of the output of the network for two input variables.
        assert len(self.architecture) == 2, "This method is designed for networks with two input neurons."
        def f(x, y):
            return self.forward([x, y])[0]  # Extract the scalar value from the vector
        P = plot3d(f, x_range, y_range, adaptive=False, color='red')
        from sage.plot.plot3d.plot3d import axes
        S = P + axes(6, color='black')
        return S.show()
        #todo later: allow slicing higher-dimensional inputs

def relu(x):
    return max(0, x)



class Tile():
    def __init__(self,N):
        self.tile = N
        self.orientation = []
        if (N == 0):
            self.numConnectionPoints = 0
            self.connectionDirections = []
        if (N in [1,2,3,4,5,6]):
            self.numConnectionPoints = 2
            if (N==1):
                self.connectionDirections = ['left','down']
            if (N==2):
                self.connectionDirections = ['right','down']
            if (N==3):
                self.connectionDirections = ['up','right']
            if (N==4):
                self.connectionDirections = ['left','up']
            if (N==5):
                self.connectionDirections = ['left','right']
            if (N==6):
                self.connectionDirections = ['up','down']
        if (N in [7,8,9,10]):
            self.numConnectionPoints = 4
            self.connectionDirections = ['up','down','left','right']
    def show(self, resolution = 5):
        if (self.tile==0):
            return line([(0,0),(1,0)], axes = False, xmin = 0, xmax = 1, ymin = 0, ymax = 1, frame = True, ticks=[[],[]], thickness=0).plot()
        if (self.tile==1):
            return arc((0,0), 1, sector=(0,pi/2), axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot()
        if (self.tile==2):
            return arc((0,0), 1, sector=(0,pi), axes = False, xmin = -2, xmax = 0, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot()
        if (self.tile==3):
            return arc((0,0), 1, sector=(pi,2*pi), axes = False, xmin = -2, xmax = 0, ymin = -2, ymax = 0, frame = True, ticks=[[],[]], thickness=resolution).plot()
        if (self.tile==4):
            return arc((0,0), 1, sector=(pi,2*pi), axes = False, xmin = 0, xmax = 2, ymin = -2, ymax = 0, frame = True, ticks=[[],[]], thickness=resolution).plot()
        if (self.tile==5):
            return line([(0,1), (1,1)], axes = False, xmin = 0, xmax = 1, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot()
        if (self.tile==6):
            return line([(1,0), (1,1)], axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 1, frame = True, ticks=[[],[]], thickness=resolution).plot()
        if (self.tile==7):
            return arc((0,0), 1, sector=(0,pi/2), axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot() + arc((2,2), 1, sector=(pi,2*pi), axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot()
        if (self.tile==8):
            return arc((0,2), 1, sector=(2*pi/3,2*pi), axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot() + arc((2,0), 1, sector=(pi,pi/2), axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot()
        if (self.tile==9):
            return line([(0,1), (2,1)], axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot() + line([(1,0), (1,.6)], axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot() + line([(1,1.4), (1,2)], axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot()
        if (self.tile==10):
            return line([(1,2), (1,0)], axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot() + line([(0,1), (.6,1)], axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot() + line([(1.4,1), (2,1)], axes = False, xmin = 0, xmax = 2, ymin = 0, ymax = 2, frame = True, ticks=[[],[]], thickness=resolution).plot()
    def isGoing(self, direction):
        # e.g. Tile(6).isGoing('up') returns True but Tile(6).isGoing('left') returns False
        # This is good for checking suitable connectivity later
        return direction in self.connectionDirections
    def zoom(self):
        # Every tile becomes 3x3 matrix
        # TODO: Later, iterate this with an input "amount"
        N = self.tile
        if (N==0):
            return [[0,0,0],[0,0,0],[0,0,0]]
        if (N==1):
            return [[0,0,0],[5,1,0],[0,6,0]]
        if (N==2):
            return [[0,0,0],[0,2,5],[0,6,0]]
        if (N==3):
            return [[0,6,0],[0,3,5],[0,0,0]]
        if (N==4):
            return [[0,6,0],[5,4,0],[0,0,0]]
        if (N==5):
            return [[0,0,0],[5,5,5],[0,0,0]]
        if (N==6):
            return [[0,6,0],[0,6,0],[0,6,0]]
        if (N==7):
            return [[0,3,1],[1,0,3],[3,1,0]]
        if (N==8):
            return [[2,4,0],[4,0,2],[0,2,4]]
        if (N==9):
            return [[0,6,0],[5,9,5],[0,6,0]]
        if (N==10):
            return [[0,6,0],[5,10,5],[0,6,0]]
    def orient(self, direction):
        # Assigns an orientation to a tile
        assert direction in self.connectionDirections #returns error if orientation not possible
        self.orientation = self.orientation + [direction]
