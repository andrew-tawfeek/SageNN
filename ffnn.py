class NN():
    # A simple feedforward neural network class.
    # It can be initialized with an architecture (list of layer sizes) and optionally weights and biases.
    # The architecture is a list where each element represents the number of neurons in that layer.
    # The weights and biases can be provided as lists of matrices and vectors respectively.
    def __init__(self, architecture, weights=None, biases=None):
        self.architecture = architecture
        # If weights are not provided, initialize them randomly.
        # If biases are not provided, initialize them randomly.
        import random
        if weights is None:
            self.weights = [
            matrix(QQ, self.architecture[i+1], self.architecture[i],
                   [[random.uniform(-1, 1) for _ in range(self.architecture[i])] for _ in range(self.architecture[i+1])])
            for i in range(len(self.architecture)-1)
            ]
        else:
            self.weights = [matrix(QQ, w) for w in weights]
        if biases is None:
            self.biases = [
            vector(QQ, [random.uniform(-1, 1) for _ in range(self.architecture[i+1])])
            for i in range(len(self.architecture)-1)
            ]
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
        #todo: display the network architecture in a more user-friendly way.
        return
    
    def binary_state(self, input_vector):
        # Returns a single vector of binary states for all hidden neurons (1 if on, 0 if off).
        if isinstance(input_vector, list):
            input_vector = vector(QQ, input_vector)
        elif not hasattr(input_vector, 'parent'):
            raise TypeError("Input must be a list or a vector.")
        assert len(input_vector) == self.architecture[0], "Input vector size must match the first layer size."
        output = input_vector
        states = []
        for i in range(len(self.architecture) - 1):
            output = self.weights[i] * output + self.biases[i]
            output = output.apply_map(relu)
            # Only collect hidden layer states (exclude output layer)
            if i < len(self.architecture) - 2:
                states.extend([1 if x > 0 else 0 for x in output])
        return vector(QQ, states)

    def plot3d(self, x_range=(-5, 5), y_range=(-5, 5)):
        # Create a 3D plot of the output of the network for two input variables.
        #assert len(self.architecture[0]) == 2, "This method is designed for networks with two input neurons."
        def f(x, y):
            return self.forward([x, y])[0]  # Extract the scalar value from the vector
        P = plot3d(f, x_range, y_range, adaptive=False, color='red')
        from sage.plot.plot3d.plot3d import axes
        S = P + axes(6, color='black')
        return S.show()
        #todo later: allow slicing higher-dimensional inputs

    def plot2d_approx(self, x_range=(-5, 5), density=100):
        # For every x in the range, compute the binary state of the network.
        # Every binary state is a differently-colored point in the plot.
        # This method approximates the binary states by sampling points in the specified range.
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(x_range[0], x_range[1], density)
        X, Y = np.meshgrid(x, y)
        
        # Dictionary to store unique binary states and their hex colors
        state_colors = {}
        color_data = np.zeros(X.shape, dtype=object)
        
        # Generate colors for unique binary states
        def generate_hex_color(state_tuple):
            if state_tuple not in state_colors:
                # Generate a unique hex color based on the binary state
                hash_val = hash(state_tuple)
                color = f"#{(hash_val & 0xFFFFFF):06x}"
                state_colors[state_tuple] = color
            return state_colors[state_tuple]
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = self.binary_state([X[i, j], Y[i, j]])
                state_tuple = tuple(state)
                color_data[i, j] = generate_hex_color(state_tuple)
        
        plt.figure(figsize=(10, 8))
        
        # Create a custom colormap for the unique states
        unique_states = list(state_colors.keys())
        unique_colors = list(state_colors.values())
        
        # Map each position to its corresponding color index
        color_indices = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = tuple(self.binary_state([X[i, j], Y[i, j]]))
                color_indices[i, j] = unique_states.index(state)
        
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(unique_colors)
        
        plt.contourf(X, Y, color_indices, levels=np.arange(-0.5, len(unique_states) + 0.5, 1), cmap=cmap)
        plt.colorbar(
            label='Binary State',
            ticks=range(len(unique_states)),
            format=plt.FuncFormatter(lambda x, p: str(unique_states[int(x)]) if int(x) < len(unique_states) else '')
        )
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.title('2D Plot of Binary States')
        plt.show()


        
def relu(x):
    return max(0, x)
