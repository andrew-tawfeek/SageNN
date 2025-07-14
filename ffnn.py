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

def relu(x):
    return max(0, x)
