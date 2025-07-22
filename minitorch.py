import numpy as np

class Module:
    """
    The base class for all neural network modules.
    Your models should also subclass this class
    """
    def __init__(self) -> None:
        # we will store the output of the forward pass here
        # this is needed for the backward pass
        self._output = None
        # we will store parameters (like weights and biases) in this list
        self._parameters = []
        # we wil store their corresponding gradients here
        self._gradients = []
    
    def forward(self, *args, **kwargs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Defines the gradient computation.
        Should be overridden by all subclasses

        Args: 
            grad_output (np.ndarray): The gradient of the loss with respect 
            to the output of this module
        """
        pass

    def parameters(self):
        """
        Returns a list of all parameters of the module.
        we will override this in Sequential to collect parmeters from sub-modules
        """
        return self._parameters

    def gradients(self):
        """
        Returns a list of all graidents of the module
        """
        return self._gradients
    
    def zero_grad(self):
        """
        Set gradients of all model parameters to zero.
        """
        for i in range(len(self._gradients)):
            self._gradients[i].fill(0.0)

    def __call__(self, *args, **kwargs) :
        """
        Makes the module callable (e.g. layer(input))
        """
        self._output = self.forward(*args, **kwargs)
        return self._output
    
class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T + B.
    This module is our equivalent of torch.nn.Linear
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: np.dtype = np.float32) :
        """
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias
                Default: True
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # 1. Initialize parameters (weights and biases)
        # we use a common initialization scheme (kaiming He) for the weights.
        # it helps with training stability.
        # the shape is (out, in) to match Pytorch's convention
        stdv = np.sqrt(2. / self.in_features)
        self.W = np.random.uniform(-stdv, stdv, (self.out_features, self.in_features)).astype(dtype)

        # store the W parameter in the Module's parameter list
        self._parameters.append(self.W)
        # create a corresponding gradient tensor, initialized to zeros
        self.gW = np.zeros_like(self.W)
        self._gradients.append(self.gW)

        if self.use_bias:
            # biases can be intialized to zero
            self.b = np.zeros((1, self.out_features), dtype=dtype)
            self._parameters.append(self.b)
            # create corresponding gradient tensor
            self.gb = np.zeros_like(self.b)
            self._gradients.append(self.gb)
        
        # we need to cache the input for the backward pass
        self._input = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass: input @ W.T + b

        Args: 
            X (np.ndarray): The input data, with shape (N, in_featurs),
                            where N is the batch size

        Returns:
            np.ndarray: The output of the layer, with shape  (N, out_features)                     
        """
        # cache the input because we'll need it for the backward pass
        self._input = X

        # perform the linear transformation
        output = self._input @ self.W.T
        if self.use_bias:
            output += self.b
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass.

        It computes three things:
        1. The gradient of the loss with respect to the weights (dL/dW)
        2. The gradient of the loss with respect to the bias (dL/db)
        3. The gradient of the loss with respect to the inputs (dL/dX)

        Args: 
            grad_output (np.ndarray): The gradient from the next layer (or the loss function).
                                      It has shape (N, out_features). this is dL/dY
        Returns: 
            np.ndarray: The gradient with respect to the input (dL/dX), which will be passed 
                        to the previous layer. It has shape (N, in_features)
        """
        # 1. Calculate the gradient with respect to the bias (dL/db)
        # The derivative of (Y) w.r.t b is 1. By the chain rule, dL/db = dL/dY * dY/db = grad_output * 1
        # Since the same bias vector 'b' was added to every sample in the batch
        # we need to sum the gradients across the batch dimension (axis=0)
        if self.use_bias:
            # Sum the gradients for each sample in the batch
            # grad_output shape: (N, out_features) -> gb shape: (1, out_features) 
            # We update the gradient in-place
            self.gb[:] = np.sum(grad_output, axis=0, keepdims=True)

        # 2. Calcualte the gradient w.r.t the weights (dL/dW)
        # Using the chain rule: dL/dW = dL/dY * dY/dW
        # The derivative dY/dW where Y = X @ W.T is X
        # So, dL/dW = grad_output.T @ X
        # Shape check: (out_features, N) @ (N, in_features) -> (out_features, in_features), which matches W's shape
        self.gW[:] = grad_output.T @ self._input

        # 3. Calculate the gradient w.r.t the input (dL/dX)
        # This is what we will return to the previous layer
        # Using the chain rule: dL/dX = dL/dY * dY/dX
        # The derivative dY/dX where Y = X @ W.T is W
        # So, dL/dX = grad_output @ W
        # Shape check: (N, out_features) @ (out_features, in_features) -> (N, in_features), which matches the input's shape
        grad_input = grad_output @ self.W

        return grad_input
    
class ReLU(Module):
    """
    Applies the Rectified Linear Unit function element-wise
    ReLU(x) = max(0, x)
    """
    def __init__(self):
        super().__init__()
        # we need to cache the input for the backward pass
        self._input = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU function

        Args:
            X (np.ndarray): the input data from the previous layer 
        
        Returns:
            np.ndarray: the input with negative values clamped to zero
        """
        # cache the input for the backward pass
        self._input = X
        # np.maximum calculates the element-wise maximum
        return np.maximum(0, self._input)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss w.r.t the input of ReLU.

        Args: 
            grad_output (np.ndarray): The gradient from the next layer (dL/dY)
                                      shape is the same as the input
        
        Returns:
            np.ndarray: the gradient w.r.t the input (dL/dX)
        """
        # The chain rule is: dL/dX = dL/dY * dY/dX
        # The derivative of ReLU is:
        # 1 if x > 0
        # -0 if x <= 0

        # create a mask for where the input was positive 
        # (self._input > 0) returns a boolean array (True/False)
        # Multiplying by 1 turns it into a binary array (1/0)
        relu_grad = (self._input > 0) * 1

        # Apply chain rule: multiply the upstream gradient by the local gradient 
        # This will pass the gradient through where the original input was positive,
        # and block it (makes it zero) where the original input was negative
        return grad_output * relu_grad
    
class Sigmoid(Module):
    """
    Applies the sigmoid function element-wise
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid function

        Args:
            X (np.ndarray): The input data
        
        Returns:
            np.ndarray: The input squashed between 0 and 1
        """
        # the output of sigmoid is calculated and stored in self._output
        # by the parent class's __call__ method
        return 1 / (1 + np.exp(-X))
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss w.r.t the input of sigmoid 
        Args:
            grad_output (np.ndarray): The gradient from the next layer (dL/dY)
        Returns:
            np.ndarray: The gradient w.r.t the input (dL/dX)
        """
        # The chain rule is: dL/dX = dL/dY * dY/dX
        # The derivative of the sigmoid function, dY/dX, can be written 
        # very elegantly in terms of its output, Y: Y * (1 - Y)
        
        # self._output was cached by the Modules __call__ method
        sigmoid_output = self._output
        sigmoid_grad = sigmoid_output * (1 - sigmoid_output)
        
        # Apply chain rule
        return grad_output * sigmoid_grad


class Sequential(Module):
    """
    A sequential container. Modules will be added to it in the order they 
    are passed in the constructor

    Example:
    model = Sequential(
        Linear(in_features=2, out_features=10),
        ReLU(),
        Linear(in_features=10, out_features=2)
    )
    """
    def __init__(self, *layers) :
        """
        Args: 
            *layers: A sequence of Module objects to be stacked.
        """
        super().__init__()
        self.layers = list(layers)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Passes the input through all layers sequentially.

        Args:
            X (np.ndarray): The initial input to the first layer.

        Returns:
            np.ndarray: The final output after last layer.
        """
        current_input = X
        for layer in self.layers:
            # the output of the one layer is input to the next 
            current_input = layer(current_input)
        # the final output of the sequence is the output of the last layer
        return current_input
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Performs backpropagation through all layers in reverse order.

        Args:
            np.ndarray: the gradient w.r.t the initial input.
                        (though this is often not used for the full model)
        """
        current_grad_output = grad_output
        for layer in reversed(self.layers):
            # the input gradients for one layer is the output gradient from the next
             current_grad_output = layer.backward(current_grad_output)
        return current_grad_output

    def parameters(self) -> list:
        """
        Collects and returns all parameters from all sub-modules
        """
        all_params = []
        for layer in self.layers:
            # list.extend adds all elements of an iterable to the list 
            all_params.extend(layer.parameters())
        return all_params
    
    def gradients(self) -> list:
        """
        Collects and returns all gradients from all the sub-modules
        """
        all_grads = []
        for layer in self.layers:
            all_grads.extend(layer.gradients())
        return all_grads
    
    def zero_grad(self):
        """
        Calls zero_grad() on all sub-modules to reset their gradients
        """
        for layer in self.layers:
            layer.zero_grad()

class MSELoss(Module):
    """
    Computes the Mean Squared Error loss.
    This is commonly used for regression tasks 
    """
    def __init__(self) :
        super().__init__() 
        # Cache the predictions and true values for the backward pass
        self._y_pred = None
        self._y_true = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the mean squared error loss
        L = (1/N) * Σ(y_pred - y_true)^2

        Args:
            y_pred (np.ndarray): The predictions from the model. Shape (N, D),
                                Where N is batch size and D is output dimension.
            y_true (np.ndarray): The ground-truth values. shape (N, D)
        
        Returns: 
            float: A single scalar value representing the loss
        """
        # Cache values needed for the backward pass
        self._y_pred = y_pred
        self._y_true = y_true

        # Number of samples in the batch
        num_samples = y_pred.shape[0]

        # Calculate the loss
        loss = np.sum((y_pred - y_true)**2) / num_samples
        return loss
    
    def backward(self) -> np.ndarray:
        """
        Computes the gradient of the MSE loss w.r.t the predictions.
        This is the first gradient in the backpropagation chain.

        Returns:
            np.ndarray: The gradient of the loss w.r.t y_pred (dL/dy_pred)
                        Shape will be the same as y_pred
        """
        # The derivative of the loss L w.r.t a single prediction y_pred_i is:
        # dL/dy_pred_i = d/dy_pred_i [ (1/N) * Σ(y_pred_j - y_true_j)^2 ]
        #              = (1/N) * 2 * (y_pred_i - y_true_i)

        # Number of samples in the batch
        num_samples = self._y_pred.shape[0]

        # Calculate the gradient
        grad_y_pred = 2 * (self._y_pred - self._y_true) / num_samples

        return grad_y_pred

class SGD:
    """
    Implements the Stochastic gradient descent (SGD) optimizer.
    It updates the parameters using the rule:
    param = param - (learning_rate * gradient)
    """
    def __init__(self, parameters: list, gradients: list, lr: float = 0.001) :
        """
        Initializes the optimizer.

        Args: 
            parameters (list): A list of model's parameters to be updated.
                               (e.g. from model.parameters())
            gradients (list) : A list of the corresponding gradients.
                               (e.g. from model.gradients())
            lr (float) : Learning rate. Default: 0.001
        """
        # Ensure that number of parameters matches the number of gradients
        if len(parameters) != len(gradients):
            raise ValueError("The number of parameters and gradients must be the same.")
        
        self.params = parameters
        self.grads = gradients
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step
        It itereates through all parameters and updates them using their gradients
        """
        # Iterates over all the parameters and their corresponding gradients
        for param, grad in zip(self.params, self.grads):
            param -= self.lr * grad