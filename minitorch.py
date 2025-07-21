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

    def __call__(self, *args: np.Any, **kwargs: np.Any) -> np.Any:
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
    def __init__(self, in_features: int, out_features: int, bias: bool = True) :
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
        self.W = np.random.uniform(-stdv, stdv, (self.out_features, self.in_features))

        # store the W parameter in the Module's parameter list
        self._parameters.append(self.W)
        # create a corresponding gradient tensor, initialized to zeros
        self.gW = np.zeros_like(self.W)
        self._gradients.append(self.gW)

        if self.use_bias:
            # biases can be intialized to zero
            self.b = np.zeros((1, self.out_features))
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