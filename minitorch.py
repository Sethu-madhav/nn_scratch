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