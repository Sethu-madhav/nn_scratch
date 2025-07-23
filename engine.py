import numpy as np
import os # we'll use this to set a default device if needed
from typing import Union

# --- Engine Selection ---

# By default, we use NumPy
xp = np
GPU_ENABLED = False

# We can use and environment variable to force CPU-only mode
# This is useful for debugging or when you want to explicitly run on the CPU
# Example: MYTORCH_DEVICE=cpu python train.py
if os.environ.get('MYTORCH_DEVICE', 'auto').lower() == 'cpu':
    print("Backend: NumPy (CPU) [Forced by user]")

else:
    try:
        # If the import succeeds, we use CuPy as our backend
        import cupy

        # Check if a GPU is actually available. CuPy can be installed without a GPU
        if cupy.is_available():
            xp = cupy
            GPU_ENABLED = True
            print("Backend: CuPy (GPU)")
        else:
            print("Backend: NumPy (CPU) [CuPy installed but no GPU found]")

    except ImportError as e:
        # If CuPy is not installed, we fall back to NumPy and inform the user
        print(f"Backend: Numpy (CPU) [CuPy not found]: {e}")

# --- Device management functions ---

def get_device(device_str: str) -> str:
    """
    Validates and returns a standardized device string 

    Args: 
        device_str (str): can be 'cuda', 'gpu', or 'cpu'

    Returns: 
        str: 'cuda' if a GPU is available and requested, otherwise 'cpu'
    """
    device_str = device_str.lower()
    if device_str in ('cuda', 'gpu'):
        if GPU_ENABLED:
            return 'cuda'
        else:
            print("Warning: Device 'cuda' was requested but not available. Falling back to 'cpu'")
            return 'cpu'
    elif device_str == 'cpu':
        return 'cpu'
    else:
        raise ValueError(f"Unknown device: '{device_str}'. Supported devices are 'cuda' and 'cpu'.")
    
def to_device(array: Union[np.ndarray, 'cupy.ndarray'], device: str) -> Union[np.ndarray, 'cupy.ndarray']:
    """
    Moves an array to the specified device.
    Converts between NumPy and CuPy arrays

    Args:
        array: The NumPy or CuPy array to move.
        device: The target device, either 'cpu', or 'cuda'
    
    Returns: 
        The array on the target device.
    """
    if device == 'cuda':
        # If the array is already a CuPy, do nothing
        if isinstance(array, xp.ndarray):
            return array
        # If its a numpy array convert it to a CuPy array
        return xp.asarray(array)
    
    elif device == 'cpu':
        # If the array is a NumPy array, do nothing.
        if isinstance(array, np.ndarray):
            return array
        # If its a cupy array, move it to the host (cpu)
        return xp.asnumpy(array)

    else:
        raise ValueError(f"Unknown device '{device}'")
    
def is_on_gpu(array: Union[np.ndarray, 'cupy.ndarray']) -> bool:
    """
    Checks if a given array is on the gpu (i.e. is a CuPy array)
    """
    # In our setup, if our backend xp is cupy, and the array is an instance
    # of xp.ndarray, then it's a cupy array
    if GPU_ENABLED:
        return isinstance(array, xp.ndarray)
    return False