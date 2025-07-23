import numpy as np
import matplotlib.pyplot as plt
# import all classes we built in our library
from minitorch import Linear, ReLU, SGD, Sequential, MSELoss, Module
from engine import xp, to_device, get_device

# 0. Choose Device
# Try setting this to 'cuda' if you have a Nvidia GPU!
# device_str = 'cpu'
device_str = 'cuda'
device = get_device(device_str)
print(f"Training on device: {device}")

# 1. Prepare the data
# Goal: learn the function y = 2x + 1
print("\nStep 1: Preparing the data")
np.random.seed(41)
DTYPE = np.float64

# Create out training data (X) and target values (Y)
# we  create 100 data points for x from -5 to 5
X_train = np.linspace(-5, 5, 90000, dtype=DTYPE).reshape(-1, 1) # Shape: (100, 1)

# The true function is y = 2x + 1
# We add some random noise to make the task more realistic
noise = np.random.normal(0, 1, X_train.shape).astype(DTYPE) # Gaussian noise 
y_train = (2 * X_train + 1 + noise).astype(dtype=DTYPE)             # Shape: (100, 1)

# Let's visualize the data to whats we're working with
plt.figure()
plt.scatter(X_train, y_train, label='Noisy training data')
plt.plot(X_train, 2*X_train+1, color='red', linestyle='--', label='True function (y=2x+1)')
plt.title('Our Synthetic dataset')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# 2. Define the Model, Optimizer and Loss_fn
print("\nStep 2: Define Model, Loss_fn and Optimizer")

# Define the model architecture using our sequential container
# A simple model: Linear -> ReLU -> Linear
# Input features: 1 (since x is a single number)
# Output features: 1 (since y is a single number)
hidden_size = 10
model = Sequential(
    Linear(in_features=1, out_features=hidden_size, dtype=DTYPE),
    ReLU(),
    Linear(in_features=hidden_size, out_features=1, dtype=DTYPE)
)

# Move the model to the selected device
print(f"Moving model to '{device}'...")
model.to(device)
# If device is 'cuda'. we can check this:
print(f"Model is on device: '{model.device}'")
if device == 'cuda':
    # The fist parameter's type should be a cupy.ndarray
    first_param = model.parameters()[0]
    print(f"Type of first model parameter: {type(first_param)}")


# Define the loss function
loss_fn = MSELoss()
print("Loss function: MSELoss")

# Define the optimizer
# We pass it model's parameters, gradients and learning rate
learning_rate = 0.05
optimizer = SGD(model.parameters(), model.gradients(), lr=learning_rate)
print(f"Optimizer: SGD with learning rate {learning_rate}")

# 3. The Training loop
print("\nStep 3: Starting the training loop")

num_epochs = 5000
# Move the dataset to the device 
print(f"Moving training data to the device...")
X_train = to_device(X_train, device=device)
y_train = to_device(y_train, device=device)
print(f"Type of X_train on device: {type(X_train)}")

for epoch in range(num_epochs):

    # 1. Zero Gradients
    model.zero_grad()

    # 2. Forward pass
    # Get the model's predictions for the training data
    y_pred = model(X_train)

    # 3. Calculate the loss
    # compare the predictions with the true values
    loss = loss_fn(y_pred, y_train)

    # 4. Backward pass
    # First get the intital gradient from the loss function
    loss_gradient = loss_fn.backward()
    # Then propagate the gradient backward through the model
    model.backward(loss_gradient)

    # 5. Update parameters
    optimizer.step()

    # Logging 
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: [{epoch}/{num_epochs}] | Loss: {loss:.4f}")

print("\nTraining finished.")

# 4. Visualize the results
print("\nStep 4: Visualizing the results")
# moving data back to CPU 
final_predictions_gpu = model(X_train)
final_predictions = to_device(final_predictions_gpu, device='cpu')

# Convert X_train and y_train back to CPU for plotting
X_train = to_device(X_train, device='cpu')
y_train = to_device(y_train, device='cpu')

print(f"y_train: {y_train[:5, :]}")
print(f"final_pred: {final_predictions[:5, :]}")

plt.figure()
plt.scatter(X_train, y_train, label='Noisy Training Data')
plt.plot(X_train, 2 * X_train + 1, color='red', linestyle='--', label='True Function')
plt.plot(X_train, final_predictions, color='green', linewidth=2, label='Model Predictions')
plt.title("Training Results")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()