import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFFNN(nn.Module):
    """
    A simple Feedforward Neural Network for our Reinforcement Learning agent.
    It takes a state representation as input and outputs action Q-values or probabilities.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initializes the neural network layers.

        Args:
            input_dim (int): The number of features in the input state.
            hidden_dim (int): The number of neurons in the hidden layer.
            output_dim (int): The number of possible actions (e.g., BUY, SELL, HOLD).
        """
        super(
            SimpleFFNN, self
        ).__init__()  # Call the constructor of the parent class (nn.Module)

        # Define the layers of the network
        self.fc1 = nn.Linear(
            input_dim, hidden_dim
        )  # First Fully Connected Layer: Input -> Hidden
        self.fc2 = nn.Linear(
            hidden_dim, hidden_dim
        )  # Second Fully Connected Layer: Hidden -> Hidden
        self.fc3 = nn.Linear(
            hidden_dim, output_dim
        )  # Third Fully Connected Layer: Hidden -> Output

        print(
            f"SimpleFFNN initialized with: Input={input_dim}, Hidden={hidden_dim}, Output={output_dim}"
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        This method describes how input data is processed through the layers.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output tensor (e.g., Q-values for each action).
        """
        # Pass through the first layer, then apply ReLU activation
        x = F.relu(self.fc1(state))
        # Pass through the second layer, then apply ReLU activation
        x = F.relu(self.fc2(x))
        # Pass through the final layer (no activation here if outputting Q-values,
        # or softmax if outputting probabilities for a policy network)
        q_values = self.fc3(x)
        return q_values


# --- Test Block (only runs when nn_models.py is executed directly) ---
if __name__ == "__main__":
    print("--- Starting SimpleFFNN Test ---")

    # Define dummy dimensions
    input_features = 10  # Example: 10 features representing market state
    hidden_neurons = 64
    output_actions = 3  # Example: BUY, SELL, HOLD

    # Create an instance of our neural network
    model = SimpleFFNN(input_features, hidden_neurons, output_actions)

    # Create a dummy input state tensor (batch_size=1, input_features=10)
    # torch.randn generates random numbers from a standard normal distribution
    dummy_state = torch.randn(1, input_features)
    print(f"\nDummy input state shape: {dummy_state.shape}")
    print(f"Dummy input state (first 5 values): {dummy_state[0, :5].numpy()}")

    # Pass the dummy state through the network to get predictions
    q_values = model(dummy_state)
    print(f"\nOutput Q-values shape: {q_values.shape}")
    print(
        f"Output Q-values: {q_values.detach().numpy()}"
    )  # .detach().numpy() to convert to numpy array

    # Get the action with the highest Q-value
    predicted_action = torch.argmax(q_values).item()
    print(f"Predicted action (index): {predicted_action}")

    print("--- SimpleFFNN Test Complete ---")
