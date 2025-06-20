import pandas as pd
import numpy as np
import torch
import random

from qmind.strategy.base_strategy import BaseStrategy
from qmind.strategy.signals import Signal
from qmind.strategy.nn_models import SimpleFFNN  # Our neural network model
from qmind.utils.logging_config import get_logger

logger = get_logger(__name__)


class RLAgentStrategy(BaseStrategy):
    """
    A Reinforcement Learning agent strategy that uses a neural network to learn a trading policy.
    Inspired by AlphaZero, but simplified to use a Deep Q-Network (DQN) like approach
    for discrete actions, without Monte Carlo Tree Search.

    This class defines the agent's decision-making process. The actual
    RL training loop (interaction with an environment, reward calculation,
    and network updates) will be handled by a separate training component.
    """

    def __init__(
        self,
        input_dim: int = 10,  # Number of features in the state representation
        hidden_dim: int = 64,  # Neurons in the hidden layer of the NN
        num_actions: int = 3,  # Number of possible actions: 0=HOLD, 1=BUY, 2=SELL
        learning_rate: float = 0.001,
        epsilon_start: float = 1.0,  # Starting exploration rate
        epsilon_end: float = 0.01,  # Minimum exploration rate
        epsilon_decay: float = 0.995,  # Decay factor for epsilon
        state_lookback: int = 5,  # How many previous bars to include in the state
    ):
        super().__init__(
            name="RL Agent Strategy",
            params={
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_actions": num_actions,
                "learning_rate": learning_rate,
                "epsilon_start": epsilon_start,
                "epsilon_end": epsilon_end,
                "epsilon_decay": epsilon_decay,
                "state_lookback": state_lookback,
            },
        )

        self.num_actions = num_actions
        self.state_lookback = state_lookback

        # Define the mapping from action index to Signal Enum
        self.action_map = {
            0: Signal.HOLD,
            1: Signal.BUY,
            2: Signal.SELL,  # For simplicity, we'll start with these three. CLOSE can be handled by logic.
        }
        if num_actions != len(self.action_map):
            raise ValueError(
                f"num_actions ({num_actions}) must match action_map size ({len(self.action_map)})"
            )

        # Initialize the neural network (our policy/Q-network)
        self.policy_net = SimpleFFNN(input_dim, hidden_dim, num_actions)

        # For demonstration, we won't initialize an optimizer here,
        # as training will be external. But typically it would be:
        # self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Exploration parameters (for training, will be dynamic)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        logger.info("RLAgentStrategy initialized with a SimpleFFNN.")
        logger.info(f"State lookback: {self.state_lookback} bars.")

    def _get_state_representation(self, data_window: pd.DataFrame) -> torch.Tensor:
        """
        Transforms a window of market data into a numerical state representation
        for the neural network. This is a crucial feature engineering step for RL.

        Args:
            data_window (pd.DataFrame): A DataFrame containing the last `self.state_lookback` bars,
                                        with columns like 'Open', 'High', 'Low', 'Close', 'Volume'.

        Returns:
            torch.Tensor: A flattened tensor representing the state.
        """
        if data_window.empty or len(data_window) < self.state_lookback:
            # Pad with zeros or handle insufficient data if necessary
            # For simplicity here, we'll return zeros for insufficient data
            logger.warning(
                f"Insufficient data for state representation. Need {self.state_lookback} bars, "
                f"got {len(data_window)}. Returning zero state."
            )
            return torch.zeros(1, self.params["input_dim"], dtype=torch.float32)

        # --- Simple State Feature Engineering (for beginners) ---
        # Flatten the last N bars' Close prices:
        state_np = data_window["Close"].values[-self.state_lookback :]

        # Add simple momentum features (e.g., price differences)
        # Ensure state_np has at least 1 element before calculating diff
        if len(state_np) > 1:
            price_diffs = np.diff(state_np)
            state_np = np.concatenate((state_np, price_diffs))

        # Add simple volume feature (e.g., last volume)
        last_volume = data_window["Volume"].iloc[-1]
        state_np = np.append(state_np, last_volume)

        # Ensure the state vector matches the expected input_dim
        # If input_dim is fixed, we need to pad/truncate or ensure features match
        if len(state_np) > self.params["input_dim"]:
            state_np = state_np[
                : self.params["input_dim"]
            ]  # Truncate if too many features
        elif len(state_np) < self.params["input_dim"]:
            state_np = np.pad(
                state_np, (0, self.params["input_dim"] - len(state_np)), "constant"
            )  # Pad with zeros

        state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(
            0
        )  # Add batch dimension
        return state_tensor

    def choose_action(self, state: torch.Tensor, explore: bool = True) -> Signal:
        """
        Chooses an action based on the current state.
        Uses an epsilon-greedy policy during training (if explore=True).

        Args:
            state (torch.Tensor): The current state representation.
            explore (bool): If True, apply epsilon-greedy for exploration.

        Returns:
            Signal: The chosen trading signal.
        """
        if explore and random.random() < self.epsilon:
            # Explore: choose a random action
            action_idx = random.randrange(self.num_actions)
            logger.debug(
                f"Exploring: chose random action {self.action_map[action_idx]}"
            )
        else:
            # Exploit: choose the action with the highest predicted Q-value
            with torch.no_grad():  # Don't calculate gradients for inference
                q_values = self.policy_net(state)
                action_idx = torch.argmax(q_values).item()
            logger.debug(f"Exploiting: chose action {self.action_map[action_idx]}")

        return self.action_map[action_idx]

    def generate_signals(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals by simulating the agent's decisions on historical data.
        This method demonstrates the agent's current policy (which is untrained initially).
        The actual RL training happens in a separate environment/training loop.

        Args:
            historical_data (pd.DataFrame): DataFrame with historical OHLCV data.

        Returns:
            pd.Series: A Series of Signal Enum members, indexed by datetime.
        """
        if historical_data.empty:
            logger.warning("Historical data is empty, no signals to generate.")
            return pd.Series(dtype="object")

        if len(historical_data) < self.state_lookback:
            logger.warning(
                f"Not enough historical data ({len(historical_data)} bars) for initial state "
                f"(need {self.state_lookback}). Returning HOLD signals."
            )
            return pd.Series(
                index=historical_data.index, data=Signal.HOLD, dtype="object"
            )

        logger.info(f"Simulating agent decisions on {len(historical_data)} bars...")

        signals = pd.Series(
            index=historical_data.index, data=Signal.HOLD, dtype="object"
        )

        # Iterate through data to generate signals for each step (simulating a live agent)
        # We start from state_lookback to ensure enough history for the first state
        for i in range(self.state_lookback - 1, len(historical_data)):
            current_window = historical_data.iloc[i - (self.state_lookback - 1) : i + 1]

            # Get state representation for the current window
            state = self._get_state_representation(current_window)

            # Choose action (no exploration during signal generation for demonstration)
            chosen_signal = self.choose_action(state, explore=False)

            signals.iloc[i] = chosen_signal

        logger.info(
            f"Generated {signals.value_counts().get(Signal.BUY, 0)} BUY signals, "
            f"{signals.value_counts().get(Signal.SELL, 0)} SELL signals, "
            f"{signals.value_counts().get(Signal.HOLD, 0)} HOLD signals."
        )
        return signals

    def on_bar(self, current_bar: pd.Series) -> Signal:
        """
        Placeholder for real-time signal generation for an incoming bar.
        This would involve getting the latest state and choosing an action.
        """
        logger.debug(
            f"On-bar method called for {current_bar.name}. Returning HOLD for now."
        )
        # In a real scenario, you'd need previous bars to form the state.
        # For simplicity, we'll just return HOLD, as a proper 'on_bar' for RL
        # would require a state history management.
        return Signal.HOLD


# --- Test Block (only runs when rl_agent_strategy.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting RLAgentStrategy Placeholder Test ---")

    from qmind.data_management.ingestion import DataIngestor

    ingestor = DataIngestor()

    # Fetch some daily data for testing (RL strategies often benefit from longer history)
    logger.info("Fetching daily data for SPY for RLAgentStrategy test...")
    spy_df = ingestor.ingest_historical_data(
        symbol="SPY",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2024-06-01",
        overwrite=False,
    )

    if not spy_df.empty:
        print("\nSPY Data for Strategy (last 5 rows):")
        print(spy_df.tail())

        # Define state parameters based on our simple feature engineering
        # 5 (close prices) + 4 (price diffs) + 1 (volume) = 10 features
        rl_agent = RLAgentStrategy(
            input_dim=10,
            hidden_dim=64,
            num_actions=3,
            state_lookback=5,  # Look back 5 bars for state representation
        )

        # Generate signals (these will be effectively random as the network is untrained)
        signals = rl_agent.generate_signals(
            spy_df.copy()
        )  # Use a copy to avoid modifying original df

        print("\nGenerated Signals (first 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].head(10))

        print("\nGenerated Signals (last 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].tail(10))

        logger.info("--- RLAgentStrategy Placeholder Test Complete ---")

    else:
        logger.error(
            "Could not fetch SPY data for RLAgentStrategy test. Check data ingestion."
        )
