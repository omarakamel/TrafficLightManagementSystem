"""
Deep Q-Network (DQN) Traffic Light Controller

This controller implements a Deep Q-Network using TensorFlow/Keras
to control traffic light phases based on queue lengths and waiting times.
"""

import os
import sys
import traci
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


def get_tl_info(tl_id):
    """Get information about a traffic light."""
    phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
    return phases, controlled_lanes


def get_queue_length(lane_id):
    """Get the queue length for a specific lane."""
    return traci.lane.getLastStepHaltingNumber(lane_id)


def get_waiting_time(lane_id):
    """Get the accumulated waiting time for vehicles on a lane."""
    return traci.lane.getWaitingTime(lane_id)


def get_emissions(lane_id):
    """Get CO2 emissions for a lane."""
    return traci.lane.getCO2Emission(lane_id)


def get_avg_speed(lane_id):
    """Get average speed for a lane."""
    return traci.lane.getLastStepMeanSpeed(lane_id)


class DQNAgent:
    """Deep Q-Network agent for traffic light control."""
    
    def __init__(self, tl_id, phases, controlled_lanes, 
                 learning_rate=0.001, gamma=0.95, epsilon=0.1, 
                 epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=32, memory_size=2000):
        """
        Initialize the DQN agent.
        
        Args:
            tl_id: Traffic light ID
            phases: List of traffic light phases
            controlled_lanes: List of lanes controlled by this traffic light
            learning_rate: Learning rate for the neural network
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum value of epsilon
            batch_size: Size of batch for training
            memory_size: Size of replay memory
        """
        self.tl_id = tl_id
        self.phases = phases
        self.controlled_lanes = controlled_lanes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Number of possible actions (phases)
        self.num_actions = len(phases)
        
        # State size (number of lanes)
        self.state_size = len(controlled_lanes)
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Build neural network model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Current state and action
        self.current_state = None
        self.current_action = None
        self.previous_total_waiting_time = 0
        
        # Training parameters
        self.train_counter = 0
        self.update_target_counter = 0
        self.update_target_frequency = 10  # Update target network every 10 steps
    
    def _build_model(self):
        """Build a neural network model for DQN."""
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.num_actions, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model with weights from the main model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def get_state(self):
        """
        Get the current state representation.
        State is represented as an array of queue lengths for each lane.
        """
        state = np.zeros(self.state_size)
        for i, lane in enumerate(self.controlled_lanes):
            state[i] = get_queue_length(lane)
        return state
    
    def get_reward(self):
        """
        Calculate reward based on the change in total waiting time.
        Negative reward for increased waiting time, positive for decreased.
        """
        total_waiting_time = sum(get_waiting_time(lane) for lane in self.controlled_lanes)
        reward = self.previous_total_waiting_time - total_waiting_time
        self.previous_total_waiting_time = total_waiting_time
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        """
        # Exploration
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_actions)
        
        # Exploitation
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def choose_action_eval(self, state):
        """
        Choose an action for evaluation (no exploration).
        """
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train the model using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )
            
            # Get current Q-values
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            
            # Update Q-value for the action
            target_f[0][action] = target
            
            # Train the model
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def step(self, training=True):
        """
        Execute one step of the DQN algorithm.
        
        Args:
            training (bool): Whether to update the model (training mode) or not (evaluation mode)
        """
        # Get current state
        next_state = self.get_state()
        
        # If this is the first step, initialize current state and action
        if self.current_state is None:
            self.current_state = next_state
            if training:
                self.current_action = self.choose_action(next_state)
            else:
                self.current_action = self.choose_action_eval(next_state)
            traci.trafficlight.setPhase(self.tl_id, self.current_action)
            return
        
        # Calculate reward
        reward = self.get_reward()
        
        if training:
            # Store experience in memory
            done = False  # In traffic simulation, episodes don't really end
            self.remember(self.current_state, self.current_action, reward, next_state, done)
            
            # Choose next action (with exploration)
            next_action = self.choose_action(next_state)
            
            # Train the model
            self.train_counter += 1
            if self.train_counter % 5 == 0:  # Train every 5 steps
                self.replay()
            
            # Update target model
            self.update_target_counter += 1
            if self.update_target_counter % self.update_target_frequency == 0:
                self.update_target_model()
        else:
            # Choose next action (without exploration)
            next_action = self.choose_action_eval(next_state)
        
        # Apply action
        traci.trafficlight.setPhase(self.tl_id, next_action)
        
        # Update current state and action
        self.current_state = next_state
        self.current_action = next_action
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the model from a file.
        
        Args:
            filepath (str): Path to load the model from
        """
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            self.target_model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file {filepath} not found")
            return False


def train(sumo_cfg, start_time, end_time, num_episodes, save_dir="models/dqn", gui=False):
    """
    Train DQN agents for traffic light control.
    
    Args:
        sumo_cfg (str): Path to the SUMO configuration file
        start_time (int): Simulation start time in seconds
        end_time (int): Simulation end time in seconds
        num_episodes (int): Number of training episodes
        save_dir (str): Directory to save trained models
        gui (bool): Whether to show the SUMO GUI
        
    Returns:
        dict: Training metrics
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize metrics storage
    training_metrics = {
        'episode_rewards': [],
        'episode_waiting_times': []
    }
    
    # Training loop
    for episode in range(num_episodes):
        print(f"Training episode {episode+1}/{num_episodes}")
        
        # Prepare the SUMO command
        if gui:
            sumo_binary = "sumo-gui"
            sumo_cmd = [
                sumo_binary,
                "-c", sumo_cfg,
                "--begin", str(start_time),
                "--end", str(end_time),
                "--step-length", "0.25",  # Match the step length in the config file
                "--no-step-log", "true",
                "--no-warnings", "true",
                "--start",  # Start simulation automatically
            ]
        else:
            sumo_binary = "sumo"
            sumo_cmd = [
                sumo_binary,
                "-c", sumo_cfg,
                "--begin", str(start_time),
                "--end", str(end_time),
                "--step-length", "0.25",  # Match the step length in the config file
                "--no-step-log", "true",
                "--no-warnings", "true",
            ]
        
        # Start the simulation
        traci.start(sumo_cmd)
        
        # Get all traffic lights
        tl_ids = traci.trafficlight.getIDList()
        
        # Create or load DQN agents for each traffic light
        agents = {}
        for tl_id in tl_ids:
            phases, controlled_lanes = get_tl_info(tl_id)
            agent = DQNAgent(tl_id, phases, controlled_lanes)
            
            # Try to load existing model if not the first episode
            if episode > 0:
                model_path = os.path.join(save_dir, f"{tl_id}.h5")
                agent.load_model(model_path)
            
            agents[tl_id] = agent
        
        # Episode metrics
        episode_total_reward = 0
        episode_waiting_times = []
        
        # Main simulation loop
        step = 0
        while step < (end_time - start_time):
            traci.simulationStep()
            
            # Collect metrics for this step
            total_waiting_time = 0
            
            # Process each traffic light
            for tl_id, agent in agents.items():
                controlled_lanes = agent.controlled_lanes
                
                # Calculate waiting time
                for lane in controlled_lanes:
                    waiting_time = get_waiting_time(lane)
                    total_waiting_time += waiting_time
                
                # Execute DQN step (training mode)
                agent.step(training=True)
                
                # Accumulate reward
                episode_total_reward += agent.get_reward()
            
            # Store waiting time for this step
            episode_waiting_times.append(total_waiting_time)
            
            step += 1
        
        # Save models after each episode
        for tl_id, agent in agents.items():
            model_path = os.path.join(save_dir, f"{tl_id}.h5")
            agent.save_model(model_path)
        
        # Store episode metrics
        training_metrics['episode_rewards'].append(episode_total_reward)
        training_metrics['episode_waiting_times'].append(np.mean(episode_waiting_times))
        
        print(f"Episode {episode+1} completed. Average waiting time: {np.mean(episode_waiting_times):.2f}")
        
        # Close the simulation
        traci.close()
    
    return training_metrics


def run(sumo_cfg, start_time, end_time, gui=False, model_dir="models/dqn", training_mode=False):
    """
    Run the DQN traffic light controller simulation.
    
    Args:
        sumo_cfg (str): Path to the SUMO configuration file
        start_time (int): Simulation start time in seconds
        end_time (int): Simulation end time in seconds
        gui (bool): Whether to show the SUMO GUI
        model_dir (str): Directory containing trained models
        training_mode (bool): Whether to run in training mode or evaluation mode
        
    Returns:
        dict: Simulation metrics including waiting times, queue lengths,
              emissions, and average speeds
    """
    # Prepare the SUMO command
    if gui:
        sumo_binary = "sumo-gui"
        sumo_cmd = [
            sumo_binary,
            "-c", sumo_cfg,
            "--begin", str(start_time),
            "--end", str(end_time),
            "--step-length", "0.25",  # Match the step length in the config file
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--start",  # Start simulation automatically
        ]
    else:
        sumo_binary = "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", sumo_cfg,
            "--begin", str(start_time),
            "--end", str(end_time),
            "--step-length", "0.25",  # Match the step length in the config file
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]
    
    # Start the simulation
    traci.start(sumo_cmd)
    
    # Get all traffic lights
    tl_ids = traci.trafficlight.getIDList()
    
    # Initialize metrics storage
    metrics = {
        'waiting_times': [],
        'queue_lengths': [],
        'emissions': [],
        'avg_speeds': []
    }
    
    # Create DQN agents for each traffic light
    agents = {}
    for tl_id in tl_ids:
        phases, controlled_lanes = get_tl_info(tl_id)
        agent = DQNAgent(tl_id, phases, controlled_lanes)
        
        # Load trained model if not in training mode
        if not training_mode:
            model_path = os.path.join(model_dir, f"{tl_id}.h5")
            if not agent.load_model(model_path):
                print(f"Warning: No trained model found for {tl_id}. Using untrained agent.")
        
        agents[tl_id] = agent
    
    # Main simulation loop
    step = 0
    while step < (end_time - start_time):
        traci.simulationStep()
        
        # Collect metrics for this step
        total_waiting_time = 0
        total_queue_length = 0
        total_emissions = 0
        total_speed = 0
        lane_count = 0
        
        # Process each traffic light
        for tl_id, agent in agents.items():
            controlled_lanes = agent.controlled_lanes
            
            # Calculate metrics for controlled lanes
            for lane in controlled_lanes:
                waiting_time = get_waiting_time(lane)
                queue = get_queue_length(lane)
                emissions = get_emissions(lane)
                speed = get_avg_speed(lane)
                
                total_waiting_time += waiting_time
                total_queue_length += queue
                total_emissions += emissions
                total_speed += speed
                lane_count += 1
            
            # Execute DQN step
            agent.step(training=training_mode)
        
        # Store metrics for this step
        if lane_count > 0:
            metrics['waiting_times'].append(total_waiting_time)
            metrics['queue_lengths'].append(total_queue_length)
            metrics['emissions'].append(total_emissions)
            metrics['avg_speeds'].append(total_speed / lane_count)
        
        step += 1
    
    # Close the simulation
    traci.close()
    
    return metrics


if __name__ == "__main__":
    # Example usage
    sumo_cfg = "../MoSTScenario-master/scenario/most.traci.sumocfg"
    
    # Training example
    # train_metrics = train(sumo_cfg, 28800, 29000, num_episodes=5)
    # print("Training completed with metrics:", train_metrics)
    
    # Evaluation example
    metrics = run(sumo_cfg, 28800, 29000, gui=False, training_mode=False)
    print("Simulation completed with metrics:", metrics)
