"""
Multi-Agent Q-Learning Traffic Light Controller

This controller implements a multi-agent approach where each traffic light
is controlled by a separate Q-learning agent with coordination mechanisms.
"""

import os
import sys
import traci
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import pickle


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


def get_neighbor_tls(tl_id, max_distance=200):
    """
    Get neighboring traffic lights within a certain distance.
    
    Args:
        tl_id: Traffic light ID
        max_distance: Maximum distance to consider a traffic light as neighbor
        
    Returns:
        List of neighboring traffic light IDs
    """
    tl_pos = traci.junction.getPosition(tl_id)
    all_tls = traci.trafficlight.getIDList()
    neighbors = []
    
    for other_tl in all_tls:
        if other_tl == tl_id:
            continue
        
        other_pos = traci.junction.getPosition(other_tl)
        distance = np.sqrt((tl_pos[0] - other_pos[0])**2 + (tl_pos[1] - other_pos[1])**2)
        
        if distance <= max_distance:
            neighbors.append(other_tl)
    
    return neighbors


class MultiAgentQLearningAgent:
    """Q-Learning agent for traffic light control in a multi-agent setting."""
    
    def __init__(self, tl_id, phases, controlled_lanes, neighbors=None, 
                 alpha=0.5, gamma=0.95, epsilon=0.1):
        """
        Initialize the Q-Learning agent.
        
        Args:
            tl_id: Traffic light ID
            phases: List of traffic light phases
            controlled_lanes: List of lanes controlled by this traffic light
            neighbors: List of neighboring traffic light IDs
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.tl_id = tl_id
        self.phases = phases
        self.controlled_lanes = controlled_lanes
        self.neighbors = neighbors if neighbors else []
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Number of possible actions (phases)
        self.num_actions = len(phases)
        
        # Initialize Q-table
        self.q_table = {}
        
        # Current state and action
        self.current_state = None
        self.current_action = None
        self.previous_total_waiting_time = 0
        
        # Coordination parameters
        self.last_phase_change = 0
        self.min_phase_duration = 10  # Minimum phase duration in seconds
    
    def get_state(self):
        """
        Get the current state representation.
        State is represented as a tuple of queue lengths for each approach
        and the current phases of neighboring traffic lights.
        """
        # Get queue lengths for controlled lanes
        queue_lengths = []
        for lane in self.controlled_lanes:
            queue = get_queue_length(lane)
            # Discretize queue length: 0, 1-3, 4-6, 7+
            if queue == 0:
                queue_bin = 0
            elif 1 <= queue <= 3:
                queue_bin = 1
            elif 4 <= queue <= 6:
                queue_bin = 2
            else:
                queue_bin = 3
            queue_lengths.append(queue_bin)
        
        # Get current phases of neighboring traffic lights
        neighbor_phases = []
        for neighbor in self.neighbors:
            phase = traci.trafficlight.getPhase(neighbor)
            neighbor_phases.append(phase)
        
        # Combine queue lengths and neighbor phases
        state = tuple(queue_lengths + neighbor_phases)
        return state
    
    def get_reward(self):
        """
        Calculate reward based on the change in total waiting time
        and coordination with neighbors.
        """
        # Calculate waiting time component
        total_waiting_time = sum(get_waiting_time(lane) for lane in self.controlled_lanes)
        waiting_time_reward = self.previous_total_waiting_time - total_waiting_time
        self.previous_total_waiting_time = total_waiting_time
        
        # Calculate coordination component
        coordination_reward = 0
        for neighbor in self.neighbors:
            # Check if this agent and neighbor have compatible phases
            my_phase = traci.trafficlight.getPhase(self.tl_id)
            neighbor_phase = traci.trafficlight.getPhase(neighbor)
            
            # Simple coordination heuristic: reward if phases are different
            # (assuming this means green waves can propagate)
            if my_phase != neighbor_phase:
                coordination_reward += 1
        
        # Combine rewards
        total_reward = waiting_time_reward + 0.2 * coordination_reward
        return total_reward
    
    def choose_action(self, state, current_step):
        """
        Choose an action using epsilon-greedy policy with coordination constraints.
        
        Args:
            state: Current state
            current_step: Current simulation step
        """
        # Check if minimum phase duration has passed
        if current_step - self.last_phase_change < self.min_phase_duration:
            # Keep current phase if minimum duration not reached
            return self.current_action
        
        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Exploitation
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        
        return np.argmax(self.q_table[state])
    
    def choose_action_eval(self, state, current_step):
        """
        Choose an action for evaluation (no exploration).
        
        Args:
            state: Current state
            current_step: Current simulation step
        """
        # Check if minimum phase duration has passed
        if current_step - self.last_phase_change < self.min_phase_duration:
            # Keep current phase if minimum duration not reached
            return self.current_action
        
        # Exploitation only
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)
        
        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state][best_next_action] - self.q_table[state][action]
        )
    
    def step(self, current_step, training=True):
        """
        Execute one step of the Q-learning algorithm.
        
        Args:
            current_step: Current simulation step
            training (bool): Whether to update the Q-table (training mode) or not (evaluation mode)
        """
        # Get current state
        next_state = self.get_state()
        
        # If this is the first step, initialize current state and action
        if self.current_state is None:
            self.current_state = next_state
            if training:
                self.current_action = self.choose_action(next_state, current_step)
            else:
                self.current_action = self.choose_action_eval(next_state, current_step)
            traci.trafficlight.setPhase(self.tl_id, self.current_action)
            self.last_phase_change = current_step
            return
        
        # Calculate reward
        reward = self.get_reward()
        
        # Update Q-table if in training mode
        if training:
            self.update_q_table(self.current_state, self.current_action, reward, next_state)
            next_action = self.choose_action(next_state, current_step)
        else:
            next_action = self.choose_action_eval(next_state, current_step)
        
        # Apply action if it's different from current
        if next_action != self.current_action:
            traci.trafficlight.setPhase(self.tl_id, next_action)
            self.last_phase_change = current_step
        
        # Update current state and action
        self.current_state = next_state
        self.current_action = next_action
    
    def save_model(self, filepath):
        """
        Save the Q-table to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the Q-table from a file.
        
        Args:
            filepath (str): Path to load the model from
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file {filepath} not found")
            return False


def train(sumo_cfg, start_time, end_time, num_episodes, save_dir="models/multi_agent", gui=False):
    """
    Train multi-agent Q-learning agents for traffic light control.
    
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
        
        # Find neighbors for each traffic light
        tl_neighbors = {}
        for tl_id in tl_ids:
            tl_neighbors[tl_id] = get_neighbor_tls(tl_id)
        
        # Create or load multi-agent Q-learning agents for each traffic light
        agents = {}
        for tl_id in tl_ids:
            phases, controlled_lanes = get_tl_info(tl_id)
            agent = MultiAgentQLearningAgent(
                tl_id, phases, controlled_lanes, tl_neighbors[tl_id]
            )
            
            # Try to load existing model if not the first episode
            if episode > 0:
                model_path = os.path.join(save_dir, f"{tl_id}.pkl")
                agent.load_model(model_path)
            
            agents[tl_id] = agent
        
        # Episode metrics
        episode_total_reward = 0
        episode_waiting_times = []
        
        # Main simulation loop
        step = 0
        while step < (end_time - start_time):
            current_time = traci.simulation.getTime()
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
                
                # Execute Q-learning step (training mode)
                agent.step(current_time, training=True)
                
                # Accumulate reward
                episode_total_reward += agent.get_reward()
            
            # Store waiting time for this step
            episode_waiting_times.append(total_waiting_time)
            
            step += 1
        
        # Save models after each episode
        for tl_id, agent in agents.items():
            model_path = os.path.join(save_dir, f"{tl_id}.pkl")
            agent.save_model(model_path)
        
        # Store episode metrics
        training_metrics['episode_rewards'].append(episode_total_reward)
        training_metrics['episode_waiting_times'].append(np.mean(episode_waiting_times))
        
        print(f"Episode {episode+1} completed. Average waiting time: {np.mean(episode_waiting_times):.2f}")
        
        # Close the simulation
        traci.close()
    
    return training_metrics


def run(sumo_cfg, start_time, end_time, gui=False, model_dir="models/multi_agent", training_mode=False):
    """
    Run the multi-agent Q-learning traffic light controller simulation.
    
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
    
    # Find neighbors for each traffic light
    tl_neighbors = {}
    for tl_id in tl_ids:
        tl_neighbors[tl_id] = get_neighbor_tls(tl_id)
    
    # Create multi-agent Q-learning agents for each traffic light
    agents = {}
    for tl_id in tl_ids:
        phases, controlled_lanes = get_tl_info(tl_id)
        agent = MultiAgentQLearningAgent(
            tl_id, phases, controlled_lanes, tl_neighbors[tl_id]
        )
        
        # Load trained model if not in training mode
        if not training_mode:
            model_path = os.path.join(model_dir, f"{tl_id}.pkl")
            if not agent.load_model(model_path):
                print(f"Warning: No trained model found for {tl_id}. Using untrained agent.")
        
        agents[tl_id] = agent
    
    # Main simulation loop
    step = 0
    while step < (end_time - start_time):
        current_time = traci.simulation.getTime()
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
            
            # Execute Q-learning step
            agent.step(current_time, training=training_mode)
        
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
