"""
Adaptive Traffic Light Controller

This controller implements a dynamic green-extension algorithm based on
real-time traffic conditions and queue thresholds.
"""

import os
import sys
import traci
import numpy as np
import pandas as pd
from collections import defaultdict


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


def get_active_lanes(tl_id, phase_index):
    """Get lanes that have green light in the current phase."""
    phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
    phase_state = phases[phase_index].state
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
    
    active_lanes = []
    for i, lane in enumerate(controlled_lanes):
        if i < len(phase_state) and phase_state[i] in ['G', 'g']:
            active_lanes.append(lane)
    
    return active_lanes


def run(sumo_cfg, start_time, end_time, gui=False):
    """
    Run the adaptive traffic light controller simulation.
    
    Args:
        sumo_cfg (str): Path to the SUMO configuration file
        start_time (int): Simulation start time in seconds
        end_time (int): Simulation end time in seconds
        gui (bool): Whether to show the SUMO GUI
        
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
    
    # Adaptive control parameters
    MIN_GREEN_TIME = 10  # Minimum green time in seconds
    MAX_GREEN_TIME = 60  # Maximum green time in seconds
    EXTENSION_TIME = 5   # Time to extend green if threshold is met
    QUEUE_THRESHOLD = 3  # Queue threshold for extension
    WAITING_TIME_THRESHOLD = 30  # Waiting time threshold for phase change
    
    # Store traffic light information
    tl_info = {}
    for tl_id in tl_ids:
        phases, controlled_lanes = get_tl_info(tl_id)
        tl_info[tl_id] = {
            'phases': phases,
            'controlled_lanes': controlled_lanes,
            'current_phase': 0,
            'min_green_time': MIN_GREEN_TIME,
            'max_green_time': MAX_GREEN_TIME,
            'elapsed_time': 0,
            'last_phase_change': 0
        }
    
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
        for tl_id in tl_ids:
            info = tl_info[tl_id]
            controlled_lanes = info['controlled_lanes']
            current_phase = info['current_phase']
            
            # Calculate metrics for all controlled lanes
            lane_queues = {}
            lane_waiting_times = {}
            for lane in controlled_lanes:
                queue = get_queue_length(lane)
                waiting_time = get_waiting_time(lane)
                emissions = get_emissions(lane)
                speed = get_avg_speed(lane)
                
                lane_queues[lane] = queue
                lane_waiting_times[lane] = waiting_time
                total_waiting_time += waiting_time
                total_queue_length += queue
                total_emissions += emissions
                total_speed += speed
                lane_count += 1
            
            # Adaptive logic for traffic light control
            info['elapsed_time'] += 1
            
            # Get active lanes (with green light) in current phase
            active_lanes = get_active_lanes(tl_id, current_phase)
            
            # Get inactive lanes (with red light) in current phase
            inactive_lanes = [lane for lane in controlled_lanes if lane not in active_lanes]
            
            # Calculate metrics for active and inactive lanes
            active_queue_sum = sum(lane_queues.get(lane, 0) for lane in active_lanes)
            inactive_queue_sum = sum(lane_queues.get(lane, 0) for lane in inactive_lanes)
            
            max_inactive_waiting_time = max([lane_waiting_times.get(lane, 0) for lane in inactive_lanes], default=0)
            
            # Decision logic
            if info['elapsed_time'] >= info['min_green_time']:
                # Check if we should extend the green phase
                if (active_queue_sum > QUEUE_THRESHOLD and 
                    info['elapsed_time'] < info['max_green_time']):
                    # Extend green time
                    pass  # Just continue with current phase
                elif (max_inactive_waiting_time > WAITING_TIME_THRESHOLD or
                      inactive_queue_sum > active_queue_sum * 1.5 or
                      info['elapsed_time'] >= info['max_green_time']):
                    # Change to next phase
                    next_phase = (current_phase + 1) % len(info['phases'])
                    traci.trafficlight.setPhase(tl_id, next_phase)
                    
                    # Update traffic light info
                    info['current_phase'] = next_phase
                    info['elapsed_time'] = 0
                    info['last_phase_change'] = step
            
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
    metrics = run(sumo_cfg, 28800, 29000, False)
    print("Simulation completed with metrics:", metrics)
