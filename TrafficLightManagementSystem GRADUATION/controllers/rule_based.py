"""
Rule-Based Traffic Light Controller

This controller implements simple IF/ELSE rules based on queue lengths
to adjust traffic light phases and durations.
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


def run(sumo_cfg, start_time, end_time, gui=False):
    """
    Run the rule-based traffic light controller simulation.
    
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
    
    # Store traffic light information
    tl_info = {}
    for tl_id in tl_ids:
        phases, controlled_lanes = get_tl_info(tl_id)
        tl_info[tl_id] = {
            'phases': phases,
            'controlled_lanes': controlled_lanes,
            'current_phase': 0,
            'phase_duration': 30,  # Default duration
            'elapsed_time': 0
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
            
            # Calculate metrics for controlled lanes
            lane_queues = {}
            for lane in controlled_lanes:
                queue = get_queue_length(lane)
                waiting_time = get_waiting_time(lane)
                emissions = get_emissions(lane)
                speed = get_avg_speed(lane)
                
                lane_queues[lane] = queue
                total_waiting_time += waiting_time
                total_queue_length += queue
                total_emissions += emissions
                total_speed += speed
                lane_count += 1
            
            # Rule-based logic for traffic light control
            info['elapsed_time'] += 1
            
            # Check if it's time to change the phase
            if info['elapsed_time'] >= info['phase_duration']:
                # Find the lanes with the longest queues
                max_queue = 0
                max_queue_phase = 0
                
                # Group lanes by phase
                phase_queues = defaultdict(int)
                for i, phase in enumerate(info['phases']):
                    phase_state = phase.state
                    for j, lane in enumerate(controlled_lanes):
                        if j < len(phase_state) and phase_state[j] in ['G', 'g']:
                            phase_queues[i] += lane_queues.get(lane, 0)
                
                # Find phase with longest queue
                for phase_idx, queue in phase_queues.items():
                    if queue > max_queue:
                        max_queue = queue
                        max_queue_phase = phase_idx
                
                # Set the new phase
                new_phase = max_queue_phase
                traci.trafficlight.setPhase(tl_id, new_phase)
                
                # Adjust duration based on queue length
                if max_queue > 10:
                    new_duration = 45  # Longer green for long queues
                elif max_queue > 5:
                    new_duration = 30  # Medium duration for medium queues
                else:
                    new_duration = 15  # Short duration for short queues
                
                # Update traffic light info
                info['current_phase'] = new_phase
                info['phase_duration'] = new_duration
                info['elapsed_time'] = 0
        
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
