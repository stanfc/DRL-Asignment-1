# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym


available = [(1, 1), (1, 1), (1, 1), (1, 1)]
def get_state(obs):
    taxi_row, taxi_col, station1_x, station1_y, station2_x, station2_y, station3_x, station3_y, station4_x, station4_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
    sxs = [station1_x, station2_x, station3_x, station4_x]
    sys = [station1_y, station2_y, station3_y, station4_y]
    stations = [(station1_x, station1_y), (station2_x, station2_y), (station3_x, station3_y), (station4_x, station4_y)]
    stations_relative = [(taxi_row - station[0], taxi_col - station[1]) for station in stations]
    stations_direction = [(int(rel[0] / abs(rel[0]) if rel[0] != 0 else 0), int(rel[1] / abs(rel[1]) if rel[1] != 0 else 0)) for rel in stations_relative]
    at_station = (taxi_row, taxi_col) in stations
    if at_station:

        for i in range(4):
            if (taxi_row, taxi_col) == stations[i]:
                if not passenger_look:
                    available[i] = (0, available[i][1])
                if not destination_look:
                    available[i] = (available[i][0], 0)

    return (stations_direction[0], stations_direction[1], stations_direction[2], stations_direction[3], available[0], available[1], available[2], available[3], at_station, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)

# Global variable to store the Q-table.
q_table = None

def reset_available():
    available = [(1, 1), (1, 1), (1, 1), (1, 1)]

def load_q_table():
    global q_table
    if q_table is None:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
    return q_table

def get_action(obs):
    """
    Given an observation, use the loaded Q-table to select an action.
    If the observation key is missing, choose a random action as a fallback.
    """
    # Load Q-table if not already loaded.
    qt = load_q_table()
    
    # Convert the observation to a unique state key.
    state_key = get_state(obs)
    
    # Check if the state key exists in the Q-table.
    if state_key in qt:
        # Exploit: select the action with the highest Q-value.
        action = int(np.argmax(np.array(q_table[state_key])))
    else:
        # Fallback: choose a random action if the state is not present.
        action = random.choice([0, 1, 2, 3, 4, 5])
    
    return action
