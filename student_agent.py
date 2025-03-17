# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym


available = [(1, 1), (1, 1), (1, 1), (1, 1)]
visited_station = (0, 0, 0, 0)
destination = -1
passenger = -1
has_passenger = 0
def get_state(obs):
    global visited_station
    taxi_row, taxi_col, station1_x, station1_y, station2_x, station2_y, station3_x, station3_y, station4_x, station4_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
    sxs = [station1_x, station2_x, station3_x, station4_x]
    sys = [station1_y, station2_y, station3_y, station4_y]
    stations = [(station1_x, station1_y), (station2_x, station2_y), (station3_x, station3_y), (station4_x, station4_y)]
    stations_relative = [(taxi_row - station[0], taxi_col - station[1]) for station in stations]
    stations_direction = []
    for station_rel in stations_relative:
        x = station_rel[0]
        y = station_rel[1]
        if station_rel[0] > 1:
            x = 1
        if station_rel[0] < -1:
            x = -1
        if station_rel[1] > 1:
            y = 1
        if station_rel[1] < -1:
            y = -1
        stations_direction.append((x, y))
    at_station = (taxi_row, taxi_col) in stations
    if (taxi_row, taxi_col) in stations and visited_station[stations.index((taxi_row, taxi_col))] == 0:
        tmp = list(visited_station)
        tmp[stations.index((taxi_row, taxi_col))] = 1
        visited_station = tuple(tmp)
    return (visited_station, stations_relative[0], stations_relative[1], stations_relative[2], stations_relative[3], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)

# Global variable to store the Q-table.
q_table = None

def reset_available():
    visited_station = (0, 0, 0, 0)
    available = [(1, 1), (1, 1), (1, 1), (1, 1)]
    has_passenger = 0

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
