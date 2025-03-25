import traci
import numpy as np
import random
import timeit
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import librosa
import numpy as np

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7

# Load your trained siren detection model
try:
    siren_model = load_model('siren_model/best_model.keras')
    print("✅ Siren model loaded in training_simulation.py!")
except Exception as e:
    print(f"Error loading siren model: {e}")

def extract_features(audio_file, max_pad_len=862):
    audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    features = np.mean(mfccs, axis=1)
    return features.reshape(1, 1, 80)

def detect_siren_for_vehicle(vehicle_id):
    # Simulate retrieving the vehicle's audio file
    # In real deployment, you'd have audio capture. In simulation, you can use a random siren file or noise file:
    audio_file = f"dynamic_sounds/ambulance.wav"
    if not os.path.exists(audio_file):
        # fallback simulation (or skip)
        return False

    features = extract_features(audio_file)
    prediction = siren_model.predict(features)
    return prediction[0][0] > 0.5

class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []

    def run(self, episode):
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1

        while self._step < self._max_steps:
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # Emergency detection logic using state flags and simulated siren detection
            emergency_action = self._emergency_override(current_state)
            if emergency_action is not None:
                action = emergency_action
            else:
                action = self._choose_action(current_state)

            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            self._simulate(self._green_duration)

            old_action = action
            old_total_wait = current_total_wait
            self._reward_episode.append(reward)

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _emergency_override(self, current_state):
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            if traci.vehicle.getTypeID(car_id) == "emergency_veh":
                edge_id = traci.vehicle.getRoadID(car_id)
                distance_to_light = 750 - traci.vehicle.getLanePosition(car_id)

                if distance_to_light < 100:
                    if detect_siren_for_vehicle(car_id):
                        print(f"🚨 Siren detected from {car_id} approaching from {edge_id}. Forcing green in 3 steps.")
                        # Wait 3 simulation steps to simulate detection processing
                        for _ in range(3):
                            traci.simulationStep()
                            self._step += 1
                        if edge_id.startswith("N2TL") or edge_id.startswith("S2TL"):
                            return 0  # NS green
                        elif edge_id.startswith("E2TL") or edge_id.startswith("W2TL"):
                            return 2  # EW green
        return None

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state):
        return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        return halt_N + halt_S + halt_E + halt_W

    def _get_state(self):
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos

            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            else:
                lane_cell = 9

            if lane_id in ["W2TL_0", "W2TL_1", "W2TL_2"]:
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id in ["N2TL_0", "N2TL_1", "N2TL_2"]:
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id in ["E2TL_0", "E2TL_1", "E2TL_2"]:
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id in ["S2TL_0", "S2TL_1", "S2TL_2"]:
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))
                state[car_position] = 1
            elif lane_group == 0:
                state[lane_cell] = 1

        directions = {"N2TL": 80, "S2TL": 81, "E2TL": 82, "W2TL": 83}
        for car_id in car_list:
            if traci.vehicle.getTypeID(car_id) == "emergency_veh":
                edge_id = traci.vehicle.getRoadID(car_id)
                for direction_prefix, state_idx in directions.items():
                    if edge_id.startswith(direction_prefix):
                        state[state_idx] = 1

        return state

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode
