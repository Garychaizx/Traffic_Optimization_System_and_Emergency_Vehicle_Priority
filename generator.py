import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        np.random.seed(seed)
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        for value in timings:
            car_gen_steps = np.append(car_gen_steps,
                                      ((self._max_steps / (max_old - min_old)) * (value - max_old) + self._max_steps))

        car_gen_steps = np.rint(car_gen_steps)

        vehicles = []

        # generate emergency vehicles at predefined times
        # emergency_schedule = [
        #     (100, "N_S"), (150, "N_S"), (250, "N_S"), (400, "S_N")
        # ]
        # for i, (depart_time, route) in enumerate(emergency_schedule):
        #     vehicles.append({
        #         "id": f"emergency_{i + 1}",
        #         "type": "emergency_veh",
        #         "route": route,
        #         "depart": depart_time,
        #         "speed": 10,
        #         "is_emergency": True
        #     })

        # Generate standard cars and some random emergency vehicles
        for car_counter, step in enumerate(car_gen_steps):
            if np.random.rand() < 0.02:  # 2% chance for random emergency vehicle
                route = np.random.choice(['W_E', 'E_W', 'N_S', 'S_N'])  # straight routes for emergency
                vehicles.append({
                    "id": f"emergency_{car_counter}",
                    "type": "emergency_veh",
                    "route": route,
                    "depart": step,
                    "speed": 20,
                    "is_emergency": True
                })
            else:
                # Standard car generation
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:
                    route_straight = np.random.choice(["W_E", "E_W", "N_S", "S_N"])
                    vehicles.append({
                        "id": f"{route_straight}_{car_counter}",
                        "type": "standard_car",
                        "route": route_straight,
                        "depart": step,
                        "speed": 10
                    })
                else:
                    route_turn = np.random.choice(["W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E"])
                    vehicles.append({
                        "id": f"{route_turn}_{car_counter}",
                        "type": "standard_car",
                        "route": route_turn,
                        "depart": step,
                        "speed": 10
                    })

        # # Generate standard cars only
        # for car_counter, step in enumerate(car_gen_steps):
        #     # Standard car generation
        #     straight_or_turn = np.random.uniform()
        #     if straight_or_turn < 0.75:
        #         route_straight = np.random.choice(["W_E", "E_W", "N_S", "S_N"])
        #         vehicles.append({
        #             "id": f"{route_straight}_{car_counter}",
        #             "type": "standard_car",
        #             "route": route_straight,
        #             "depart": step,
        #             "speed": 10
        #         })
        #     else:
        #         route_turn = np.random.choice(["W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E"])
        #         vehicles.append({
        #             "id": f"{route_turn}_{car_counter}",
        #             "type": "standard_car",
        #             "route": route_turn,
        #             "depart": step,
        #             "speed": 10
        #         })

        # Sort by depart time
        vehicles.sort(key=lambda x: x["depart"])

        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5"/>
            <vType  accel="1.0" decel="4.5" id="emergency_veh" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" guiShape="emergency"/>
            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for v in vehicles:
                print(
                    f'    <vehicle id="{v["id"]}" type="{v["type"]}" route="{v["route"]}" depart="{int(v["depart"])}" departLane="random" departSpeed="{v["speed"]}" />',
                    file=routes)

            print("</routes>", file=routes)

