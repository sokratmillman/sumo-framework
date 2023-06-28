import argparse
import os
import sys

import pandas as pd
import numpy as np

from sumo_framework.environment.env import SumoEnvironment



if __name__ == "__main__":
    mode = "choose_next_phase"

    env = SumoEnvironment(
        net_file="nets/Moscow/osm.net.xml",
        route_file=(
            "nets/Moscow/routes_chumakova.rou.xml,"
            "nets/Moscow/routes_moskvitina.rou.xml,"
            "nets/Moscow/routes_valuevskoe.rou.xml,"
            "nets/Moscow/routes_habarova.rou.xml,"
            "nets/Moscow/routes_atlasova.rou.xml"
            ),
        begin_time=36000,
        sim_max_time=40000,
        min_green=20,
        max_green=100,
        delta_time=5,
        additional_sumo_cmd=f"-a nets/Moscow/tls_cycles.evening.add.xml",
        mode=mode,
        use_gui=True,
    )

    initial_state = env.reset()
    done = {"__all__": False}
    actions = {}
    
    while not done["__all__"]:
        obs, _, done, i = env.step(actions)
        actions = {}
        for ts in obs.keys():
            pressures = obs[ts]["pressure"]
            actions[ts] = np.argmax(pressures)  

    env.close()

