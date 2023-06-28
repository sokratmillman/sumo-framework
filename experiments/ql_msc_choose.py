import argparse
import os
import sys

import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_framework.environment.env import SumoEnvironment
from sumo_framework.agents import CustomAgent, create_greedy_policy


if __name__ == "__main__":
    alpha = 0.02
    gamma = 0.99
    decay = 1
    runs = 1
    episodes = 100
    fixed = False
    mode = "choose_next_phase"
    epsilon = 0.1
    # cluster = 3

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
        mode=mode, # ["switch"|"choose_next_phase"|"phases_duration"]
        use_gui=True,
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()[0]
        ql_agents = {
            ts: CustomAgent(
                initial_state=env.encode(initial_states, ts),
                state_space=env.observation_spaces[ts],
                action_space=env.action_spaces(ts),
                alpha=alpha,
                discount_factor=gamma,
                exploration_strategy=create_greedy_policy(epsilon=epsilon),
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()[0]
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states, ts)

            infos = []
            done = {"__all__": False}
            if fixed:
                while not done["__all__"]:
                    _, _, done, _ = env.step({})
            else:
                while not done["__all__"]:
                    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                    s, r, done, info = env.step(action=actions)

                    for agent_id in s.keys():
                        ql_agents[agent_id].learn(next_state=env.encode(s, agent_id), reward=r[agent_id])

            env.save_csv(f"outputs/msc_choose/ql-msc_run{run}_{mode}_mode", episode)

    env.close()
