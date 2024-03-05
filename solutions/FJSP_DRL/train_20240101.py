# -*- coding: utf-8 -*-
# @Time    : 2024/1/1 15:36
# @Author  : Hongxinag Zhang
# @Email   : hongxiang@my.swjtu.edu.cn
# @File    : train_20240101.py
# @Software: PyCharm

import argparse
import os
import logging
from solutions.helper_functions import load_parameters
from data_parsers import parser_fajsp, parser_fjsp, parser_jsp_fsp
from solutions.helper_functions import load_job_shop_env, load_FJSPEnv_from_case
# from solutions.FJSP_DRL.load_data import nums_detec
from solutions.FJSP_DRL.case_generator import CaseGenerator
# from solutions.FJSP_DRL.env import FJSPEnv_new
from solutions.FJSP_DRL.env_new import FJSPEnv_new
import solutions.FJSP_DRL.PPO_model as PPO_model
import random
import time
import sys
import torch
import numpy as np
# from visdom import Visdom
from pathlib import Path
# Add the base path to the Python module search path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

# import PPO_model

PARAM_FILE = str(base_path) + "/configs/FJSP_DRL.toml"

def initialize_device(parameters: dict) -> torch.device:
    device_str = "cpu"
    if parameters['test_parameters']['device'] == "cuda":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)

def nums_detec(lines):
    """
    Count the number of jobs, machines and operations
    """

    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i] != "\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes

def main(param_file: str = PARAM_FILE):
    # print(os.path.abspath('..'))
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    device = initialize_device(parameters)

    # Configure PyTorch's default device
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if device.type == 'cuda' else 'torch.FloatTensor')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Extract parameters
    env_parameters = parameters["env_parameters"]
    model_parameters = parameters["model_parameters"]
    train_parameters = parameters["train_parameters"]
    test_parameters = parameters["test_parameters"]

    model_parameters["actor_in_dim"] = model_parameters["out_size_ma"] * 2 + model_parameters["out_size_ope"] * 2
    model_parameters["critic_in_dim"] = model_parameters["out_size_ma"] + model_parameters["out_size_ope"]

    num_jobs = env_parameters['num_jobs']
    num_machines = env_parameters['num_mas']

    opes_per_job_min = int(num_machines * 0.8)
    opes_per_job_max = int(num_machines * 1.2)
    print(num_jobs, num_machines)

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_parameters, train_parameters, num_envs=env_parameters["batch_size"])

    # Training part
    start_time = time.time()
    Env_training = None
    for i in range(1, train_parameters["max_iterations"]+1):
        if (i - 1) % train_parameters["parallel_iter"] == 0:
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_machines, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            JSPEnv_in_this_batch = [load_FJSPEnv_from_case(case.get_case(j)[0], num_jobs, num_machines) for j in range(env_parameters["batch_size"])]
            Env_training = FJSPEnv_new(JSPEnv_in_this_batch, env_parameters)
            print(Env_training.num_jobs, Env_training.num_opes)

        for j in range(env_parameters['batch_size']):
            Env_training.JSP_instance[j]._name = 'TrainStep' + str(i) + '-' + "case"+str(j+1)

        # Get state and completion signal
        state = Env_training.state
        done = False
        dones = Env_training.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = Env_training.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("step:", i, "spend_time:", time.time() - last_time)
        print([Env_training.JSP_instance[i].makespan for i in range(Env_training.batch_size)])

        Env_training.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_parameters["update_timestep"] == 0:
            loss, reward = model.update(memories, env_parameters, train_parameters)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()

        #############
        # Validate, every train_paras["save_timestep"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FJSP_DRL")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)