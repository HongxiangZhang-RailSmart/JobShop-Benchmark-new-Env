# GITHUB REPO: https://github.com/songwenas12/fjsp-drl

# Code based on the paper:
# "Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning"
# by Wen Song, Xinyang Chen, Qiqiang Li and Zhiguang Cao
# Presented in IEEE Transactions on Industrial Informatics, 2023.
# Paper URL: https://ieeexplore.ieee.org/document/9826438

import time
import copy
import os

import torch
from pathlib import Path
import gym

import solutions.FJSP_DRL.PPO_model as PPO_model

base_path = Path(__file__).resolve().parents[2]


def get_validate_env(env_paras, train_paras):
    """
    Generate and return the validation environment from the validation set ()
    """
    file_path = str(base_path) + '/data/' + train_paras["validation_folder"]
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path + valid_data_files[i]
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=env_paras, data_source='file')
    return env


def validate(env_paras, env_validate, model_policy):
    """
    Validate the policy during training, and the process is similar to test
    """
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    env_validate.reset()
    state = env_validate.state
    done = False
    dones = env_validate.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, rewards, dones, _ = env_validate.step(actions)
        done = dones.all()
    gantt_result = env_validate.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error!")
    makespan = copy.deepcopy(env_validate.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env_validate.makespan_batch)
    env_validate.reset()
    print('validating time: ', time.time() - start, '\n')
    return makespan, makespan_batch
