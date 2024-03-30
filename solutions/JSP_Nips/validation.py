from solutions.JSP_Nips.JSSP_Env import SJSSP
from solutions.JSP_Nips.mb_agg import g_pool_cal
from solutions.JSP_Nips.agent_utils import greedy_select_action
import numpy as np
import torch
from solutions.helper_functions import load_parameters
from pathlib import Path
import sys

base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

param_file = str(base_path) + "/configs/Nips_JSP.toml"
parameters = load_parameters(param_file)
env_parameters = parameters["env_parameter"]
model_parameters = parameters["network_parameter"]
train_parameters = parameters["train_parameter"]


def validate(vali_set, model):
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]

    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES)
    device = torch.device(env_parameters["device"])
    g_pool_step = g_pool_cal(graph_pool_type=model_parameters["graph_pool_type"],
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)
    make_spans = []
    # rollout using model
    for data in vali_set:
        adj, fea, candidate, mask = env.reset(data)
        rewards = - env.initQuality
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            with torch.no_grad():
                pi, _ = model(x=fea_tensor,
                              graph_pool=g_pool_step,
                              padded_nei=None,
                              adj=adj_tensor,
                              candidate=candidate_tensor.unsqueeze(0),
                              mask=mask_tensor.unsqueeze(0))
            # action = sample_select_action(pi, candidate)
            action = greedy_select_action(pi, candidate)
            adj, fea, reward, done, candidate, mask = env.step(action.item())
            rewards += reward
            if done:
                break
        make_spans.append(rewards - env.posRewards)
        # print(rewards - env.posRewards)
    return np.array(make_spans)



