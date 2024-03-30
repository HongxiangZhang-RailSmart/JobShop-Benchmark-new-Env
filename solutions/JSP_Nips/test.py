import time
import sys
import numpy as np
# from solutions.JSP_Nips.Params import configs
from solutions.JSP_Nips.PPO_model import PPO, Memory
from solutions.JSP_Nips.mb_agg import *
from solutions.JSP_Nips.agent_utils import *
from solutions.JSP_Nips.env_test import NipsJSPEnv_test as Env_test
from solutions.helper_functions import convert_uni_instance_to_lines as Converter
from solutions.helper_functions import load_NipsJSPEnv_from_file, load_NipsJSPEnv_from_case, load_parameters
import torch

from pathlib import Path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

param_file = str(base_path) + "/configs/Nips_JSP.toml"
parameters = load_parameters(param_file)
env_parameters = parameters["env_parameter"]
model_parameters = parameters["network_parameter"]
train_parameters = parameters["train_parameter"]
test_parameters = parameters["test_parameter"]

device = torch.device(env_parameters["device"])

def main(test_mode, instance_class):
    N_JOBS_Test = test_parameters["Pn_j"]
    N_MACHINES_Test = test_parameters["Pn_m"]
    LOW = test_parameters["low"]
    HIGH = test_parameters["high"]
    SEED = test_parameters["seed"]
    N_JOBS_Policy = test_parameters["Nn_j"]
    N_MACHINES_Policy = test_parameters["Nn_m"]

    file_path_instance = str(base_path) + '/data/jsp/'
    case_test_JSM = []
    envs_test = []
    if test_mode == 'benchmark':
        if instance_class == 'dmu' or instance_class == 'tai':
            dataLoaded = np.load(file_path_instance+'nips/'+ test_parameters["instance_name"] + '.npy')
            for i in range(dataLoaded.shape[0]):
                this_data = dataLoaded[i][0], dataLoaded[i][1]
                lines = Converter(this_data, N_JOBS_Test, N_MACHINES_Test)
                case_test_JSM.append(load_NipsJSPEnv_from_case(lines, N_JOBS_Test, N_MACHINES_Test))
                envs_test.append(Env_test(n_j=N_JOBS_Test, n_m=N_MACHINES_Test))
        if instance_class == 'adams' or instance_class == 'taillard':
            this_JSM = load_NipsJSPEnv_from_file('/jsp/'+instance_class+'/'+test_parameters["instance_name"])
            N_JOBS_Test = len(this_JSM.jobs)
            N_MACHINES_Test = len(this_JSM.machines)
            case_test_JSM.append(this_JSM)
            envs_test.append(Env_test(n_j=N_JOBS_Test, n_m=N_MACHINES_Test))
    else:
        dataLoaded = np.load(file_path_instance+'nips/' + instance_class + str(N_JOBS_Test) + '_' + str(N_MACHINES_Test) + '_Seed' + str(
            SEED) + '.npy')
        for i in range(dataLoaded.shape[0]):
            this_data = dataLoaded[i][0], dataLoaded[i][1]
            lines = Converter(this_data, N_JOBS_Test, N_MACHINES_Test)
            case_test_JSM.append(load_NipsJSPEnv_from_case(lines, N_JOBS_Test, N_MACHINES_Test))
            envs_test.append(Env_test(n_j=N_JOBS_Test, n_m=N_MACHINES_Test))

    ppo = PPO(train_parameters["lr"], train_parameters["gamma"], train_parameters["k_epochs"], train_parameters["eps_clip"],
              n_j=N_JOBS_Test,
              n_m=N_MACHINES_Test,
              num_layers=model_parameters["num_layers"],
              neighbor_pooling_type=model_parameters["neighbor_pooling_type"],
              input_dim=model_parameters["input_dim"],
              hidden_dim=model_parameters["hidden_dim"],
              num_mlp_layers_feature_extract=model_parameters["num_mlp_layers_feature_extract"],
              num_mlp_layers_actor=model_parameters["num_mlp_layers_actor"],
              hidden_dim_actor=model_parameters["hidden_dim_actor"],
              num_mlp_layers_critic=model_parameters["num_mlp_layers_critic"],
              hidden_dim_critic=model_parameters["hidden_dim_critic"])
    file_path_policy = str(base_path) + '/save/JSP_Nips/'+str(N_JOBS_Policy) + '_' + str(N_MACHINES_Policy) + '_' +\
                       str(LOW) + '_' + str(HIGH)+'.pth'
    ppo.policy.load_state_dict(torch.load(file_path_policy))
    g_pool_step = g_pool_cal(graph_pool_type=model_parameters["graph_pool_type"],
                             batch_size=torch.Size([1, N_JOBS_Test * N_MACHINES_Test, N_JOBS_Test * N_MACHINES_Test]),
                             n_nodes=N_JOBS_Test * N_MACHINES_Test,
                             device=device)

    result = []
    for i in range(len(envs_test)):
        test_case = case_test_JSM[i]
        test_env = envs_test[i]

        adj, fea, candidate, mask = test_env.reset(test_case)
        ep_reward = - test_env.JSM_max_endTime
        # print('init', test_env.JSM_max_endTime)
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                   graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                # action = sample_select_action(pi, omega)
                action = greedy_select_action(pi, candidate)

            adj, fea, reward, done, candidate, mask = test_env.step(action)
            ep_reward += reward

            if done:
                break

        print('Instance' + str(i + 1) + ' makespan:', -ep_reward + test_env.posRewards)
        result.append(-ep_reward + test_env.posRewards)

    print(result)


if __name__ == '__main__':
    total1 = time.time()
    main('generatedData', 'generatedData')
    total2 = time.time()
    # print(total2 - total1)