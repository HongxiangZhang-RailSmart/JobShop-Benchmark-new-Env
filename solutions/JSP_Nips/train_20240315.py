import time
import sys
import numpy as np
from Params import configs
from PPO_model import PPO, Memory
from agent_utils import select_action
from mb_agg import *
from validation import validate
from env_test import NipsJSPEnv_test as Env_test
from uniform_instance_gen import uni_instance_gen
from pathlib import Path

base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))
device = torch.device(configs.device)


def main():

    envs_jsm = []
    for _ in range(configs.num_envs):
        this_env = Env_test(n_j=configs.n_j, n_m=configs.n_m)
        envs_jsm.append(this_env)

    data_generator = uni_instance_gen
    file_path = str(base_path) + '/data/jsp/nips/'
    dataLoaded = np.load(file_path+'generatedData' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(
        configs.np_seed_validation) + '.npy')
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))

    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    memories = [Memory() for _ in range(configs.num_envs)]

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=configs.n_j,
              n_m=configs.n_m,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    print(ppo.policy)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                             n_nodes=configs.n_j * configs.n_m,
                             device=device)
    # training loop
    log = []
    validation_log = []
    optimal_gaps = []
    optimal_gap = 1
    record = 100000
    for i_update in range(configs.max_updates):

        t3 = time.time()

        ep_rewards = [0 for _ in range(configs.num_envs)]
        adj_envs = []
        fea_envs = []
        candidate_envs = []
        mask_envs = []

        for i, env in enumerate(envs_jsm):
            this_data = data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high)
            adj, fea, candidate, mask = env.reset(this_data)

            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality
        # rollout the env
        while True:
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
            # print(envs[0].number_of_tasks)
            # print('shape', candidate_tensor_envs[0].shape, mask_tensor_envs[0].shape)
            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                for i in range(configs.num_envs):
                    # print('old', fea_tensor_envs[i].shape, adj_tensor_envs[i].shape)

                    pi, xxx = ppo.policy_old(x=fea_tensor_envs[i],
                                             graph_pool=g_pool_step,
                                             padded_nei=None,
                                             adj=adj_tensor_envs[i],
                                             candidate=candidate_tensor_envs[i].unsqueeze(0),
                                             mask=mask_tensor_envs[i].unsqueeze(0))
                    # print('old', pi.shape,xxx.shape)
                    action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                    action_envs.append(action)
                    a_idx_envs.append(a_idx)

            adj_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []
            # Saving episode data
            for i in range(configs.num_envs):
                memories[i].adj_mb.append(adj_tensor_envs[i])
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(a_idx_envs[i])

                adj, fea, reward, done, candidate, mask = envs_jsm[i].step(action_envs[i].item())

                adj_envs.append(adj)
                fea_envs.append(fea)
                candidate_envs.append(candidate)
                mask_envs.append(mask)
                ep_rewards[i] += reward
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)
            if envs_jsm[0].done():
                break
        for j in range(configs.num_envs):
            ep_rewards[j] -= envs_jsm[j].posRewards
            print(str(j) + ":", ep_rewards[j], envs_jsm[j].JSM_max_endTime, envs_jsm[j].JobShopModule.makespan)


        loss, v_loss = ppo.update(memories, configs.n_j * configs.n_m, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_rewards_all_env])
        if (i_update + 1) % 100 == 0:
            file_writing_obj = open(
                './' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(
                    configs.high) + '.txt', 'w')
            file_writing_obj.write(str(log))

        # log results
        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
            i_update + 1, mean_rewards_all_env, v_loss))

        # validate and save use mean performance
        t4 = time.time()
        if (i_update + 1) % 100 == 0:
            vali_result = - validate(vali_data, ppo.policy).mean()
            validation_log.append(vali_result)
            if vali_result < record:
                torch.save(ppo.policy.state_dict(), './{}.pth'.format(
                    str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))
                record = vali_result
            print('The validation quality is:', vali_result)
            file_writing_obj1 = open(
                './' + 'vali_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(
                    configs.high) + '.txt', 'w')
            file_writing_obj1.write(str(validation_log))
        t5 = time.time()

        # print('Training:', t4 - t3)
        # print('Validation:', t5 - t4)


if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    # print(total2 - total1)

