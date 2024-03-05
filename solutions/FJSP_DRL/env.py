# GITHUB REPO: https://github.com/songwenas12/fjsp-drl

# Code based on the paper:
# "Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning"
# by Wen Song, Xinyang Chen, Qiqiang Li and Zhiguang Cao
# Presented in IEEE Transactions on Industrial Informatics, 2023.
# Paper URL: https://ieeexplore.ieee.org/document/9826438

import random
import sys
import copy

import gym
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from solutions.FJSP_DRL.load_data import load_fjs, nums_detec, load_fjs_case
from scheduling_environment.jobShop import JobShop
from solutions.FJSP_DRL.load_data import load_fjs_from_sim


# Add the base path to the Python module search path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time


def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    """
    Convert job features into operation features (such as dimension)
    """
    return feat_job_batch.gather(1, opes_appertain_batch)

class FJSPEnv_new():

    def __init__(self, JobShop_module, env_paras):
        # static
        self.show_mode = env_paras["show_mode"]  # Result display mode (deprecated in the final experiment)
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs
        self.num_mas = env_paras["num_mas"]  # Number of machines
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch

        self.JSP_instance: list[JobShop] = JobShop_module

        # load instance
        num_data = 8  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        for i in range(self.batch_size):
            self.num_opes = max(self.num_jobs, len(self.JSP_instance[i].operations))

        # Extract features from each JobShop module
        for i in range(self.batch_size):
            raw_features = load_fjs_from_sim(self.JSP_instance[i], self.num_mas, self.num_opes)
            # print(raw_features[0].shape)
            for j in range(num_data):
                tensors[j].append(raw_features[j].to(self.device))

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], self.num_mas))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)

        # self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        # self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        # self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]

        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)

        # new features (generated from JobShop module)
        self.JSM_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.JSM_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.JSM_time = copy.deepcopy(self.time)
        self.JSM_mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                 fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.JSM_mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.JSM_mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)


    def step(self, actions):
        """
        Environment transition function, based on JobShop module
        """
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1

        for index in range(self.batch_size):
            ope_idx = opes[index].item()
            mac_idx = mas[index].item()
            env = self.JSP_instance[index]
            operation = env.operations[ope_idx]
            duration = operation.processing_times[mac_idx]
            env.schedule_operation_on_machine(operation, mac_idx, duration)
            env.get_job(operation.job_id).scheduled_operations.append(operation)


        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1)
        self.JSM_feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1)
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1,
                                opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        # Update 'Number of unscheduled operations in the job'
        start_ope_idx = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope_idx = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope_idx[i]:end_ope_idx[i] + 1] -= 1

        # Update 'Number of unscheduled operations in the job' - use JobShop
        for instance_idx in range(self.batch_size):
            job_idx = jobs[instance_idx].item()
            unscheduled_opes = 0
            for each_ope in self.JSP_instance[instance_idx].get_job(job_idx).operations:
                if each_ope.scheduling_information.__len__() == 0:
                    unscheduled_opes += 1
            start_ope_idx = self.num_ope_biases_batch[self.batch_idxes, jobs]
            end_ope_idx = self.end_ope_biases_batch[self.batch_idxes, jobs]
            self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 3, start_ope_idx[instance_idx]:end_ope_idx[instance_idx] + 1] = unscheduled_opes
            # print('test1', unscheduled_opes,
            #       self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 3, start_ope_idx[instance_idx]:end_ope_idx[instance_idx] + 1],
            #       self.feat_opes_batch[self.batch_idxes[instance_idx], 3, start_ope_idx[instance_idx]:end_ope_idx[instance_idx] + 1])

        # Update 'Start time' and 'Job completion time'
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5,
                      :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                                   self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() \
                         * un_scheduled  # estimate start time of unscheduled opes
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        # print(start_times[0,:], estimate_times[0,:])
        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                          self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch,self.opes_appertain_batch[self.batch_idxes, :])

        # Update 'Start time' and 'Job completion time' - use JobShop
        self.JSM_feat_opes_batch[self.batch_idxes, 5, opes] = self.JSM_time[self.batch_idxes]
        for instance_idx in range(self.batch_size):
            for each_ope in self.JSP_instance[instance_idx].operations:
                if each_ope.scheduling_information.__len__() == 0:
                    if each_ope.predecessors.__len__() == 0:
                        est_start_time = self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 5, each_ope.operation_id]
                    else:
                        if each_ope.predecessors[0].scheduling_information.__len__() == 0:
                            est_start_time = self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 5, each_ope.predecessors[0].operation_id] + self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 2, each_ope.predecessors[0].operation_id]
                        else:
                            est_start_time = each_ope.predecessors[0].scheduling_information['start_time']  + each_ope.predecessors[0].scheduling_information['processing_time']
                else:
                    est_start_time = each_ope.scheduling_information['start_time']
                self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 5, each_ope.operation_id] = est_start_time

            for each_job in self.JSP_instance[instance_idx].jobs:
                est_end_times = [(self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 5, ope_in_job.operation_id] + self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 2, ope_in_job.operation_id]) for ope_in_job in each_job.operations]
                job_end_time = max(est_end_times)
                for ope_of_job in each_job.operations:
                    self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 4, ope_of_job.operation_id] = job_end_time
            # print('-------------------------------------------')
            # print(self.JSP_instance[instance_idx].instance_name, 'test_ope_feature_5', self.N[self.batch_idxes[instance_idx]], self.time[self.batch_idxes[instance_idx]].item())
            # print('Std', self.feat_opes_batch[self.batch_idxes[instance_idx], 5, :])
            # print('JSM', self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 5, :])
            # print(self.feat_opes_batch[self.batch_idxes[instance_idx], 5, :].equal(self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 5, :]))
            # print(self.JSP_instance[instance_idx].instance_name, 'test_ope_feature_4',
            #       self.N[self.batch_idxes[instance_idx]], self.time[self.batch_idxes[instance_idx]].item())
            # print('Std', self.feat_opes_batch[self.batch_idxes[instance_idx], 4, :])
            # print('JSM', self.JSM_feat_opes_batch[self.batch_idxes[instance_idx], 4, :])

        # Update partial schedule (state)
        # self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas), dim=1)
        # self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        # self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
        #                                                self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        # Update feature vectors of machines
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time == 0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time - using JobShop Module
        for instance_idx in range(self.batch_size):
            this_env = self.JSP_instance[instance_idx]
            cur_times = []
            for each_job in this_env.jobs:
                if len(each_job.scheduled_operations) == len(each_job.operations):
                    continue
                next_ope = each_job.operations[len(each_job.scheduled_operations)]
                for each_mach_id in next_ope.optional_machines_id:
                    # if schedule the next operation of this job on an optional machine, the earlist start time
                    # operation available: predecessor operation end
                    # machine available: last assigned operation end
                    cur_times.append(max(next_ope.finishing_time_predecessors, this_env.get_machine(each_mach_id).next_available_time, self.JSM_time[instance_idx]))
            self.JSM_time[instance_idx] = min(cur_times, default=self.JSM_time[instance_idx])

        # Update feature vectors of machines - using JobShop module
        self.JSM_feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :],dim=1).float()

        for instance_idx in range(self.batch_size):
            for each_mach in self.JSP_instance[instance_idx].machines:
                workload = sum([ope_on_mach.scheduled_duration for ope_on_mach in each_mach.scheduled_operations])
                cur_time = self.JSM_time[self.batch_idxes[instance_idx], None]
                workload = min(cur_time, workload)
                self.JSM_feat_mas_batch[self.batch_idxes[instance_idx], 2, each_mach.machine_id] = workload / (cur_time + 1e-9)
                self.JSM_feat_mas_batch[self.batch_idxes[instance_idx], 1, each_mach.machine_id] = each_mach.next_available_time
        print('-------------------------------------------')
        print(self.JSP_instance[5].instance_name, 'test_mas_feature_2', self.N[self.batch_idxes[5]].item(), self.time[self.batch_idxes[5]].item())
        print('Std', self.feat_mas_batch[self.batch_idxes[5], 1, :])
        print('JSM', self.JSM_feat_mas_batch[self.batch_idxes[5], 1, :])


        # self.mask_job_procing_batch, self.mask_job_finish_batch, self.mask_ma_procing_batch
        JSM_mask_jp_list = []
        JSM_mask_jf_list = []
        JSM_mask_mp_list = []
        for instance_idx in range(self.batch_size):
            JSM_mask_jp_list.append([True if this_job.next_ope_earlist_begin_time > self.JSM_time[instance_idx] else False
                                     for this_job in self.JSP_instance[instance_idx].jobs])
            JSM_mask_jf_list.append([True if this_job.operations.__len__() == this_job.scheduled_operations.__len__() else False
                 for this_job in self.JSP_instance[instance_idx].jobs])
            JSM_mask_mp_list.append([True if this_mach.next_available_time > self.JSM_time[instance_idx] else False
                 for this_mach in self.JSP_instance[instance_idx].machines])
        self.JSM_mask_job_procing_batch = torch.tensor(JSM_mask_jp_list, dtype=torch.bool)
        self.JSM_mask_job_finish_batch = torch.tensor(JSM_mask_jf_list, dtype=torch.bool)
        self.JSM_mask_ma_procing_batch = torch.tensor(JSM_mask_mp_list, dtype=torch.bool)


        max_makespan = torch.max(self.JSM_feat_opes_batch[:, 4, :], dim=1)[0]
        self.reward_batch = self.makespan_batch - max_makespan
        self.makespan_batch = max_makespan

        # Update the vector for uncompleted instances
        mask_finish = (self.N + 1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        self.state.update(self.batch_idxes, self.JSM_feat_opes_batch, self.JSM_feat_mas_batch, self.proc_times_batch,
                          self.ope_ma_adj_batch, self.JSM_mask_job_procing_batch, self.JSM_mask_job_finish_batch,
                          self.JSM_mask_ma_procing_batch,self.ope_step_batch, self.JSM_time)
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        """
        Check if there are still O-M pairs to be processed
        """
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                           self.proc_times_batch.size(
                                                                                               2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(
            torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
            dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        """
        Transit to the next time
        """
        # need to transit
        flag_need_trans = (flag_trans_2_next_time == 0) & (~self.done_batch)
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # remain available_time greater than current time
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        # Return the minimum value of available_time (the time to transit to)
        c = torch.min(b, dim=1)[0]
        # Detect the machines that completed (at above time)
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def reset(self):
        """
        Reset the environment to its initial state
        """
        for i in range(self.batch_size):
            self.JSP_instance[i].reset()

        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)
        self.JSM_feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.JSM_feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.JSM_time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                 fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)
        # self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        # self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        # self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.JSM_mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                 fill_value=False)
        self.JSM_mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)
        self.JSM_mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)
        return self.state

    def get_idx(self, id_ope, batch_id):
        """
        Get job and operation (relative) index based on instance index and operation (absolute) index
        """
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

'''
    def validate_gantt(self):
        """
        Verify whether the schedule is feasible
        """
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i]) - 1):
                        break
                    if ma_gantt[i][j][2] > ma_gantt[i][j + 1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2] - ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i] + j]
                    step_next = schedule[num_ope_biases[i] + j + 1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0] == 1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch
'''



