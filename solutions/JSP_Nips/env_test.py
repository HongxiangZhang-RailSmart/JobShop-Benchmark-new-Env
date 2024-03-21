import copy
import sys
import numpy as np
from Params import configs
from mb_agg import *
from uniform_instance_gen import uni_instance_gen
from pathlib import Path
from solutions.helper_functions import load_NipsJSPEnv_from_file, load_NipsJSPEnv_from_case
from scheduling_environment.jobShop import JobShop
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))
device = torch.device(configs.device)


def convert_uni_instance_to_lines(generated_data: np.ndarray):
    lines = []
    # line_1_size = '{0}\t{1}\n'.format(configs.n_j, configs.n_m)
    # lines.append(line_1_size)
    process_time = generated_data[0]
    assigned_machine = generated_data[1]

    for i in range(configs.n_j):
        line_job = []
        for j in range(configs.n_m):
            line_job.append(assigned_machine[i, j]-1) # In generated data, machine id is indexed from 1.
            line_job.append(process_time[i, j])
        str_line = " ".join([str(val) for val in line_job])
        lines.append(str_line + '\n')

    return lines

class NipsJSPEnv_test():

    def __init__(self, n_j: int, n_m: int):

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.JobShopModule: JobShop = None


    def reset(self, data):
        lines = convert_uni_instance_to_lines(data)
        self.JobShopModule = load_NipsJSPEnv_from_case(lines, configs.n_j, configs.n_m)

        self.step_count = 0
        self.m = data[-1]  # specify an operation on which machine
        self.dur = data[0].astype(np.single)  # the duration of processing an operation on the assigned machine

        # record action history
        self.partial_sol_sequeence = []
        self.posRewards = 0

        # initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.JSM_adj = self_as_nei + conj_nei_up_stream

        # initialize features
        self.JSM_LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.JSM_max_endTime = self.JSM_LBs.max() if not configs.init_quality_flag else 0
        self.JSM_finished_mark = np.zeros_like(self.m, dtype=np.single)
        self.initQuality = self.JSM_LBs.max() if not configs.init_quality_flag else 0
        fea = np.concatenate((self.JSM_LBs.reshape(-1, 1) / configs.et_normalize_coef,
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.JSM_finished_mark.reshape(-1, 1)), axis=1)
        # initialize feasible omega
        self.JSM_omega = self.first_col.astype(np.int64)
        # initialize mask
        self.JSM_mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        return self.JSM_adj, fea, self.JSM_omega, self.JSM_mask

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect

        ope_to_schedule = self.JobShopModule.get_operation(action)

        if len(ope_to_schedule.scheduling_information) == 0:
            self.partial_sol_sequeence.append(action)
            self.step_count += 1

            assigned_mach = list(ope_to_schedule.processing_times.keys())[0]
            process_time = list(ope_to_schedule.processing_times.values())[0]
            self.JobShopModule.schedule_operation_on_machine_earliest(ope_to_schedule, assigned_mach, process_time)
            job_id = ope_to_schedule.job_id
            ope_idx_in_job = ope_to_schedule.job.operations.index(ope_to_schedule)
            self.JSM_finished_mark[job_id, ope_idx_in_job] = 1

            self.JSM_adj[ope_to_schedule.operation_id] = 0
            self.JSM_adj[ope_to_schedule.operation_id, ope_to_schedule.operation_id] = 1
            if ope_idx_in_job != 0:
                self.JSM_adj[ope_to_schedule.operation_id, ope_to_schedule.operation_id-1] = 1
            machine = self.JobShopModule.get_machine(assigned_mach)
            ope_idx_in_machine = machine.scheduled_operations.index(ope_to_schedule)
            if ope_idx_in_machine > 0:
                prede_ope_id = machine.scheduled_operations[ope_idx_in_machine - 1].operation_id
                self.JSM_adj[ope_to_schedule.operation_id, prede_ope_id] = 1
            if ope_idx_in_machine < len(machine.scheduled_operations) - 1:
                succe_ope_id = machine.scheduled_operations[ope_idx_in_machine + 1].operation_id
                self.JSM_adj[succe_ope_id, ope_to_schedule.operation_id] = 1
                if ope_idx_in_machine > 0:
                    self.JSM_adj[succe_ope_id, prede_ope_id] = 0

            if action not in self.last_col:
                self.JSM_omega[job_id] += 1
            else:
                self.JSM_mask[job_id] = 1

            self.JSM_LBs[job_id, ope_idx_in_job] = ope_to_schedule.scheduling_information.get('end_time')
            for i in range(ope_idx_in_job + 1, len(ope_to_schedule.job.operations)):
                next_ope = ope_to_schedule.job.operations[i]
                pure_process_time = list(next_ope.processing_times.values())[0]
                self.JSM_LBs[job_id, i] = self.JSM_LBs[job_id, i-1] + pure_process_time

        # prepare for return
        feature_JSM = np.concatenate((self.JSM_LBs.reshape(-1, 1) / configs.et_normalize_coef,
                              self.JSM_finished_mark.reshape(-1, 1)), axis=1)

        reward_JSM = - (self.JSM_LBs.max() - self.JSM_max_endTime)

        if reward_JSM == 0:
            reward_JSM = configs.rewardscale
            self.posRewards += reward_JSM

        self.JSM_max_endTime = self.JSM_LBs.max()

        return self.JSM_adj, feature_JSM, reward_JSM, self.done(), self.JSM_omega, self.JSM_mask


if __name__ == '__main__':
    data_generator = uni_instance_gen
    test_data = data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high)

    print(configs.n_j, configs.n_m, configs.low, configs.high)
    print(test_data)

    test_file = '/jsp/adams/abz5'
    JSM_test = load_NipsJSPEnv_from_file(test_file)
    print(len(JSM_test.jobs), len(JSM_test.machines), len(JSM_test.operations))
    print(JSM_test.get_job(1).operations[2].processing_times)

    lines = convert_uni_instance_to_lines(test_data)
    print(lines)
    JSM2 = load_NipsJSPEnv_from_case(lines, configs.n_j, configs.n_m)
    print(JSM2.get_job(14).operations[1].processing_times)
    print(JSM2.get_operation(45).processing_times)
