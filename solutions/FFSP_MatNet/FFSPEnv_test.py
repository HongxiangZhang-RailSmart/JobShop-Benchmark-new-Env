import copy
from dataclasses import dataclass
import torch
import itertools  # for permutation list

from solutions.FFSP_MatNet.FFSProblemDef import get_random_problems
from data_parsers.parser_ffsp import parse_from_tensor as convert_data

from typing import List
from scheduling_environment.machine import Machine
from scheduling_environment.job import Job
from scheduling_environment.operation import Operation
from scheduling_environment.jobShop import JobShop

# For Gantt Chart
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class Reset_State:
    problems_list: list
    # len(problems_list) = stage_cnt
    # problems_list[current_stage].shape: (batch, job, machine_cnt_list[current_stage])
    # float type


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    #--------------------------------------
    step_cnt: int = 0
    stage_idx: torch.Tensor = None
    # shape: (batch, pomo)
    stage_machine_idx: torch.Tensor = None
    # shape: (batch, pomo)
    job_ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, job+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class FFSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.stage_cnt = env_params['stage_cnt']
        self.machine_cnt_list = env_params['machine_cnt_list']
        self.total_machine_cnt = sum(self.machine_cnt_list)
        self.job_cnt = env_params['job_cnt']
        self.process_time_params = env_params['process_time_params']
        self.pomo_size = env_params['pomo_size']
        self.sm_indexer = _Stage_N_Machine_Index_Converter(self)
        self.step_count = 0

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems_list = None
        # len(problems_list) = stage_cnt
        # problems_list[current_stage].shape: (batch, job, machine_cnt_list[current_stage])
        self.job_durations = None
        # shape: (batch, job+1, total_machine)
        # last job means NO_JOB ==> duration = 0

        # Dynamic
        ####################################
        self.time_idx = None
        # shape: (batch, pomo)
        self.sub_time_idx = None  # 0 ~ total_machine_cnt-1
        # shape: (batch, pomo)
        self.machine_idx = None  # must update according to sub_time_idx
        # shape: (batch, pomo)

        self.schedule = None
        # shape: (batch, pomo, machine, job+1)
        # records start time of each job at each machine
        self.JSM_machine_wait_step = None
        # shape: (batch, pomo, machine)
        # How many time steps each machine needs to run, before it become available for a new job
        self.JSM_job_location = None
        # shape: (batch, pomo, job+1)
        # index of stage each job can be processed at. if stage_cnt, it means the job is finished (when job_wait_step=0)
        self.JSM_job_wait_step = None
        # shape: (batch, pomo, job+1)
        # how many time steps job needs to wait, before it is completed and ready to start at job_location
        self.JSM_finished = None
        # shape: (batch, pomo)

        # STEP-State
        ####################################
        self.step_state = None

        self.JSM_module_batch_pomo: List[List[JobShop]] = []

    def load_problems(self, batch_size):
        self.batch_size = batch_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        problems_INT_list = get_random_problems(batch_size, self.stage_cnt, self.machine_cnt_list,
                                                self.job_cnt, self.process_time_params)

        problems_list = []
        for stage_num in range(self.stage_cnt):
            stage_problems_INT = problems_INT_list[stage_num]
            stage_problems = stage_problems_INT.clone().type(torch.float)
            problems_list.append(stage_problems)
        self.problems_list = problems_list

        self.job_durations = torch.empty(size=(self.batch_size, self.job_cnt+1, self.total_machine_cnt),
                                         dtype=torch.long)
        # shape: (batch, job+1, total_machine)
        self.job_durations[:, :self.job_cnt, :] = torch.cat(problems_INT_list, dim=2)
        self.job_durations[:, self.job_cnt, :] = 0

        JSM_cases = convert_data(self.job_durations, self.stage_cnt, self.machine_cnt_list)
        self.JSM_module_batch_pomo = []
        for i in range(self.batch_size):
            JSM_pomo = []
            this_JSM_case = JSM_cases[i]
            for j in range(self.pomo_size):
                new_case = copy.deepcopy(this_JSM_case)
                JSM_pomo.append(new_case)
            self.JSM_module_batch_pomo.append(JSM_pomo)

    def load_problems_manual(self, problems_INT_list):
        # problems_INT_list[current_stage].shape: (batch, job, machine_cnt_list[current_stage])

        self.batch_size = problems_INT_list[0].size(0)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        problems_list = []
        for stage_num in range(self.stage_cnt):
            stage_problems_INT = problems_INT_list[stage_num]
            stage_problems = stage_problems_INT.clone().type(torch.float)
            problems_list.append(stage_problems)
        self.problems_list = problems_list

        self.job_durations = torch.empty(size=(self.batch_size, self.job_cnt+1, self.total_machine_cnt),
                                         dtype=torch.long)
        # shape: (batch, job+1, total_machine)
        self.job_durations[:, :self.job_cnt, :] = torch.cat(problems_INT_list, dim=2)
        self.job_durations[:, self.job_cnt, :] = 0

        JSM_cases = convert_data(self.job_durations, self.stage_cnt, self.machine_cnt_list)
        self.JSM_module_batch_pomo = []

        for i in range(self.batch_size):
            JSM_pomo = []
            this_JSM_case = JSM_cases[i]
            for j in range(self.pomo_size):
                new_case = copy.deepcopy(this_JSM_case)
                JSM_pomo.append(new_case)
            self.JSM_module_batch_pomo.append(JSM_pomo)


    def reset(self):
        self.time_idx = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.sub_time_idx = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.machine_idx = self.sm_indexer.get_machine_index(self.POMO_IDX, self.sub_time_idx)
        # shape: (batch, pomo)

        self.schedule = torch.full(size=(self.batch_size, self.pomo_size, self.total_machine_cnt, self.job_cnt+1),
                                   dtype=torch.long, fill_value=-999999)
        # shape: (batch, pomo, machine, job+1)
        self.JSM_machine_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.total_machine_cnt),
                                             dtype=torch.long)
        # shape: (batch, pomo, machine)
        self.JSM_job_location = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        # shape: (batch, pomo, job+1)
        self.JSM_job_wait_step = torch.zeros(size=(self.batch_size, self.pomo_size, self.job_cnt+1), dtype=torch.long)
        # shape: (batch, pomo, job+1)
        self.JSM_finished = torch.full(size=(self.batch_size, self.pomo_size), dtype=torch.bool, fill_value=False)
        # shape: (batch, pomo)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)

        reward = None
        done = None

        for case_pomo_list in self.JSM_module_batch_pomo:
            for each_case in case_pomo_list:
                each_case.reset()

        return Reset_State(self.problems_list), reward, done

    def pre_step(self):
        self._update_step_state()
        self.step_state.step_cnt = 0

        reward = None
        done = False
        return self.step_state, reward, done

    def sub_step(self, job_idx, batch_idx, pomo_idx, current_time, machine_id, dur):
        sim = self.JSM_module_batch_pomo[batch_idx][pomo_idx]

        # for i in range(self.total_machine_cnt):
        #     self.JSM_machine_wait_step[batch_idx, pomo_idx, i] = 0 if sim.get_machine(i).next_available_time - current_time < 0 else sim.get_machine(i).next_available_time - current_time

        if job_idx == self.job_cnt:
            self.JSM_job_location[batch_idx, pomo_idx, job_idx] += 1
            self.JSM_job_wait_step[batch_idx, pomo_idx, job_idx] = 0
            return

        operation_idx_to_schedule = len(sim.get_job(job_idx).scheduled_operations)
        operation = sim.get_job(job_idx).operations[operation_idx_to_schedule]

        sim.schedule_operation_on_machine_at_certain_time(operation, machine_id, operation.processing_times[machine_id], current_time)

        self.JSM_machine_wait_step[batch_idx, pomo_idx, machine_id] = operation.processing_times[machine_id]
        self.JSM_job_location[batch_idx, pomo_idx, job_idx] = operation_idx_to_schedule+1
        self.JSM_job_wait_step[batch_idx, pomo_idx, job_idx] = operation.processing_times[machine_id]
        # if len(sim.scheduled_operations) == len(sim.operations):
        #     self.JSM_finished[batch_idx, pomo_idx] = True



    def step(self, job_idx):
        # job_idx.shape: (batch, pomo)
        job_length2 = self.job_durations[self.BATCH_IDX, job_idx, self.machine_idx]
        for i in range(self.batch_size):
            for j in range(self.pomo_size):
                    action_job_id = job_idx[i, j].item()
                    action_machine_id = self.machine_idx[i, j].item()
                    current_time = int(self.time_idx[i, j].item())
                    self.sub_step(action_job_id,  i, j, current_time, action_machine_id, job_length2[i, j])

        self.schedule[self.BATCH_IDX, self.POMO_IDX, self.machine_idx, job_idx] = self.time_idx
        self.JSM_finished = (self.JSM_job_location[:, :, :self.job_cnt] == self.stage_cnt).all(dim=2)
        done = self.JSM_finished.all()

        if done:
            pass  # do nothing. do not update step_state, because it won't be used anyway
        else:
            self._move_to_next_machine()
            self._update_step_state()

        if done:
            reward = -self._get_makespan()  # Note the MINUS Sign ==> We want to MAXIMIZE reward
        else:
            reward = None

        return self.step_state, reward, done

    def _move_to_next_machine(self):

        b_idx = torch.flatten(self.BATCH_IDX)
        # shape: (batch*pomo,) == (not_ready_cnt,)
        p_idx = torch.flatten(self.POMO_IDX)
        # shape: (batch*pomo,) == (not_ready_cnt,)
        # ready = torch.flatten(self.finished)
        ready = torch.flatten(self.JSM_finished)
        # shape: (batch*pomo,) == (not_ready_cnt,)

        b_idx = b_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)
        p_idx = p_idx[~ready]
        # shape: ( (NEW) not_ready_cnt,)

        while ~ready.all():
            new_sub_time_idx = self.sub_time_idx[b_idx, p_idx] + 1
            # shape: (not_ready_cnt,)
            step_time_required = new_sub_time_idx == self.total_machine_cnt
            # shape: (not_ready_cnt,)
            self.time_idx[b_idx, p_idx] += step_time_required.long()
            new_sub_time_idx[step_time_required] = 0
            self.sub_time_idx[b_idx, p_idx] = new_sub_time_idx
            new_machine_idx = self.sm_indexer.get_machine_index(p_idx, new_sub_time_idx)
            self.machine_idx[b_idx, p_idx] = new_machine_idx


            ####################################
            machine_wait_steps = self.JSM_machine_wait_step[b_idx, p_idx, :]
            # shape: (not_ready_cnt, machine)
            machine_wait_steps[step_time_required, :] -= 1
            machine_wait_steps[machine_wait_steps < 0] = 0
            self.JSM_machine_wait_step[b_idx, p_idx, :] = machine_wait_steps

            job_wait_steps = self.JSM_job_wait_step[b_idx, p_idx, :]
            # shape: (not_ready_cnt, job+1)
            job_wait_steps[step_time_required, :] -= 1
            job_wait_steps[job_wait_steps < 0] = 0
            self.JSM_job_wait_step[b_idx, p_idx, :] = job_wait_steps

            # machine_ready = self.machine_wait_step[b_idx, p_idx, new_machine_idx] == 0
            machine_ready = self.JSM_machine_wait_step[b_idx, p_idx, new_machine_idx] == 0
            # shape: (not_ready_cnt,)

            new_stage_idx = self.sm_indexer.get_stage_index(new_sub_time_idx)
            # shape: (not_ready_cnt,)
            job_ready_1 = (self.JSM_job_location[b_idx, p_idx, :self.job_cnt] == new_stage_idx[:, None])
            # shape: (not_ready_cnt, job)
            job_ready_2 = (self.JSM_job_wait_step[b_idx, p_idx, :self.job_cnt] == 0)
            # shape: (not_ready_cnt, job)
            job_ready = (job_ready_1 & job_ready_2).any(dim=1)
            # shape: (not_ready_cnt,)

            ready = machine_ready & job_ready
            # shape: (not_ready_cnt,)

            b_idx = b_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)
            p_idx = p_idx[~ready]
            # shape: ( (NEW) not_ready_cnt,)


    def _update_step_state(self):
        self.step_state.step_cnt += 1

        self.step_state.stage_idx = self.sm_indexer.get_stage_index(self.sub_time_idx)
        # shape: (batch, pomo)
        self.step_state.stage_machine_idx = self.sm_indexer.get_stage_machine_index(self.POMO_IDX, self.sub_time_idx)
        # shape: (batch, pomo)

        job_loc = self.JSM_job_location[:, :, :self.job_cnt]
        # shape: (batch, pomo, job)
        job_wait_t = self.JSM_job_wait_step[:, :, :self.job_cnt]
        # shape: (batch, pomo, job)

        job_in_stage = job_loc == self.step_state.stage_idx[:, :, None]
        # shape: (batch, pomo, job)
        job_not_waiting = (job_wait_t == 0)
        # shape: (batch, pomo, job)
        job_available = job_in_stage & job_not_waiting
        # shape: (batch, pomo, job)

        job_in_previous_stages = (job_loc < self.step_state.stage_idx[:, :, None]).any(dim=2)
        # shape: (batch, pomo)
        job_waiting_in_stage = (job_in_stage & (job_wait_t > 0)).any(dim=2)
        # shape: (batch, pomo)
        wait_allowed = job_in_previous_stages + job_waiting_in_stage + self.JSM_finished
        # shape: (batch, pomo)

        self.step_state.job_ninf_mask = torch.full(size=(self.batch_size, self.pomo_size, self.job_cnt+1),
                                                   fill_value=float('-inf'))
        # shape: (batch, pomo, job+1)
        job_enable = torch.cat((job_available, wait_allowed[:, :, None]), dim=2)
        # shape: (batch, pomo, job+1)
        self.step_state.job_ninf_mask[job_enable] = 0
        # shape: (batch, pomo, job+1)

        # self.step_state.finished = self.finished
        self.step_state.finished = self.JSM_finished
        # shape: (batch, pomo)

    def _get_makespan(self):

        job_durations_perm = self.job_durations.permute(0, 2, 1)
        # shape: (batch, machine, job+1)
        end_schedule = self.schedule + job_durations_perm[:, None, :, :]
        # shape: (batch, pomo, machine, job+1)

        end_time_max, _ = end_schedule[:, :, :, :self.job_cnt].max(dim=3)
        # shape: (batch, pomo, machine)
        end_time_max, _ = end_time_max.max(dim=2)
        # shape: (batch, pomo)

        return end_time_max

    def draw_Gantt_Chart(self, batch_i, pomo_i):

        job_durations = self.job_durations[batch_i, :, :]
        # shape: (job, machine)
        schedule = self.schedule[batch_i, pomo_i, :, :]
        # shape: (machine, job)

        total_machine_cnt = self.total_machine_cnt
        makespan = self._get_makespan()[batch_i, pomo_i].item()

        # Create figure and axes
        fig,ax = plt.subplots(figsize=(makespan/3, 5))
        cmap = self._get_cmap(self.job_cnt)

        plt.xlim(0, makespan)
        plt.ylim(0, total_machine_cnt)
        ax.invert_yaxis()

        plt.plot([0, makespan], [4, 4], 'black')
        plt.plot([0, makespan], [8, 8], 'black')

        for machine_idx in range(total_machine_cnt):

            duration = job_durations[:, machine_idx]
            # shape: (job)
            machine_schedule = schedule[machine_idx, :]
            # shape: (job)

            for job_idx in range(self.job_cnt):

                job_length = duration[job_idx].item()
                job_start_time = machine_schedule[job_idx].item()
                if job_start_time >= 0:
                    # Create a Rectangle patch
                    rect = patches.Rectangle((job_start_time,machine_idx),job_length,1, facecolor=cmap(job_idx))
                    ax.add_patch(rect)

        ax.grid()
        ax.set_axisbelow(True)
        plt.show()

    def _get_cmap(self, color_cnt):

        colors_list = ['red', 'orange', 'yellow', 'green', 'blue',
                       'purple', 'aqua', 'aquamarine', 'black',
                       'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chocolate',
                       'coral', 'cornflowerblue', 'darkblue', 'darkgoldenrod', 'darkgreen',
                       'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
                       'darkorchid', 'darkred', 'darkslateblue', 'darkslategrey', 'darkturquoise',
                       'darkviolet', 'deeppink', 'deepskyblue', 'dimgrey', 'dodgerblue',
                       'forestgreen', 'gold', 'goldenrod', 'gray', 'greenyellow',
                       'hotpink', 'indianred', 'khaki', 'lawngreen', 'magenta',
                       'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
                       'mediumpurple',
                       'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
                       'navy', 'olive', 'olivedrab', 'orangered',
                       'orchid',
                       'palegreen', 'paleturquoise', 'palevioletred', 'pink', 'plum', 'powderblue',
                       'rebeccapurple',
                       'rosybrown', 'royalblue', 'saddlebrown', 'sandybrown', 'sienna',
                       'silver', 'skyblue', 'slateblue',
                       'springgreen',
                       'steelblue', 'tan', 'teal', 'thistle',
                       'tomato', 'turquoise', 'violet', 'yellowgreen']

        cmap = ListedColormap(colors_list, N=color_cnt)

        return cmap


class _Stage_N_Machine_Index_Converter:
    def __init__(self, env):
        assert env.machine_cnt_list == [4, 4, 4]
        assert env.pomo_size == 24

        machine_SUBindex_0 = torch.tensor(list(itertools.permutations([0, 1, 2, 3])))
        machine_SUBindex_1 = torch.tensor(list(itertools.permutations([0, 1, 2, 3])))
        machine_SUBindex_2 = torch.tensor(list(itertools.permutations([0, 1, 2, 3])))
        self.machine_SUBindex_table = torch.cat((machine_SUBindex_0, machine_SUBindex_1, machine_SUBindex_2), dim=1)
        # machine_SUBindex_table.shape: (pomo, total_machine)


        starting_SUBindex = [0, 4, 8]
        machine_order_0 = machine_SUBindex_0 + starting_SUBindex[0]
        machine_order_1 = machine_SUBindex_1 + starting_SUBindex[1]
        machine_order_2 = machine_SUBindex_2 + starting_SUBindex[2]
        self.machine_table = torch.cat((machine_order_0, machine_order_1, machine_order_2), dim=1)

        self.stage_table = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    def get_stage_index(self, sub_time_idx):
        return self.stage_table[sub_time_idx]

    def get_machine_index(self, POMO_IDX, sub_time_idx):
        # POMO_IDX.shape: (batch, pomo)
        # sub_time_idx.shape: (batch, pomo)
        return self.machine_table[POMO_IDX, sub_time_idx]
        # shape: (batch, pomo)

    def get_stage_machine_index(self, POMO_IDX, sub_time_idx):
        return self.machine_SUBindex_table[POMO_IDX, sub_time_idx]
