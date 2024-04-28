from pathlib import Path
import re
from typing import List
from scheduling_environment.machine import Machine
from scheduling_environment.job import Job
from scheduling_environment.operation import Operation
from scheduling_environment.jobShop import JobShop

import torch

def parse_from_tensor(job_durations: torch.Tensor, num_of_stage: int, machine_num_list: List) -> List[JobShop]:
    batch_size = job_durations.shape[0]
    num_of_job = job_durations.shape[1]-1
    num_of_machine = sum(machine_num_list)

    machine_index_list = [0]
    machine_idx = 0
    for i in range(len(machine_num_list)):
        machine_idx += machine_num_list[i]
        machine_index_list.append(machine_idx)

    all_instance_JSM = []

    for i in range(batch_size):
        case = JobShop()
        case.set_nr_of_jobs(num_of_job)
        case.set_nr_of_machines(num_of_machine)
        precedence_relations = {}
        for j in range(num_of_machine):
            case.add_machine((Machine(j)))

        for job_id in range(0, num_of_job):
            job = Job(job_id)
            for op_idx in range(0, num_of_stage):
                operation_id = job_id * num_of_stage + op_idx
                operation = Operation(job, job_id, operation_id)
                optional_machine_id = [idx for idx in range(machine_index_list[op_idx], machine_index_list[op_idx+1])]
                for machine_id in optional_machine_id:
                    duration = job_durations[i, job_id, machine_id].item()
                    operation.add_operation_option(machine_id, duration)
                job.add_operation(operation)
                case.add_operation(operation)
                if op_idx != 0:
                    precedence_relations[operation_id] = [case.get_operation(operation_id-1)]
            case.add_job(job)

        for operation in case.operations:
            if operation.operation_id not in precedence_relations.keys():
                precedence_relations[operation.operation_id] = []
            operation.add_predecessors(precedence_relations[operation.operation_id])

        sequence_dependent_setup_times = [[[0 for r in range(len(case.operations))] for t in range(len(case.operations))] for m in range(num_of_machine)]

        case.add_precedence_relations_operations(precedence_relations)
        case.add_sequence_dependent_setup_times(sequence_dependent_setup_times)

        all_instance_JSM.append(case)

    test_case = all_instance_JSM[0]
    # print(job_durations[0, : , :])
    # for job in test_case.jobs:
    #     print(f'job {job.job_id} has {len(job.operations)} operations:')
    #     for operation in job.operations:
    #         print(f'\t operation {operation.operation_id}, duration: {operation.processing_times}, predecessors: {[op.operation_id for op in operation.predecessors]}')

    return all_instance_JSM
