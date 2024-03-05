from .operation import Operation
from typing import List


class Job:
    def __init__(self, job_id: int):
        self._job_id: int = job_id
        self._operations: List[Operation] = []
        self._scheduled_operations: List[Operation] = []

    def add_operation(self, operation: Operation):
        """Add an operation to the job."""
        self._operations.append(operation)

    def reset(self):
        self._scheduled_operations = []

    @property
    def nr_of_ops(self) -> int:
        """Return the number of jobs."""
        return len(self._operations)

    @property
    def operations(self) -> List[Operation]:
        """Return the list of operations."""
        return self._operations

    @property
    def job_id(self) -> int:
        """Return the job's id."""
        return self._job_id

    @property
    def scheduled_operations(self) -> List[Operation]:
        return self._scheduled_operations

    @property
    def next_ope_earlist_begin_time(self):
        return max([operation.scheduled_end_time for operation in self.scheduled_operations], default=0)

    def get_operation(self, operation_id):
        """Return operation object with operation id."""
        for operation in self._operations:
            if operation.operation_id == operation_id:
                return operation
