# from typing import Callable

import numpy as np

from beetroots.approx_optim.approach_type.abstract_approach_type import ApproachType


class GridSearchApproach(ApproachType):
    def optimization(
        self,
        first_points: list[np.ndarray],
        init_points: int,
        n_iter: int,
    ):
        pass

    def plot_cost_map(self):
        pass
