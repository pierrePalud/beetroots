# Theta: MTM - Log(Aux): PMALA
NUMBA_NUM_THREADS=16 python beetroots/simulations/astro/problems/toy_case/toy_case_nn_hierarchical_log-aux.py nn_N10_fixed_angle_hierarchical_log-aux_mtm-pmala.yaml ./data/toycases ./data/models .
# Theta: MTM PMALA - Log(Aux): MTM
NUMBA_NUM_THREADS=16 python beetroots/simulations/astro/problems/toy_case/toy_case_nn_hierarchical_log-aux.py nn_N10_fixed_angle_hierarchical_log-aux_pmalamtm-mtm.yaml ./data/toycases ./data/models .
# Joint
NUMBA_NUM_THREADS=16 python beetroots/simulations/astro/problems/toy_case/toy_case_nn_hierarchical_log-aux.py nn_N10_fixed_angle_hierarchical_log-aux_joint.yaml ./data/toycases ./data/models .


# Hierarchical aux (linear)
NUMBA_NUM_THREADS=24 python src/beetroots/simulations/astro/problems/toy_case/toy_case_nn_hierarchical.py nn_N10_fixed_angle_hierarchical.yaml ./data/toycases ./data/models .
