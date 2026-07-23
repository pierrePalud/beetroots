"""Utilities for loading MCMC checkpoints and restarting simulations."""

import os
from typing import Dict, Optional, Tuple
import h5py


def detect_output_folder(input_path: str) -> bool:
    """Detect if the input path is an output folder from a previous simulation run.
    
    Parameters
    ----------
    input_path : str
        Path to check
        
    Returns
    -------
    bool
        True if the path is a valid output folder structure
    """
    if not os.path.isdir(input_path):
        return False
    
    # Check for required subdirectories
    required_dirs = ['data', 'raw']
    for dir_name in required_dirs:
        if not os.path.isdir(os.path.join(input_path, dir_name)):
            return False
    
    # Check for input YAML file
    yaml_file = os.path.join(input_path, 'data', 'input', 'input_file.yaml')
    if not os.path.isfile(yaml_file):
        return False
    
    return True


def extract_checkpoint_info(output_folder: str) -> Optional[Dict[str, str]]:
    """Extract checkpoint information from output folder structure.
    
    Parameters
    ----------
    output_folder : str
        Path to the output folder
        
    Returns
    -------
    dict or None
        Dictionary containing:
        - 'yaml_file': full path to the YAML file
        - 'var_name': variable name (e.g., 'hierarchical_aux_theta')
        - 'mcmc_run_id': MCMC run number (e.g., 'mcmc_0')
        - 'checkpoint_file': full path to HDF5 checkpoint file
        or None if checkpoint not found
    """
    yaml_file = os.path.join(output_folder, 'data', 'input', 'input_file.yaml')
    
    if not os.path.isfile(yaml_file):
        return None
    
    raw_dir = os.path.join(output_folder, 'raw')
    
    # Find first variable with checkpoint
    for var_name in os.listdir(raw_dir):
        var_path = os.path.join(raw_dir, var_name)
        if not os.path.isdir(var_path):
            continue
        
        for mcmc_id in os.listdir(var_path):
            mcmc_path = os.path.join(var_path, mcmc_id)
            if not os.path.isdir(mcmc_path):
                continue
            
            checkpoint_file = os.path.join(mcmc_path, 'mc_chains.hdf5')
            if os.path.isfile(checkpoint_file):
                return {
                    'yaml_file': yaml_file,
                    'var_name': var_name,
                    'mcmc_run_id': mcmc_id,
                    'checkpoint_file': checkpoint_file,
                }
    
    return None


def load_checkpoint_data(checkpoint_file: str) -> Dict:
    """Load kernel and model_check checkpoint data from HDF5 file.
    
    Parameters
    ----------
    checkpoint_file : str
        Path to the HDF5 checkpoint file
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'kernels': dict of kernel checkpoint states
        - 'model_check': dict of model checking statistics
        - 'last_iteration': last iteration number
    """
    checkpoint_data = {
        'kernels': {},
        'model_check': {},
        'last_iteration': 0,
    }
    
    try:
        with h5py.File(checkpoint_file, 'r') as f:
            # Load kernels checkpoint
            if 'kernels' in f:
                kernels_group = f['kernels']
                for kernel_name in kernels_group.keys():
                    kernel_group = kernels_group[kernel_name]
                    checkpoint_data['kernels'][kernel_name] = {}
                    for param_name in kernel_group.keys():
                        # Convert to Python native types
                        data = kernel_group[param_name][()]
                        checkpoint_data['kernels'][kernel_name][param_name] = data
            
            # Load model_check checkpoint
            if 'model_check' in f:
                model_check_group = f['model_check']
                for key in model_check_group.keys():
                    # Convert to Python native types
                    data = model_check_group[key][()]
                    checkpoint_data['model_check'][key] = data
            
            # Get last iteration from any variable's history
            # We look for the last saved iteration number
            for var_name in ['full_conditional_theta', 'full_conditional_auxiliary']:
                if var_name in f:
                    var_group = f[var_name]
                    if 'list_var' in var_group:
                        list_var = var_group['list_var']
                        checkpoint_data['last_iteration'] = list_var.shape[0]
                        break
    
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_file}: {e}")
    
    return checkpoint_data


def prepare_restart_params(
    output_folder: str,
    original_params: Dict,
) -> Tuple[str, Dict, Optional[Dict]]:
    """Prepare parameters for restarting from a checkpoint.
    
    Parameters
    ----------
    output_folder : str
        Path to the output folder from previous run
    original_params : dict
        Parameters loaded from YAML file
        
    Returns
    -------
    tuple
        - data_path: path to data folder (for load_params)
        - params: updated parameters dict
        - checkpoint_data: loaded checkpoint data or None
    """
    checkpoint_info = extract_checkpoint_info(output_folder)
    if checkpoint_info is None:
        raise ValueError(f"No valid checkpoint found in {output_folder}")
    
    checkpoint_data = load_checkpoint_data(checkpoint_info['checkpoint_file'])
    
    data_path = os.path.join(output_folder, 'data', 'input')
    
    return data_path, original_params, checkpoint_data
