import os
import re
import numpy as np
import pandas as pd
from IPython.display import display, HTML


def parse_expname(expname):
    pattern = r'^experiment([A-Za-z0-9]*)_' # Define a regex pattern to match 'experiment' followed by any characters until '_'
    exparam_string = re.sub(pattern, '', expname)
    exparam_string = exparam_string.replace('num_', 'n').replace('batch_', 'batch').replace('grf_', 'g').replace('dist_', 'd').replace('maxL_', 'mL')
    param_assignments = exparam_string.split("_")

    params = {}
    for assignment in param_assignments:
        key, value = assignment.split('=')
        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value

    match = re.search(pattern, expname) # Extract 'VE' or 'VP'
    if match:
        params['SDE'] = match.group(1)

    return params


def load_gm_data(dirs):
    """
    Load Gaussian mixture experiment results from a list of directories and return them as a Pandas DataFrame.

    This function is intended for experiments where entropy and log density are known exactly.
    It parses experiment folders named 'experiment*', loads associated `.npz` result files, computes summary
    statistics such as average loss, actual and neural entropy estimates, and optionally the KL divergence.
    
    Args:
        dirs (list of str): List of directory paths containing experiment subfolders.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an experiment, containing parsed hyperparameters 
                      and computed metrics such as loss, entropy, and KL divergence (if available).
    """
    data = []

    for base_dir in dirs:
        for expname in os.listdir(base_dir):
            if expname.startswith('experiment'):
                losses_and_entropies = np.load(os.path.join(base_dir, expname, 'losses_and_entropies.npz'))
                losses = losses_and_entropies['losses']
                sorted_entropy = losses_and_entropies.get('sorted_entropy', None)
                total_entropy_analytic = losses_and_entropies['total_entropy_analytic']
                total_entropy_neural = losses_and_entropies['total_entropy_neural']

                exparams = parse_expname(expname)
                epochs = exparams['epochs']
                exparams['loss'] = np.mean(losses[max(0,epochs-10):epochs]) # Average the last 10 epochs up to the current one.
                exparams['Stot_actual'] = total_entropy_analytic[-2]
                exparams['Stot_neural'] = total_entropy_neural[-2]
                
                if sorted_entropy is not None:
                    # If there are three entries the first is the actual log density and we can compute KL.
                    sorted_lhood = np.stack([sorted_entropy[:,0], np.sum(sorted_entropy[:,1:], axis=-1)], axis=-1)
                    KL = np.mean(sorted_lhood[:,0] - sorted_lhood[:,1])
                    exparams['KL'] = KL
                
                data.append(exparams)

    return pd.DataFrame(data)


def load_entropies(dirs):
    data = []

    for base_dir in dirs:
        for expname in os.listdir(base_dir):
            if expname.startswith('experiment'):
                losses_and_entropies = np.load(os.path.join(base_dir, expname, 'losses_and_entropies.npz'))
                # losses = losses_and_entropies['losses']
                entropy_rate_analytic = losses_and_entropies.get('entropy_rate_analytic', None)
                total_entropy_analytic = losses_and_entropies.get('total_entropy_analytic', None)
                entropy_rate_neural = losses_and_entropies['entropy_rate_neural']
                total_entropy_neural = losses_and_entropies['total_entropy_neural']

                exparams = parse_expname(expname)
                exparams['er_analytic'] = entropy_rate_analytic
                exparams['te_analytic'] = total_entropy_analytic
                exparams['er_neural'] = entropy_rate_neural
                exparams['te_neural'] = total_entropy_neural
                data.append(exparams)

    return pd.DataFrame(data)


def load_images_data(dirs):
    """
    Load image-based experiment results from a list of directories and return them as a Pandas DataFrame.

    This function is designed for experiments on image datasets, where ground-truth log densities are 
    unavailable and entropy must be estimated. It parses experiment folders named 'experiment*', loads 
    `.npz` result files, computes summary statistics such as average loss, neural entropy, and cross-entropy.

    Args:
        dirs (list of str): List of directory paths containing experiment subfolders.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an experiment, containing parsed hyperparameters 
                      and computed metrics such as loss, estimated entropy, and cross-entropy (if available).
    """
    data = []

    for base_dir in dirs:
        for expname in os.listdir(base_dir):
            if expname.startswith('experiment'):
                losses_and_entropies = np.load(os.path.join(base_dir, expname, 'losses_and_entropies.npz'))
                losses = losses_and_entropies['losses']
                unsorted_entropy = losses_and_entropies.get('unsorted_entropy', None) # For image experiments.
                total_entropy_neural = losses_and_entropies.get('total_entropy_neural', None)

                exparams = parse_expname(expname)
                epochs = exparams['epochs']
                exparams['loss'] = np.mean(losses[max(0,epochs-10):epochs]) # Average the last 10 epochs up to the current one.
                exparams['Stot_neural'] = total_entropy_neural[-2]
                
                # Compute the cross entropy since actual log densities are unavailable.
                if unsorted_entropy is not None:
                    CE = -np.mean(np.sum(unsorted_entropy, axis=-1))
                    exparams['CE'] = CE
                
                data.append(exparams)

    return pd.DataFrame(data)


def load_images(dirs, m=10, sortby='epochs='):
    """
    Load first m images from each 'samples_gen_neural.npy' in experiment folders,
    sorted by numeric value following `sortby` string in folder names.

    Args:
        dirs: list of base directories
        m: number of images per experiment
        sortby: string to search for (e.g., 'epochs=')

    Returns:
        images: NumPy array of shape (m * E, 28, 28)
    """
    entries = []

    # Collect matching folders and sort keys
    for base_dir in dirs:
        for expname in os.listdir(base_dir):
            if expname.startswith('experimentEM_seed=1'): # Change
                match = re.search(rf'{re.escape(sortby)}(\d+)', expname)
                if match:
                    sort_value = int(match.group(1))
                    full_path = os.path.join(base_dir, expname, 'samples_gen_neural.npy')
                    if os.path.exists(full_path):
                        entries.append((sort_value, full_path))

    # Sort by extracted number
    entries.sort(key=lambda x: x[0])

    images = []
    for _, path in entries:
        samples = np.load(path)
        # images.append(samples[:m])  # take first m
        images.append(samples[-m:])  # take last m

    if not images:
        raise RuntimeError("No images loaded. Check naming and paths.")

    # Organize images in the way make_grid expects.
    images = np.array(images)               # shape: (E, m, 28, 28, 1)
    images = np.transpose(images, (1, 0, 2, 3, 4))  # shape: (m, E, 28, 28, 1)
    images = images.reshape(-1, 28, 28, 1)          # shape: (m * E, 28, 28, 1)
    return images