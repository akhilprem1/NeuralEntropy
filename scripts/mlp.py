# Runs experiments with Gaussian mixtures on a diffusion model with an MLP core.
# These experiments are run with large number of training samples and low D. We will also
# compute the evolution over epochs of the entropies and KL.

# WARNING: Do not run multiple instances of this script on the same save_dir simultaneously.
# The intermediate model weights are stored at a checkpoint and can interfere in different
# instances of this program.

import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as lnn
from jax import jit, random
from tqdm.notebook import tqdm
import distrax
from functools import partial
import argparse
import ast
import itertools
import time
import gm
import model as dm
import diffusion_util as du
import train as tu
import util
from scipy.integrate import solve_ivp, cumulative_trapezoid

#------------------------------------------------------------------------

def get_arguments():
    # We use ast.literal_eval to safely evaluate the string as a Python expression,
    # which allows it to convert the string representation of a list into an actual
    # Python list. No spaces between list commas because those are interpreted as
    # separate arguments.
    def parse_list(s):
        try:
            return ast.literal_eval(s)
        except:
            raise argparse.ArgumentTypeError("Must be a valid Python list")
        
    # Create the parser, and add two option-based arguments
    parser = argparse.ArgumentParser(description='This script runs experiments on MLP-based diffusion models with training data from Gaussian mixtures.')
    parser.add_argument('--D', type=parse_list, required=True, help='List of data dimensions, e.g. [2,3,4]')
    parser.add_argument('--num_samples', type=parse_list, required=True, help='List of training data sizes, e.g. [2048, 4096, 8192]')
    parser.add_argument('--batch_size', type=parse_list, required=True, help='List of batch sizes, e.g. [4, 16, 32]')
    parser.add_argument('--epochs_list', type=parse_list, required=True, help='List of number of training epochs, e.g. [50, 100, 150, 200]')
    parser.add_argument('--num_steps', type=parse_list, required=True, help='List of number of MC throws per sample during training, e.g. [10, 100, 1000]')
    parser.add_argument('--maxL_prefactor', type=parse_list, required=True, help='List of booleans. True if maximum likelihood prefactor must be used.')
    parser.add_argument('--seed', type=parse_list, required=True, help='List of seed values to initialize NN and to train.')
    parser.add_argument('--dist_scale', type=parse_list, required=True, help='Scales of the distribution.')
    parser.add_argument('--save_dir', type=str, required=True, help='The path save weights and likelihoods.')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    return args

#------------------------------------------------------------------------

def deploy(save_dir, epochs_list, exparams):

    D = exparams['D']
    maxL_prefactor = exparams['maxL_prefactor'] # Specified when creating the model object.
    del exparams['maxL_prefactor'] # No need to pass this to run_experiment().

    # A checkpoint path HAS to be specified so that optimizer state and keys are saved.
    if save_dir:
        base_dir = os.path.abspath(save_dir)
        checkpoint_path = os.path.join(base_dir, "checkpoint.msgpack")

    print(f'Running experiment for: {exparams}.')


    ###### Perform experiments with the VP process. This is VPx with kappa=1. ######
    dmodelEM = dm.DiffusionVPx(features=[512, 256, D],
                                mapping_size=256,
                                num_dimensions=D,
                                beta_min=0.1,
                                beta_max=16,
                                kappa=1,
                                x_embed=True, # NOTE: We use Fourier features for low D!
                                maxL_prefactor=maxL_prefactor)
    
    losses_list, sorted_entropy_list, er_analytic, te_analytic, er_neural_list, te_neural_list = \
        run_experiment(dmodelEM, epochs_list, checkpoint_path, **exparams)
    
    for i in range(len(epochs_list)):
        exparams['maxL_prefactor'] = maxL_prefactor
        exparams['epochs'] = epochs_list[i]
        save_experiment(save_dir, 'experimentVP', exparams,
                        losses_list[i], sorted_entropy_list[i],
                        er_analytic, te_analytic,
                        er_neural_list[i], te_neural_list[i])
    
    # IMPORTANT: Delete checkpoint to avoid its use between experiments with different parameters.
    os.remove(checkpoint_path)

    del exparams['maxL_prefactor']
    del exparams['epochs']
    # Delete the epochs key-value pair for the next experiment.
    # run_experiment expects epochs_list as an argument, not epochs.
    # The latter is only for save_experiment.


    ###### Perform experiments with the SL process. ######
    dmodelSL = dm.DiffusionSL(features=[512, 256, D],
                                            mapping_size=256,
                                            num_dimensions=D,
                                            Sigma_0=0.1,
                                            x_embed=True,
                                            maxL_prefactor=maxL_prefactor)
    
    losses_list, sorted_entropy_list, er_analytic, te_analytic, er_neural_list, te_neural_list = \
        run_experiment(dmodelSL, epochs_list, checkpoint_path, **exparams)
    
    for i in range(len(epochs_list)):
        exparams['maxL_prefactor'] = maxL_prefactor
        exparams['epochs'] = epochs_list[i]
        save_experiment(save_dir, 'experimentSL', exparams,
                        losses_list[i], sorted_entropy_list[i],
                        er_analytic, te_analytic,
                        er_neural_list[i], te_neural_list[i])
    
    os.remove(checkpoint_path)

    del exparams['maxL_prefactor']
    del exparams['epochs']


    ###### Perform experiments with the VPx process. ######
    dmodelEM = dm.DiffusionVPx(features=[512, 256, D],
                                mapping_size=256,
                                num_dimensions=D,
                                beta_min=0.1,
                                beta_max=16,
                                kappa=0.1, # Changed from 1 to 0.1 for VPx
                                x_embed=True,
                                maxL_prefactor=maxL_prefactor)
    
    losses_list, sorted_entropy_list, er_analytic, te_analytic, er_neural_list, te_neural_list = \
        run_experiment(dmodelEM, epochs_list, checkpoint_path, **exparams)
    
    for i in range(len(epochs_list)):
        exparams['maxL_prefactor'] = maxL_prefactor
        exparams['epochs'] = epochs_list[i]
        save_experiment(save_dir, 'experimentVPx', exparams,
                        losses_list[i], sorted_entropy_list[i],
                        er_analytic, te_analytic,
                        er_neural_list[i], te_neural_list[i])
    
    os.remove(checkpoint_path)

    del exparams['maxL_prefactor']
    del exparams['epochs']


#------------------------------------------------------------------------

def run_experiment(dmodel, epochs_list, checkpoint_path, seed, D, num_samples, batch_size, num_steps, dist_scale):

    ml_key = random.PRNGKey(seed) # Controls the initialization of the neural network, and the shuffling of training batches.
    dist_key = jax.random.PRNGKey(32) # Controls the data distribution.
    mc_samples_key = jax.random.PRNGKey(24) # Controls the samples used for entropy rate and KL calculations.
    mc_eval_key = jax.random.PRNGKey(45) # Controls the randomness of the entropy rate calculation.

    # Parameters
    n_r = 5  # Number of components
    box_size = 4 + dist_scale * 4
    scale = 1 + dist_scale

    # Create the distribution, prior, and score for downstream experiments.
    dist = gm.make_distribution_flexible(dist_key, D, n_r, box_size, scale)
    samples = dist.sample(seed=dist_key, sample_shape=(num_samples,))
    samples = np.asarray(samples).copy() # We use a PyTorch data loader which does not understand JAX arrays.
    train_dataset = util.DataD(samples)

    # Compute the ideal entropy production curve.
    score_analytic = gm.score_builder(dist, dmodel)
    entropy_prod_rate_analytic = lambda x, s : 0.5 * dmodel.sigma_at(s)**2 * jnp.sum((dmodel.grad_logp_eq(x,s) - score_analytic(x, s)) ** 2, axis=-1)
    
    mc_batch_size = 10000
    num_steps_likelihood = 10000
    mc_samples = dist.sample(seed=mc_samples_key, sample_shape=(mc_batch_size,))
    actual_lhood = dist.log_prob(mc_samples)

    print('Computing analytic entropy evolution.')
    entropy_rate_analytic, times = du.mc_entropy_rate(mc_eval_key, mc_samples, dmodel, entropy_prod_rate_analytic)
    total_entropy_analytic = cumulative_trapezoid(entropy_rate_analytic, times)

    ###### Train the model. ######
    params = dmodel.init(ml_key, samples, np.ones(samples.shape[0]))

    # We want to feed train_diffusion the *increments* the epochs after the first
    # epoch. We form an array of these increments, and stick in the first epoch.
    epochs_array = np.sort(epochs_list) # Make sure the epochs are in ascending order.
    diff_epochs = np.diff(epochs_array)
    diff_epochs = np.insert(diff_epochs, 0, epochs_array[0])
    losses_list = []
    losses_all = []
    sorted_entropy_list = []
    entropy_rate_neural_list = []
    total_entropy_neural_list = []

    for epochs in diff_epochs:
        print('Training the model.')
        params, losses = tu.train_diffusion_with_checkpoints(key=ml_key,
                                                            model=dmodel,
                                                            params=params,
                                                            learning_rate=0.0001,
                                                            epochs=epochs,
                                                            train_dataset=train_dataset,
                                                            batch_size=batch_size,
                                                            num_steps=num_steps,
                                                            checkpoint_path=checkpoint_path)
        
        losses_all.extend(jnp.array(losses).tolist()) # Append to the loss history.
        losses_list.append(losses_all)


        ###### Compute the log densities for a set of samples. ######
        print('Computing log densities.')
        etheta = lambda x, s : dmodel.apply(params, x, s)
        _, lh_subkey = jax.random.split(mc_samples_key)
        density_helper_EM = du.DensityHelperEM(lh_subkey, dmodel, D, etheta, epsT=1e-5) # Stop shy of T=1 for SLDM.
        path_ent, prior_ent = density_helper_EM.mc_likelihoodOU(mc_samples, num_steps_likelihood) # Use mc_samples for likelihood.

        # Sort the array by actual likelihood.
        combined_arr = list(zip(actual_lhood, path_ent, prior_ent))
        sorted_entropy = np.array(sorted(combined_arr, key=lambda x : x[0]))
        

        ###### Compute the neural entropy production rate for the model. ######
        entropy_prod_rate_neural = lambda x, s : 0.5 * dmodel.sigma_at(s)**2 * jnp.sum(dmodel.apply(params, x, s)**2, axis=-1)
        
        print('Computing neural entropy evolution.')
        start = 1
        entropy_rate_neural, times = du.mc_entropy_rate(mc_eval_key, mc_samples, dmodel, entropy_prod_rate_neural)
        total_entropy_neural = cumulative_trapezoid(entropy_rate_neural[start:], times[start:])

        sorted_entropy_list.append(sorted_entropy)
        entropy_rate_neural_list.append(entropy_rate_neural)
        total_entropy_neural_list.append(total_entropy_neural)

    return losses_list, sorted_entropy_list, entropy_rate_analytic, total_entropy_analytic, entropy_rate_neural_list, total_entropy_neural_list

#------------------------------------------------------------------------

def save_experiment(save_dir, prefix, exparams, losses, sorted_entropy, er_analytic, te_analytic, er_neural, te_neural):
    if save_dir:
        base_dir = os.path.abspath(save_dir)
    else:
        base_dir = os.getcwd()

    os.makedirs(save_dir, exist_ok=True)
    exparam_str = "_".join([f"{k}={v}" for k, v in exparams.items()])
    experiment_dir = os.path.join(base_dir, f"{prefix}_{exparam_str}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the losses and entropies.
    losses_and_entropies_fname = os.path.join(experiment_dir, f"losses_and_entropies.npz")
    np.savez(losses_and_entropies_fname,
             losses=losses,
             sorted_entropy=sorted_entropy,
             entropy_rate_analytic=er_analytic,
             total_entropy_analytic=te_analytic,
             entropy_rate_neural=er_neural,
             total_entropy_neural=te_neural)
    
    # We do not save the model weigths because there are too many.

#------------------------------------------------------------------------

def main():
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95' # Decides what percentage of total memory must be allocated.

    args = get_arguments()

    # Convert to list, otherwise the iterator will be exhausted after looping through it once.
    combinations = list(itertools.product(args.maxL_prefactor, args.seed, args.D, args.num_samples, args.batch_size, args.num_steps, args.dist_scale))
    exparam_names = ['maxL_prefactor', 'seed', 'D', 'num_samples', 'batch_size', 'num_steps', 'dist_scale']
    # NOTE: exparams are the parameters of each experiments, not to be confused with
    # the neural network params.

    start_time = time.time()

    # Do the experiments for each combination of parameters.
    for combo in combinations:
        exparams = dict(zip(exparam_names, combo))
        deploy(args.save_dir, args.epochs_list, exparams)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert seconds to hours, minutes, seconds and format as as HH:MM:SS.
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds) # Format as HH:MM:SS
    print(f"Script execution time: {time_str}")


if __name__ == "__main__":
    main()