# Trains a diffusion model with a simple U-net core. Stops at specified epochs to
# generate samples, compute log densities, and the entropy curves.

import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as lnn
from jax import jit, random
from tqdm.notebook import tqdm
from functools import partial
import argparse
import ast
import itertools
import time
import distrax
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from scipy.integrate import solve_ivp, cumulative_trapezoid
import model as dm
import diffusion_util as du
import train as tu
import util
import unet

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
    parser = argparse.ArgumentParser(description='Runs experiments on a diffusion model with the MNIST dataset.')
    # parser.add_argument('--labels', type=parse_list, required=True, help='List of MNIST class labels to use for training dataset. Must include 0.')
    parser.add_argument('--batch_size', type=parse_list, required=True, help='List of batch sizes, e.g. [4, 16, 32].')
    parser.add_argument('--epochs_list', type=parse_list, required=True, help='List of number of training epochs, e.g. [50, 100, 150, 200]')
    parser.add_argument('--num_steps', type=parse_list, required=True, help='List of number of MC throws per sample during training, e.g. [10, 100, 1000]')
    parser.add_argument('--maxL_prefactor', type=parse_list, required=True, help='List of booleans. True if maximum likelihood prefactor must be used.')
    parser.add_argument('--seed', type=parse_list, required=True, help='List of seed values to initialize NN and to train.')
    parser.add_argument('--num_samples', type=parse_list, default=[10000], help='List of number of training samples per class, e.g. [10, 100, 1000]')
    parser.add_argument('--save_dir', type=str, required=True, help='The path to save weights and likelihoods.')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    return args


#------------------------------------------------------------------------

def deploy(save_dir, epochs_list, exparams):

    maxL_prefactor = exparams['maxL_prefactor'] # Specified when creating the model object.
    del exparams['maxL_prefactor'] # No need to pass this to run_experiment().

    print(f'Running experiment for: {exparams}.')

    # A checkpoint path HAS to be specified so that optimizer state and keys are saved.
    if save_dir:
        base_dir = os.path.abspath(save_dir)
        ml_seed = exparams['seed']
        checkpoint_path = os.path.join(base_dir, f"checkpointEM_seed={ml_seed}.msgpack")

    # The VP/VPx model.
    neuralCoreUnet= unet.NeuralCoreUnet()
    dmodelEM_Img = dm.DiffusionImagesVP(neural_core=neuralCoreUnet,
                                        beta_min=0.1,
                                        beta_max=16,
                                        kappa=1,
                                        maxL_prefactor=maxL_prefactor)
    
    losses_list, samples_gen_list, unsorted_entropy_list, er_neural_list, te_neural_list = \
        run_experiment(dmodelEM_Img, epochs_list, checkpoint_path, **exparams)
    
    for i in range(len(epochs_list)):
        exparams['maxL_prefactor'] = maxL_prefactor
        exparams['epochs'] = epochs_list[i]
        save_experiment(save_dir, 'experimentVP', exparams,
                        losses_list[i], samples_gen_list[i],
                        unsorted_entropy_list[i],
                        er_neural_list[i], te_neural_list[i])
    
    # IMPORTANT: Delete checkpoint to avoid its use between experiments with different parameters.
    os.remove(checkpoint_path)

    del exparams['maxL_prefactor']
    del exparams['epochs']
    # # Delete the epochs key-value pair for the next experiment.
    # # run_experiment expects epochs_list as an argument, not epochs.
    # # The latter is only for save_experiment.

    # Save a different checkpoint for SL so we retain the checkpoints at the end.
    if save_dir:
        base_dir = os.path.abspath(save_dir)
        ml_seed = exparams['seed']
        checkpoint_path = os.path.join(base_dir, f"checkpointSL_seed={ml_seed}.msgpack")

    # The SL model.
    neuralCoreUnet= unet.NeuralCoreUnet()
    dmodelSL_Img = dm.DiffusionImagesSL(neural_core=neuralCoreUnet,
                                        Sigma_0=0.1,
                                        maxL_prefactor=maxL_prefactor)
    
    losses_list, samples_gen_list, unsorted_entropy_list, er_neural_list, te_neural_list = \
        run_experiment(dmodelSL_Img, epochs_list, checkpoint_path, **exparams)
    
    for i in range(len(epochs_list)):
        exparams['maxL_prefactor'] = maxL_prefactor
        exparams['epochs'] = epochs_list[i]
        save_experiment(save_dir, 'experimentSL', exparams,
                        losses_list[i], samples_gen_list[i],
                        unsorted_entropy_list[i],
                        er_neural_list[i], te_neural_list[i])
    
    os.remove(checkpoint_path)

    del exparams['maxL_prefactor']
    del exparams['epochs']


#------------------------------------------------------------------------

def run_experiment(dmodel, epochs_list, checkpoint_path, seed, num_samples, batch_size, num_steps):

    ml_key = random.PRNGKey(seed) # Controls the initialization of the neural network, and the shuffling of training batches.
    mc_samples_key = jax.random.PRNGKey(24)
    mc_eval_key = jax.random.PRNGKey(45)
    gen_key = jax.random.PRNGKey(67)

    shape = dmodel.neural_core.img_shape
    D = jnp.prod(jnp.array(shape))
    mc_batch_size = 1000 # Number of test samples to use for entropy rate and cross entropy calculations.
    num_steps_mc = 500 # Number of time steps for entropy rate.
    num_steps_likelihood = 10000 # Number of samples for log density calculations.
    num_gen = 100 # Number of samples to generate from p_theta.

    base_dataset = MNIST(
                      '../files', 
                      train=True, 
                      transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.permute(1, 2, 0))  # Rearrange to [28, 28, 1]
                        ]), 
                      download=True
                    )

    # Filter the dataset to include only images of the digits specified digits. It MUST include 0 because of how one_hot works.
    labels = np.arange(10) # [0,1,2,3] # Adjust this for faster testing.

    # Select the first num_samples samples of the specified labels.
    limited_indices = []
    for label in labels:
        label_indices = (base_dataset.targets == label).nonzero(as_tuple=True)[0]
        actual_N = min(num_samples, len(label_indices))
        selected = label_indices[:actual_N]
        limited_indices.append(selected)

    indices = torch.cat(limited_indices)
    filtered_dataset = torch.utils.data.Subset(base_dataset, indices)
    train_dataset = util.ImageOnlyDataset(filtered_dataset)

    # Create the MNIST test dataset with the same transform
    test_dataset = MNIST(
                        '../files', 
                        train=False,  # Set to False to load the test dataset
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.permute(1, 2, 0))  # Rearrange to [H, W, ch]
                        ]), 
                        download=True
                        )
    
    indices = torch.where(torch.isin(test_dataset.targets, torch.tensor(labels)))[0]
    filtered_dataset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = DataLoader(filtered_dataset, batch_size=mc_batch_size, shuffle=False)
    test_samples, _ = next(iter(test_loader))
    mc_samples = jnp.asarray(test_samples) # JAX does not handle torch tensors.

    ###### Train the model. ######
    fake_input = jnp.ones((batch_size * num_steps, 28, 28, 1))
    fake_time = jnp.ones(fake_input.shape[0])
    params = dmodel.init(ml_key, fake_input, fake_time)

    # We want to feed train_diffusion the *increments* the epochs after the first
    # epoch. We form an array of these increments, and stick in the first epoch.
    epochs_array = np.sort(epochs_list) # Make sure the epochs are in ascending order.
    diff_epochs = np.diff(epochs_array)
    diff_epochs = np.insert(diff_epochs, 0, epochs_array[0])
    losses_list = []
    losses_all = []
    samples_gen_list = []
    unsorted_entropy_list = []
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

        ###### Generate some samples. ######
        print('Generating samples.')

        p0 = distrax.MultivariateNormalDiag(loc=jnp.zeros(shape), scale_diag=jnp.ones(shape) * dmodel.marginal_prob_std(1))
        samples_0 = p0.sample(seed=gen_key, sample_shape=(num_gen,))
        score_neural = lambda x, s : dmodel.grad_logp_eq(x, s) + dmodel.apply(params, x, s)
        samples_evolution = du.transport_to_data(samples_0, dmodel, score_neural, endtime=0.99999, num_steps=2) # Stop shy of T=1 for SL.
        samples_gen = samples_evolution[-1]


        ###### Compute the log densities for a set of samples. ######
        print('Computing log densities.')

        etheta = lambda x, c, s : dmodel.apply(params, x, s) # We supply a dummy class variable because the density helper expects it. 
        _, lh_subkey = jax.random.split(mc_samples_key)
        density_helper_EM = du.DensityHelperImagesAndClassesEM(lh_subkey, dmodel, D, etheta, epsT=1e-5) # Stop shy of T=1 for SLDM.
        path_ent, prior_ent = density_helper_EM.mc_likelihoodOU(mc_samples, classes=jnp.zeros(1), num_steps=num_steps_likelihood)
        unsorted_entropy = list(zip(path_ent, prior_ent))


        ###### Compute the neural entropy production rate for the model. ######
        print('Computing neural entropy evolution.')

        entropy_prod_rate_neural = lambda x, s : 0.5 * dmodel.sigma_at(s)**2 * jnp.sum(dmodel.apply(params, x, s)**2, axis=(1,2,3))
        start = 1
        entropy_rate_neural, times = du.mc_entropy_rate(mc_eval_key, mc_samples, dmodel, entropy_prod_rate_neural, num_steps=num_steps_mc)
        total_entropy_neural = cumulative_trapezoid(entropy_rate_neural[start:], times[start:])

        samples_gen_list.append(samples_gen)
        unsorted_entropy_list.append(unsorted_entropy)
        entropy_rate_neural_list.append(entropy_rate_neural)
        total_entropy_neural_list.append(total_entropy_neural)

    return losses_list, samples_gen_list, unsorted_entropy_list, entropy_rate_neural_list, total_entropy_neural_list


#------------------------------------------------------------------------

def save_experiment(save_dir, prefix, exparams, losses, samples_gen, unsorted_entropy, er_neural, te_neural):
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
             unsorted_entropy=unsorted_entropy,
             entropy_rate_neural=er_neural,
             total_entropy_neural=te_neural)
    
    # Save generated images.
    samples_gen_fname = os.path.join(experiment_dir, f"samples_gen_neural.npy")
    np.save(samples_gen_fname, np.array(samples_gen))
    
    # We do not save the model weigths because there are too many.


#------------------------------------------------------------------------

def main():
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95' # Decides what percentage of total memory must be allocated.

    args = get_arguments()

    # Convert to list, otherwise the iterator will be exhausted after looping through it once.
    combinations = list(itertools.product(args.maxL_prefactor, args.seed, args.num_samples, args.batch_size, args.num_steps))
    exparam_names = ['maxL_prefactor', 'seed', 'num_samples', 'batch_size', 'num_steps']
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