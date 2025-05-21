import jax
import jax.numpy as jnp
from jax import jit, random
import numpy as np
from scipy.integrate import solve_ivp
import flax.jax_utils as jax_utils
from functools import partial
from tqdm import tqdm
# from tqdm.notebook import tqdm


# ----------------------------------------------------------------------
# A method that generates new samples by solving the probability flow ODE.

def transport_to_data(samples, model, score, endtime=1, num_steps=2):
    t_span = (0, endtime-0.001)
    t_eval = np.linspace(*t_span, num_steps)

    shape = samples.shape

    def derivative(t, x):
        s = endtime - t
        x = x.reshape(shape) # Put it back in the original shape to compute score.
        s_arr = jnp.ones(shape[0]) * s # Score builder expects a time value per sample.
        f = - model.bplus(x, s_arr) + 0.5 * model.sigma_at(s)**2 * score(x, s_arr)
        return f.reshape((-1,))

    x0 = samples.reshape(-1,) # Flatten
    sol = solve_ivp(derivative, t_span, x0, method='RK45', t_eval=t_eval)
    y_arr = np.transpose(sol.y).reshape(num_steps, *shape)

    return y_arr


# ----------------------------------------------------------------------
# A method to compute the entropy production rate.

def mc_entropy_rate(key, x, model, entropy_prod_rate, endtime=1, num_steps=100):
    eps=1e-5
    times = jnp.linspace(eps, endtime, num_steps)
    entropy_prod_rate_arr = []

    # Determine how many trailing singleton dimensions are needed
    extra_dims = x.ndim - 1  # subtract batch dimension
    expand = lambda a : a[(...,) + (None,) * extra_dims]

    for time in (pbar := tqdm(times, desc='time step', leave=True)):
        key, subkey = random.split(key)
        fixed_t = jnp.ones(x.shape[0]) * time
        z = random.normal(subkey, x.shape)
        std = expand(model.marginal_prob_std(fixed_t))  # shape [B, 1, ..., 1]
        mu = expand(model.mu(fixed_t))

        y = x * mu + z * std
        rate = jnp.mean(entropy_prod_rate(y, fixed_t))

        entropy_prod_rate_arr.append(rate)

    return entropy_prod_rate_arr, times


# ----------------------------------------------------------------------
# Classes that help with log density calculations in D-dimensional MLP models.

class DensityHelper():
    def __init__(self, key, model, D, eps0=1e-5, epsT=0):
        self.key = key
        self.sigma = model.sigma_at
        self.bplus = model.bplus
        self.mu = model.mu
        self.marginal_prob_std = model.marginal_prob_std
        self.S0 = (D/2) * jnp.log(2 * jnp.pi * jnp.e * model.marginal_prob_std(1)**2)
        self.eps0 = eps0
        self.epsT = epsT

    def mc_path_entropyOU_fn(self, key, samples, num_steps):
        x = jnp.tile(samples[:, jnp.newaxis, :], (1, num_steps, 1)) # shape is [num_samples, num_steps, D]
        key, subkey = random.split(key)
        random_s = random.uniform(subkey, x.shape[:-1], minval=self.eps0, maxval=1.-self.epsT)
        key, subkey = random.split(key)
        z = random.normal(subkey, x.shape)
        std = self.marginal_prob_std(random_s)
        y = x * self.mu(random_s)[...,None] + z * std[..., None]
        return -jnp.mean(self.path_entropyOU(x, y, random_s), axis=-1) # Average over num_steps

    ###### Function to handle large entropy calculations ###### 
    # # mc_path_entropy_fn allocates too much memory when the number of
    # # samples and/or steps are large. We break down the calculation
    # # into epochs to avoid this.
    def mc_likelihoodOU(self, samples, num_steps):
        # The data in each epoch will have shape [num_samples, steps_per_epoch, D].
        # We choose to have num_samples * steps_per_epoch to be 10^5 to keep memory
        # usage reasonable.
        epochs = samples.shape[0] * num_steps // (10**5)
        steps_per_epoch = num_steps // epochs

        # jit-ing leads to ~10x speed up. The num_steps argument needs to be partialed
        # otherwise jax complains about not knowing it at compile time.
        mc_path_entropy = jit(partial(self.mc_path_entropyOU_fn, num_steps=steps_per_epoch))
        
        # Compute the entropy terms in batches of size steps_per_epoch.
        path_ent = 0
        prior_ent = 0
        num_items = 0
        key = self.key

        for i in (pbar := tqdm(range(epochs), desc='mc iter', leave=True)):
            key, subkey = random.split(key)  
            path_ent += mc_path_entropy(subkey, samples) * steps_per_epoch
            num_items += steps_per_epoch

        path_ent = path_ent / num_items
        prior_ent = jnp.zeros_like(path_ent) - self.S0

        return path_ent, prior_ent

class DensityHelperSM(DensityHelper):
    def __init__(self, key, model, D, stheta, eps0=1e-5, epsT=0):
        self.stheta = stheta
        super().__init__(key, model, D, eps0, epsT)

    # x, y has shape [N, num_dimensions] and s has shape [N,] .
    # sigma, beta, expintbeta, and marginal_prob_std return values with shape [N,]
    # This is why we do [...,None]. N can be a multidimensional shape itself.
    def path_entropyOU(self, x, y, s):
        grad_log_Q = -(y - x * self.mu(s)[...,None]) / (self.marginal_prob_std(s) ** 2)[...,None]
        return 0.5 * self.sigma(s)**2 * jnp.sum(self.stheta(y,s) ** 2, axis=-1) \
            + jnp.sum((self.bplus(y, s) - (self.sigma(s)**2)[...,None] * self.stheta(y,s)) * grad_log_Q, axis=-1)

# A class to help with likelihood calculations for the entropy matching model.
# The only change is the formula for path entropy.
class DensityHelperEM(DensityHelper):
    def __init__(self, key, model, D, etheta, eps0=1e-5, epsT=0):
        self.etheta = etheta
        super().__init__(key, model, D, eps0, epsT)

    def path_entropyOU(self, x, y, s):
        grad_log_Q = -(y - x * self.mu(s)[...,None]) / (self.marginal_prob_std(s) ** 2)[...,None]
        bplus_minus_u_scaled = 2 * self.bplus(y,s) / (self.sigma(s)**2)[...,None] + self.etheta(y,s)
        u = - self.bplus(y,s) - (self.sigma(s)**2)[...,None] * self.etheta(y,s)
        return 0.5 * self.sigma(s)**2 * jnp.sum(bplus_minus_u_scaled ** 2, axis=-1) + jnp.sum(u * grad_log_Q, axis=-1)


# ----------------------------------------------------------------------
# Classes that help with log density calculations in conditional image models.

class DensityHelperImagesAndClasses():
    def __init__(self, key, model, D, eps0=1e-5, epsT=0.):
        self.key = key
        self.sigma = model.sigma_at
        self.bplus = model.bplus
        self.mu = model.mu
        self.marginal_prob_std = model.marginal_prob_std
        self.S0 = (D/2) * jnp.log(2 * jnp.pi * jnp.e * model.marginal_prob_std(1)**2)
        self.eps0 = eps0
        self.epsT = epsT

    # num_steps is number of throws per sample, per class.
    # If a class_embedder is available we make the copies of the image with all class embeddings.
    def mc_path_entropyOU_fn(self, key, samples, classes, num_steps):
        x = jnp.tile(samples[:, jnp.newaxis, jnp.newaxis, :, :, :], (1, classes.shape[0], num_steps, 1, 1, 1)) # shape is [num_samples, num_classes, num_steps, w, h, ch]
        x = x.reshape(-1, *x.shape[-3:]) # shape is [num_samples * num_classes * num_steps, w, h, ch]
        c = jnp.tile(classes[jnp.newaxis, :, jnp.newaxis], (samples.shape[0], 1, num_steps)).reshape(-1) # shape is [num_samples * num_classes * num_steps]
        
        key, subkey = jax.random.split(key)
        random_s = random.uniform(subkey, (samples.shape[0], num_steps), minval=self.eps0, maxval=1.-self.epsT)
        random_s = jnp.tile(random_s[:, jnp.newaxis, :], (1, classes.shape[0], 1))
        random_s = random_s.reshape(-1)
        key, subkey = jax.random.split(key)
        z = random.normal(subkey, (samples.shape[0], num_steps, *x.shape[-3:]))
        z = jnp.tile(z[:, jnp.newaxis, :, :, :, :], (1, classes.shape[0], 1, 1, 1, 1))
        z = z.reshape(-1, *x.shape[-3:])

        std = self.marginal_prob_std(random_s)
        y = x * self.mu(random_s)[..., None, None, None] + z * std[..., None, None, None]
        pe = self.path_entropyOU(x, y, c, random_s) # shape is [num_samples * num_classes * num_steps]
        pe = pe.reshape(samples.shape[0], classes.shape[0], num_steps) # shape is [num_samples, num_classes, num_steps]
        return -jnp.mean(pe, axis=-1) # shape is [num_samples, num_classes]

    ###### Function to handle large entropy calculations ###### 
    # # mc_path_entropy_fn and mc_prior_entropy_fn allocates too much memory
    # # when the number of samples and/or steps are large. We break down the
    # # calculation into epochs was to avoid this. The argument max_throws
    # # should be understood as num_steps_per_epoch_per_device.
    def mc_likelihoodOU(self, samples, classes, num_steps, max_throws=10**3):
        # Each epoch processes data of shape [num_samples, num_classes, steps_per_epoch, w, h, ch].
        # We choose to have num_samples * num_classes * steps_per_epoch to be max_throws * num_devices
        # to keep memory usage reasonable. If more than one device is available, we can think of it as
        # a single device that can handle max_throws * num_devices. For e.g. if we have two devices
        # we only need half the epochs since both devices average over the same steps_per_epoch.
        assert max_throws >= samples.shape[0] * classes.shape[0], f'max_throws must be greater than N * n_c.'
        num_devices = jax.device_count()
        total_throws = samples.shape[0] * classes.shape[0] * num_steps
        epochs = total_throws // (max_throws * num_devices)
        steps_per_epoch_per_device = num_steps // (epochs * num_devices)

        print(f'Total throws = N * n_c * n_s = {samples.shape[0]} * {classes.shape[0]} * {num_steps} = {total_throws}')
        print(f'Epochs = Total throws / (m_t * n_d) = {epochs}')
        print(f'Steps per epoch per device = n_s / (Epochs * n_d) = {steps_per_epoch_per_device}')

        # Compute the entropy terms in batches of size steps_per_epoch.
        path_ent = 0
        prior_ent = 0
        num_items = 0
        key = self.key

        p_mc_path_entropy = jax.pmap(partial(self.mc_path_entropyOU_fn, num_steps=steps_per_epoch_per_device)) # returns shape [num_devices, num_samples, num_classes]

        samples = jax_utils.replicate(samples)
        classes = jax_utils.replicate(classes)

        for i in (pbar := tqdm(range(epochs), desc='mc iter', leave=True)):
            key, subkey = random.split(key)
            keys = jnp.tile(subkey[jnp.newaxis, :], (num_devices, 1)) # Send the same key to both devices.
            path_ent += jnp.sum(p_mc_path_entropy(keys, samples, classes), axis=0) * steps_per_epoch_per_device
            num_items += steps_per_epoch_per_device * num_devices

        path_ent = path_ent / num_items
        prior_ent = jnp.zeros_like(path_ent) - self.S0 # Fixed sign.

        return path_ent, prior_ent


class DensityHelperImagesAndClassesSM(DensityHelperImagesAndClasses):
    def __init__(self, key, model, D, stheta, eps0=1e-5, epsT=0.):
        self.stheta = stheta
        super().__init__(key, model, D, eps0, epsT)

    # x, y has shape [N, num_dimensions] and s has shape [N,] .
    # sigma, beta, expintbeta, and marginal_prob_std return values with shape [N,]
    # This is why we do [...,None]. N can be a multidimensional shape itself.
    def path_entropyOU(self, x, y, c, s):
        grad_log_Q = -(y - x * self.mu(s)[..., None, None, None]) / (self.marginal_prob_std(s) ** 2)[..., None, None, None]
        return 0.5 * self.sigma(s)**2 * jnp.sum(self.stheta(y,c,s) ** 2, axis=(1,2,3)) \
            + jnp.sum((self.bplus(y, s) - (self.sigma(s)**2)[..., None, None, None] * self.stheta(y,c,s)) * grad_log_Q, axis=(1,2,3))

# A class to help with likelihood calculations for the entropy matching model.
# The only change is the formula for path entropy.
class DensityHelperImagesAndClassesEM(DensityHelperImagesAndClasses):
    def __init__(self, key, model, D, etheta, eps0=1e-5, epsT=0.):
        self.etheta = etheta
        super().__init__(key, model, D, eps0, epsT)

    def path_entropyOU(self, x, y, c, s):
        grad_log_Q = -(y - x * self.mu(s)[..., None, None, None]) / (self.marginal_prob_std(s) ** 2)[..., None, None, None]
        bplus_minus_u_scaled = 2 * self.bplus(y,s) / (self.sigma(s)**2)[..., None, None, None] + self.etheta(y,c,s)
        u = - self.bplus(y,s) - (self.sigma(s)**2)[..., None, None, None] * self.etheta(y,c,s)
        return 0.5 * self.sigma(s)**2 * jnp.sum(bplus_minus_u_scaled ** 2, axis=(1,2,3)) + jnp.sum(u * grad_log_Q, axis=(1,2,3))