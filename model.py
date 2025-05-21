import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import jit, random
from typing import Sequence
import optax
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial


# ------------------------------------------------------------------------
# # The MLP based models.

class DiffusionVPx(nn.Module):
  features: Sequence[int]
  mapping_size: int
  num_dimensions: int # Dimensionality of the data vectors. Time not included.
  # sigma: float
  beta_min : float
  beta_max : float
  kappa : float = 1.0
  x_embed : bool = True
  maxL_prefactor : bool = False
  grf_scale_x : float = 10.0
  grf_scale_s : float = 10.0

  @nn.compact
  def __call__(self, x, s):
    B_x = self.grf_scale_x * self.param('B_x', nn.initializers.normal(), (self.mapping_size, self.num_dimensions))
    B_s = self.grf_scale_s * self.param('B_s', nn.initializers.normal(), (self.mapping_size, 1))
    # Stop gradients from flowing through B_x and B_t. [NEW in Redux] 
    B_x = jax.lax.stop_gradient(B_x)
    B_s = jax.lax.stop_gradient(B_s)
    B_x = B_x if self.x_embed else None

    embed = self.input_mapping(s[..., None], B_s) # Convert from [batch_size,] to [batch_size, 1]
    embed = nn.Dense(embed.shape[-1])(embed) # embed.shape[-1] = 2 * (mapping_size)
    embed = nn.sigmoid(embed)
    pos = self.input_mapping(x, B_x)
    pos = nn.Dense(pos.shape[-1])(pos) # This definitely helps improve learned scores.
    pos = nn.sigmoid(pos)
    h = pos

    for feat in self.features[:-1]:
        tau = nn.Dense(feat)(embed)
        h = nn.Dense(feat)(h)
        h += tau
        h = nn.LayerNorm()(h)
        h = nn.sigmoid(h)

    # No time embedded in the last step, following Song's code.
    h = nn.Dense(self.features[-1])(h)

    # Normalize the output.
    # h = h / jnp.expand_dims(self.marginal_prob_std(t), -1) #
    return h

  # Fourier feature mapping
  def input_mapping(self, x, B):
    if B is None:
      return x
    else:
      x_proj = (2.*jnp.pi*x) @ B.T
      return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

  # VP SDE with kappa
  def beta_at(self, s):
    return self.beta_min + (self.beta_max - self.beta_min) * s
  
  def bplus(self, x, s):
    return - self.beta_at(s)[..., None] * x / 2

  def sigma_at(self, s):
    return self.kappa * jnp.sqrt(self.beta_at(s))

  # This is the square root of expintbeta.
  def mu(self, s):
    return jnp.exp(- 0.5 * self.beta_min * s - 0.25 * (self.beta_max - self.beta_min) * (s ** 2))

  def marginal_prob_std(self, s):
    return self.kappa * jnp.sqrt((1 - self.mu(s)**2))
  
  def grad_logp_eq(self, x, s):
    return - x / self.kappa**2

  # The entropy matching loss.
  def loss(self, params, x, key, eps=1e-5, num_steps=1):
    x = jnp.tile(x[:, jnp.newaxis, :], (1, num_steps, 1)) # shape is [batch_size, num_steps, 2]
    key, subkey = random.split(key) # [NEW in Redux]
    random_s = random.uniform(subkey, x.shape[:-1]) * (1. - eps) + eps
    key, subkey = random.split(key)
    z = random.normal(subkey, x.shape)
    std = self.marginal_prob_std(random_s) # shape is [batch_size, num_steps]
    perturbed_x = x * self.mu(random_s)[..., None] + z * std[..., None] # Different for OU
    etheta = self.apply(params, perturbed_x, random_s)
    prefactor = (0.5 * (self.sigma_at(random_s)/std) ** 2) ** int(self.maxL_prefactor) # Maximum likelihood.
    return jnp.mean(prefactor * jnp.sum(((-perturbed_x/self.kappa**2 + etheta) * std[..., None] + z) ** 2, axis=-1))


class DiffusionSL(nn.Module):
  features: Sequence[int]
  mapping_size: int
  num_dimensions: int # Dimensionality of the data vectors. Time not included.
  Sigma_0 : float
  x_embed : bool = True
  maxL_prefactor : bool = False
  grf_scale_x : float = 10.0
  grf_scale_s : float = 10.0

  @nn.compact
  def __call__(self, x, s):
    B_x = self.grf_scale_x * self.param('B_x', nn.initializers.normal(), (self.mapping_size, self.num_dimensions))
    B_s = self.grf_scale_s * self.param('B_s', nn.initializers.normal(), (self.mapping_size, 1))
    # Stop gradients from flowing through B_x and B_t. [NEW in Redux] 
    B_x = jax.lax.stop_gradient(B_x)
    B_s = jax.lax.stop_gradient(B_s)
    B_x = B_x if self.x_embed else None

    embed = self.input_mapping(s[..., None], B_s) # Convert from [batch_size,] to [batch_size, 1]
    embed = nn.Dense(embed.shape[-1])(embed) # embed.shape[-1] = 2 * (mapping_size)
    embed = nn.sigmoid(embed)
    pos = self.input_mapping(x, B_x)
    pos = nn.Dense(pos.shape[-1])(pos) # This definitely helps improve learned scores.
    pos = nn.sigmoid(pos)
    h = pos

    for feat in self.features[:-1]:
        tau = nn.Dense(feat)(embed)
        h = nn.Dense(feat)(h)
        h += tau
        h = nn.LayerNorm()(h)
        h = nn.sigmoid(h)

    # No time embedded in the last step, following Song's code.
    h = nn.Dense(self.features[-1])(h)

    # Normalize the output.
    # h = h / jnp.expand_dims(self.marginal_prob_std(t), -1) #
    return h

  # Fourier feature mapping
  def input_mapping(self, x, B):
    if B is None:
      return x
    else:
      x_proj = (2.*jnp.pi*x) @ B.T
      return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
  
  def bplus(self, x, s):
    return - x / (1 - s[..., None])

  def sigma_at(self, s):
    return self.Sigma_0 * jnp.sqrt(2/(1 - s))
  
  def mu(self, s):
    return 1-s

  def marginal_prob_std(self, s):
    return self.Sigma_0 * jnp.sqrt(1 - (1-s)**2)
  
  def grad_logp_eq(self, x, s):
    return - x / self.Sigma_0**2

  # The entropy matching loss.
  def loss(self, params, x, key, eps=1e-5, num_steps=1):
    x = jnp.tile(x[:, jnp.newaxis, :], (1, num_steps, 1)) # shape is [batch_size, num_steps, 2]
    key, subkey = random.split(key) # [NEW in Redux]
    random_s = random.uniform(subkey, x.shape[:-1]) * (1. - 2.*eps) + eps # Stop shy of T=1.
    key, subkey = random.split(key)
    z = random.normal(subkey, x.shape)
    std = self.marginal_prob_std(random_s) # shape is [batch_size, num_steps]
    perturbed_x = x * self.mu(random_s)[..., None] + z * std[..., None] # Different for OU
    etheta = self.apply(params, perturbed_x, random_s)
    prefactor = (0.5 * (self.sigma_at(random_s)/std) ** 2) ** int(self.maxL_prefactor) # Maximum likelihood.
    return jnp.mean(prefactor * jnp.sum(((-perturbed_x/self.Sigma_0**2 + etheta) * std[..., None] + z) ** 2, axis=-1))
  


# # The image models.
# ------------------------------------------------------------------------


# Base class for the unconditional diffusion models.
class DiffusionImages():
  def __init__(self, neural_core, maxL_prefactor=False, eps0=1e-5, epsT=0):
    self.neural_core = neural_core
    self.maxL_prefactor = maxL_prefactor

    self.eps0 = eps0
    self.epsT = epsT

  def init(self, key, x, s):
    D = jnp.prod(jnp.array(x.shape[-3:]))
    self.S0 = 0.5 * D * jnp.log(2 * jnp.pi * jnp.e * self.marginal_prob_std(1)**2)
    
    return self.neural_core.init(key, x, s)
  
  def apply(self, params, x, s):
    return self.neural_core.apply(params, x, s)

  # The entropy matching loss.
  def loss(self, params, x, key, num_steps=1):
    x = jnp.tile(x[:, jnp.newaxis, :, :, :], (1, num_steps, 1, 1, 1)) # shape is [batch_size, num_steps, 2]
    x = x.reshape(-1, *x.shape[-3:]) # Flatten all but last 3 dimensions, so shape is [batch_size * num_steps w, h, channels]

    key, subkey = random.split(key)
    random_s = random.uniform(subkey, (*x.shape[:-3],), minval=self.eps0, maxval=1.-self.epsT) # shape is [batch_size * num_steps]
    key, subkey = random.split(key)
    z = random.normal(subkey, x.shape)
    std = self.marginal_prob_std(random_s)

    perturbed_x = x * self.mu(random_s)[..., None, None, None] + z * std[..., None, None, None] # Different for OU
    etheta = self.apply(params, perturbed_x, random_s)
    prefactor = (0.5 * (self.sigma_at(random_s)/std) ** 2) ** int(self.maxL_prefactor) # Maximum likelihood.
    return jnp.mean(prefactor * jnp.sum(((self.grad_logp_eq(perturbed_x, random_s) + etheta) * std[..., None, None, None] + z) ** 2, axis=(1,2,3)))


# We inherit from the above base class to implement the different diffusion processses.
class DiffusionImagesVP(DiffusionImages):
  def __init__(self, neural_core, beta_min, beta_max, kappa, maxL_prefactor=False):
    """
    Initializes the diffusion model with a provided neural network module
    and the diffusion process parameters.
    """
    super().__init__(neural_core, maxL_prefactor)
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.kappa = kappa
  
  # VP SDE with kappa
  def beta_at(self, s):
    return self.beta_min + (self.beta_max - self.beta_min) * s
  
  def bplus(self, x, s):
    return - self.beta_at(s)[..., None, None, None] * x / 2

  def sigma_at(self, s):
    return self.kappa * jnp.sqrt(self.beta_at(s))

  # This is the square root of expintbeta.
  def mu(self, s):
    return jnp.exp(- 0.5 * self.beta_min * s - 0.25 * (self.beta_max - self.beta_min) * (s ** 2))

  def marginal_prob_std(self, s):
    return self.kappa * jnp.sqrt((1 - self.mu(s)**2))
  
  def grad_logp_eq(self, x, s):
    return - x / self.kappa**2


class DiffusionImagesSL(DiffusionImages):
  def __init__(self, neural_core, Sigma_0, maxL_prefactor=False):
    """
    Initializes the diffusion model with a provided neural network module
    and the diffusion process parameters.
    """
    # Stop shy of T=1 for SLDM.
    super().__init__(neural_core, maxL_prefactor, eps0=1e-5, epsT=1e-5)
    self.Sigma_0 = Sigma_0

  def bplus(self, x, s):
    return - x / (1 - s[..., None, None, None])

  def sigma_at(self, s):
    return self.Sigma_0 * jnp.sqrt(2/(1 - s))
  
  def mu(self, s):
    return 1-s

  def marginal_prob_std(self, s):
    return self.Sigma_0 * jnp.sqrt(1 - (1-s)**2)
  
  def grad_logp_eq(self, x, s):
    return - x / self.Sigma_0**2
    

#------------------------------------------------------------------------
# The conditional entropy matching model (VP) with class conditioning.
# To be used with the U-net with self-attention.
class DiffusionImagesCondVP():
  def __init__(self, neural_core, classes, prior_prob, beta_min, beta_max):
    """
    Initializes the diffusion model with a provided U-net + SA module
    and the diffusion process parameters.
    """
    self.neural_core = neural_core
    self.classes = classes
    self.prior_prob = prior_prob # p(c)
    self.num_classes = len(prior_prob)

    self.beta_min = beta_min
    self.beta_max = beta_max

  def init(self, key, x, c, s):
    D = jnp.prod(jnp.array(x.shape[-3:]))
    self.S0 = 0.5 * D * jnp.log(2 * jnp.pi * jnp.e)
    
    return self.neural_core.init(key, x, c, s, train=False)
  
  def apply(self, params, x, c, s):
    return self.neural_core.apply(params, x, c, s, train=False)

  # VP SDE
  def beta_at(self, t):
    return self.beta_min + (self.beta_max - self.beta_min) * t
  
  def bplus(self, x, t):
    return - self.beta_at(t)[..., None, None, None] * x / 2

  def sigma_at(self, t):
    return jnp.sqrt(self.beta_at(t))

  # This is the square root of expintbeta.
  def mu(self, t):
    return jnp.exp(- 0.5 * self.beta_min * t - 0.25 * (self.beta_max - self.beta_min) * (t ** 2))

  def marginal_prob_std(self, t):
    return jnp.sqrt((1 - self.mu(t)**2))
  
  def grad_logp_eq(self, x, s):
    return -x
  
  def reverse_drift(self, params, x, c, t):
    return - self.beta_at(t)[..., None, None, None] / 2 * x \
      + self.sigma_at(t)[..., None, None, None] ** 2 * self.apply(params, x, c, t)

  # The OG loss for conditional generation.
  def loss(self, params, x, c, key, eps=1e-5, num_steps=1):
    x = jnp.tile(x[:, jnp.newaxis, :, :, :], (1, num_steps, 1, 1, 1)) # shape is [batch_size, num_steps, w, h, channels]
    x = x.reshape(-1, *x.shape[-3:]) # Flatten all but last 3 dimensions, so shape is [batch_size * num_steps w, h, channels]
    c = jnp.repeat(c, num_steps) # Same as jnp.tile(c[:, jnp.newaxis], (1, num_steps)).reshape(-1)
    
    # TODO: Should we be tiling random_t or using a different random t? Tiling makes sure the steps are evaluated at the same time for all classes.
    key, subkey = jax.random.split(key)
    random_t = random.uniform(subkey, (*x.shape[:-3],), minval=eps, maxval=1.) # shape is [batch_size * num_steps]
    key, subkey = jax.random.split(key)
    z = random.normal(subkey, x.shape)
    Sigma = self.marginal_prob_std(random_t)

    y = x * self.mu(random_t)[..., None, None, None] + z * Sigma[..., None, None, None]
    key, dropout_key = random.split(key)
    etheta = self.neural_core.apply(params, y, c, random_t, train=True, rngs={'dropout': dropout_key}) # Dropout is applied during training.

    # We implement the OG diffusion model loss.
    KL_bound = jnp.sum(((-y + etheta) * Sigma[..., None, None, None] + z) ** 2, axis=(1,2,3)) # Variance dropped OG loss, shape is [batch_size * num_steps]
    return jnp.mean(KL_bound)