import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import jit, random
from typing import Sequence, Tuple
from absl import logging
import jax.numpy as jnp
import numpy as np
from tqdm.notebook import tqdm
from typing import Optional, Tuple


#------------------------------------------------------------------------
# The simple U-net and its parts used for MNIST experiments.

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  embed_dim: int
  scale: float = 30.
  @nn.compact
  def __call__(self, x): 
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), 
                 (self.embed_dim // 2, ))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""  
  output_dim: int  
  
  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.output_dim)(x)[:, None, None, :]


# The U-net and the associated parameters. Works for images with shape (28, 28, 1).
class NeuralCoreUnet(nn.Module):
  channels: Tuple[int] = (32, 64, 128, 256)
  mapping_size: int = 256
  grf_scale_t : float = 30.0
  img_shape : tuple = (28, 28, 1)

  @nn.compact
  def __call__(self, x, s): 
    # The swish activation function
    act = nn.swish

    # Obtain the Gaussian random feature embedding for t   
    embed = act(nn.Dense(self.mapping_size)(
        GaussianFourierProjection(embed_dim=self.mapping_size, scale=self.grf_scale_t)(s)))
        
    # Encoding path
    # 'VALID' adds no padding. Same as leaving out the padding argument completely.
    h1 = nn.Conv(self.channels[0], (3, 3), (1, 1), padding='VALID', use_bias=False)(x)   
    ## Incorporate information from t
    h1 += Dense(self.channels[0])(embed)
    ## Group normalization
    h1 = nn.GroupNorm(4)(h1)    
    h1 = act(h1)
    h2 = nn.Conv(self.channels[1], (3, 3), (2, 2), padding='VALID', use_bias=False)(h1)
    h2 += Dense(self.channels[1])(embed)
    h2 = nn.GroupNorm()(h2)        
    h2 = act(h2)
    h3 = nn.Conv(self.channels[2], (3, 3), (2, 2), padding='VALID', use_bias=False)(h2)
    h3 += Dense(self.channels[2])(embed)
    h3 = nn.GroupNorm()(h3)
    h3 = act(h3)
    h4 = nn.Conv(self.channels[3], (3, 3), (2, 2), padding='VALID', use_bias=False)(h3)
    h4 += Dense(self.channels[3])(embed)
    h4 = nn.GroupNorm()(h4)    
    h4 = act(h4)

    # Decoding path
    h = nn.Conv(self.channels[2], (3, 3), (1, 1), padding=((2, 2), (2, 2)),
                  input_dilation=(2, 2), use_bias=False)(h4)    
    ## Skip connection from the encoding path
    h += Dense(self.channels[2])(embed)
    h = nn.GroupNorm()(h)
    h = act(h)
    h = nn.Conv(self.channels[1], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h3], axis=-1)
                  )
    h += Dense(self.channels[1])(embed)
    h = nn.GroupNorm()(h)
    h = act(h)
    h = nn.Conv(self.channels[0], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                  input_dilation=(2, 2), use_bias=False)(
                      jnp.concatenate([h, h2], axis=-1)
                  )    
    h += Dense(self.channels[0])(embed)    
    h = nn.GroupNorm()(h)
    h = act(h)
    h = nn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
        jnp.concatenate([h, h1], axis=-1)
    )

    return h
  

#------------------------------------------------------------------------
# The U-net with self-attention layers used for CIFAR-10.

nonlinearity = nn.swish
Normalize = nn.normalization.GroupNorm


def get_timestep_embedding(timesteps, embedding_dim,
                           max_time=1000., dtype=jnp.float32):
  """Build sinusoidal embeddings (from Fairseq).

  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".

  Args:
    timesteps: jnp.ndarray: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    max_time: float: largest time input
    dtype: data type of the generated embeddings

  Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  timesteps *= (1000. / max_time)

  half_dim = embedding_dim // 2
  emb = np.log(10000) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
  emb = timesteps.astype(dtype)[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


def nearest_neighbor_upsample(x):
  B, H, W, C = x.shape  # pylint: disable=invalid-name
  x = x.reshape(B, H, 1, W, 1, C)
  x = jnp.broadcast_to(x, (B, H, 2, W, 2, C))
  return x.reshape(B, H * 2, W * 2, C)


class ResnetBlock(nn.Module):
  """Convolutional residual block."""

  dropout: float
  out_ch: Optional[int] = None
  resample: Optional[str] = None

  @nn.compact
  def __call__(self, x, *, emb, deterministic):
    B, _, _, C = x.shape  # pylint: disable=invalid-name
    assert emb.shape[0] == B and len(emb.shape) == 2
    out_ch = C if self.out_ch is None else self.out_ch

    h = nonlinearity(Normalize(name='norm1')(x))
    if self.resample is not None:
      updown = lambda z: {
          'up': nearest_neighbor_upsample(z),
          'down': nn.avg_pool(z, (2, 2), (2, 2))
      }[self.resample]
      h = updown(h)
      x = updown(x) # AP: So x and h have the same H and W and we can add it later.
    h = nn.Conv(
        features=out_ch, kernel_size=(3, 3), strides=(1, 1), name='conv1')(h)

    # add in timestep/class embedding
    emb_out = nn.Dense(features=2 * out_ch, name='temb_proj')(
        nonlinearity(emb))[:, None, None, :]
    scale, shift = jnp.split(emb_out, 2, axis=-1)
    h = Normalize(name='norm2')(h) * (1 + scale) + shift
    # rest
    h = nonlinearity(h)
    h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
    h = nn.Conv(
        features=out_ch,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv2')(h)

    if C != out_ch:
      x = nn.Dense(features=out_ch, name='nin_shortcut')(x)

    assert x.shape == h.shape
    logging.info(
        '%s: x=%r emb=%r resample=%r',
        self.name, x.shape, emb.shape, self.resample)
    return x + h

class AttnBlock(nn.Module):
  """Self-attention residual block."""

  num_heads: Optional[int]
  head_dim: Optional[int]

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable

    if self.head_dim is None:
      assert self.num_heads is not None
      assert C % self.num_heads == 0
      num_heads = self.num_heads
      head_dim = C // num_heads
    else:
      assert self.num_heads is None
      assert C % self.head_dim == 0
      head_dim = self.head_dim
      num_heads = C // head_dim

    h = Normalize(name='norm')(x)

    assert h.shape == (B, H, W, C)
    h = h.reshape(B, H * W, C)
    q = nn.DenseGeneral(features=(num_heads, head_dim), name='q')(h)
    k = nn.DenseGeneral(features=(num_heads, head_dim), name='k')(h)
    v = nn.DenseGeneral(features=(num_heads, head_dim), name='v')(h)
    assert q.shape == k.shape == v.shape == (B, H * W, num_heads, head_dim)
    h = nn.dot_product_attention(query=q, key=k, value=v)
    assert h.shape == (B, H * W, num_heads, head_dim)
    h = nn.DenseGeneral(
        features=C,
        axis=(-2, -1),
        kernel_init=nn.initializers.zeros,
        name='proj_out')(h)
    assert h.shape == (B, H * W, C)
    h = h.reshape(B, H, W, C)
    assert h.shape == x.shape
    logging.info(
        '%s: x=%r num_heads=%d head_dim=%d',
        self.name, x.shape, num_heads, head_dim)
    return x + h


class UNetWithSelfAttention(nn.Module):
  """A UNet architecture."""

  num_classes: int
  ch: int
  emb_ch: int
  out_ch: int
  ch_mult: Tuple[int]
  num_res_blocks: int
  attn_resolutions: Tuple[int]
  num_heads: Optional[int]
  dropout: float

  # logsnr_input_type: str
  # logsnr_scale_range: Tuple[float, float] = (-10., 10.)

  resblock_resample: bool = False
  head_dim: Optional[int] = None  # alternative to num_heads

  @nn.compact
  def __call__(self, x, c, s, *, train):
    B, H, W, _ = x.shape  # pylint: disable=invalid-name
    assert H == W
    assert x.dtype in (jnp.float32, jnp.float64)
    # assert logsnr.shape == (B,) and logsnr.dtype in (jnp.float32, jnp.float64)
    assert s.shape == (B,) and s.dtype in (jnp.float32, jnp.float64)
    num_resolutions = len(self.ch_mult)
    ch = self.ch
    emb_ch = self.emb_ch

    # # Timestep embedding
    # if self.logsnr_input_type == 'linear':
    #   logging.info('LogSNR representation: linear')
    #   logsnr_input = (logsnr - self.logsnr_scale_range[0]) / (
    #       self.logsnr_scale_range[1] - self.logsnr_scale_range[0])
    # elif self.logsnr_input_type == 'sigmoid':
    #   logging.info('LogSNR representation: sigmoid')
    #   logsnr_input = nn.sigmoid(logsnr)
    # elif self.logsnr_input_type == 'inv_cos':
    #   logging.info('LogSNR representation: inverse cosine')
    #   logsnr_input = (jnp.arctan(jnp.exp(-0.5 * jnp.clip(logsnr, -20., 20.)))
    #                   / (0.5 * jnp.pi))
    # else:
    #   raise NotImplementedError(self.logsnr_input_type)

    # emb = get_timestep_embedding(logsnr_input, embedding_dim=ch, max_time=1.)
    emb = get_timestep_embedding(s, embedding_dim=ch, max_time=1.)
    emb = nn.Dense(features=emb_ch, name='dense0')(emb)
    emb = nn.Dense(features=emb_ch, name='dense1')(nonlinearity(emb))
    assert emb.shape == (B, emb_ch)

    # Class embedding
    assert self.num_classes >= 1
    if self.num_classes > 1:
      logging.info('conditional: num_classes=%d', self.num_classes)
      assert c.shape == (B,) and c.dtype == jnp.int32
      c_emb = jax.nn.one_hot(c, num_classes=self.num_classes, dtype=x.dtype)
      c_emb = nn.Dense(features=emb_ch, name='class_emb')(c_emb)
      assert c_emb.shape == emb.shape == (B, emb_ch)
      emb += c_emb
    else:
      logging.info('unconditional: num_classes=%d', self.num_classes)
    del c

    # Downsampling
    hs = [nn.Conv(
        features=ch, kernel_size=(3, 3), strides=(1, 1), name='conv_in')(x)]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = ResnetBlock(
            out_ch=ch * self.ch_mult[i_level],
            dropout=self.dropout,
            name=f'down_{i_level}.block_{i_block}')(
                hs[-1], emb=emb, deterministic=not train)
        if h.shape[1] in self.attn_resolutions:
          h = AttnBlock(
              num_heads=self.num_heads,
              head_dim=self.head_dim,
              name=f'down_{i_level}.attn_{i_block}')(h)
        hs.append(h)
      # Downsample
      if i_level != num_resolutions - 1:
        hs.append(self._downsample(
            hs[-1], name=f'down_{i_level}.downsample', emb=emb, train=train))

    # Middle
    h = hs[-1]
    h = ResnetBlock(dropout=self.dropout, name='mid.block_1')(
        h, emb=emb, deterministic=not train)
    h = AttnBlock(
        num_heads=self.num_heads, head_dim=self.head_dim, name='mid.attn_1')(h)
    h = ResnetBlock(dropout=self.dropout, name='mid.block_2')(
        h, emb=emb, deterministic=not train)

    # Upsampling
    for i_level in reversed(range(num_resolutions)):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks + 1):
        h = ResnetBlock(
            out_ch=ch * self.ch_mult[i_level],
            dropout=self.dropout,
            name=f'up_{i_level}.block_{i_block}')(
                jnp.concatenate([h, hs.pop()], axis=-1),
                emb=emb, deterministic=not train)
        if h.shape[1] in self.attn_resolutions:
          h = AttnBlock(
              num_heads=self.num_heads,
              head_dim=self.head_dim,
              name=f'up_{i_level}.attn_{i_block}')(h)
      # Upsample
      if i_level != 0:
        h = self._upsample(
            h, name=f'up_{i_level}.upsample', emb=emb, train=train)
    assert not hs

    # End
    h = nonlinearity(Normalize(name='norm_out')(h))
    h = nn.Conv(
        features=self.out_ch,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)
    assert h.shape == (*x.shape[:3], self.out_ch)
    return h

  def _downsample(self, x, *, name, emb, train):
    B, H, W, C = x.shape  # pylint: disable=invalid-name
    if self.resblock_resample:
      x = ResnetBlock(
          dropout=self.dropout, resample='down', name=name)(
              x, emb=emb, deterministic=not train)
    else:
      x = nn.Conv(features=C, kernel_size=(3, 3), strides=(2, 2), name=name)(x)
    assert x.shape == (B, H // 2, W // 2, C)
    return x

  def _upsample(self, x, *, name, emb, train):
    B, H, W, C = x.shape  # pylint: disable=invalid-name
    if self.resblock_resample:
      x = ResnetBlock(
          dropout=self.dropout, resample='up', name=name)(
              x, emb=emb, deterministic=not train)
    else:
      x = nearest_neighbor_upsample(x)
      x = nn.Conv(features=C, kernel_size=(3, 3), strides=(1, 1), name=name)(x)
    assert x.shape == (B, H * 2, W * 2, C)
    return x