import jax.random as random
import jax.numpy as jnp
import jax
from augmax.geometric import GeometricTransformation, LazyCoordinates


class StochasticWarp(GeometricTransformation):
    """
    Applies an elastic image transformation with probability p.
    Modified slightly from the augmax Warp transformation.
    See Warp in augmax and ElasticTransform in pytorch for additional details.
    """

    def __init__(self, p: float = 1.0, strength: int = 5, coarseness: int = 32):
        super().__init__()
        self.strength = strength
        self.coarseness = coarseness
        self.probability = p

    def transform_coordinates(
        self, rng: jnp.ndarray, coordinates: LazyCoordinates, invert=False
    ):
        do_apply = random.bernoulli(rng, self.probability)
        rng, _ = random.split(rng)

        H, W = coordinates.final_shape
        H_, W_ = H // self.coarseness, W // self.coarseness

        coordshift_coarse = do_apply * self.strength * random.normal(rng, [2, H_, W_])
        coordshift = jnp.tensordot(
            coordinates._current_transform[:2, :2], coordshift_coarse, axes=1
        )
        coordshift = jax.image.resize(coordshift, (2, H, W), method="bicubic")
        coordinates.apply_pixelwise_offsets(coordshift)


class RandomTranslate(GeometricTransformation):
    """
    Horizontally and vertically translate an image by an amount sampled from the
    uniform distribution on the interval [0, x_max] (horizontal) and [0, ymax] (vertical).
    Modified slightly from the Translate transformation in the augmax package.
    """

    def __init__(self, x_max, y_max, x_min=0, y_min=0):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def transform_coordinates(
        self, key: jnp.ndarray, coordinates: LazyCoordinates, invert=False
    ):
        dx = random.uniform(key, shape=[1], minval=self.x_min, maxval=self.x_max)[0]
        key, _ = random.split(key)
        dy = -random.uniform(key, shape=[1], minval=self.y_min, maxval=self.y_max)[0]
        transform = jnp.array([[1, 0, -dy], [0, 1, -dx], [0, 0, 1]])
        coordinates.push_transform(transform)
