"""
Defines problem domains

Each domain class must inherit from the Domain base class.
Each domain class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax, pdb
import jax.numpy as jnp
import numpy as np
import scipy.stats
from jax import random
from scipy.spatial.distance import cdist
from scipy.stats import truncnorm
from scipy.stats import qmc

from fbpinns import networks


class Domain:
    """Base domain class to be inherited by different domain classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""

        # below parameters need to be defined
        static_params = {
            "xd":None,# dimensionality of x
            }
        raise NotImplementedError

    @staticmethod
    def sample_interior(all_params, key, sampler, batch_shape):
        """Samples interior of domain.
        Returns x_batch points in interior of domain"""
        raise NotImplementedError

    @staticmethod
    def sample_boundaries(all_params, key, sampler, batch_shapes):
        """Samples boundaries of domain.
        Returns (x_batch, ...) tuple of points for each boundary"""
        raise NotImplementedError

    @staticmethod
    def norm_fn(all_params, x):
        """"Applies norm function, for a SINGLE point with shape (xd,)"""# note only used for PINNs, FBPINN norm function defined in Decomposition
        raise NotImplementedError




# class RectangularDomainND(Domain):
#
#     @staticmethod
#     def init_params(xmin, xmax):
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#
#         static_params = {
#             "xd":xd,
#             "xmin":jnp.array(xmin),
#             "xmax":jnp.array(xmax),
#             }
#         return static_params, {}
#
#     @staticmethod
#     def sample_interior(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_samplerND(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_start1d(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler1NDD(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_boundary1d(all_params, key, sampler, batch_shape, loc):
#
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler1NDDD(key, sampler, xmin, xmax, batch_shape, loc)
#     @staticmethod
#     def sample_start2d(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler2NDD(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_boundary2d(all_params, key, sampler, batch_shape, loc):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler2NDDD(key, sampler, xmin, xmax, batch_shape, loc)
#
#     @staticmethod
#     def sample_boundaries(all_params, key, sampler, batch_shapes):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         xd = all_params["static"]["domain"]["xd"]
#
#         assert len(batch_shapes) == 2*xd# total number of boundaries
#         x_batches = []
#         for i in range(xd):
#             ic = jnp.array(list(range(i))+list(range(i+1,xd)), dtype=int)
#             for j,v in enumerate([xmin[i], xmax[i]]):
#                 batch_shape = batch_shapes[2*i+j]
#                 if len(ic):
#                     xmin_, xmax_ = xmin[ic], xmax[ic]
#                     key, subkey = jax.random.split(key)
#                     x_batch_ = RectangularDomainND._rectangle_samplerND(subkey, sampler, xmin_, xmax_, batch_shape)# (n, xd-1)
#                     x_batch = v*jnp.ones((jnp.prod(jnp.array(batch_shape)),xd), dtype=float)
#                     x_batch = x_batch.at[:,ic].set(x_batch_)
#                 else:
#                     assert len(batch_shape) == 1
#                     x_batch = v*jnp.ones(batch_shape+(1,), dtype=float)
#                 x_batches.append(x_batch)
#         return x_batches
#
#     @staticmethod
#     def norm_fn(all_params, x):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
#         x = networks.norm(mu, sd, x)
#         return x
#
#     @staticmethod
#     def _rectangle_samplerND(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin, xmax, b) for xmin,xmax,b in zip(xmin, xmax, batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)# (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1,-1)), xmax.reshape((1,-1))
#             x_batch = xmin + (xmax - xmin)*s
#
#         return jnp.array(x_batch)
#
#     @staticmethod
#     def _rectangle_sampler1NDD(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin[i], xmax[i], b) if i != 1 else jnp.array([xmin[i]]) for i, b in
#                   enumerate(batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
#
#     @staticmethod
#     def _rectangle_sampler1NDDD(key, sampler, xmin, xmax, batch_shape, loc):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             assert xmin[0] <= loc <= xmax[0], "loc must be within the range defined by xmin[0] and xmax[0]"
#             xs = [
#                 jnp.array([loc]) if i == 0 else  # 对于第一个维度，在loc处取值
#                 jnp.linspace(xmin[i], xmax[i], b)  # 对于其他维度（包括第二个维度），按均匀间隔取样
#                 for i, b in enumerate(batch_shape)
#             ]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
#     @staticmethod
#     def _rectangle_sampler2NDD(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin[i], xmax[i], b) if i != 2 else jnp.array([xmin[i]]) for i, b in
#                   enumerate(batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
#
#     @staticmethod
#     def _rectangle_sampler2NDDD(key, sampler, xmin, xmax, batch_shape, loc):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             assert xmin[0] <= loc <= xmax[0], "loc must be within the range defined by xmin[0] and xmax[0]"
#             xs = [
#                 jnp.array([loc]) if i == 1 else  # 对于第一个维度，在loc处取值
#                 jnp.linspace(xmin[i], xmax[i], b)  # 对于其他维度（包括第二个维度），按均匀间隔取样
#                 for i, b in enumerate(batch_shape)
#             ]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
class RectangularDomainND(Domain):

    @staticmethod
    def init_params(xmin, xmax):

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)

        static_params = {
            "xd":xd,
            "xmin":jnp.array(xmin),
            "xmax":jnp.array(xmax),
            }
        return static_params, {}

    @staticmethod
    def sample_interior(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_samplerND(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def norm_fn(all_params, x):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
        x = networks.norm(mu, sd, x)
        return x

    @staticmethod
    def _rectangle_samplerND(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
            xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
            x_batch = xx.reshape((-1, xd))
        else:
            if sampler == "halton":
                # use scipy as not implemented in jax (!)
                r = scipy.stats.qmc.Halton(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "sobol":
                r = scipy.stats.qmc.Sobol(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "uniform":
                s = jax.random.uniform(key, (np.prod(batch_shape), xd))

            xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
            x_batch = xmin + (xmax - xmin) * s

        return jnp.array(x_batch)

    def sample_interior_cycle(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_samplerND_cycle(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_start2d_cycle(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler2NDD_cycle(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_boundary2d_cycle(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler2NDDD_cycle(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def _rectangle_samplerND_cycle(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
            xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
            x_batch = xx.reshape((-1, xd))
        else:
            if sampler == "halton":
                # use scipy as not implemented in jax (!)
                r = scipy.stats.qmc.Halton(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "sobol":
                r = scipy.stats.qmc.Sobol(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "uniform":
                s = jax.random.uniform(key, (np.prod(batch_shape), xd))

            xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
            x_batch = xmin + (xmax - xmin) * s

        # After generating x_batch
        xmin = xmin[:2]
        xmax = xmax[:2]
        x_batch_xy = x_batch[:, :2]
        x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
        y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
        xy_center = np.array([[x_center, y_center]])
        # Compute the center of the rectangle
        side_lengths = xmax - xmin
        radius = np.min(side_lengths) / 4.0  # Use the shorter side's fifth as radius

        # Filter out points that fall within the circle
        distances = cdist(x_batch_xy, xy_center, metric='euclidean')
        mask = distances > radius  # Points outside the circle


#        x_coords = x_batch_xy[:,0]
#        y_coords = x_batch_xy[:,1]
#        side_length = radius
#        left_boundary = x_center - side_length / 2
#        right_boundary = x_center + side_length / 2
#        top_boundary = y_center + side_length / 2
#        bottom_boundary = y_center - side_length / 2
#        mask = (x_coords >= left_boundary) & (x_coords <= right_boundary) & \
#           (y_coords >= bottom_boundary) & (y_coords <= top_boundary)
#        mask = ~mask
#
#        mask = mask.reshape(-1,1)
        x_filtered = x_batch[mask.all(axis=1)]

        return jnp.array(x_filtered)
    @staticmethod
    def _rectangle_sampler2NDD_cycle(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            xs = [jnp.linspace(xmin[i], xmax[i], b) if i != 2 else jnp.array([xmin[i]]) for i, b in
                  enumerate(batch_shape)]
            xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
            x_batch = xx.reshape((-1, xd))
        else:
            if sampler == "halton":
                # use scipy as not implemented in jax (!)
                r = scipy.stats.qmc.Halton(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "sobol":
                r = scipy.stats.qmc.Sobol(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "uniform":
                s = jax.random.uniform(key, (np.prod(batch_shape), xd))

            xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
            x_batch = xmin + (xmax - xmin) * s

        # After generating x_batch
        xmin = xmin[:2]
        xmax = xmax[:2]
        x_batch_xy = x_batch[:, :2]
        x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
        y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
        xy_center = np.array([[x_center, y_center]])
        # Compute the center of the rectangle
        side_lengths = xmax - xmin
        radius = np.min(side_lengths) / 4.  # Use the shorter side's fifth as radius

        # Filter out points that fall within the circle
        distances = cdist(x_batch_xy, xy_center, metric='euclidean')
        mask = distances > radius  # Points outside the circle

#        x_coords = x_batch_xy[:,0]
#        y_coords = x_batch_xy[:,1]
#        side_length = radius
#        left_boundary = x_center - side_length / 2
#        right_boundary = x_center + side_length / 2
#        top_boundary = y_center + side_length / 2
#        bottom_boundary = y_center - side_length / 2
#        mask = (x_coords >= left_boundary) & (x_coords <= right_boundary) & \
#           (y_coords >= bottom_boundary) & (y_coords <= top_boundary)
#        mask = ~mask
#        mask = mask.reshape(-1,1)

        x_filtered = x_batch[mask.all(axis=1)]
        return jnp.array(x_filtered)

    def _rectangle_sampler2NDDD_cycle(key, sampler, xmin, xmax, batch_shape):
        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
            xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
            x_batch = xx.reshape((-1, xd))
        else:
            if sampler == "halton":
                # use scipy as not implemented in jax (!)
                r = scipy.stats.qmc.Halton(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "sobol":
                r = scipy.stats.qmc.Sobol(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "uniform":
                s = jax.random.uniform(key, (np.prod(batch_shape), xd))

            xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
            x_batch = xmin + (xmax - xmin) * s

        # After generating x_batch
        xmin = xmin[:2]
        xmax = xmax[:2]
        x_batch_xy = x_batch[:, :2]
        x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
        y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
        xy_center = np.array([[x_center, y_center]])
        # Compute the center of the rectangle
        side_lengths = xmax - xmin
        radius = np.min(side_lengths) / 4.0  # Use the shorter side's fifth as radius

        # Filter out points that fall within the circle
#        distances = cdist(x_batch_xy, xy_center, metric='cityblock')
#        mask1 = jnp.abs(distances - radius) <= 0.001  # Points outside the circle

        x_coords = x_batch_xy[:,0]
        y_coords = x_batch_xy[:,1]
        side_length = radius
        left_boundary = x_center - side_length / 2
        right_boundary = x_center + side_length / 2
        top_boundary = y_center + side_length / 2
        bottom_boundary = y_center - side_length / 2
#        mask = ((jnp.abs(x_cod - x_center - 0.5* radius) <= 0.001 or jnp.abs(x_cod - x_center + 0.5* radius) <= 0.001) and y_cod <= y_center + 0.5*radius and y_cod >=y_center -0.5*radius) or \
 #          ((jnp.abs(y_cod - y_center - 0.5* radius) <= 0.001 or jnp.abs(y_cod - y_center + 0.5* radius) <= 0.001) and x_cod <= x_center + 0.5*radius and x_cod >=x_center -0.5*radius)
#        mask = ((jnp.abs(x_coords - left_boundary)<=0.001 | jnp.abs(x_coords - right_boundary)<=0.001) & \
#           (y_coords >= bottom_boundary) & (y_coords <= top_boundary) ) | \
#           ((jnp.abs(y_coords - bottom_boundary)<=0.001 | jnp.abs(y_coords - top_boundary)<=0.001) & \
#           (x_coords >= left_boundary) & (x_coords <= right_boundary) )
#        mask = mask.reshape(-1,1)
                
        
      #  pdb.set_trace()
      #  x_filtered = x_batch[mask.all(axis=1)]
        def generate_square_boundary_points(center, side_length, num_points_per_side=10):
          """
          Generate points on the boundary of a square.

          Args:
          - center: Tuple (x, y) representing the center of the square.
          - side_length: Length of the side of the square.
          - num_points_per_side: Number of points to generate on each side.

          Returns:
          - boundary_points: Array of shape (4 * num_points_per_side, 2) containing the boundary points.
          """
          # Calculate the coordinates of the vertices of the square
          half_side_length = side_length / 2
          vertices = [(center[0] - half_side_length *1 , center[1] - half_side_length),  # Bottom left
                      (center[0] + half_side_length *1, center[1] - half_side_length),  # Bottom right
                      (center[0] + half_side_length *1, center[1] + half_side_length),  # Top right
                      (center[0] - half_side_length *1, center[1] + half_side_length)]  # Top left

          # Generate points along each side of the square
          boundary_points = []
          for i in range(4):
              start_point = vertices[i]
              end_point = vertices[(i + 1) % 4]
              side_points = np.linspace(start_point, end_point, num_points_per_side + 1)
              boundary_points.extend(side_points[:-1].tolist())  # Exclude the last point to avoid duplicates

          return boundary_points

        def generate_circular_boundary_points(center, r, num_points=10):
          cx = center[0]
          cy = center[1]
          theta = np.linspace(0, 2*np.pi, num_points)
          x = cx + r * np.cos(theta)
          y = cy + r * np.sin(theta)
          circle_points = np.column_stack((x, y))
          return circle_points.tolist()
#        pdb.set_trace()
        #pboundary = generate_square_boundary_points([x_center, y_center], side_length, 100)
        pboundary = generate_circular_boundary_points([x_center, y_center], radius, 400)

 #       pdb.set_trace()
        def foo(input_list):
          expanded_list = []
          hw = np.linspace(0, 1, 40)
          for sublist in input_list:
            for i in hw:
              expanded_list.append(sublist + [i])
          return expanded_list
        x_filtered = foo(pboundary)
  #      pdb.set_trace()
        return jnp.array(x_filtered)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(0)


    domain = RectangularDomainND
    sampler = "halton"


    # 1D

    xmin, xmax = jnp.array([-1,]), jnp.array([2,])
    batch_shape = (10,)
    batch_shapes = ((3,),(4,))

    ps_ = domain.init_params(xmin, xmax)
    all_params = {"static":{"domain":ps_[0]}, "trainable":{"domain":ps_[1]}}
    x_batch = domain.sample_interior(all_params, key, sampler, batch_shape)
    x_batches = domain.sample_boundaries(all_params, key, sampler, batch_shapes)

    plt.figure()
    plt.scatter(x_batch, jnp.zeros_like(x_batch))
    for x_batch in x_batches:
        print(x_batch.shape)
        plt.scatter(x_batch, jnp.zeros_like(x_batch))
    plt.show()


    # 2D

    xmin, xmax = jnp.array([0,1]), jnp.array([1,2])
    batch_shape = (10,20)
    batch_shapes = ((3,),(4,),(5,),(6,))

    ps_ = domain.init_params(xmin, xmax)
    all_params = {"static":{"domain":ps_[0]}, "trainable":{"domain":ps_[1]}}
    x_batch = domain.sample_interior(all_params, key, sampler, batch_shape)
    x_batches = domain.sample_boundaries(all_params, key, sampler, batch_shapes)

    plt.figure()
    plt.scatter(x_batch[:,0], x_batch[:,1])
    for x_batch in x_batches:
        print(x_batch.shape)
        plt.scatter(x_batch[:,0], x_batch[:,1])
    plt.show()




