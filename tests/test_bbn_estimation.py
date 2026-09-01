import jax
import jax.numpy as jnp
import pytest

from examples.bbn_estimation import main as bbn


def _have_cuda() -> bool:
    try:
        from numba import cuda
    except ImportError:
        return False
    return bool(cuda.is_available())


@pytest.mark.skipif(not _have_cuda(), reason="numba.cuda unavailable")
def test_bbn_log_likelihood_vmap_is_finite():
    key = jax.random.key(0)
    positions = jax.random.uniform(key, (8, 2)) * (bbn.HI - bbn.LO) + bbn.LO

    values = jax.jit(jax.vmap(bbn.log_likelihood))(positions)

    assert values.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(values)))
