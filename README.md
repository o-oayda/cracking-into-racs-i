# cracking-into-racs-i

Run `uv sync --extra cuda` for the optional cuda backend, which is useful for
efficiently generating simulations for the posterior predictive residuals plot.
In this case, be sure to pass this flag to snakemake to take advantage of jax
GPU-accelerated simulations:
```
--config posterior_predictive_backend=jax-cuda \
    posterior_predictive_batch_size=5
```
Otherwise, the `numpy` backend with `catsim`'s `batch_simulate` will be used.
In this case, specify as a number of threads to use with a flag to `snakemake`,
e.g. `--set-threads posterior_predictive_residuals=8` 
