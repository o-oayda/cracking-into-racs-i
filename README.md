# cracking-into-racs-i

## Installation
### Requirements
- `uv` project manager
- Python 3.12
- Unix-like system, Windows is untested

### Set up
Run `uv sync` to create a virtual environment and install the base dependencies.
Alternatively, run with the flag `uv sync --extra cuda` for the optional cuda
backend, which is useful for efficiently generating simulations for
the posterior predictive residuals plot. Otherwise, the numpy backend will be
used; this will get the job done but will take quite a bit longer on the 
`posterior_predictive_residuals` rule.

After this, you must download the model artifacts from the
[Releases](https://github.com/o-oayda/cracking-into-racs-i/releases). Then,
place the zips (do not extract) into the `artifacts/cache` folder.
You will also need to copy the data products into the `data` folder.

## Reproducing the plots
For a very slow workflow or smoke test, run
```
uv run snakemake all --cores 1
```

If you have the cuda deps setup and an NVIDIA GPU, be sure to pass this flag to
snakemake to take advantage of jax GPU-accelerated simulations:
```
uv run snakemake all \
    --config posterior_predictive_backend=jax-cuda \
    posterior_predictive_batch_size=5 \
    --cores 1
```

Otherwise, the `numpy` backend with `catsim`'s `batch_simulate` will be used.
In this case, you can specify a number of threads to use like so:
```
uv run snakemake all \
    --config posterior_predictive_backend=numpy \
    --set-threads posterior_predictive_residuals=8 \
    --cores 1
```
This will speed up the production of the simulation for the residuals plot by
distributing the jobs over many workers via `joblib`.
