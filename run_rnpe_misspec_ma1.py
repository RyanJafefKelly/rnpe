from flowjax.flows import CouplingFlow
from flowjax.flows import BlockNeuralAutoregressiveFlow
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import StandardNormal
from flowjax.train.data_fit import fit_to_data
import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, HMC, MixedHMC, init_to_value
from rnpe.denoise import spike_and_slab_denoiser, spike_and_slab_denoiser_hyperprior
from rnpe.tasks import MisspecifiedMA1
# from rnpe.metrics import calculate_metrics
from time import time
import pickle
import argparse
import os
import matplotlib.pyplot as plt

def rescale_results(res):
    x_mean, x_std = res["scales"]["x_mean"], res["scales"]["x_std"]
    theta_mean, theta_std = res["scales"]["theta_mean"], res["scales"]["theta_std"]

    res["data"]["x"] = res["data"]["x"] * x_std + x_mean
    res["data"]["y"] = res["data"]["y"] * x_std + x_mean
    res["data"]["theta"] = res["data"]["theta"] * theta_std + theta_mean
    res["data"]["theta_true"] = res["data"]["theta_true"] * theta_std + theta_mean
    res["mcmc_samples"]["x"] = res["mcmc_samples"]["x"] * x_std + x_mean

    res["posterior_samples"]["NPE"] = (
        res["posterior_samples"]["NPE"] * theta_std + theta_mean
    )
    res["posterior_samples"]["RNPE"] = (
        res["posterior_samples"]["RNPE"] * theta_std + theta_mean
    )
    return res



def add_spike_and_slab_error(
    key: random.PRNGKey, x: jnp.ndarray, slab_scale: float, spike_scale: float = 0.01
):
    keys = random.split(key, 3)
    misspecified = random.bernoulli(keys[0], shape=x.shape)
    spike = random.normal(keys[2], shape=x.shape) * spike_scale
    slab = random.cauchy(keys[1], shape=x.shape) * slab_scale
    return x + misspecified * slab + (1 - misspecified) * spike


def run_rnpe_misspec_ma1(args):
    print('sss')
    seed = args.seed
    folder_name = "res/misspec_ma1/seed_{}/".format(seed)

    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)

    denoiser = spike_and_slab_denoiser_hyperprior
    misspecified = True
    n_sim = 10_000
    key, sub_key = random.split(random.PRNGKey(seed))

    # get simulated data
    misspec_ma1 = MisspecifiedMA1()
    data = misspec_ma1.generate_dataset(key=sub_key,
                                    n=n_sim,
                                    misspecified=misspecified)

    # Train marginal likelihood flow
    pseudo_true_param = jnp.array([0.0])
    key, flow_key, train_key = random.split(key, 3)
    theta_dims = 1
    summary_dims = 2
    base_dist = StandardNormal((summary_dims,))
    x_flow = BlockNeuralAutoregressiveFlow(flow_key,
                                        base_dist=base_dist,  # TODO?
                                        # cond_dim=summary_dims,
                                        nn_depth=1,
                                        nn_block_dim=summary_dims*8)
    max_epochs = 50
    show_progress = True
    x_flow, x_losses = fit_to_data(
        train_key,
        dist=x_flow,
        x=data["x"],
        # condition=data["x"],
        learning_rate=0.01,
        max_epochs=max_epochs,
        show_progress=show_progress,
    )

    # TODO: being a bit generous with initial ... stop painful initialisation
    init = init_to_value(
        values={"x": data["y"], "misspecified": jnp.ones(2)}
    )

    kernel = MixedHMC(
        HMC(denoiser,
            trajectory_length=1,
            init_strategy=init,
            target_accept_prob=0.95,)
    )

    mcmc_warmup = 20_000
    mcmc_samples = 100_000
    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_warmup,
        num_samples=mcmc_samples,
        progress_bar=show_progress,
    )

    key, mcmc_key = random.split(key)
    slab_scale = 0.25
    model_kwargs = {"y_obs": data["y"],
                    "flow": x_flow,
                    "slab_scale": slab_scale}

    mcmc.run(mcmc_key, **model_kwargs)
    mcmc.print_summary()
    # Carry out posterior inference
    key, flow_key, train_key = random.split(key, 3)
    base_dist = StandardNormal((theta_dims,))
    transformer = RationalQuadraticSpline(knots=10, interval=5)
    posterior_flow = CouplingFlow(key=flow_key,
                                  base_dist=base_dist,
                                  transformer=transformer,
                                  cond_dim=data["x"].shape[1],
                                  flow_layers=5,
                                  nn_width=50)

    posterior_flow, npe_losses = fit_to_data(
        key=train_key,
        dist=posterior_flow,
        x=data["theta"].reshape((-1, 1)),
        condition=data["x"],
        max_epochs=max_epochs,
        learning_rate=0.0005,
        show_progress=show_progress,
    )

    key, subkey = random.split(key)
    noisy_sims = add_spike_and_slab_error(key, data["x"], slab_scale)

    noisy_posterior_flow, npe_losses = fit_to_data(
        key=train_key,
        dist=posterior_flow,
        x=data["theta"].reshape((-1, 1)),
        condition=noisy_sims,
        max_epochs=max_epochs,
        learning_rate=0.0005,
        show_progress=show_progress,
    )

    posterior_samples = 10_000
    denoised = mcmc.get_samples()["x"]
    key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
    denoised_subset = random.permutation(subkey1, denoised)[: posterior_samples]
    robust_npe_samples = posterior_flow.sample(subkey2,
                                               sample_shape=(),
                                               condition=denoised_subset)
    #Todo!!!!! fix below
    mcmc_samples = mcmc.get_samples()['x']
    naive_npe_samples = posterior_flow.sample(
        subkey3,
        sample_shape=(),
        # data["y"],
        condition=mcmc_samples
    )
    noisy_npe_samples = noisy_posterior_flow.sample(
        subkey4,
        sample_shape=(),
        # data["y"],
        condition=mcmc_samples
    )

    results = {
        "data": data,
        "mcmc_samples": mcmc.get_samples(),
        "posterior_samples": {
            "RNPE": robust_npe_samples,
            "NPE": naive_npe_samples,
            "NNPE": noisy_npe_samples,
        },
        "scales": misspec_ma1.scales,
        "losses": {"x": x_losses, "theta|x": npe_losses},
    }
    results = rescale_results(results)

    fname = f"{folder_name}thetas.pkl"

    with open(fname, "wb") as f:
        pickle.dump(results, f)

    rnpe_samples = results['posterior_samples']['RNPE']
    for i in range(1):
        plt.hist(rnpe_samples[:, i].flatten(), bins=50)
        plt.savefig(f"{folder_name}rnpe_samples_{str(i)}.png")
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_rnpe_misspec_ma1.py',
        description='Run inference on misspecified MA(1) example with RNPE.',
        epilog='Example: python run_rnpe_misspec_ma1.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_rnpe_misspec_ma1(args)