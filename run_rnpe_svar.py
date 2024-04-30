import os
import jax.random as random
import argparse

from rnpe.denoise import spike_and_slab_denoiser, spike_and_slab_denoiser_hyperprior
from rnpe.tasks import SVAR


def run_rnpe_svar(args):
    seed = args.seed
    folder_name = "res/svar/seed_{}/".format(seed)
    is_exists = os.path.exists(folder_name)
    if not is_exists:
        os.makedirs(folder_name)

    denoiser = spike_and_slab_denoiser_hyperprior
    misspecified = False
    n_sim = 10_000
    key, sub_key = random.split(random.PRNGKey(seed))

    svar = SVAR()
    data = svar.generate_dataset(key=sub_key,
                                n=n_sim,
                                misspecified=misspecified)


    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_rnpe_svar.py',
        description='Run inference on SVAR example with RNPE.',
        epilog='Example: python run_rnpe_svar.py'
        )
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    run_rnpe_svar(args)
