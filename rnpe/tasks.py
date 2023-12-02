from abc import ABC, abstractmethod
import os
import jax.numpy as jnp
import numpy as onp
from numpyro import distributions as dist
from jax import random
import jax
import numba


class Task(ABC):
    @property
    def name(self):
        return type(self).__name__

    @abstractmethod
    def sample_prior(self, key: random.PRNGKey, n: int):
        "Draw n samples from the prior."
        pass

    @abstractmethod
    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray):
        "Carry out simulations."
        pass

    @abstractmethod
    def generate_observation(self, key: random.PRNGKey, misspecified=True):
        "Generate misspecified pseudo-observed data. Returns (theta_true, y)."
        pass

    def generate_dataset(
        self, key: random.PRNGKey, n: int, scale=True, misspecified=True,
    ):
        "Generate optionally scaled dataset with pseudo-observed value and simulations"
        theta_key, x_key, obs_key = random.split(key, 3)
        theta = self.sample_prior(theta_key, n)
        x = self.simulate(x_key, theta)
        nan_idx = remove_nans_and_warn_idx(x)
        x = x[~nan_idx]
        theta = theta[~nan_idx]
        theta_true, y, y_raw = self.generate_observation(
            obs_key, misspecified=misspecified
        )

        if scale:
            theta, theta_true, x, y = self.scale(theta, theta_true, x, y)

        data = {
            "theta": theta,
            "theta_true": theta_true,
            "x": x,
            "y": y,
            "y_raw": y_raw,
        }

        return data

    def scale(self, theta, theta_true, x, y):
        "Center and scale data, using mean and std from theta and x."
        theta_means, theta_stds = theta.mean(axis=0), theta.std(axis=0)
        theta = (theta - theta_means) / theta_stds
        theta_true = (theta_true - theta_means) / theta_stds

        x_means, x_stds = x.mean(axis=0), x.std(axis=0)
        x = (x - x_means) / x_stds
        y = (y - x_means) / x_stds

        self.scales = {
            "theta_mean": theta_means,
            "theta_std": theta_stds,
            "x_mean": x_means,
            "x_std": x_stds,
        }
        return theta, theta_true, x, y

    def in_prior_support(self, theta):
        return jnp.full(theta.shape[0], True)

    def get_true_posterior_samples(self, key: random.PRNGKey, y: jnp.ndarray, n: int):
        raise NotImplementedError(
            "This task does not have a method for sampling the true posterior implemented."
        )


class Gaussian(Task):
    """Task to infer mean of Gaussian using samples, with misspecified std.
    See https://arxiv.org/pdf/1708.01974.pdf."""

    def __init__(
        self,
        x_raw_dim: int = 100,
        prior_var: float = 25,
        likelihood_var: float = 1,
        dgp_var: float = 2,
    ):
        self.x_raw_dim = x_raw_dim
        self.prior_var = prior_var
        self.likelihood_var = likelihood_var
        self.dgp_var = dgp_var
        self.theta_names = [r"$\mu$"]
        self.tractable_posterior = True
        self.x_names = [r"$\mu$", r"$\sigma^2$"]

    def sample_prior(self, key: random.PRNGKey, n: int):
        return random.normal(key, (n, 1)) * jnp.sqrt(self.prior_var)

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray):
        x_demean = random.normal(key, (theta.shape[0], self.x_raw_dim)) * jnp.sqrt(
            self.likelihood_var
        )
        x = x_demean + theta
        x = jnp.column_stack((x.mean(axis=1), x.var(axis=1)))
        return x

    def generate_observation(self, key: random.PRNGKey, misspecified=True):
        theta_key, y_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)
        var = self.dgp_var if misspecified else self.likelihood_var
        y_demean = random.normal(
            y_key, (theta_true.shape[0], self.x_raw_dim)
        ) * jnp.sqrt(var)
        y_raw = y_demean + theta_true
        y = jnp.column_stack((y_raw.mean(axis=1), y_raw.var(axis=1)))
        return theta_true[0, :], y[0, :], y_raw

    def get_true_posterior_samples(self, key: random.PRNGKey, y: jnp.ndarray, n: int):
        "Ensure observation is not scaled."
        mu, std = self._get_true_posterior_mu_std(y[0])
        theta = random.normal(key, (n, 1))*std + mu
        return theta

    def _get_true_posterior_mu_std(self, obs_mean, use_dgp=True):
        "If use_dgp gets posterior under data generating process, otherwise uses the misspecified model."
        l_var = self.dgp_var if use_dgp else self.likelihood_var
        p_var = self.prior_var
        n = self.x_raw_dim
        mu = ((obs_mean * n) / l_var) * ((1 / p_var + n / l_var) ** (-1))
        std = jnp.sqrt((1 / p_var + n / l_var) ** (-1))
        return mu, std.item()


class MisspecifiedMA1(Task):
    """
    Task to infer parameters of a misspecified MA(1) model.
    """

    def __init__(self, n_obs: int = 100, prior_range: tuple = (-1.0, 1.0)):
        self.n_obs = n_obs
        self.prior_range = prior_range
        self.theta_names = [r"$t1$"]  # Assuming t1 is the parameter of interest
        self.tractable_posterior = False  # Assuming the posterior is not analytically tractable
        self.x_names = [r"Autocovariance lag 0", r"Autocovariance lag 1"]

    def sample_prior(self, key: random.PRNGKey, n: int):
        low, high = self.prior_range
        return dist.Uniform(low, high).sample(key=key, sample_shape=(n,))

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray):
        # Here, theta represents the t1 parameter of the MA(1) model
        w = dist.Normal(0, 1).sample(key=key, sample_shape=(theta.shape[0], self.n_obs + 2))
        x = w[:, 2:] + theta[:, None] * w[:, 1:-1]
        s0 = self.autocov(x, lag=0)
        s1 = self.autocov(x, lag=1)
        return jnp.column_stack((s0, s1))

    def autocov(self,
                x: jnp.ndarray,
                lag: int = 1
                ) -> jnp.ndarray:
        """Compute the autocovariance of a given sequence at a specified lag.

        Args:
            x (jnp.ndarray): The sequence for which the autocovariance is to
                            be computed.
            lag (int, optional): The lag at which to compute the autocovariance.
                                Defaults to 1.


        Returns:
            jnp.ndarray: A JAX array containing the autocovariance at the specified lag.
        """
        x = jnp.atleast_2d(x)
        if lag == 0:
            C = jnp.mean(x[:, :] ** 2, axis=1)
        else:
            C = jnp.mean(x[:, lag:] * x[:, :-lag], axis=1)

        return C


    def calculate_summaries(self, x: jnp.ndarray) -> jnp.ndarray:
        """Calculate summary statistics for misspec MA(1) example."""
        s0 = self.autocov(x, lag=0)
        s1 = self.autocov(x, lag=1)
        return jnp.squeeze(jnp.array([s0, s1]))

    def generate_observation(self, key: random.PRNGKey, misspecified=True):
        theta_key, y_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)  # NOTE: ... nonsense
        # Simulate observation from the true model or a misspecified model
        if misspecified:
            y = self.true_dgp(y_key)
            y = self.calculate_summaries(y)
        else:
            y = self.simulate(y_key, theta_true)
        return theta_true, y, None

    def true_dgp(self,
                 key: random.PRNGKey,
                w: float = -0.736,
                rho: float = 0.9,
                sigma_v: float = 0.36,
                batch_size: int = 1,
                n_obs: int = 100
                ) -> jnp.ndarray:
        """Sample from a stochastic volatility model using a normally distributed shock term.

        Args:
            key (PRNGKeyArray): a PRNGKeyArray for reproducibility.
            w (float, optional): Defaults to -0.736.
            rho (float, optional): Defaults to 0.9.
            sigma_v (float, optional): Defaults to 0.36.
            batch_size (int, optional): Defaults to 1.
            n_obs (int, optional): Defaults to 100.

        Returns:
            jnp.ndarray: samples generated by the model.
        """
        h_mat = jnp.zeros((batch_size, n_obs))
        y_mat = jnp.zeros((batch_size, n_obs))

        w_vec = jnp.repeat(w, batch_size)
        rho_vec = jnp.repeat(rho, batch_size)
        sigma_v_vec = jnp.repeat(sigma_v, batch_size)

        key, subkey = random.split(key)
        h_mat = h_mat.at[:, 0].set(w_vec + dist.Normal(0, 1).sample(key=subkey,
                                                                    sample_shape=(batch_size,)) * sigma_v_vec)
        key, subkey = random.split(key)
        y_mat = y_mat.at[:, 0].set(jnp.exp(h_mat[:, 0]/2) * dist.Normal(0, 1).sample(key=subkey,
                                                                                    sample_shape=(batch_size,)))

        for i in range(1, n_obs):
            key, subkey = random.split(key)
            h_mat = h_mat.at[:, i].set(w_vec + rho_vec * h_mat[:, i-1] + dist.Normal(0, 1).sample(key=subkey,
                                                                                                sample_shape=(batch_size,)) * sigma_v_vec)
            key, subkey = random.split(key)
            y_mat = y_mat.at[:, i].set(jnp.exp(h_mat[:, i]/2)*dist.Normal(0, 1).sample(key=subkey,
                                                                                    sample_shape=(batch_size,)))

        return y_mat

class GaussianLinear(Task):
    def __init__(
        self,
        dim: int = 10,
        prior_var: float = 0.1,
        likelihood_var: float = 0.1,
        error_var: float = 0.1,
    ) -> None:
        self.tractable_posterior = True
        self.dim = dim
        self.prior_var = prior_var
        self.likelihood_var = likelihood_var
        self.error_var = error_var
        self.x_names = [fr"x_{i}" for i in range(dim)]
        self.theta_names = [fr"\theta_{i}" for i in range(dim)]

    def sample_prior(self, key: random.PRNGKey, n: int):
        return random.normal(key, (n, self.dim)) * jnp.sqrt(self.prior_var)

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray):
        return random.normal(key, theta.shape) * jnp.sqrt(self.likelihood_var) + theta

    def generate_observation(self, key: random.PRNGKey, misspecified=True):
        var = (
            self.likelihood_var + self.error_var
            if misspecified
            else self.likelihood_var
        )
        theta_key, y_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)
        y = random.normal(y_key, theta_true.shape) * jnp.sqrt(var) + theta_true
        return theta_true.reshape(-1), y.reshape(-1), y.reshape(-1) # No summary stats here so raw==summarised

    def _get_true_posterior_mu_std(self, y, use_dgp=True):
        "Obs either 1d vector or 2d for multiple i.i.d samples."
        l_var = (
            self.likelihood_var + self.error_var if use_dgp else self.likelihood_var
        )
        var = (self.prior_var*l_var)/(self.prior_var + l_var)
        mu = var*(1/l_var)*y
        return mu, jnp.full(self.dim, jnp.sqrt(var))

    def get_true_posterior_samples(self, key: random.PRNGKey, y: jnp.ndarray, n: int):
        "Ensure observation is not scaled."
        mu, std = self._get_true_posterior_mu_std(y)
        theta = random.normal(key, (n, self.dim))*std + mu
        return theta


class SIR(Task):
    """Prior is uniform [0, 0.5] with constraint such that beta > gamma. Note
    this example requires julia, and may take a minute or two to compile."""

    def __init__(self, julia_env_path=".", misspecify_multiplier=0.95):
        self.julia_env_path = julia_env_path
        self.misspecify_multiplier = misspecify_multiplier
        self.x_names = [
            "Mean",  # Mean infections
            "Median",  # Median infections
            "Max",  # Max infections
            "Max Day",  # Day of max infections
            "Half Day",  # Day where half of cumulative total number of infections reached
            "Autocor",  # Autocorrelation lag 1
        ]
        self.theta_names = [r"$\beta$", r"$\gamma$"]
        self.tractable_posterior = False

    def sample_prior(self, key: random.PRNGKey, n: int):
        u1key, u2key = random.split(key)
        x = jnp.sqrt(random.uniform(u1key, (n,))) / 2
        y = random.uniform(u2key, (n,)) * x
        return jnp.column_stack((x, y))

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray, summarise=True):
        "Simulates 365 days of sir model."
        key = key[1].item()
        # temporarily save parameters to file to be accessed from julia
        jl_script_f = os.path.dirname(os.path.realpath(__file__)) + "/julia/SIR.jl"
        tmp_theta_f = f"_temp_theta_{key}.npz"
        tmp_x_f = f"_temp_x_{key}.npz"
        jnp.savez(tmp_theta_f, theta=theta)

        try:
            # Run julia simulation script, outputing to file
            command = f"""
            julia --project={self.julia_env_path} {jl_script_f} --seed={key}  \
                --theta_path={tmp_theta_f} --output_path={tmp_x_f}
            """
            os.system(command)
            x = jnp.load(tmp_x_f)["x"]
        finally:
            for f in [tmp_theta_f, tmp_x_f]:
                try:
                    os.remove(f)
                except OSError:
                    pass

        if summarise:
            x = self.summarise(x)
        return x

    def summarise(self, x):
        @jax.jit
        @jax.vmap
        def autocorr_lag1(x):
            x1 = x[:-1]
            x2 = x[1:]
            x1_dif = x1 - x1.mean()
            x2_dif = x2 - x2.mean()
            numerator = (x1_dif * x2_dif).sum()
            denominator = jnp.sqrt((x1_dif ** 2).sum() * (x2_dif ** 2).sum())
            return numerator / denominator

        def cumulative_day(x, q):
            "Day when q proportion of total infections was reached."
            prop_i = (jnp.cumsum(x, axis=1).T / jnp.sum(x, axis=1)).T
            return jnp.argmax(prop_i > q, axis=1)

        summaries = [
            jnp.log(x.mean(axis=1)),
            jnp.log(jnp.median(x, axis=1)),
            jnp.log(jnp.max(x, axis=1)),
            jnp.log(jnp.argmax(x, axis=1) + 1),  # +1 incase 0 is max_day
            jnp.log(cumulative_day(x, 0.5)),
            autocorr_lag1(x),
        ]

        summaries = jnp.column_stack(summaries)
        return summaries

    def generate_observation(self, key: random.PRNGKey, misspecified=True):
        theta_key, y_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)
        theta_true = jnp.array([[0.15, 0.1]])
        y_raw = self.simulate(y_key, theta_true, summarise=False)
        y_raw = self.misspecify(y_raw) if misspecified else y_raw
        y = self.summarise(y_raw)
        return theta_true[0, :], y[0, :], y_raw

    def misspecify(self, x):
        x = onp.array(x)
        x = x.copy()
        sat_idx, sun_idx, mon_idx = [range(i, 365, 7) for i in range(1, 4)]
        sat_new = x[:, sat_idx] * self.misspecify_multiplier
        sun_new = x[:, sun_idx] * self.misspecify_multiplier
        missed_cases = (x[:, sat_idx] - sat_new) + (x[:, sun_idx] - sun_new)
        mon_new = x[:, mon_idx] + missed_cases
        for idx, new in zip([sat_idx, sun_idx, mon_idx], [sat_new, sun_new, mon_new]):
            x[:, idx] = new
        return jnp.array(x)

    def in_prior_support(self, theta):
        a = jnp.all(theta > 0, axis=1) & jnp.all(theta < 0.5, axis=1)
        b = theta[:, 0] > theta[:, 1]
        return a & b


class CS(Task):
    def __init__(self):
        self.theta_names = [r"$\lambda_c$", r"$\lambda_p$", r"$\lambda_d$"]
        self.x_names = [
            "N Stromal",
            "N Cancer",
            "Mean Min Dist",
            "Max Min Dist",
        ]
        self.cell_rate_lims = {"minval": 200, "maxval": 1500}
        self.parent_rate_lims = {"minval": 3, "maxval": 20}
        self.daughter_rate_lims = {"minval": 10, "maxval": 20}
        self.tractable_posterior = False

    def simulate(
        self,
        key: random.PRNGKey,
        theta: jnp.array,
        summarise: bool = True,
        necrosis: bool = False,
    ):
        theta = onp.array(theta)
        onp.random.seed(key[1].item())

        x = []
        for row in theta:
            xi = self._simulate(
                cell_rate=row[0],
                cancer_parent_rate=row[1],
                cancer_daughter_rate=row[2],
                necrosis=necrosis,
            )
            if summarise is True:
                xi = self.summarise(*xi)
            x.append(xi)

        if summarise:
            x = jnp.row_stack(x)

        return x

    def _simulate(
        self,
        cell_rate: float = 1000,
        cancer_parent_rate: float = 5,
        cancer_daughter_rate: float = 30,
        necrosis=False,
    ):
        num_cells = onp.random.poisson(cell_rate)
        cells = onp.random.uniform(size=(num_cells, 2))

        num_cancer_parents = onp.random.poisson(cancer_parent_rate) + 1
        cancer_parents = onp.random.uniform(0, 1, size=(num_cancer_parents, 2))

        num_cancer_daughters = (
            onp.random.poisson(cancer_daughter_rate, (num_cancer_parents,)) + 1
        )

        dists = dists_between(cells, cancer_parents)  # n_cells by n_parents
        radii = self.n_nearest_dists(dists, num_cancer_daughters)
        is_cancer = (dists <= radii).any(axis=1)

        if necrosis:
            has_necrosis = onp.random.binomial(1, p=0.75, size=(num_cancer_parents,))
            has_necrosis = has_necrosis.astype(bool)
            if has_necrosis.sum() > 0:
                bl_array = dists[:, has_necrosis] < (radii[has_necrosis] * 0.8)
                necrotized = onp.any(bl_array, axis=1)
                cells, is_cancer = (
                    cells[~necrotized],
                    is_cancer[~necrotized],
                )

        return cells, is_cancer

    def summarise(self, cells, is_cancer, threshold_n_stromal: int = 50):
        """Calculate summary statistics, threshold_n_stromal limits the number
        of stromal cells (trade off for efficiency)."""
        num_cancer = is_cancer.sum()

        if num_cancer == is_cancer.shape[0]:
            print("Warning, no stromal cells. Returning nan for summary statistics.")
            return jnp.full(len(self.x_names), jnp.nan)

        num_stromal = (~is_cancer).sum()
        threshold_num_stromal = min(threshold_n_stromal, num_stromal)
        cancer = cells[is_cancer]
        stromal = cells[~is_cancer][:threshold_num_stromal]

        dists = dists_between(stromal, cancer)

        min_dists = dists.min(axis=1)
        mean_nearest_cancer = min_dists.mean()
        max_nearest_cancer = min_dists.max()

        summaries = [
            num_stromal,
            num_cancer,
            mean_nearest_cancer,
            max_nearest_cancer,
        ]

        return jnp.array(summaries)

    def generate_observation(self, key: random.PRNGKey, misspecified=True):
        theta_key, y_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)
        y_raw = self.simulate(y_key, theta_true, necrosis=misspecified, summarise=False)
        y = self.summarise(*y_raw[0])
        return jnp.squeeze(theta_true), jnp.squeeze(y), y_raw

    def sample_prior(self, key: random.PRNGKey, n: int):
        keys = random.split(key, 3)
        cell_rate = random.uniform(keys[0], (n,), **self.cell_rate_lims)
        cancer_parent_rate = random.uniform(keys[1], (n,), **self.parent_rate_lims)
        cancer_daughter_rate = random.uniform(keys[2], (n,), **self.daughter_rate_lims,)
        return jnp.column_stack([cell_rate, cancer_parent_rate, cancer_daughter_rate])

    def n_nearest_dists(self, dists, n_points):
        "Minimum distance containing n points. n_points to match axis dists in axis 1."
        assert dists.shape[1] == len(n_points)
        d_sorted = onp.partition(dists, kth=n_points, axis=0)
        min_d = d_sorted[n_points, onp.arange(dists.shape[1])]
        return min_d

    def in_prior_support(self, theta):
        bools = []
        limits = [self.cell_rate_lims, self.parent_rate_lims, self.daughter_rate_lims]
        for col, lims in zip(theta.T, limits):
            in_support = (col > lims["minval"]) & (col < lims["maxval"])
            bools.append(in_support)
        return jnp.column_stack(bools).all(axis=1)


@numba.njit(fastmath=True)
def dists_between(a, b):
    "Returns a.shape[0] by b.shape[0] l2 norms between rows of arrays."
    dists = []
    for ai in a:
        for bi in b:
            dists.append(onp.linalg.norm(ai - bi))
    return onp.array(dists).reshape(a.shape[0], b.shape[0])


def remove_nans_and_warn(x):
    nan_rows = jnp.any(jnp.isnan(x), axis=1)
    n_nan = nan_rows.sum()
    if n_nan > 0:
        x = x[~nan_rows]
        print(f"Warning {n_nan} simulations contained NAN values have been removed.")
    return x

def remove_nans_and_warn_idx(x):
    # TODO: ryan func
    nan_rows = jnp.any(jnp.isnan(x), axis=1)
    # n_nan = nan_rows.sum()
    # if n_nan > 0:
    #     x = x[~nan_rows]
    #     print(f"Warning {n_nan} simulations contained NAN values have been removed.")
    return nan_rows



class ContaminatedSLCP(Task):
    """
    Class to represent the contaminated simple likelihood complex posterior (SLCP) model.
    """

    def __init__(self, num_draws: int = 5, misspec_level: float = 1.0):
        self.num_draws = num_draws
        self.misspec_level = misspec_level
        self.theta_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\theta_5$"]
        self.prior_bounds = (-3.0, 3.0)

    def sample_prior(self, key: random.PRNGKey, n: int):
        low, high = self.prior_bounds
        return dist.Uniform(low, high).sample(key=key, sample_shape=(n, 5))

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray, misspecified: bool = False):
        num_samples = theta.shape[0]
        samples = jnp.zeros((num_samples, self.num_draws * 2))

        for i in range(num_samples):
            t1, t2, t3, t4, t5 = theta[i]
            m_theta = jnp.array([t1, t2])
            s1, s2 = t3 ** 2, t4 ** 2
            rho = jnp.tanh(t5)
            cov_mat = jnp.array([[s1, rho * s1 * s2], [rho * s1 * s2, s2]])
            draw_key, key = random.split(key)
            sample = dist.MultivariateNormal(m_theta, cov_mat).sample(key=draw_key, sample_shape=(self.num_draws,)).flatten()
            samples = samples.at[i].set(sample)
            # if jnp.any(jnp.isnan(sample)):
            #     samples = samples.at[i].set(1e+4*jnp.ones_like(sample))

            if misspecified:
                samples = samples.at[i].set(self.apply_misspecification(samples[i], key))
        return samples

    def apply_misspecification(self, x, key):
        noise_scale = 100
        noise_key, _ = random.split(key)
        noise = dist.Normal(0, noise_scale).sample(key=noise_key, sample_shape=x.shape)
        return x + self.misspec_level * noise

    def calculate_summary_statistics(self, x):
        return x  # In this model, data is used directly as summary statistics.

    def generate_observation(self, key: random.PRNGKey, misspecified: bool = True):
        theta_key, sim_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)
        theta_true = jnp.array([[0.7, -2.9, -1.0, -0.9,  0.6]])  # TODO: MANUALLY SET
        observation = self.simulate(sim_key, theta_true, misspecified)
        return theta_true, observation, None
