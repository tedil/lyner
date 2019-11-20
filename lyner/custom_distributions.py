import numba
import numpy as np
from numba import jit
from scipy.optimize import minimize
from scipy.special import gammainc, gammaincc, gammaln as gamln
from scipy.stats import rv_continuous
from scipy.stats._discrete_distns import nbinom_gen
from scipy.stats._distn_infrastructure import argsreduce


class negbinom_gen(rv_continuous):
    """A negative binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    Negative binomial distribution describes a sequence of i.i.d. Bernoulli
    trials, repeated until a predefined, non-random number of successes occurs.

    The probability mass function of the number of failures for `nbinom` is::

       nbinom.pmf(k) = choose(k+n-1, n-1) * p**n * (1-p)**k

    for ``k >= 0``.

    `nbinom` takes ``n`` and ``p`` as shape parameters where n is the number of
    successes, whereas p is the probability of a single success.

    %(after_notes)s

    %(example)s

    """

    # def _rvs(self, mu, sigma):
    #     return self._random_state.negative_binomial(n, p, self._size)

    def _argcheck(self, mu, sigma):
        return (mu >= 0) & (sigma >= - 1 / mu)

    def _pdf(self, x, mu, sigma):
        return np.exp(self._logpdf(x, mu, sigma))
        # _sigma = 1 / sigma
        # return gamma(x + sigma) / (gamma(_sigma) * gamma(x + 1)) * np.power((1 / (1 + mu * sigma)), _sigma) * np.power(
        #     (mu / (_sigma + mu)), x)

    def _logpdf(self, x, mu, sigma):
        # if has_theano:
        #     r = tlogpmf(x, mu, sigma)
        #     return r
        _sigma = 1 / sigma
        coeff = gamln(x + _sigma) - gamln(_sigma) - gamln(x + 1)
        return coeff - _sigma * np.log1p(mu * sigma) + x * np.log(mu) + x * np.log(sigma) - x * np.log1p(mu * sigma)

    def _stats(self, mu, sigma):
        mu = mu
        var = mu + sigma * np.power(mu, 2)
        # g1 = (Q + P) / np.sqrt(n * P * Q)
        # g2 = (1.0 + 6 * P * Q) / (n * P * Q)
        return mu, var, None, None

    def _negbinom_nllh(self, P, x):
        mu, sigma = P
        return -(negbinom._logpdf(x, mu, sigma)).sum()

    def fit(self, data, x0=None):
        if x0 is None:
            av = np.median(data)
            va = np.var(data)
            x0 = av, va
        return minimize(self._negbinom_nllh, x0, args=data, method='Nelder-Mead',
                        # bounds=[(self.a, np.max(args)), (0, np.inf)]
                        ).x


negbinom = negbinom_gen(name='nbinom', shapes='mu, sigma')


class laisson_gen(rv_continuous):
    def _argcheck(self, mu, b):
        return (mu >= 0) & (b >= 1)

    # @jit(locals={'mu_theta': numba.float64[:], 's1': numba.float64[:], 's2': numba.float64[:],
    #              'x1': numba.float64[:], 'x2': numba.float64[:], 'result': numba.float64[:]})
    def _pdf(self, x, mu, b):
        mu_b = mu / b
        x_0 = x <= 1e-09
        x_v = ~x_0
        x1 = x * x_0
        x2 = x * x_v

        # left half of laplace distribution used for the case x close to 0
        # needed because laisson is inaccurate close to 0
        s1 = 0.5 / b * np.exp(-(mu - x1) / b)

        s2_a = np.exp(mu_b) * np.power(1. + 1 / b, -x2) * gammaincc(x2, mu + mu_b)
        s2_b = np.exp(-mu_b) * np.power(1. - 1 / b, -x2) * gammainc(x2, mu - mu_b)
        s2 = 0.5 / b * (s2_a + s2_b)

        s1 = s1 * x_0
        s1[s1 == np.nan] = np.inf
        s2 = s2 * x_v
        s2[s2 == np.nan] = np.inf
        return s1 + s2

    @jit(locals={'mu_theta': numba.float64})
    def _logpdf(self, x, mu, b):
        mu_b = mu / b
        s_a = np.exp(mu_b) * np.power(1 + 1 / b, -x) * gammaincc(x, mu + mu_b)
        s_b = np.exp(-mu_b) * np.power(1 - 1 / b, -x) * gammainc(x, mu - mu_b) if b >= 1 else 0
        s = np.log(s_a + s_b) - np.log(b) - np.log(2)
        return s

    def pdf(self, x, *args, **kwds):
        """
        Probability density function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(np.asarray, (x, loc, scale))
        args = tuple(map(np.asarray, args))
        dtyp = np.find_common_type([x.dtype, np.float64], [])
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x) & (scale > 0)
        cond = cond0 & cond1
        output = np.zeros(np.shape(cond), dtyp)
        np.putmask(output, (1 - cond0) + np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,) + args + (scale,)))
            s, goodargs = goodargs[-1], goodargs[:-1]
            # use trapezoidal integration rule to estimate normalization factor
            # # end = (np.max(x) + np.max(goodargs[1]) + 2 * np.max(goodargs[2]) + 1) * 4
            #
            # end = np.max([np.max(x) + np.max(goodargs[2]), 1000])
            # num_segments = int(end * 1.666)
            # r = np.linspace(self.a + 1e-07,
            #                 end,
            #                 num_segments)
            # norm_scale = np.array([scale[0]] * num_segments)
            # norm_args = [np.array([arg[0]] * num_segments) for arg in goodargs]
            # len_scale = len(scale)
            # scale = norm_scale * np.trapz(self._pdf(r, *norm_args[1:]), r)[:len_scale]
            mu = goodargs[1]
            b = goodargs[2]
            s = 1 - 0.5 * np.exp((0 - mu) / b)
            np.place(output, cond, self._pdf(*goodargs) / s)
        if output.ndim == 0:
            return output[()]
        return output

    @jit(locals={'ratio': numba.float64, 'nsteps': numba.float64, 'upper': numba.float64,
                 'b': numba.float64, 'mu': numba.float64, 's': numba.float64[:], 'c': numba.float64[:]})
    def _cdf_single(self, x, *args):
        mu, b = args
        ratio = 2
        upper = np.max([x, mu * b, 2 * mu ** 2 + x, (x + mu) * 2])
        nsteps = np.max([(x + upper) // ratio, 5])
        s = np.linspace(0, x, nsteps, dtype=np.float64)
        s = np.trapz(np.nan_to_num(laisson.pdf(s, mu, b)), s)
        c = np.linspace(x, upper, nsteps, dtype=np.float64)
        c = np.trapz(np.nan_to_num(laisson.pdf(c, mu, b)), c)
        return s / (s + c)

    def negative_llh(self, x0, x):
        mu, b = x0
        return -np.sum(np.log(self._pdf(x, mu, b)))

    def fit(self, data, x0=None):
        if not x0:
            x0 = np.median(data), np.var(data)
        mu, b = minimize(self.negative_llh, x0, args=data, method='Nelder-Mead',
                         # bounds=[(self.a, np.max(args)), (1e-07, 1.0 - 1e-07)]
                         ).x
        return mu, b


laisson = laisson_gen(a=0, name="laisson", shapes='mu, b')
pseudo_laisson = laisson_gen(name="pseudo-laisson", shapes='mu, b')  # piecewise distribution: -inf → 0: laplace, 0 → inf: laisson


def _nbinom2_nllh(P, x):
    n, p = P
    return -(nbinom2.logpmf(x, n, p)).sum()


def fit(self, x0, *args):
    return minimize(_nbinom2_nllh, x0, args=args, method='Nelder-Mead').x


nbinom2_gen = nbinom_gen
nbinom2_gen.fit = fit
nbinom2 = nbinom2_gen(a=0, name="nbinom2")


def _negbinom_nllh(P, x):
    mu, sigma = P
    return -negbinom._logpmf(x, mu, sigma).sum()


def _fit_negbinom(g, x0):
    mu0, sigma0 = x0
    mu, sigma = negbinom.fit((mu0, sigma0), g).x
    return mu, sigma


def _fit_laisson(g, x0):
    mu0, theta0 = x0
    mu, b = laisson.fit((mu0, theta0), g).x
    return mu, b


def fit_distribution(matrix, distribution, x0=None):
    def fit_distr_parallel(distribution, values, x0=x0):
        import multiprocessing
        from joblib import Parallel, delayed
        if x0:
            return Parallel(n_jobs=multiprocessing.cpu_count())(delayed(distribution.fit)(v[~np.isnan(v)], x0) for x0, v in zip(x0, values))
        else:
            if distribution in {laisson, negbinom}:
                x0 = list(zip([np.nanmedian(arr) for arr in matrix], [np.nanvar(arr) for arr in matrix]))
                return Parallel(n_jobs=multiprocessing.cpu_count())(delayed(distribution.fit)(v[~np.isnan(v)], x0) for x0, v in zip(x0, values))
            else:
                return Parallel(n_jobs=multiprocessing.cpu_count())(delayed(distribution.fit)(v[~np.isnan(v)]) for v in values)

    def estimate_fit_ml(distribution, x0=x0):
        return fit_distr_parallel(distribution, matrix, x0)

    return estimate_fit_ml(distribution=distribution, x0=x0)
