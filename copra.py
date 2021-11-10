#! /usr/bin/env python3
"""
copra -   COnstructing Proxy Records from Age-models (COPRA)

          This module implements the heuristic framework put forward in [1],
          allowing users to estimate an ensemble of age models and
          consequently an ensemble of proxy record models based on initial
          age v/s depth and proxy v/s depth measurements. In principle, the
          current implementation is, in some ways, an extension of the ideas
          described in [1] and these new developments will be soon
          communicated in an upcoming publication.

[1] Breitenbach, Rehfeld, Goswami, et. al., COnstructing Proxy Records from
Age Models (COPRA) (2012),_Climate of the Past_, 8, 1765–1779

"""
# Created: Sun Sep 20, 2020  08:26pm 
# Last modified: Tue Nov 09, 2021  10:59am
#
# Copyright (C) 2020  Bedartha Goswami <bedartha@gmail.com> This program is
# free software: you can redistribute it and/or modify it under the terms of
# the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


import sys
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def agemodels(age, proxy, nens, max_iter=10000,
              interp_kind="linear", extrapolate=False, covariance=None):
    """
    Estimate age model ensemble from age v/s depth and proxy v/s depth data.

    Parameters
    ----------
    age : pandas.DataFrame
          dataframe containing age measurements; should contain three columns
          named as "depth", "age", and "error"
    proxy : pandas.DataFrame
          dataframe containing proxy measurements; should contain three columns
          named as "depth" and "proxy"
    nens : int
           size of the age model ensemble
    max_iter : int
               maximum number of iterations to try and achieve the required
               ensemble size
    interp_kind : string
                  interpolation type; passed on to scipy.interpolate.interp1d
    extrapolate : boolean
                  should the interpolation step also include extrapolation
    covariance : 2D numpy.ndarray
                 covariance matrix of the errors of age measurements; if None,
                 the age measurement errors are assumed to uncorrelated

    Returns
    -------
    agemodels : pandas.DataFrame
                dataframe with nens + 1 columns; the first column "depth" has
                the same depth values as in proxy["depth"], the remaining
                columns "age model N", where N is the N-th age model contains
                the corresponding age for each depth

    """
    # extract data into numpy arrays for convenience
    ## age measurements
    az = age["depth"].to_numpy()
    aa = age["age"].to_numpy()
    ae = age["error"].to_numpy()
    n_am = len(az)
    ## proxy measurements
    pz = proxy["depth"].to_numpy()
    pp = proxy["proxy"].to_numpy()
    n_pm = len(pz)

    # check if dates are ordered properly
    strati = np.all(np.sign(az[1:] - az[:-1]).astype("bool"))
    assert strati, "Stratigraphic order is violated by input data set!"

    # check if extreme proxy depths fall in the range of c14 depths
    chk1 = pz.min() >= az.min()
    chk2 = pz.max() <= az.max()
    if not chk1 * chk2:
        print("Warning! Proxy depths not in range of 14C measurements. " +
              "Extrapolating results!")
        extrapolate = True

    # use a while loop to obtain required number of ensemble members
    print("estimating age models ...")
    ## parse args
    ES = nens
    MAXITER = max_iter
    if extrapolate:
        fv = "extrapolate"
    else:
        fv = np.nan
    ## core loop
    agemods = []
    NUMITER = 0
    FACTOR = 1
    ES_LOCAL = ES * FACTOR
    mvn_mean = aa.copy()
    if covariance is None:
        mvn_cov = (ae ** 2) * np.eye(n_am)
    else:
        mvn_cov = covariance
    while (len(agemods) < ES) and (NUMITER <= MAXITER):
        # sample using multivariate normal
        sample = np.random.multivariate_normal(mvn_mean, mvn_cov, ES_LOCAL)

        # use scipy.interpolate.CubicSpline to get ages at proxy depths
        a_interp = np.zeros((ES_LOCAL, n_pm))
        for i in range(ES_LOCAL):
            curr_samp = sample[i, :]
            f_az = interp1d(az, curr_samp, kind="linear",
                            bounds_error=False,
                            fill_value=fv)
            res = f_az(pz)
            a_interp[i] = res

        # keep only stratigraphically correct models
        diff = np.diff(a_interp, axis=1)
        sign = np.sign(diff)
        cond = sign == 1.
        stra = np.all(cond, axis=1)
        a_interp = a_interp[stra]

        # keep only those models that have positive BP ages
        neg = np.min(a_interp, axis=1) < 0.
        a_interp = a_interp[~neg] 

        # add to the main list
        for mod in a_interp:
            agemods.append(mod)
        NUMITER += 1


    # the while loop might return a higher number of age models so return only
    # the first ES age models
    if len(agemods) > ES:
        agemods = agemods[:ES]
    agemods = np.array(agemods, dtype="float")

    # create dataframe for age model ensemble
    names = ["depth"]
    names.extend(["age model %d" % (i + 1) for i in range(ES)])
    agemodels = pd.DataFrame(np.c_[pz.T, agemods.T], columns=names)

    return agemodels


def proxyens(agemodels, proxy, ageres=10, agelims=None, nan_policy="strict"):
    """
    Estimates proxy record ensemble from given age model ensemble

    Parameters
    ----------
    agemodels : pandas.DataFrame
                dataframe containing ensemble of age models as given by
                copra.agemodels; the first column "Depth" has
                the same depth values as in proxy["depth"], the remaining
                columns "Age Model N", where N is the N-th age model contains
                the corresponding age for each depth 
    proxy : pandas.DataFrame
          dataframe containing proxy measurements; should contain three columns
          named as "depth" and "proxy"
    ageres : int
             integer specifying the sampling frequency along the time axis for
             the final proxy records; default is 10, i.e. every 10 years
    agelims : list, tuple, or array-like
              pair of values specifying the start and end of the age axis in
              the form `(start age, end age)`. The resulting age array will be
              created as a regular linearly spaced grid between `start age` and
              `end age` (inclusive) with a spacing of `ageres`.

    Returns
    -------
    proxyens : pandas.DataFrame
               dataframe with nens + 1 columns; the first column "age" has
               the age sampled every ``ageres`` years, the remaining
               columns "proxy record N", where N is the N-th age model contains
               the corresponding proxy value for each sampled age. NaNs
               signify those proxy values where the corresponding sampled age
               lie outside the age range of that particular age model.


    """
    # extract data to numpy arrays for convenience
    ## age models
    az = agemodels["depth"].to_numpy()
    aa = agemodels[agemodels.columns[1:]].to_numpy().T
    ES = aa.shape[0]
    ## proxy measurements
    pz = proxy["depth"].to_numpy()
    pp = proxy["proxy"].to_numpy()
    n_pm = len(pz)

    # create proxy-depth interpolation object
    f_pz = interp1d(pz, pp, bounds_error=False, fill_value=np.nan)

    # check
    assert len(az) == n_pm, "Age models lengths are different from proxy"

    # create regularly spaced age array
    if agelims == None:
        start_age, end_age = np.floor(aa.min()), np.ceil(aa.max())
    elif len(agelims) == 2:
        start_age, end_age = agelims[0], agelims[1]
    else:
        raise "Error: Limits of age axis incorrectly specified!"
    end_age += 0.0001 * ageres         # add buffer to include the end_age value
    age = np.arange(start_age, end_age, ageres)

    # loop over age models and interpolate the proxy record on regular grid
    prxrec = np.zeros((ES, len(age)))
    for i in range(ES):
        curr_agemod = aa[i]
        f_za = interp1d(curr_agemod, az,
                        bounds_error=False, fill_value="extrapolate")
        curr_depth = f_za(age)
        prxrec[i] = f_pz(curr_depth)

    # implement NaN policy
    len_age = len(age)
    percent_nans = (np.isnan(prxrec).sum(axis=0) * 100) / ES
    nan_policy_opts = {
                        "strict": 0,
                        "moderate": 5,
                        "relaxed": 10,
                        "easygoing": 25,
                        "none": 100,
            }
    nonan_idx = np.where(percent_nans <= nan_policy_opts[nan_policy])[0]
    prxrec = prxrec[:, nonan_idx]
    age = age[nonan_idx]
    if len(nonan_idx) < len_age:
        print("Warning: Some proxy depths were outside of age model depths.")
        print("\tThese values were assigned NaN values.")
        print("\tRemoving times with NaNs as per 'nan_policy = %s'" % nan_policy)
        print("\tPrescribed age limits (BP) = (%.1f, %.1f)" % (start_age, end_age))
        print("\tFinal age limits (BP) = (%.1f, %.1f)" % (age[0], age[-1]))
        if nan_policy != "strict":
            print("Warning: Chance of up to %d %% of records with NaNs" \
                    % nan_policy_opts[nan_policy])


    # create dataframe for the proxy record ensemble
    names = ["age"]
    names.extend(["proxy record %d" % (i + 1) for i in range(ES)])

    return pd.DataFrame(np.c_[age.T, prxrec.T], columns=names)


def proxygp(proxyens):
    """
    Returns a Gaussian Process approximation of the proxy ensemble#
    """
    x = proxyens["age"].to_numpy()
    y = np.nanmean(proxyens.iloc[:, 1:].to_numpy(), axis=1)
    sigma_n = np.nanstd(proxyens.iloc[:, 1:].to_numpy(), axis=1)

    # define kernel hyper-parameters
    l = np.nanmean(np.diff(x))
    sigma_f = (np.nanmax(y) - np.nanmin(y)) ** 2

    # define kernel object
    kernel1 = ConstantKernel(constant_value=sigma_f,
                             constant_value_bounds=(1e-3, 1e3))
    kernel2 = RBF(length_scale=l, length_scale_bounds=(1e-3, 1e3))
    kernel = kernel1 * kernel2

    # define GP regressor object
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n ** 2,
                                  optimizer=None)

    # format data shapes for the regressor
    X = x.reshape(len(x), 1)

    # fit to observations
    gp.fit(X, y)

    return gp


def synthcore(M=10, K=1000, ae_top=1, ae_bot=500, ae_incr="linear"):
    """
    Simulate synthetic sedimentary core and make age and proxy measurements.
    """
    def get_model(N=5000, zmax=150, z0=100, Amax=11000, k=5E-2):
        """
        Returns a simulated growth model and proxy signal.
        """
        # simulate core growth as a monotonic logistic function
        print("simulate logistic growth of sedimentary archive ...")
        z = np.linspace(0., zmax, N)
        a = Amax / (1. + np.exp(-k * (z - z0)))
        age_true, depth_true = a, z

        # simulate proxy signal as a noisy sinusoid
        print("simulate proxy signal as noisy sinusoid ...")
        A1, A2 = 2.5, 2.5
        T1, T2 = 100., 1000.
        e1, e2 = 0.75, 7.5
        p_hif = A1 * np.sin((2. * np.pi) * (a / T1)) + e1 * np.random.randn(N)
        p_lof = A2 * np.sin((2. * np.pi) * (a / T2)) + e2 * np.random.randn(N)
        p = p_hif + p_lof
        proxy_true = p

        # create dictionary
        print("create dictionary for model ...")
        model = {
                "depth": depth_true,
                "age": age_true,
                "proxy": proxy_true,
                }

        # create DataFrame
        print("create DataFrame from dictionary ...")
        model = pd.DataFrame(model)

        return model


    def get_measurements(model, M=10, K=1000,
                         ae_top=1, ae_bot=500, ae_incr="linear"):
        """
        Returns proxy-depth and age-depth measurements on simulated paleo-archive
        """
        # load model data
        z = model["depth"].to_numpy()
        a = model["age"].to_numpy()
        p = model["proxy"].to_numpy()
        N = len(z)

        # make age measurements
        print("make age measurements ...")
        # AME = ame
        # M = 10
        az = np.linspace(z[0], z[-1] + 0.001, M)
        # ae = AME * np.linspace(0., 2, M)
        # ae = 4 * (ame / 100) * np.logspace(0, 2, M)
        if ae_incr == "linear":
            ae = np.linspace(ae_top, ae_bot, M)
        elif ae_incr == "exp":
            ae = np.logspace(np.log(ae_top), np.log(ae_bot), M, base=np.exp(1))
        f_az = interp1d(z, a, bounds_error=False, fill_value="extrapolate")
        aa = f_az(az)
        aa = np.random.normal(loc=aa, scale=ae)

        # make proxy measurements
        print("make proxy measurements ...")
        # K = 1000
        i = int(np.floor(N / K))
        pp = p[::i]
        pz = z[::i]

        # create dictionaries model and measurements
        print("create dictionaries for model and measurements ...")
        age_measurements = {
                "depth": az,
                "age": aa,
                "error": ae,
                }
        proxy_measurements = {
                "depth": pz,
                "proxy": pp,
                }

        # create pandas DataFrames from dictionaries
        print("create dataframes from dictionaries ...")
        dataframes = [
                pd.DataFrame(age_measurements),
                pd.DataFrame(proxy_measurements),
                ]

        return dataframes

    # get model
    model = get_model()
    dataframes = get_measurements(model, M=M,
                                  ae_top=ae_top, ae_bot=ae_bot, ae_incr="exp"
                                  )

    return model, dataframes
