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
# Last modified: Tue Jun 08, 2021  11:26pm
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
                dataframe with nens + 1 columns; the first column "Depth" has
                the same depth values as in proxy["depth"], the remaining
                columns "Age Model N", where N is the N-th age model contains
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
    print("while loop ...")
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
    names = ["Depth"]
    names.extend(["Age Model %d" % (i + 1) for i in range(ES)])
    agemodels = pd.DataFrame(np.c_[pz.T, agemods.T], columns=names)

    return agemodels


def proxyrecords(agemodels, proxy, ageres=10):
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

    Returns
    -------
    proxyrecs : pandas.DataFrame
                dataframe with nens + 1 columns; the first column "Age" has
                the age sampled every ``ageres`` years, the remaining
                columns "Proxy Record N", where N is the N-th age model contains
                the corresponding proxy value for each sampled age. NaNs
                signify those proxy values where the corresponding sampled age
                lie outside the age range of that particular age model.


    """
    # extract data to numpy arrays for convenience
    ## age models
    az = agemodels["Depth"].to_numpy()
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
    age = np.arange(np.floor(aa.min()), np.ceil(aa.max()), 10)

    # loop over age models and interpolate the proxy record on regular grid
    prxrec = np.zeros((ES, len(age)))
    for i in range(ES):
        curr_agemod = aa[i]
        f_za = interp1d(curr_agemod, az,
                        bounds_error=False, fill_value="extrapolate")
        curr_depth = f_za(age)
        prxrec[i] = f_pz(curr_depth)

    # create dataframe for the proxy record ensemble
    names = ["Age"]
    names.extend(["Proxy Record %d" % (i + 1) for i in range(ES)])
    proxyrecords = pd.DataFrame(np.c_[age.T, prxrec.T], columns=names)

    return proxyrecords
