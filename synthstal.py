#! /usr/bin/env python3
"""
Simulates a synthetic sedimentary core and makes age and proxy measurements
===========================================================================

"""
# Last modified: Mon Nov 08, 2021  04:53pm
#
# Copyright (C) 2020  Bedartha Goswami <bedartha.goswami@uni-tuebingen.de>
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


import pandas as pd
import numpy as np

from scipy.interpolate import interp1d


def get_model(N=5000, zmax=150, z0=100, Amax=11000, k=5E-2):
    """
    Returns a simulated growth model and proxy signal.
    """
    # simulate core growth as a monotonic logistic function
    print("simulate logistic growth of sedimentary archive ...")
    # N = 5000
    z = np.linspace(0., zmax, N)
    # growth_params = {
    #         "L": 11000.,     # maximum possible age
    #         "z0": 100.,      # depth value for growth midpoint
    #         "k": 5E-2,       # steepness of growth
    #         }
    # z0, L, k = growth_params["z0"], growth_params["L"], growth_params["k"]
    a = Amax / (1. + np.exp(-k * (z - z0)))
    age_true, depth_true = a, z

    # simulate proxy signal as a noisy sinusoid
    print("simulate proxy signal as noisy sinusoid ...")
    A1, A2 = 2.5, 2.5
    T1, T2 = 100., 1000.
    e1, e2 = 0.75, 2.5
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


def get_measurements(model, M=10, K=1000, ae_top=1, ae_bot=500, ae_incr="linear"):
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


