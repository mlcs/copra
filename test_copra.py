"""
Test the COPRA code
===================

This file is not added to the git
"""
# (c) Bedartha Goswami
# bedartha.goswami@uni-tubeingen.de

import copra
import synthstal
import matplotlib.pyplot as pl

# get synthetic model measurements
model = synthstal.get_model()
dataframes = synthstal.get_measurements(model)
age_df, proxy_df = dataframes[0], dataframes[1]

# run COPRA
agemods = copra.agemodels(age_df, proxy_df, nens=100, extrapolate=True)
print(agemods)
prxrecs = copra.proxyrecords(agemods, proxy_df, ageres=50, agelims=(100, 9500))

# plot results
fig = pl.figure(figsize=[12., 4.])
ax1 = fig.add_axes([0.05, 0.10, 0.25, 0.80])
ax2 = fig.add_axes([0.35, 0.10, 0.60, 0.80])
ax1.plot(agemods.iloc[:, 1:], agemods['depth'], "-", c="0.5", alpha=0.10)
ax1.errorbar(age_df['age'], age_df['depth'], xerr=age_df['error'], fmt="ro",
             capsize=4, ms=4)
ax1.set_ylim(ax1.get_ylim()[::-1])
ax2.plot(prxrecs['age'], prxrecs.iloc[:, 1:], "o", c="0.5", ms=2, alpha=0.01)
num_nans = copra.np.isnan(prxrecs.iloc[:, 1:].to_numpy()).sum(axis=1)
ax2_ = ax2.twinx()
ax2_.plot(prxrecs['age'], num_nans, "ko-")
pl.show()
