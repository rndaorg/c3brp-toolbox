import numpy as np
from const.const import mu_earth_sun
from crbrp.collinear import collinear_lagrange_points


# Test with Sun-Earth system
L1, L2, L3 = collinear_lagrange_points(mu_earth_sun)
print(L1, L2, L3)