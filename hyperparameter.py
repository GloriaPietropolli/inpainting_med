"""
hyperparameter for the implementation
"""

batch = 1
number_channel = 4  # 1: temp, 2:salinity, 3:doxy, 4: chla
latitude_interval = (36, 44)
longitude_interval = (2, 9)
depth_interval = (0, 3)
year_interval = (2015, 2016)
year = 2015
resolution = (12, 12, 0.1)
kindof = 'dumb'
if kindof == 'float':
    channels = [0, 1, 2]
if kindof == 'sat':
    channels = [3]
if kindof == 'flat_sat':
    channels = [3]
if kindof == 'model2015':
    channels = [0, 1, 2, 3]

dumb_list = 1  # on
if dumb_list:
    kindof = 'dumb'
    channels = [0, 1, 2, 3]
