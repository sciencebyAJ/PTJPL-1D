
# coding: utf-8

# # PTJPL for point forcing datasets
# ***************************************************************
# <left> The below code is from [Fisher et al., 2008](http://josh.yosh.org/publications/Fisher%20et%20al%202008%20-%20Global%20estimates%20of%20the%20land-atmosphere%20water%20flux.pdf)
# <left> Adjustments are made for application at daily timesteps
# <left> This version of code was authored by: AJ Purdy
# <left> Major Contributions for this code are from Gregory Halverson & Grayson Badgley
# <left> Contact:  ajpurdy@uci.edu     
# ***************************************************************
# 
#     Input variables within DATAFRAME:     
#         air_temperature: air temperature near the surface (C)
#         air_temperature_mean: daily average air temperature near the surface (K)        
#         RH_day_min: minimum daily relative humidity(%) 
#            replacement if not avail --> daily minimum vapor pressure (Pa)
#         ndvi_mean: average Normalized Difference Vegetation Index for day        
#         optimum_temperature: phenologically optimum temperature (K)
#         fAPARmax: maximum fraction of photosynthetically active radiation (unitless)   
#         net_radiation: instantaneous net radiation in (W/m2) 
#         daily_radiation: daily net radiation in (W/m2)
#         
#     Returned:
#         A dataset is returned from this script containing the following variables:
#         evapotranspiration: total evapotranspiration (W/m2)
#         interception_evaporation: intercepted evaporation (W/m2) 
#         soil_evaporation: evaporation from soil (W/m2)
#         canopy_transpiration: transpiration from canopy (W/m2)                       
#         potential_evapotranspiration: potential evapotranspiration (W/m2)
#         
# ***************************************************************
# 

# In[2]:

import datetime
import glob
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import exp, log
import os
import pandas as pd
pd.options.mode.chained_assignment = None

# FILE PATHWAYS
data_path = 'data/'
figs_path = 'figs/'
# ----------------------- MODEL IS IN THE LIBRARY REFERENCED HERE ---------------------- 
from ptjpl_lib import *
# ----------------------------- NOTEBOOK SPECIFIC COMMANDS ----------------------------- 
# % matplotlib inline
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:90% !important; }</style>"))

os.chdir(data_path);
fList = glob.glob('*.csv');
fNameMMS = fList[0]; 
datMMS = pd.read_csv(fNameMMS); 
df_MMS = datMMS.set_index('Time'); 
df_model_MMS = ptjpl(df_MMS)
os.chdir('../')

plt.figure();
df_model_MMS.evapotranspiration.rolling(4,2).mean().plot(label='PTJPL')
df_model_MMS.LE_FC.rolling(4,2).mean().plot(label='LE$_{observation}$')
plt.legend(ncol=2, loc = 3, fontsize = 7)
plt.title('Morgan Monroe State Forest')
plt.ylabel('$Wm^{-2}$', fontsize=14)
plt.savefig(figs_path+'US-MMS_total_ET.png')

plt.figure()
df_model_MMS.canopy_transpiration.plot()
df_model_MMS.interception_evaporation.plot()
df_model_MMS.soil_evaporation.plot()
plt.legend(ncol=3, fontsize=7)
plt.ylabel('$Wm^{-2}$', fontsize=14)
plt.title('Morgan Monroe State Forest')
plt.savefig(figs_path+'US-MMS_ET_components.png')




# In[ ]:



