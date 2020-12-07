# original python script by Gregory Halverson:  gregory.halverson@gmail.com
__author__ = 'Gregory Halverson'
# updates to include soil moisture by AJ Purdy: adamjpurdy@gmail.com
__author2__ = 'AJ Purdy'

# This is where all the ptjpl functions originate from
# set up this such that output is a dataframe and each necessary variable for fine tuning model for different fluxnet sites with cosmos data
import numpy as np
import numpy
from numpy.ma import exp, log
import csv
import datetime
import pandas as pd
# import FLUXNET_META

DEFAULT_AVERAGING_PERIOD = 30
DEFAULT_FLOOR_SATURATION_VAPOR_PRESSURE = True

# Priestley-Taylor coefficient alpha
PRIESTLEY_TAYLOR_ALPHA = 1.26
BETA = 1.0
PSYCHROMETRIC_GAMMA = 0.0662 # Pa/K # http://www.fao.org/docrep/x0490e/x0490e07.htm
KRN = 0.6
KPAR = 0.5

# calculate Soil-Adjusted Vegetation Index from Normalized Difference Vegetation Index
# using linear relationship
def savi_from_ndvi(ndvi):
    SAVI_MULT = 0.45
    SAVI_ADD = 0.132
    savi = ndvi * SAVI_MULT + SAVI_ADD
    
    return savi

# calculate fraction of absorbed photosynthetically active radiation
# from Soil-Adjusted Vegetation Index using linear relationship
def fAPAR_from_savi(savi):
    FAPAR_MULT = 1.3632
    FAPAR_ADD = -0.048
    
    fAPAR = savi * FAPAR_MULT + FAPAR_ADD
    
    return fAPAR


# calculate fraction of absorbed photosynthetically active radiation
# from Normalized Difference Vegetation Index
def fAPAR_from_ndvi(ndvi):
    savi = savi_from_ndvi(ndvi)
    fAPAR = fAPAR_from_savi(savi)
    
    return fAPAR


# calculate fraction of intercepted photosynthetically active radiation
# from Normalized Difference Vegetation Index
def fIPAR_from_ndvi(ndvi):
    FIPAR_ADD = -0.05
    fIPAR = ndvi + FIPAR_ADD
    
    return fIPAR


# saturation vapor pressure in kPa from air temperature in celsius
def saturation_vapor_pressure_from_air_temperature(air_temperature):
    SVP_BASE = 0.611
    SVP_MULT = 17.27
    SVP_ADD = 237.7
    svp = SVP_BASE * exp((air_temperature * SVP_MULT) / (air_temperature + SVP_ADD))
    
    return svp


# calculate slope of saturation vapor pressure to air temperature
# in pascals over kelvin
def delta_from_air_temperature(air_temperature):
    return 4098 * (0.6108 * exp(17.27 * air_temperature / (237.7 + air_temperature))) / (air_temperature + 237.3) ** 2


# cut max and min values to defined values
def enforce_boundaries(matrix, lower_bound, upper_bound):
    matrix[matrix < lower_bound] = np.nan
    matrix[matrix > upper_bound] = np.nan
    
    return matrix

# cut max and min values and set min to nan
def clean_RH(DATA, min_data, max_data):
    ''' returns cleaned data based on pre-defined min and max values '''
    DATA[DATA>max_data] = np.nan;
    DATA[DATA<min_data] = np.nan;

    return DATA


# calculate the green canopy fraction from FAPAR and FIPAR
def fG_fun(fAPAR, fIPAR):
    
    return fAPAR/fIPAR

# calculate plant moisture constraint from fAPAR and amximum fAPAR
def fM_fun(fAPAR,fAPARMAX):
    
    return fAPAR/fAPARMAX

# theory trying to capture temp at peak photosynthesis (wet, green, high vpd, high radiation)
def Topt_fun(RN,TMAX,SAVI,VPD):
    ''' returns the optimal temperature
        inputs 30 day averages of: RN, TACTUAL, SAVI, VPD'''
    topt_mask = ~np.isnan(np.array(RN)) & ~np.isnan(TMAX) & ~np.isnan(SAVI) & ~np.isnan(VPD)
    
    PHEN_RAW = np.ones(np.shape(RN)); PHEN_RAW[:]=np.nan;

    VPD_g1 = VPD;VPD_g1[VPD<0.5]=0.5;
    PHEN_RAW[topt_mask] = RN[topt_mask]*TMAX[topt_mask]*SAVI[topt_mask]/VPD_g1[topt_mask]

    max_idx = PHEN_RAW.argmax();
    T_opt = TMAX[max_idx];
    
    if np.isnan(T_opt):
        T_opt = np.nanmax(np.nanmax(TMAX))-5;

    return T_opt

# calculate the relative stress from Temperature
# this is for high temperatures

def fT_fun(Tmax,Topt):
    '''plant temperature constraint '''
    fT = np.exp(-((Tmax-Topt)/Topt)**2)
    fT[Tmax<-5]=0.05; # <---- add a cold temperature constraint
    return fT

# calculate the relative surfae wetness
# returns a scalar between 1 and 0;
def fWET_fun(RH):
    '''RETURNS: fraction of wet area
        INPUTS: average relative humidity'''
    return RH**4

# calculate soil moisture function from RH and VPD
# returns a scalar between 1 and 0;
def fSM_fun(RH, VPD):
    '''RETURNS: soil moisture constraint
        INPUTS: relative humidity, vapor pressure deficit'''
    beta = 1.0;

    return np.power(RH,(VPD/beta))

def LE_2_ETcm(LE_Wm2):
    '''
        This tool converts Latent Energy to Evapotranspiration
        INPUT DATA:  LE_2_ET (W/m2)
        OUTPUT DATA: ET_cm (cm/day)
        '''
    lambda_e = 2.460*10**6      # J kg^-1
    roe_w = 1000                # kg m^-3
    m_2_cm = 100                # convert m to mm
    s_2_30m = 60*30*48          # multiply by s in 30 min multiply by 48 to get cm/day
    mask = ~np.isnan(LE_Wm2)
    ET_cm = np.empty(LE_Wm2.shape)
    ET_cm[:] = np.NAN
    ET_cm[mask] = (LE_Wm2[mask]*(m_2_cm*s_2_30m)/(lambda_e*roe_w))
    return ET_cm


#def gamma_fun(e_sat,T_C):
#    gamma_mult  = 240.97*17.502; gamma_add = 240.97;
#    return (gamma_mult*e_sat)/(np.power((T_C+gamma_add),2)); # anon func



# calculate Soil-Adjusted Vegetation Index from Normalized Difference Vegetation Index
# using linear relationship
def savi_from_ndvi(ndvi):
    SAVI_MULT = 0.45
    SAVI_ADD = 0.132
    savi = ndvi * SAVI_MULT + SAVI_ADD
    
    return savi


# calculate fraction of absorbed photosynthetically active radiation
# from Soil-Adjusted Vegetation Index using linear relationship
def fAPAR_from_savi(savi):
    FAPAR_MULT = 1.3632
    FAPAR_ADD = -0.048
    
    fAPAR = savi * FAPAR_MULT + FAPAR_ADD
    
    return fAPAR


# calculate fraction of absorbed photosynthetically active radiation
# from Normalized Difference Vegetation Index
def fAPAR_from_ndvi(ndvi):
    savi = savi_from_ndvi(ndvi)
    fAPAR = fAPAR_from_savi(savi)
    
    return fAPAR


# calculate fraction of intercepted photosynthetically active radiation
# from Normalized Difference Vegetation Index
def fIPAR_from_ndvi(ndvi):
    FIPAR_ADD = -0.05
    fIPAR = ndvi + FIPAR_ADD
    
    return fIPAR

# calculate slope of saturation vapor pressure to air temperature
# in pascals over kelvin
def delta_from_air_temperature(air_temperature):
    return 4098 * (0.6108 * exp(17.27 * air_temperature / (237.7 + air_temperature))) / (air_temperature + 237.3) ** 2


def enforce_boundaries(matrix, lower_bound, upper_bound):
    matrix[matrix < lower_bound] = lower_bound
    matrix[matrix > upper_bound] = upper_bound
    
    return matrix

def filter_bad_values(matrix, lower_bound, upper_bound):
    matrix[matrix < lower_bound] = np.nan
    matrix[matrix > upper_bound] = np.nan
    
    return matrix


DEFAULT_FLOOR_SATURATION_VAPOR_PRESSURE=1.;
def ptjpl(AA,
          verbose=True,
          floor_saturation_vapor_pressure=DEFAULT_FLOOR_SATURATION_VAPOR_PRESSURE):
    """
        :AA: is a dataframe from where each variable listed below is extracted
        I have attached a csv file containing what each variable name is provided.  

        :param air_temperature_K:
        Numpy matrix of air temperature near the surface in kelvin
        :param air_temperature_mean_K:
        Numpy matrix of average air temperature near the surface in kelvin
        :param water_vapor_pressure_mean_Pa:
        Numpy matrix of average vapor pressure in pascals
        :param ndvi_mean:
        Numpy matrix of average Normalized Difference Vegetation Index
        :param optimum_temperature:
        Numpy matrix of phenologically optimum temperature
        :param fAPARmax:
        Numpy matrix of maximum fraction of photosynthetically active radiation
        :param net_radiation:
        Numpy matrix of instantaneous net radiation in watts per square meter
        :param daily_radiation:
        Numpy matrix of daily net radiation in watts per square meter
        :param verbose:
        Flag to output activity to console
        :param floor_saturation_vapor_pressure:
        Option to floor calculation of saturation vapor pressure at 1 to avoid anomalous output
        :return:
        Dataframe with:
        evapotranspiration, PTJPL original model
        potential_evapotranspiration, T and S and I
        daily_evapotranspiration, PTJPL original scaled with EF and adiation
        """
    air_temperature_K =       np.array(AA.TA+273.15)          # (K)
    air_temperature_mean_K =  np.array(AA.TA_day_mean+273.15) # (K)
    ndvi_mean=                np.array(AA.NDVI)               # (0-1.0)
    net_radiation=            np.array(AA.NETRAD);            # (W/m2)
    daily_radiation=          np.array(AA.NETRAD_day)         # (W/m2)
    RH =                      np.array(AA.RH_day_min)         # (%), Can run with vapor pressure see comments out below

    #     water_vapor_pressure_mean_Pa =np.array(AA.VPD_day_max)*100# set VPD of input dataset to Pa
    #     optimum_temperature=18; # use the function Topt_fun to return this value [ constrain to +5 o C, if below a certain value, set fT to 1.0]
    #     fAPARmax=0.65 # grab this value from MODIS timeseries dataset

    # convert temperatures from kelvin to celsius
    air_temperature = air_temperature_K - 273
    air_temperature_mean = air_temperature_mean_K - 273
    
    #     scale water vapor pressure from Pa to kPa
    #     water_vapor_pressure_mean = water_vapor_pressure_mean_Pa * 0.001
    
    # calculate surface wetness values
    if verbose:
        print('calculating surface wetness values [%]')
        print('calculating vapor pressure deficit [kPa]')
    
    # ***REPLACED THIS WITH ACTUAL MEASTUREMENTS FROM TOWERS:
    # calculate relative humidity from water vapor pressure and saturation vapor pressure
    #     relative_humidity = water_vapor_pressure_mean / saturation_vapor_pressure
    # upper bound of relative humidity is one, results higher than one are capped at one
    # WE INSTEAD HAVE RH SO WE CALCULATE VPD FROM THIS:
    relative_humidity = filter_bad_values(RH,0.,1.)
    
    # calculate saturation vapor pressure in kPa from air temperature in celcius
    saturation_vapor_pressure = saturation_vapor_pressure_from_air_temperature(air_temperature_mean)#[kPA]
    
    # floor saturation vapor pressure at 1
    if floor_saturation_vapor_pressure:
        saturation_vapor_pressure[saturation_vapor_pressure < 1] = 1
    water_vapor_pressure_mean = RH*saturation_vapor_pressure;
    
    # calculate vapor pressure deficit from water vapor pressure
    vapor_pressure_deficit = saturation_vapor_pressure - water_vapor_pressure_mean # [kPa]
    
    # lower bound of vapor pressure deficit is zero, negative values replaced with nodata
    vapor_pressure_deficit[vapor_pressure_deficit < 0] = np.nan
    
    # calculate relative surface wetness from relative humidity
    AA['RH_roll']=relative_humidity; 
    relative_surface_wetness = relative_humidity ** 4
    relative_surface_wetness[air_temperature<=0]=0.; # FROZEN WATER
    # calculate slope of saturation to vapor pressure curve Pa/K
    delta = delta_from_air_temperature(air_temperature)
    
    # calculate vegetation values
    if verbose:
        print('calculating vegetation values')
    
    # calculate fAPAR & fAPARmax from NDVI mean
    fAPAR = fAPAR_from_ndvi(ndvi_mean)
    fAPAR = enforce_boundaries(fAPAR,0.,1.)
    fAPARmax = np.nanmax(fAPAR)
    AA['fAPAR']=fAPAR;

    # calculate fIPAR from NDVI mean
    fIPAR = fIPAR_from_ndvi(ndvi_mean)
    AA['fIPAR']=fIPAR
    # calculate green canopy fraction (fg) from fAPAR and fIPAR, constrained between zero and one
    green_canopy_fraction = enforce_boundaries(fAPAR / fIPAR, 0, 1)
    
    # calculate plant moisture constraint (fM) from fraction of photosynthetically active radiation,
    # constrained between zero and one
    plant_moisture_constraint = enforce_boundaries(fAPAR / fAPARmax, 0, 1)
    AA['VPD_roll']=vapor_pressure_deficit;     
    # calculate soil moisture constraint from relative humidity and vapor pressure deficit,
    # constrained between zero and one
    soil_moisture_constraint = enforce_boundaries(AA.RH_roll.rolling(14,1).mean() ** (AA.VPD_roll.rolling(14,1).mean() / BETA), 0, 1)
    soil_moisture_constraint[air_temperature<0]=0.00;
    AA['soil_moisture_constraint']=soil_moisture_constraint;
    # take fractional vegetation cover from fraction of photosynthetically active radiation
    fractional_vegetation_cover = enforce_boundaries(fIPAR, 0, 1)
    
    # calculate SAVI from NDVI
    savi_mean = savi_from_ndvi(ndvi_mean)
    AA['savi']=savi_mean;
    AA['VPD']= vapor_pressure_deficit

    # calculate optimum_temperature from flux tower:
    optimum_temperature = Topt_fun(np.array(AA.NETRAD_day.rolling(30,1).mean()),np.array(AA.TA_day_max.rolling(30,1).mean()),np.array(AA['savi'].rolling(30,1).mean()),np.array(AA['VPD'].rolling(30,1).mean()))

    if verbose:
        print('calculating plant optimum temperature')
        print 'optimum temp: ' +str(optimum_temperature)+ ' C'

    # calculate plant temperature constraint (fT) from optimal phenology
    plant_temperature_constraint = fT_fun(air_temperature_mean,optimum_temperature)
    # exp(-(((air_temperature_mean - optimum_temperature) / optimum_temperature) ** 2))

    # calculate leaf area index : now extract from MODIS dataset
    leaf_area_index = -log(1 - fIPAR) * (1 / KPAR)
    # leaf_area_index = np.array(AA.LAI);

    # calculate delta / (delta + gamma)
    epsilon = delta / (delta + PSYCHROMETRIC_GAMMA)
    
    # soil evaporation
    if verbose:
        print('calculating soil evaporation')

    # caluclate net radiation of the soil from leaf area index
    soil_net_radiation = net_radiation * exp(-KRN * leaf_area_index)

    # calculate instantaneous soil heat flux from net radiation and fractional vegetation cover
    soil_heat_flux = net_radiation * (0.05 + (1 - fractional_vegetation_cover) * 0.265)
    # change the above to METRIC G

    soil_heat_flux[soil_heat_flux < 0] = 0;
    soil_heat_flux[soil_heat_flux > 0.35 * soil_net_radiation] = 0.35 * soil_net_radiation[soil_heat_flux > 0.35 * soil_net_radiation]

    # calculate soil evaporation (LEs) from relative surface wetness, soil moisture constraint,
    # priestley taylor coefficient, epsilon = delta / (delta + gamma), net radiation of the soil,
    # and soil heat flux
    soil_evaporation = (relative_surface_wetness + soil_moisture_constraint * (1 - relative_surface_wetness)) * PRIESTLEY_TAYLOR_ALPHA * epsilon * (soil_net_radiation - soil_heat_flux)
    #     soil_evaporation[numpy.isnan(soil_evaporation)] = 0
    soil_evaporation[soil_evaporation < 0] = numpy.nan

    # canopy transpiration
    if verbose:
        print('calculating canopy transpiration')

    # calculate net radiation of the canopy from net radiation of the soil
    canopy_net_radiation = net_radiation - soil_net_radiation

    # calculate canopy transpiration (LEc) from priestley taylor, relative surface wetness,
    # green canopy fraction, plant temperature constraint, plant moisture constraint,
    # epsilon = delta / (delta + gamma), and net radiation of the canopy
    canopy_transpiration = PRIESTLEY_TAYLOR_ALPHA * (1 - relative_surface_wetness) * green_canopy_fraction * plant_temperature_constraint * plant_moisture_constraint * epsilon * canopy_net_radiation
    canopy_transpiration[numpy.isnan(canopy_transpiration)] = 0
    canopy_transpiration[canopy_transpiration < 0] = 0

    # interception evaporation
    if verbose:
        print('calculating interception evaporation')
    # calculate interception evaporation (LEi) from relative surface wetness, priestley taylor coefficient,
    # epsilon = delta / (delta + gamma), and net radiation of the canopy
    interception_evaporation = relative_surface_wetness * PRIESTLEY_TAYLOR_ALPHA * epsilon * canopy_net_radiation
    interception_evaporation[interception_evaporation < 0] = 0
    
    # combined evapotranspiration
    if verbose:
        print('combining evapotranspiration')
    # combine soil evaporation (LEs), canopy transpiration (LEc), and interception evaporation (LEi)
    # into instantaneous evapotranspiration (LE)
    evapotranspiration = soil_evaporation + canopy_transpiration + interception_evaporation
    evapotranspiration[evapotranspiration > net_radiation] = net_radiation[evapotranspiration > net_radiation]
    evapotranspiration[numpy.isinf(evapotranspiration)] = numpy.nan
    evapotranspiration[evapotranspiration < 0] = numpy.nan

    # daily evapotranspiration
    if verbose:
        print('calculating daily evapotranspiration')
    # calculate evaporative fraction (EF) from evapotranspiration, net radiation, and soil heat flux
    evaporative_fraction = evapotranspiration / (net_radiation - soil_heat_flux)

    # calculate daily evapotranspiration from daily net radiation and evaporative fraction
    daily_evapotranspiration = daily_radiation * evaporative_fraction
    
    # potential evapotranspiration
    if verbose:
        print('calculating potential evapotranspiration')
    # calculate potential evapotranspiration (pET) from priestley taylor coefficient,
    # epsilon = delta / (delta + gamma), net radiation, and soil heat flux
    potential_evapotranspiration = PRIESTLEY_TAYLOR_ALPHA * epsilon * (net_radiation - soil_heat_flux)
    # potential_transpiration      = PRIESTLEY_TAYLOR_ALPHA * epsilon * (canopy_net_radiation)
    # potential_evaporation        = PRIESTLEY_TAYLOR_ALPHA * epsilon * (soil_net_radiation - soil_heat_flux)
    
    # append new data to array
    results = AA;
    results['evapotranspiration'] =           evapotranspiration    
    # potential et added to results
    results['potential_evapotranspiration']=  potential_evapotranspiration
    # components evapotranspriation added for partitioning study
    results['canopy_transpiration'] =         canopy_transpiration
    results['interception_evaporation'] =     interception_evaporation
    results['soil_evaporation'] =             soil_evaporation
    return results

def filter_nan(s,o):
        """
        this functions removed the data  from simulated and observed data
        whereever the observed data contains nan

        this is used by all other functions, otherwise they will produce nan as
        output
        """
        import numpy as np
        data = np.array([s,o])
        data = np.transpose(data)
        data = data[~np.isnan(data).any(1)]
        return data[:,0],data[:,1]
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def rmse(s,o):
        """
        Root Mean Squared Error
        input:
                s: simulated
                o: observed
        output:
                rmses: root mean squared error
        """
        import numpy as np
        s,o = filter_nan(s,o)
        return np.sqrt(np.mean((s-o)**2))
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def no_nans(A1, A2):
    ''' returns the mask of nans for 2 arrays'''
    import numpy as np
    mask = ~np.isnan(A1) & ~np.isnan(A2);
    return mask

def R2_fun(s,o):
    """
    R^2 or Correlation coefficient^0.5
    input:
            s: simulated
            o: observed
    output:
            R^2
    """
    import numpy as np
    m_o_d = no_nans(np.array(o),np.array(s));
    stats_o_d = linregress(np.array(o)[m_o_d],np.array(s)[m_o_d])
    slope_o_d = stats_o_d[0]; int_o_d = stats_o_d[1]; r2_o_d_ = stats_o_d[2]**2;
    return r2_o_d_
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def KT(s,o):
    import scipy.stats
    """
    Kendalls Tao 
    input:
        s: simulated
        o: observed
    output:
        tau: Kendalls Tao
        p-value
    """
    s,o = filter_nan(s,o)
    tao = scipy.stats.stats.kendalltau(s, o)[0];
    pvalue = scipy.stats.stats.kendalltau(s, o)[1];
    return tao
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def lin_regress(X, Y):
    import numpy as np
    x, y = filter_nan(X,Y);
    # Regress ET and SR_in to obtain relationship to gap-fill NaN's in ET
    A = np.vstack([x,np.ones(len(x))]).T
    slope, intercept, rvalue, pvalue, stderr = np.linalg.lstsq(A,y)[0]
    return slope, intercept
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def FORCE_CLOSE(Rn, G, LE, H):
    '''
        This tool closes the SEB according to Twine et al & Wilson et al
        INPUT DATA:  Rn (W/m2); G  (W/m2); LE (W/m2); H (W/m2)
        OUTPUT DATA: An array with [H_fc (W/m2), LE_fc (W/m2)]
        '''
    if np.count_nonzero(~np.isnan(G))==0:
        x = np.array(Rn)
    else:
        x = np.array(Rn) - np.array(G)
    y = np.array(LE) + np.array(H)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask][np.newaxis]
    y = y[mask]
    cr = np.linalg.lstsq(x.T,y)
    H_fc = (1/cr[0][0])*H
    LE_fc = (1/cr[0][0])*LE
    return LE_fc