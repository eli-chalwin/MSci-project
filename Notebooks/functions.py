#import libraries 
import openghg 
from openghg.analyse import ModelScenario
from openghg.retrieve import search_footprints, search_flux, get_footprint, get_flux, get_obs_surface, search, search_bc, get_bc
from matplotlib import pyplot as plt
import numpy as np
from cartopy import crs as ccrs
import pandas as pd
import xarray as xr
import skill_metrics as sm
from sklearn.metrics import root_mean_squared_error

#function that retrieves and plots observational data for a given site and inlet 
def obs(site, inlet, start_date=None, end_date=None):
    ds_obs = get_obs_surface(site=site, species="ch4", inlet=inlet, store="obs_paris_2024_07_store")
    ds_obs = ds_obs.data 
  
    if start_date is None and end_date is None:
        ds_obs_1M = ds_obs.resample(time="1M").mean()  
    else:
        time_slice = slice(start_date, end_date)
        ds_obs_1M = ds_obs.sel(time=time_slice).resample(time="1M").mean()
    
    ds_obs_1M = ds_obs_1M.mf 
    return ds_obs_1M

#function for plotting 
def plot(model, title, x, y):
    model.plot()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()  #add grid
    plt.show() #show

#function for finding the time overlap of the flux, footprint and bc data
def overlap(site, inlet):
    bc = get_bc(species="ch4", store="old_shared_store_zarr", bc_input = "camsv22r2_daily", domain = "europe")
    bc_start_date = bc.metadata.get("start_date")
    bc_end_date = bc.metadata.get("end_date")
    flux = get_flux(species="ch4", domain="europe", store="old_shared_store_zarr", source = "edgar-annual-total")
    flux_start_date = flux.metadata.get("start_date")
    flux_end_date = flux.metadata.get("end_date")
    fp = get_footprint(site=site,store="old_shared_store_zarr", inlet=inlet, domain ="europe", species="inert")
    fp_start_date = fp.metadata.get("start_date")
    fp_end_date = fp.metadata.get("end_date")
    start_date = max(fp_start_date, flux_start_date, bc_start_date)
    end_date = min(fp_end_date, flux_end_date, bc_end_date)
    start_date = pd.to_datetime(start_date).tz_localize(None).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).tz_localize(None).strftime('%Y-%m-%d')
    return start_date, end_date

#function that sets up an openghg modelscenario and adds the relevant data and calculates everything 
def scenario(site, inlet, return_type = "model", start_date=None, end_date=None):
    model_scenario = ModelScenario()   #set up model scenario 
    model_scenario.add_flux(species="ch4", domain="europe", store="old_shared_store_zarr")   #add flux
    model_scenario.add_bc(species="ch4", domain="europe", store="old_shared_store_zarr", bc_input = "camsv22r2_daily")  #add boundary conditions 
    model_scenario.footprint = get_footprint(site= site, inlet=inlet, domain="europe",species='inert' , store="old_shared_store_zarr")  #add footprint for desired site and inlet
    model_scenario.add_obs(site=site,species="ch4",inlet=inlet,store="obs_paris_2024_07_store") #add observed data 
    
    if start_date is None and end_date is None:
        model = model_scenario.calc_modelled_obs(resample_to="1M", cache=True, recalculate=False) #model obs data without a baseline
    else:
        model = model_scenario.calc_modelled_obs(resample_to="1M", cache=True, recalculate=False) #model obs data without a baseline
        model = model.sel(time=slice(start_date,end_date))
   
    if start_date is None and end_date is None:
        baseline = model_scenario.calc_modelled_baseline(resample_to="1M", output_units=1e-09, cache=True, recalculate=False) #calculate baseline with monthly average     
    else:
        baseline = model_scenario.calc_modelled_baseline(resample_to="1M", output_units=1e-09, cache=True, recalculate=False) #calculate baseline with monthly average     
        baseline = baseline.sel(time=slice(start_date,end_date))
   
    combined = model + baseline 

    if return_type == "model":
        return model
    elif return_type == "baseline":
        return baseline
    elif return_type == "combined":
        return combined
    else:
        raise ValueError("Invalid return_type specified. Choose 'model' or 'baseline'.")

#function that calculates and returns standard deviation, rmse and correlation 
def stats(obs, predicted):
    difference = obs - predicted 
    std = difference.std() 
    std = std.values
    std = std.item()
    rmse = root_mean_squared_error(obs, predicted)
    correlation = xr.corr(obs, predicted, dim='time')
    correlation = correlation.compute()
    correlation = correlation.values
    correlation = correlation.item()

    return std, rmse, correlation 

#function that plots a taylor diagram 
def taylor(std, rmse, corr, sites):
    if isinstance(std, (int, float)):
        std =[std]  # Convert to a list if it's a number
    else:
        pass  # Do nothing if it's already a list
    if isinstance(rmse, (int, float)):
        rmse = [rmse]  # Convert to a list if it's a number
    else:
        pass  # Do nothing if it's already a list
    if isinstance(corr, (int, float)):
        corr =[corr]  # Convert to a list if it's a number
    else:
        pass  # Do nothing if it's already a list
    if isinstance(sites, (str)):
        sites =[sites]  # Convert to a list if it's a number
    else:
        pass  # Do nothing if it's already a list
    std_dev_taylor = [0.0]
    rmse_taylor = [0.0]
    correlation_taylor = [1.0]
    std_dev_taylor.extend(std)
    rmse_taylor.extend(rmse)
    correlation_taylor.extend(corr)
    std_dev_taylor = np.array(std_dev_taylor)
    rmse_taylor = np.array(rmse_taylor)
    correlation_taylor = np.array(correlation_taylor)
    label = ['Non-Dimensional Observation']
    label.extend(sites)
    sm.taylor_diagram(std_dev_taylor, rmse_taylor, correlation_taylor, markerLabel = label)
