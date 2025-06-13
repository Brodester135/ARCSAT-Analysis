import numpy as np
import polars
import seaborn
import glob
import gc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
import astropy.units as u

def determine_ls_period(times, fluxes):
    
    frequency, power = LombScargle(times, fluxes).autopower()
    best_freq = frequency[np.argmax(power)]
    
    if not hasattr(best_freq, 'unit'):
        best_freq = best_freq / u.day
    period = 1/best_freq
    ls = period.to(u.h) * 2
        
    formatted_freq = f"{best_freq.value:.3g}"
    formatted_period = f"{period:.3g}"
    formatted_ls = f"{ls:.3g}"
    
    plt.plot(frequency, power)
    plt.axvline(x=best_freq.value, label=f"Period: {formatted_period}", color='red')
    plt.xlabel("Freq (1/day)")
    plt.title(f"System period: {formatted_ls}")
    plt.xlim(0, 50)
    plt.legend()
    plt.savefig("Lomb_Scargle_fplot.png")
    plt.show()
        
    return ls


def calc_ingress_egress(params, period_days):
    
    primary = params["primary"]
    mid_phase = primary["mid_phase"]
    duration = primary["duration_days"]
    dur_purr = dur/period_days
    print("Ingress phase: ", mid_phase-dur_purr/2)
    print("Egress phase: ", mid_phase+dur_purr/2)
    print("Ingress (MJD): ", primary["mid_time_mjd"]-duration/2)
    print("Egress (MJD): ", primary["mid_time_mjd"]+duration/2)