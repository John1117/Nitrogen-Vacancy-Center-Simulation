import numpy as np
import pandas as pd
from  scipy import constants
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import cauchy


def plot_ODMR_data(frequency, brightness, condition):
    plt.figure(figsize=(20,5))
    plt.rc('axes', linewidth=2)
    plt.title(condition, fontsize=25)
    plt.xlabel('Frequency (MHz)', fontsize=25)
    plt.ylabel('Brightness', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tick_params(length=6, width=2)
    plt.grid()
    plt.plot(frequency, brightness, '.', c=(0,0,0,0.2), markersize=15)
    plt.show()


class ODMR():
    def __init__(self, dtype=np.complex128):
        self.dtype = dtype

    def Lorentz_fn(self, x, x0, gamma):
        return cauchy.pdf(x, x0, gamma) * np.pi * gamma

    def Lorentz_ODMR_fn(self, frequency, parameter):
        odmr = parameter[0]
        for i in range(8):
            odmr -= parameter[1+i] * self.Lorentz_fn(frequency, parameter[9+i], parameter[17+i])
        return odmr
                
    def Lorentz_ODMR_fitting_fn(
            self, 
            frequency, 
            baseline, 
            peak_brightness_1, 
            peak_brightness_2, 
            peak_brightness_3, 
            peak_brightness_4, 
            peak_brightness_5, 
            peak_brightness_6, 
            peak_brightness_7, 
            peak_brightness_8, 
            peak_freqency_1, 
            peak_freqency_2, 
            peak_freqency_3, 
            peak_freqency_4, 
            peak_freqency_5, 
            peak_freqency_6, 
            peak_freqency_7, 
            peak_freqency_8,
            gamma0,
            gamma1,
            gamma2,
            gamma3,
            gamma4,
            gamma5,
            gamma6,
            gamma7
        ):
        return baseline - (
             peak_brightness_1 * self.Lorentz_fn(frequency, peak_freqency_1, gamma0)
            +peak_brightness_2 * self.Lorentz_fn(frequency, peak_freqency_2, gamma1)
            +peak_brightness_3 * self.Lorentz_fn(frequency, peak_freqency_3, gamma2)
            +peak_brightness_4 * self.Lorentz_fn(frequency, peak_freqency_4, gamma3)
            +peak_brightness_5 * self.Lorentz_fn(frequency, peak_freqency_5, gamma4)
            +peak_brightness_6 * self.Lorentz_fn(frequency, peak_freqency_6, gamma5)
            +peak_brightness_7 * self.Lorentz_fn(frequency, peak_freqency_7, gamma6)
            +peak_brightness_8 * self.Lorentz_fn(frequency, peak_freqency_8, gamma7)
            )

    def fit(self, experimental_frequency, experimental_brightness, baseline, experimental_peak_brightness, experimental_peak_frequency, gamma=5):
        n_peak = len(experimental_peak_brightness)
        if n_peak < 8:
            experimental_peak_brightness = np.concatenate((experimental_peak_brightness, np.repeat(baseline, 8-n_peak)))
            experimental_peak_frequency = np.concatenate((experimental_peak_frequency, np.zeros(8-n_peak)))
            upper_bound = np.repeat(np.inf, 25)
        for i in range(8-n_peak):
            upper_bound[n_peak+i+1] = 1e-7
            parameter_0 = np.concatenate(([baseline], baseline-experimental_peak_brightness, experimental_peak_frequency, np.repeat(gamma, 8)))
            optimal_parameter = curve_fit(frequency=self.Lorentz_ODMR_fitting_fn, xdata=experimental_frequency, ydata=experimental_brightness, p0=parameter_0, bounds=(0, upper_bound))[0]
    
        plt.figure(figsize=(20,5))
        plt.plot(experimental_frequency, experimental_brightness, 'k.')
        for i in range(n_peak):
            peak_parameter = optimal_parameter.copy()
            peak_parameter[1:9] = 0
            peak_parameter[i+1] = optimal_parameter[i+1]
            peak_frequency = np.linspace(optimal_parameter[i+9]-6*optimal_parameter[i+17], optimal_parameter[i+9]+6*optimal_parameter[i+17])
            plt.plot(peak_frequency, self.Lorentz_ODMR_fn(peak_frequency, peak_parameter), 'm--')
        plt.plot(experimental_frequency, self.Lorentz_ODMR_fn(experimental_frequency, optimal_parameter), 'b-') 
        plt.show()

        optimal_parameter[1:9] = optimal_parameter[0] - optimal_parameter[1:9]
        return optimal_parameter[0], optimal_parameter[1:1+n_peak], optimal_parameter[9:9+n_peak], optimal_parameter[17:17+n_peak]