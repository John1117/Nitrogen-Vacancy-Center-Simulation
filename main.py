# %%
import numpy as np
import pandas as pd

from ODMR import plot_ODMR_data, ODMR
from ODRS import plot_ODRS_data, normalize_ODRS, ODRS


# %%
ODMR_old_data = pd.read_csv('ODMR_old_data.csv')
experimental_frequency_old = ODMR_old_data['Freq (MHz)'].values
experimental_brightness_old = {}
experimental_brightness_old['0'] = ODMR_old_data['B_0mA'].values
experimental_brightness_old['10'] = ODMR_old_data['B_10mA'].values
experimental_brightness_old['20'] = ODMR_old_data['B_20mA'].values
experimental_brightness_old['40'] = ODMR_old_data['B_40mA'].values
experimental_brightness_old['60'] = ODMR_old_data['B_60mA'].values
experimental_brightness_old['80'] = ODMR_old_data['B_80mA'].values

for condition, experimental_brightness in experimental_brightness_old.items():
    plot_ODMR_data(experimental_frequency_old, experimental_brightness, condition)

# %%
ODMR_new_data = pd.read_excel('ODMR_new_data.xlsx')
experimental_frequency_new = ODMR_new_data['Freq (MHz)'].values
experimental_brightness_new = {}
experimental_brightness_new['0'] = ODMR_new_data['B_0mA'].values
experimental_brightness_new['15'] = ODMR_new_data['B_15mA'].values
experimental_brightness_new['30'] = ODMR_new_data['B_30mA'].values
experimental_brightness_new['50'] = ODMR_new_data['B_50mA'].values
experimental_brightness_new['70'] = ODMR_new_data['B_70mA'].values
experimental_brightness_new['90'] = ODMR_new_data['B_90mA'].values
experimental_brightness_new['110'] = ODMR_new_data['B_110mA'].values

for condition, experimental_brightness in experimental_brightness_new.items():
    plot_ODMR_data(experimental_frequency_new, experimental_brightness, condition)

# %%
ODMR_deg_data = pd.read_csv('ODMR_deg_data.csv')
experimental_frequency_deg = ODMR_deg_data['Freq_48G (MHz)'].values
experimental_brightness_deg = {}
experimental_brightness_deg['48'] = ODMR_deg_data['ODMR_48G_norm'].values
for condition, experimental_brightness in experimental_brightness_deg.items():
    plot_ODMR_data(experimental_frequency_deg, experimental_brightness, condition)

# %%
ODRS_nondeg_data = pd.read_csv('ODRS_nondeg_data.csv')
experitental_time_arr_nondeg = ODRS_nondeg_data['tW_us_long (microsecond)'].values
experitental_brightness_arr_nondeg = {}
experitental_brightness_arr_nondeg['0'] = ODRS_nondeg_data['0_Gauss'].values
experitental_brightness_arr_nondeg['25'] = ODRS_nondeg_data['25_Gauss'].values
experitental_brightness_arr_nondeg['40'] = ODRS_nondeg_data['40_Gauss'].values
experitental_brightness_arr_nondeg['50'] = ODRS_nondeg_data['50_Gauss'].values
experitental_brightness_arr_nondeg['80'] = ODRS_nondeg_data['80_Gauss'].values
experitental_brightness_arr_nondeg['100'] = ODRS_nondeg_data['100_Gauss'].values
experitental_brightness_arr_nondeg['142'] = ODRS_nondeg_data['142_Gauss'].values
for condition, experitental_brightness_arr in experitental_brightness_arr_nondeg.items():
    plot_ODRS_data(experitental_time_arr_nondeg, experitental_brightness_arr, condition)

# %%
ODRS_deg_data = pd.read_csv('ODRS_deg_data.csv')
experitental_time_arr_deg = ODRS_deg_data['tW_us_long (microsecond)'].values
experitental_brightness_arr_deg = {}
experitental_brightness_arr_deg['48'] = ODRS_deg_data['48G noMW'].values
for condition, experitental_brightness_arr in experitental_brightness_arr_deg.items():
    plot_ODRS_data(experitental_time_arr_deg, experitental_brightness_arr, condition)

# %%
experimental_peak_frequency = np.array([2690, 2758, 2787, 2850, 2935, 2985, 3012, 3062])
experimental_peak_brightness = np.array([1.895e6, 1.915e6, 1.917e6, 1.920e6, 1.920e6, 1.915e6, 1.915e6, 1.905e6])
ODMR.fit(experimental_frequency_new, experimental_brightness_new['90'], 1.935e6, experimental_peak_brightness, experimental_peak_frequency)


# %%
for condition, experitental_brightness_arr in experitental_brightness_arr_nondeg.items():
    nmlzd_b_sqn_xpr = normalize_ODRS(experitental_time_arr_nondeg, experitental_brightness_arr)
    plot_ODRS_data(experitental_time_arr_nondeg, nmlzd_b_sqn_xpr, condition)

for condition, experitental_brightness_arr in experitental_brightness_arr_deg.items():
    nmlzd_b_sqn_xpr = normalize_ODRS(experitental_time_arr_deg, experitental_brightness_arr)
    plot_ODRS_data(experitental_time_arr_deg, nmlzd_b_sqn_xpr, condition)

# %%
microwave_parameters = []
max_time = 1000
experitental_time_arr_nondeg_segment = experitental_time_arr_nondeg[experitental_time_arr_nondeg<max_time]
experitental_brightness_arr_nondeg_segment = normalize_ODRS(experitental_time_arr_nondeg, experitental_brightness_arr_nondeg['0'])[experitental_time_arr_nondeg<max_time]
simulation_time_arr = np.arange(experitental_time_arr_nondeg[0], max_time, 0.1)

odrs = ODRS()
odrs.set_initial_density_matrix(experitental_time_arr_nondeg, experitental_brightness_arr_nondeg['0'])
odrs.get_degenerates(np.array([2861.14087356, 2861.14091075, 2861.14113023, 2861.1411302 ,
                               2874.75493047, 2876.83370652, 2878.31873119, 2877.00858183]),
                     np.array([9.33702047, 9.33721266, 9.3389599 , 9.33896087,
                               8.27854233, 7.144524  , 6.77207363, 7.03051746]), 1000000, 1)
resonance_frequency = np.array([2861.14087356, 2861.14091075, 2861.14113023, 2861.1411302 ,
                  2874.75493047, 2876.83370652, 2878.31873119, 2877.00858183])


odrs.RK45_relative_tolerance = 1e-5
odrs.RK45_absolute_tolerance = 1e-8
odrs.diamond.Gaussian_stdev = 20
odrs.min_NVC_distanse = 1.5
odrs.interaction_radius = 100
odrs.brightness_bound = [-np.inf,np.inf]

gamma_1 = 1/6000
gamma_1_prime = 1/6000
gamma_2 = 0

odrs.n_config = 1
odrs.n_sampling = 10

simulation_brightness_arr = odrs.simulate(simulation_time_arr, resonance_frequency, microwave_parameters, [gamma_1, gamma_1_prime, gamma_2])

arrs = [[experitental_time_arr_nondeg_segment, experitental_brightness_arr_nondeg_segment, 'o', (0,0,0,0.2), 'Xpr'],
        [simulation_time_arr, simulation_brightness_arr.T, '-', (0,0,1,0.1), None],
        [simulation_time_arr, simulation_brightness_arr.mean(axis=0), '-', (0,0,1,1), 'Simu']]