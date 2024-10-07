import numpy as np
import pandas as pd
from numpy import random
from scipy import linalg as linalg
from scipy import constants as constants
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
from scipy.stats import cauchy as cauchy



def plot_ODRS_data(time, brightness, condition):
    plt.figure(figsize=(20,5))
    plt.rc('axes', linewidth=2)
    plt.title(condition, fontsize=25)
    plt.xlabel('Time (us)', fontsize=25)
    plt.ylabel('Contrast', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tick_params(length=6, width=2)
    plt.grid()

    plt.plot(time, brightness, '.', c=(0,0,0,0.2), markersize=15)
    plt.show()


def normalize_ODRS(time, brightness, equilibrium_time=5000):
    initial_brightness = brightness[0]
    equilibrium_brightness = brightness[time>equilibrium_time].mean()
    return (brightness - equilibrium_brightness) / (initial_brightness - equilibrium_brightness)


class NanoDiamond():
    def __init__(self, dtype=np.complex128, diamond_radius=50, n_NVC=500, Gaussian_stdev=14):
        self.dtype = dtype
        self.diamond_radius = diamond_radius
        self.n_NVC = n_NVC
        self.Gaussian_stdev = Gaussian_stdev
        self.config = None

    def generate_config(self):
        config = []
        while len(config) < self.n_NVC:
            decay_rates = random.randn(3)*self.Gaussian_stdev
            if linalg.norm(decay_rates) < self.diamond_radius:
                i = random.randint(low=0, high=4)
                a = self.generate_quantization_axis(i)
                config.append([i, decay_rates, a])
        self.config = config
        return config

    def get_density_nm3(self, gamma_1=0, r2=None):
        if r2 and r2 < self.diamond_radius:
            if self.config == None:
                self.generate_config()
            n_NVC = 0
            for _, decay_rates, _ in self.config:
                if gamma_1 < linalg.norm(decay_rates) < r2:
                    n_NVC += 1
        else:
            r2 = self.diamond_radius
            n_NVC = self.n_NVC
        return n_NVC / (4 * np.pi / 3 * (r2**3 - gamma_1**3))

    def get_density_ppm(self, gamma_1=0, r2=None):
        if r2 and r2 < self.diamond_radius:
            if self.config==None:
                self.generate_config()
            n_NVC = 0
            for _, decay_rates, _ in self.config:
                if gamma_1 < linalg.norm(decay_rates) < r2:
                    n_NVC += 1
        else:
            r2 = self.diamond_radius
            n_NVC = self.n_NVC
        n_cell = (4 * np.pi / 3 * (r2**3 - gamma_1**3)) / 0.3567**3
        n_particle = n_cell * 8
        return n_NVC / n_particle * 1e6

    def get_average_distance(self, gamma_1=0, r2=None):
        if r2 and r2 < self.diamond_radius:
            if self.config==None:
                self.generate_config()
            n_NVC = 0
            for _, decay_rates , _ in self.config:
                if gamma_1 < linalg.norm(decay_rates) < r2:
                    n_NVC += 1
        else:
            r2 = self.diamond_radius
            n_NVC = self.n_NVC
        if n_NVC==0:
            return f'No NVC within decay_rates=[{gamma_1}, {r2}]nm.'
        else:
            return (4 * np.pi / 3 * (r2**3 - gamma_1**3) / n_NVC)**(1 / 3)

    def generate_first_quantization_axis(self):
        e = np.eye(N=3, dtype=self.dtype).T
        sign = random.choice([1,-1])
        phi = random.rand()*2*np.pi
        return (self.Rz(phi)@(sign*e)).T

    def generate_quantization_axis(self, i):
        theta = [0,1,1,1][i]*np.arccos(-1/3)
        phi = [0,0,1,2][i]*np.pi*2/3
        return (self.Rz(phi)@self.Ry(theta)@(self.generate_first_quantization_axis().T)).T

    def Ry(self, theta): 
        return np.array([[np.cos(theta),0,np.sin(theta)],
                        [0,1,0],
                        [-np.sin(theta),0,np.cos(theta)]])

    def Rz(self, theta):
        return np.array([[np.cos(theta),-np.sin(theta),0],
                        [np.sin(theta),np.cos(theta),0],
                        [0,0,1]])


class ODRS():
    def __init__(self, dtype=np.complex128, diamond=NanoDiamond()):  
        self.dtype = dtype
        self.diamond = diamond
        self.rotating_wave_approx_threshold = 10.0
        self.brightness_bound = [-0.2, 1.0]
        self.min_NVC_distanse = 0.2522
        self.interaction_radius = 10
        self.initial_density_matrix = None
        self.n_config = 5
        self.n_sampling = 10
        self.RK45_relative_tolerance = 1e-3
        self.RK45_absolute_tolerance = 1e-6

        # constants
        self.dipolar_intensity = (constants.physical_constants['electron gyromag. ratio'][0]**2)*constants.mu_0*(constants.hbar**2)/(4*np.pi)*(1e9**3/1e6/constants.h)
        self.gyromagnetic_ratio_NVC = -2.0029*constants.physical_constants['Bohr magneton'][0]*(1e-6/constants.h)

        #v ertices of tetrahedron
        self.tetrahedral_vectors = np.array([
            [0,0,1],
            [(8/9)**0.5,0,-1/3],
            [-(2/9)**0.5,(2/3)**0.5,-1/3],
            [-(2/9)**0.5,-(2/3)**0.5,-1/3]], dtype=dtype)
        
        # spin-1 Pauli operator
        self.Pauli_Z = np.array([
            [1,0,0], 
            [0,0,0], 
            [0,0,-1]], dtype=dtype)
        
        # flip operator
        self.flip_zero_to_plus = np.array([
            [0,1,0], 
            [0,0,0], 
            [0,0,0]], dtype=dtype) #|+><0|
        
        self.flip_plus_to_zero = np.array([
            [0,0,0], 
            [1,0,0], 
            [0,0,0]], dtype=dtype) #|0><+|
        
        self.flip_zero_to_minus = np.array([
            [0,0,0], 
            [0,0,0], 
            [0,1,0]], dtype=dtype) #|-><0|
        
        self.flip_minus_to_zero = np.array([
            [0,0,0], 
            [0,0,1], 
            [0,0,0]], dtype=dtype) #|0><-|
        
        self.flip_minus_to_plus = np.array([
            [0,0,1], 
            [0,0,0], 
            [0,0,0]], dtype=dtype) #|+><-|
        
        self.flip_plus_to_minus = np.array([
            [0,0,0], 
            [0,0,0], 
            [1,0,0]], dtype=dtype) #|-><+|

    def trace(self, A):
        return np.einsum('ii', A, dtype=self.dtype)

    def product(self, A, B, C=np.eye(3)):
        return np.einsum('...ij, ...jk, ...kl', A, B, C, dtype=self.dtype)

    def conjugate(self, A): 
        return np.array(A, dtype=self.dtype).T.conj() 

    def sample(self, config):
        origin = np.ones(3)*self.diamond.diamond_radius
        while linalg.norm(origin) > self.diamond.diamond_radius:
            origin = random.randn(3)*self.diamond.Gaussian_stdev

        surroundings = []
        for i in range(4):
            center_position, min_distance = None, np.inf
            for j, x, a in config:
                distance = linalg.norm(x - origin)
                if j==i and distance < min_distance:
                    center_position, min_distance = x, distance
        
            surroundings.append([])
            for j, x, a in config:
                distance = linalg.norm(x - center_position)
                if self.min_NVC_distanse <= distance <= self.interaction_radius:
                    surroundings[i].append([j, x-center_position, a])
        return surroundings

    def get_dipolar_Hamiltonian(self, density_matrix, dipolar_coef):
        dipolar_Hamiltonian = np.zeros_like(density_matrix)
        for i in range(4):
            for j in range(4):
                dipolar_Hamiltonian[i] += (
                     dipolar_coef[i,j,0]*(density_matrix[j,0,0]-density_matrix[j,2,2])*self.Pauli_Z
                    +dipolar_coef[i,j,3]*density_matrix[j,1,0]*self.flip_plus_to_zero
                    +dipolar_coef[i,j,1]*density_matrix[j,2,1]*self.flip_zero_to_minus
                    +dipolar_coef[i,j,4]*density_matrix[j,0,1]*self.flip_zero_to_plus
                    +dipolar_coef[i,j,2]*density_matrix[j,1,2]*self.flip_minus_to_zero
                    +dipolar_coef[i,j,5]*density_matrix[j,1,0]*self.flip_minus_to_zero
                    +dipolar_coef[i,j,7]*density_matrix[j,2,1]*self.flip_zero_to_plus
                    +dipolar_coef[i,j,6]*density_matrix[j,0,1]*self.flip_zero_to_minus
                    +dipolar_coef[i,j,8]*density_matrix[j,1,2]*self.flip_plus_to_zero)
        return dipolar_Hamiltonian

    def get_Lindblad_term(self, gamma_1, gamma_1_prime, gamma_2, density_matrix):
        Lindblad_term = np.zeros_like(density_matrix)
        for i in range(4):
            Lindblad_term[i] = [
                [gamma_1*(density_matrix[i,1,1]-density_matrix[i,0,0])+gamma_1_prime*(density_matrix[i,2,2]-density_matrix[i,0,0]), 
                 -(1.5*gamma_1+0.5*gamma_1_prime+0.5*gamma_2)*density_matrix[i,0,1], 
                 -(gamma_1+gamma_1_prime+2*gamma_2)*density_matrix[i,0,2]],
                [-(1.5*gamma_1+0.5*gamma_1_prime+0.5*gamma_2)*density_matrix[i,1,0], 
                 gamma_1*(density_matrix[i,0,0]+density_matrix[i,2,2]-2*density_matrix[i,1,1]), 
                 -(1.5*gamma_1+0.5*gamma_1_prime+0.5*gamma_2)*density_matrix[i,1,2]],
                [-(gamma_1+gamma_1_prime+2*gamma_2)*density_matrix[i,2,0], 
                 -(1.5*gamma_1+0.5*gamma_1_prime+0.5*gamma_2)*density_matrix[i,2,1], 
                 gamma_1*(density_matrix[i,1,1]-density_matrix[i,2,2])+gamma_1_prime*(density_matrix[i,0,0]-density_matrix[i,2,2])]
            ]
        return Lindblad_term

    def get_derivative_of_density_matrix(self, time, density_matrix, microwave_Hamiltonian, dipolar_coef, decay_rates):
        density_matrix = self.vector_to_matrix(density_matrix)
        dipolar_Hamiltonian = self.get_dipolar_Hamiltonian(time, density_matrix, dipolar_coef)
        Lindblad_term = self.get_Lindblad_term(decay_rates[0], decay_rates[1], decay_rates[2], density_matrix)
        derivative_of_density_matrix = -1j*(self.product(microwave_Hamiltonian+dipolar_Hamiltonian, density_matrix)-self.product(density_matrix, microwave_Hamiltonian+dipolar_Hamiltonian))+Lindblad_term
        return self.matrix_to_vector(derivative_of_density_matrix)

    def vector_to_matrix(self, vector):
        return (vector[0:36]+1j*vector[36:72]).reshape((4,3,3))

    def matrix_to_vector(self, matrix):
        return np.concatenate((matrix.reshape(36).real, matrix.reshape(36).imag), axis=0)

    def get_microwave_Hamiltonian(self, resonance_frequency, microwave_parameters):
        microwave_Hamiltonian = np.zeros((4,3,3), dtype=self.dtype)
        for microwave_frequency, microwave_vector in microwave_parameters:
            detuning = microwave_frequency - resonance_frequency
            Rabi_frequency = self.gyromagnetic_ratio_NVC * np.einsum('...i,i', self.tetrahedral_vectors, microwave_vector)
            for i in range(4):
                zero_minus_detuning = detuning[i] if abs(detuning[i]) < self.rotating_wave_approx_threshold else 0 #detuning of |0> and |->
                zero_plus_detuning = detuning[-i-1] if abs(detuning[-i-1]) < self.rotating_wave_approx_threshold else 0 #detuning of |0> and |+>
                zero_minus_Rabi_frequency = Rabi_frequency[i] if abs(detuning[i]) < self.rotating_wave_approx_threshold else 0 #Rabi freq. of |0> and |->
                zero_plus_Rabi_frequency = Rabi_frequency[i] if abs(detuning[-i-1]) < self.rotating_wave_approx_threshold else 0 #Rabi freq. of |0> and |+>
                microwave_Hamiltonian[i] += [
                    [-zero_plus_detuning, zero_plus_Rabi_frequency/2, 0],
                    [zero_plus_Rabi_frequency/2 , 0, zero_minus_Rabi_frequency/2],
                    [0, zero_minus_Rabi_frequency/2, -zero_minus_detuning]]
        return microwave_Hamiltonian

    
    def simulate(self, time_arr, resonance_frequency, microwave_parameters, decay_rates):

        def brightness_lower_bound(time, density_matrix, microwave_Hamiltonian, dipolar_coef, decay_rates):
            brightness = self.density_matrix_to_brightness(self.vector_to_matrix(density_matrix))
            return brightness - self.brightness_bound[0]
        brightness_lower_bound.terminal = True

        def brightness_upper_bound(time, density_matrix, microwave_Hamiltonian, dipolar_coef, decay_rates):
            brightness = self.density_matrix_to_brightness(self.vector_to_matrix(density_matrix))
            return brightness-self.brightness_bound[1]
        brightness_upper_bound.terminal = True

        ensemble_brightness_arr = np.zeros((self.n_config * self.n_sampling, len(time_arr)))
        n = 0
        while n < (self.n_config * self.n_sampling):
            t = time.time()
            if (n % self.n_sampling) == 0:
                config = self.diamond.generate_config()
            initial_vector = self.matrix_to_vector(self.initial_density_matrix)
        
            microwave_Hamiltonian = self.get_microwave_Hamiltonian(resonance_frequency, microwave_parameters)
            dipolar_coef = self.assemble_dipolar_coef(self.sample(config))
            vector_arr = solve_ivp(
                fun = self.get_derivative_of_density_matrix, 
                t_span = (0, time_arr[-1]), 
                y0 = initial_vector, 
                method = 'RK45', 
                t_eval = time_arr, 
                rtol = self.RK45_relative_tolerance, 
                atol = self.RK45_absolute_tolerance, 
                args = (microwave_Hamiltonian, dipolar_coef, decay_rates), 
                events = [brightness_lower_bound, brightness_upper_bound]
            )
            if len(vector_arr.y.T) < len(time_arr):
                print('dvg')
                continue
            ensemble_brightness_arr[n] = np.array([self.density_matrix_to_brightness(self.vector_to_matrix(vector)) for vector in vector_arr.y.T])
            print(f'n = {n+1}: t = {time.time()-t:.3f}')
            n += 1
        return ensemble_brightness_arr

    def set_initial_density_matrix(self, time_arr, brightness_arr, initial_time=None, equilibrium_time=5000, asymmetry=0.5**3):
        if initial_time:
            for time, brightness in zip(time_arr, brightness_arr):
                if time > initial_time:
                    initial_brightness = brightness
                    break
        else:
            for brightness in brightness_arr:
                if brightness < 1.0:
                    initial_brightness = brightness
                    break
        
        equilibrium_brightness = brightness_arr[time_arr > equilibrium_time].mean()
        dark_state = (equilibrium_brightness-1/3)*3/2
        zero_state_population = (initial_brightness-dark_state)/(1-dark_state)*np.ones((4,1))
        plus_state_population = np.clip(a=(1-zero_state_population)*(0.5+random.randn(4,1)*asymmetry), a_min=0, a_max=zero_state_population)
        population = np.concatenate([plus_state_population, zero_state_population, 1-plus_state_population-zero_state_population], axis=1)
        self.initial_density_matrix = np.einsum('...i,...j', population**0.5, population**0.5)

    def assemble_dipolar_coef(self, surroundings):
        coef = np.zeros((4,4,9), dtype=self.dtype)
        for i in range(4):
            ai = self.diamond.generate_quantization_axis(i)
            for j, x, aj in surroundings[i]:
                d = linalg.norm(x)
                x = x / d
                coef_Jr = self.dipolar_intensity / d**3
                coef_Sz = 3*np.dot(x,ai[2])*np.dot(x,aj[2])-np.dot(ai[2],aj[2])
                coef_FF = 0.5*(3*np.dot(x,ai[0])*np.dot(x,aj[0])-np.dot(ai[0],aj[0])+3*np.dot(x,ai[1])*np.dot(x,aj[1])-np.dot(ai[1],aj[1])
                        +1j*(3*np.dot(x,ai[0])*np.dot(x,aj[1])-np.dot(ai[0],aj[1])-3*np.dot(x,ai[1])*np.dot(x,aj[0])+np.dot(ai[1],aj[0])))
                coef_DF = 0.5*(3*np.dot(x,ai[0])*np.dot(x,aj[0])-np.dot(ai[0],aj[0])-3*np.dot(x,ai[1])*np.dot(x,aj[1])+np.dot(ai[1],aj[1])
                        -1j*(3*np.dot(x,ai[0])*np.dot(x,aj[1])-np.dot(ai[0],aj[1])+3*np.dot(x,ai[1])*np.dot(x,aj[0])-np.dot(ai[1],aj[0])))
                
                coef[i,j,0] += coef_Jr*coef_Sz
                coef[i,j,1] += self.degenerates[i,j]*coef_Jr*coef_FF
                coef[i,j,2] += self.degenerates[j,i]*coef_Jr*self.conjugate(coef_FF)
                coef[i,j,3] += self.degenerates[7-i,7-j]*coef_Jr*coef_FF
                coef[i,j,4] += self.degenerates[7-j,7-i]*coef_Jr*self.conjugate(coef_FF)
                coef[i,j,5] += self.degenerates[i,7-j]*coef_Jr*coef_DF
                coef[i,j,6] += self.degenerates[7-j,i]*coef_Jr*self.conjugate(coef_DF)
                coef[i,j,7] += self.degenerates[7-i,j]*coef_Jr*coef_DF
                coef[i,j,8] += self.degenerates[j,7-i]*coef_Jr*self.conjugate(coef_DF)
        return coef

    def density_matrix_to_brightness(self, p):
        return np.einsum('i,...ii', np.array([-0.5,1,-0.5]), abs(p)).mean(axis=0)

    def get_degenerates(self, frequency, decay_rates, n_sampling, degenerate_threshold):
        degenerates = np.zeros((8,8))
        for i in range(8):
            for j in range(8):
                if i==j or i+j==7:
                    frequency_diff = abs(cauchy.rvs(frequency[i],decay_rates[i],n_sampling)-cauchy.rvs(frequency[j],decay_rates[j],n_sampling))
                else:
                    frequency_diff = abs(cauchy.rvs(frequency[i],decay_rates[i],n_sampling)-cauchy.rvs(frequency[j],decay_rates[j],n_sampling))
                degenerates[i,j] = len(frequency_diff[frequency_diff<degenerate_threshold])/n_sampling
        self.degenerates = (degenerates+degenerates.T)/2

    def get_stretched_exponential(self, time_arr, decay_rate):
        return np.exp(-(decay_rate*time_arr)**(1/2))
