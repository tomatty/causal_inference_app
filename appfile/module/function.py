import streamlit as st
import os
import io

import numpy as np
import pandas as pd

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm


class Function:
    def __init__(self, loc=4, scale=0.8, seed=1):
        self.population = stats.norm(loc=loc, scale=scale)
        self.seed = seed
        self.loc = loc
        self.scale = scale
        np.random.seed(self.seed)


    def calc_sample_mean(self, size, iteration):
        sample_mean_array = np.zeros(iteration)
        for i in range(iteration):
            sample_loop = self.population.rvs(size=size)
            sample_mean_array[i] = np.mean(sample_loop)
        return sample_mean_array


    def sample_mean_size(self, max_sample_size):
        size_array = np.arange(start=10, stop=max_sample_size, step=100)
        sample_mean_array_size = np.zeros(len(size_array))
        for i in range(len(size_array)):
            sample_loop = self.population.rvs(size_array[i])
            sample_mean_array_size[i] = np.mean(sample_loop)
        return size_array, sample_mean_array_size


    def sample_var_gen(self, sample_size=10, iteration=10000):
        sample_var_array = np.zeros(iteration)
        for i in range(iteration):
            sample_loop = self.population.rvs(size=sample_size)
            sample_var_array[i] = np.var(sample_loop, ddof=0)
        mean_sample_var = round(np.mean(sample_var_array),3)
        return sample_var_array, mean_sample_var


    def unbiased_var_gen(self, sample_size=10, iteration=10000):
        unbias_var_array = np.zeros(iteration)
        for i in range(iteration):
            sample_loop = self.population.rvs(size=sample_size)
            unbias_var_array[i] = np.var(sample_loop, ddof=1)
        mean_unbias_var = round(np.mean(unbias_var_array),3)
        return unbias_var_array, mean_unbias_var


    def sample_unbias_var(self, max_sample_size):
        size_array = np.arange(start=10, stop=max_sample_size, step=100)

        sample_var_array_size = np.zeros(len(size_array))
        for i in range(len(size_array)):
            sample_loop = self.population.rvs(size_array[i])
            sample_var_array_size[i] = np.var(sample_loop, ddof=0)

        unbias_var_array_size = np.zeros(len(size_array))
        for i in range(len(size_array)):
            sample_loop = self.population.rvs(size_array[i])
            unbias_var_array_size[i] = np.var(sample_loop, ddof=1)

        return size_array, sample_var_array_size, unbias_var_array_size


    def chi_gen(self, n, sigma=0.8, iteration=10000):
        n = n
        sigma = self.scale
        chi2_value_array = np.zeros(iteration)
        for i in range(0, iteration):
            sample = self.population.rvs(size=n)
            u2 = np.var(sample, ddof=1)
            chi2 = (n - 1) * u2 / sigma ** 2
            chi2_value_array[i] = chi2
        x = np.arange(start=0, stop=50.1, step=0.1)
        chi2_distribution = stats.chi2.pdf(x=x, df=n - 1)
        return x, chi2_distribution, chi2_value_array


    def std_sample_mean(self, sample_size, sigma=0.8, iteration=10000):
        n = sample_size
        sigma = self.scale
        z_value_array = np.zeros(iteration)
        for i in range(0, iteration):
            sample = self.population.rvs(size=n)
            x_bar = np.mean(sample)
            bar_sigma = sigma / np.sqrt(n)
            z_value_array[i] = (x_bar - self.loc) / bar_sigma
        x = np.arange(start=-6, stop=6.1, step=0.1)
        z_distribution = stats.norm.pdf(x=x, loc=0, scale=1)
        return x, z_distribution, z_value_array


    def t_dist_gen(self, sample_size, iteration=10000):
        n = sample_size
        t_value_array = np.zeros(iteration)
        for i in range(0, iteration):
            sample = self.population.rvs(size=n)
            x_bar = np.mean(sample)
            u = np.std(sample, ddof=1)
            se = u / np.sqrt(n)
            t_value_array[i] = (x_bar - self.loc) / se
        x = np.arange(start=-6, stop=6.1, step=0.1)
        t_distribution = stats.t.pdf(x=x, df=n - 1)
        return x, t_distribution, t_value_array


    def f_dist_gen(self, m_size, n_size, iteration=10000):
        m = m_size
        n = n_size
        f_value_array = np.zeros(iteration)
        for i in range(0, iteration):
            sample_x = self.population.rvs(m_size)
            sample_y = self.population.rvs(n_size)
            u2_x = np.var(sample_x, ddof=1)
            u2_y = np.var(sample_y, ddof=1)
            f_value_array[i] = u2_x / u2_y
        x = np.arange(start=0, stop=6.1, step=0.1)
        f_distribution = stats.f.pdf(x=x, dfn=m - 1, dfd=n - 1)
        return x, f_distribution, f_value_array
    

    def welch_df(self, u2_bef, u2_aft, m, n):
        numerator = (u2_bef/m + u2_aft/n)**2
        denominator = (u2_bef**2 / (m**2 * (m - 1))) + (u2_aft**2 / (n**2 * (n - 1)))
        return numerator / denominator