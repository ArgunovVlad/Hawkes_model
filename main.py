"""
The code implements the paper "Contemporary LOB: Presence of influential MO"
"""

import random as rd
import math

import csv
import pickle
from scipy.optimize import minimize
from scipy.stats import kstest
import statsmodels.api as sm

# Actions to work with the simulated data
SIMULATE_UNIVARIATE = False  # Simulates the univariate Hawkes on simulated data
ESTIMATE_UNIVARIATE = False  # Estimates the univariate Hawkes on simulated data
UNIVARIATE_TESTS = False  # Performs the specification tests for univariate Hawkes on simulated data

BIVARIATE_SIMULATE = False  # Simulates the bivariate Hawkes on simulated data
BIVARIATE_ESTIMATE = False  # Estimates the bivariate Hawkes on simulated data
BIVARIATE_TESTS = False  # Performs the specification tests for bivariate Hawkes on simulated data

# Actions to work with market data
CONVERT_DATA = False  # Convert the market data for readable for Python format
RETRIEVE_DATA = True  # Retrieve the tractable for Python market data from the "data" folder

ESTIMATE_DATA_BIVARIATE = False  # Estimates the bivariate Hawkes on market data
RETRIEVE_ESTIMATES_BIVARIATE = False  # Retrieve the estimated coefficients of bivariate model
TESTS_BIVARIATE = False  # Performs the specification tests for bivariate Hawkes on market data
# Note, in order to perform tests on bivariate data, both RETRIEVE_DATA
# and RETRIEVE_ESTIMATES_DATA_BIVARIATE must be true.

ESTIMATE_DATA_UNIVARIATE_BUY = False  # Estimates the univariate Hawkes (buy-side) on market data
ESTIMATE_DATA_UNIVARIATE_SELL = False  # Estimates the univariate Hawkes (sell-side) on market data
RETRIEVE_ESTIMATES_UNIVARIATE = False  # Retrieve the estimated coefficients of univariate model
TEST_UNIVARIATE_DATA_BUY = False  # Performs the specification tests for univariate Hawkes (buy-side) on market data
TEST_UNIVARIATE_DATA_SELL = False  # Performs the specification tests for univariate Hawkes (sell-side) on market data
# Note, in order to perform tests on univariate data, both RETRIEVE_DATA
# and RETRIEVE_ESTIMATES_DATA_UNIVARIATE must be true.


"""
This section implements the simulation and estimation of the univariate Hawkes process.
"""


def intensity_exponential_kernel(mu, alpha, beta, history, tn):
    """
    Returns the conditional intensity of a process given a deterministic function and
    a kernel
    :param mu: Deterministic function of the intensity
    :param alpha: Parameter alpha of the kernel
    :param beta: Parameter beta of the kernel
    :param history: Simulation history of the process
    :param tn: Time at which intensity is estimated
    :return: Current intensity of the Hawkes process
    """
    return mu + sum([alpha * math.exp(- beta * (tn - event_time)) for event_time in history
                     if tn >= event_time])


def sim_univariate_hawkes_exp_kernel(mu, alpha, beta, tn):
    """
    Simulates the univariate Hawkes process using some deterministic function
    :param mu: Deterministic function of the intensity of Hawkes process
    :param alpha: Parameter needed for kernel
    :param beta: Parameter needed for kernel
    :param tn: Simulation time
    :return: History of a simulated counting process
    """
    # Define the structure which keeps the simulated results
    simulation = []
    t = 0
    while t < tn:
        # Set the upper bound for intensity following the specified kernel
        intensity = intensity_exponential_kernel(mu, alpha, beta, simulation, t)
        # Sample the inter-arrival time
        u = rd.random()
        dt = - math.log(u) / intensity
        # Update the current time
        t += dt
        # Determine whether to accept the event
        s = rd.uniform(0, intensity)
        acceptance_level = intensity_exponential_kernel(mu, alpha, beta, simulation, t)
        if s < acceptance_level:
            simulation.append(t)
    return simulation


def term_a_univariate_hawkes(data, stop_time, beta):
    """
    Computes term A in MLE of univariate Hawkes process.
    :param data:
    :param stop_time:
    :param beta:
    :return:
    """
    a = 0
    event_number = 0
    time = data[0]
    while time < stop_time:
        a += math.exp(- beta * (stop_time - time))
        event_number += 1
        time = data[event_number]
    return a


def term_b_univariate_hawkes(data, stop_time, beta):
    """
    Computes term B in MLE of univariate Hawkes process.
    :param data:
    :param stop_time:
    :param beta:
    :return:
    """
    b = 0
    event_number = 0
    time = data[0]
    while time < stop_time:
        b += (stop_time - time) * math.exp(- beta * (stop_time - time))
        event_number += 1
        time = data[event_number]
    return b


def gradient_alpha_univariate_hawkes(data, mu, beta):
    """
    The function computes the partial derivative of alpha of a univariate hawkes process
    List of floats :param data:
    Float :param mu:
    Float :param beta:
    Float :return:
    """
    tn = data[-1]
    sum_1 = 0
    sum_2 = 0
    for i in range(len(data)):
        time = data[i]
        sum_1 += math.exp(- beta * (tn - time)) - 1
        sum_2 += (term_a_univariate_hawkes(data, time, beta) / (mu + term_a_univariate_hawkes(data, time, beta)))
    return (1 / beta) * sum_1 + sum_2


def gradient_beta_univariate_hawkes(data, mu, beta, alpha):
    """
    The function computes the partial derivative of beta of a univariate hawkes process
    :param data:
    :param mu:
    :param beta:
    :param alpha:
    :return:
    """
    tn = data[-1]
    sum_1 = 0
    sum_2 = 0
    for i in range(len(data)):
        time = data[i]
        sum_1 += ((1 / beta) * (tn - time) + (1 / (beta ** 2))) * math.exp(- beta * (tn - time))
        sum_2 += (alpha * term_b_univariate_hawkes(data, time, beta)) / (
                mu + alpha * term_a_univariate_hawkes(data, time, beta))
    return - alpha * sum_1 - sum_2


def gradient_mu_univariate_hawkes(data, mu, beta, alpha):
    """
    The function computes the partial derivative of beta of a univariate hawkes process
    :param data:
    :param mu:
    :param beta:
    :param alpha:
    :return:
    """
    tn = data[-1]
    sum_1 = 0
    for i in range(len(data)):
        time = data[i]
        sum_1 += (1 / (mu + alpha * term_a_univariate_hawkes(data, time, beta)))
    return - tn + sum_1


def likelihood_uni_exp(data, mu, alpha, beta):
    if (mu <= 0) or (alpha <= 0) or (beta <= 0):
        return float("-inf")
    tn = data[-1]
    sum_1 = 0
    sum_2 = 0
    for i in range(len(data)):
        time = data[i]
        sum_1 += math.exp(- beta * (tn - time)) - 1
        check_sum_2 = mu + alpha * term_a_univariate_hawkes(data, time, beta)
        if check_sum_2 <= 0:
            return float("-inf")
        sum_2 += math.log(check_sum_2)
    return - mu * tn + (alpha / beta) * sum_1 + sum_2


def negative_likelihood_uni_exp(data, mu, alpha, beta):
    return -likelihood_uni_exp(data, mu, alpha, beta)


def maximise_likelihood_uni_hawkes(times, initial_guess):
    """
    Maximises the conditional likelihood of the univariate Hawkes.
    :param times: The univariate Hawkes process of times
    :param initial_guess: Initial guess for the solver
    :return:
    """

    def negative_likelihood_uni_exp_scipy(scipy_x):
        mu = scipy_x[0]
        alpha = scipy_x[1]
        beta = scipy_x[2]
        return negative_likelihood_uni_exp(times, mu, alpha, beta)

    def jacobian_mu_scipy(scipy_x):
        mu = scipy_x[0]
        alpha = scipy_x[1]
        beta = scipy_x[2]
        return - gradient_mu_univariate_hawkes(times, mu, beta, alpha)

    def jacobian_alpha_scipy(scipy_x):
        mu = scipy_x[0]
        beta = scipy_x[2]
        return - gradient_alpha_univariate_hawkes(times, mu, beta)

    def jacobian_beta_scipy(scipy_x):
        mu = scipy_x[0]
        alpha = scipy_x[1]
        beta = scipy_x[2]
        return - gradient_beta_univariate_hawkes(times, mu, beta, alpha)

    results = minimize(fun=negative_likelihood_uni_exp_scipy,
                       x0=initial_guess,
                       method="BFGS",
                       jac=[jacobian_mu_scipy, jacobian_alpha_scipy, jacobian_beta_scipy])
    return results


"""
The section implements simulation and estimation of the bivariate Hawkes process. 
"""


def compute_exponential_kernel_constant_base_bivariate(coefficients, history, t, number_process):
    """
    Computes the intensity of the bivariate Hawkes process
    :param number_process: The number of the Hawkes process considered.
    :param coefficients: Row of coefficients of the form [mu, beta, alpha_same, alpha_diff]
    :param history: History of events
    :param t: Current time
    :return: Value of intensity
    """
    if number_process == 1:
        current_process = history[0]
        other_process = history[1]
    if number_process == 2:
        current_process = history[1]
        other_process = history[0]
    assert len(coefficients) == 4, "Coefficients for kernel are given incorrectly."
    intensity = coefficients[0] + sum([coefficients[2] *
                                       math.exp(- coefficients[1] *
                                                (t - event_time)) for event_time in current_process
                                       if t >= event_time])
    intensity += sum([coefficients[3] *
                      math.exp(- coefficients[1] *
                               (t - event_time)) for event_time in other_process
                      if t >= event_time])
    return intensity


def sim_bivariate_hawkes(coefficients, tn):
    """
    Simulates the bivariate Hawkes.
    :param coefficients: It is the list of the form
    [mu1, beta1, alpha11, alpha12, mu2, beta2, alpha22, alpha21]
    :param tn: Final time
    :return:
    """
    simulation = [[], []]
    coefficients1 = coefficients[:4]
    coefficients2 = coefficients[4:]
    s = 0
    while s < tn:
        lambda_bar = sum([compute_exponential_kernel_constant_base_bivariate(coefficients1, simulation, s, 1),
                          compute_exponential_kernel_constant_base_bivariate(coefficients2, simulation, s, 2)])
        u = rd.random()
        w = - math.log(u) / lambda_bar
        s += w
        d = rd.random()
        intensities = [compute_exponential_kernel_constant_base_bivariate(coefficients1, simulation, s, 1),
                       compute_exponential_kernel_constant_base_bivariate(coefficients2, simulation, s, 2)]
        sum_intensities = sum(intensities)
        d_lambda_bar = d * lambda_bar
        if d_lambda_bar <= sum_intensities:
            k = 0
            if d_lambda_bar > intensities[0]:
                k = 1
            simulation[k].append(s)

    return simulation


def gradient_mu(data, mu, beta, alpha_same, alpha_diff, part_likelihood):
    """
    Calculates the partial derivative of mu from likelihood function.
    :param data:
    :param mu:
    :param beta:
    :param alpha_same:
    :param alpha_diff:
    :param part_likelihood:
    :return:
    """
    if part_likelihood == 1:
        current_data = data[0]
        other_data = data[1]
    if part_likelihood == 2:
        current_data = data[1]
        other_data = data[0]

    tn = max(current_data[-1], other_data[-1])

    final_summation = 0
    for event in current_data[1:]:
        summation_same = [math.exp(- beta * (event - previous_event))
                          for previous_event in current_data if previous_event < event]
        summation_diff = [math.exp(- beta * (event - previous_event))
                          for previous_event in other_data if previous_event < event]
        final_summation += (mu + alpha_same * sum(summation_same) + alpha_diff * sum(summation_diff)) ** (-1)

    return - tn + final_summation


def gradient_alpha_same(data, mu, beta, alpha_same, alpha_diff, part_likelihood):
    """
    Calculates the partial derivative of alpha_same of the likelihood function.
    :param data:
    :param mu:
    :param beta:
    :param alpha_same:
    :param alpha_diff:
    :param part_likelihood:
    :return:
    """
    if part_likelihood == 1:
        current_data = data[0]
        other_data = data[1]
    if part_likelihood == 2:
        current_data = data[1]
        other_data = data[0]

    tn = max(current_data[-1], other_data[-1])
    summation_1 = [(1 - math.exp(- beta * (tn - event))) for event in current_data]

    summation_2 = 0
    for event in current_data[1:]:
        summation_same = [math.exp(- beta * (event - previous_event))
                          for previous_event in current_data if previous_event < event]
        summation_diff = [math.exp(- beta * (event - previous_event))
                          for previous_event in other_data if previous_event < event]
        new_term = (mu + alpha_same * sum(summation_same) + alpha_diff * sum(summation_diff)) ** (-1)
        new_term *= sum(summation_same)
        summation_2 += new_term

    return (- 1 / beta) * sum(summation_1) + summation_2


def gradient_alpha_diff(data, mu, beta, alpha_same, alpha_diff, part_likelihood):
    """
    Calculates the partial derivative of alpha_diff from the likelihood function.
    :param data:
    :param mu:
    :param beta:
    :param alpha_same:
    :param alpha_diff:
    :param part_likelihood:
    :return:
    """
    if part_likelihood == 1:
        current_data = data[0]
        other_data = data[1]
    if part_likelihood == 2:
        current_data = data[1]
        other_data = data[0]

    tn = max(current_data[-1], other_data[-1])
    summation_1 = [(1 - math.exp(- beta * (tn - event))) for event in other_data]

    summation_2 = 0
    for event in current_data[1:]:
        summation_same = [math.exp(- beta * (event - previous_event))
                          for previous_event in current_data if previous_event < event]
        summation_diff = [math.exp(- beta * (event - previous_event))
                          for previous_event in other_data if previous_event < event]
        new_term = (mu + alpha_same * sum(summation_same) + alpha_diff * sum(summation_diff)) ** (-1)
        new_term *= sum(summation_diff)
        summation_2 += new_term

    return (- 1 / beta) * sum(summation_1) + summation_2


def gradient_beta(data, mu, beta, alpha_same, alpha_diff, part_likelihood):
    """
    Calculates the partial derivative of beta from the likelihood function.
    :param data:
    :param mu:
    :param beta:
    :param alpha_same:
    :param alpha_diff:
    :param part_likelihood:
    :return:
    """
    if part_likelihood == 1:
        current_data = data[0]
        other_data = data[1]
    if part_likelihood == 2:
        current_data = data[1]
        other_data = data[0]

    tn = max(current_data[-1], other_data[-1])

    summation_1 = [((1 + beta * (tn - event)) * math.exp(- beta * (tn - event)) - 1) for event in current_data]
    summation_2 = [((1 + beta * (tn - event)) * math.exp(- beta * (tn - event)) - 1) for event in other_data]
    summation_3 = 0

    for event in current_data[1:]:
        summation_same = [math.exp(- beta * (event - previous_event))
                          for previous_event in current_data if previous_event < event]
        summation_same_beta = [(previous_event - event) * math.exp(- beta * (event - previous_event))
                               for previous_event in current_data if previous_event < event]
        summation_diff = [math.exp(- beta * (event - previous_event))
                          for previous_event in other_data if previous_event < event]
        summation_diff_beta = [(previous_event - event) * math.exp(- beta * (event - previous_event))
                               for previous_event in other_data if previous_event < event]
        new_term = (mu + alpha_same * sum(summation_same) + alpha_diff * sum(summation_diff)) ** (-1)
        new_term *= alpha_same * sum(summation_same_beta) + alpha_diff * sum(summation_diff_beta)
        summation_3 += new_term
    return (- alpha_same) * (beta ** (-2)) * sum(summation_1) + (- alpha_diff) * (beta ** (-2)) * sum(
        summation_2) + summation_3


def likelihood(data, mu, beta, alpha_same, alpha_diff, part_likelihood):
    """
    Calculates the partial likelihood function of the bivariate Hawkes process.
    :param data:
    :param mu:
    :param beta:
    :param alpha_same:
    :param alpha_diff:
    :param part_likelihood:
    :return:
    """
    if mu <= 0:
        return float("-inf")
    if beta <= 0:
        return float("-inf")
    if alpha_same < 0:
        return float("-inf")
    if alpha_diff < 0:
        return float("-inf")

    if part_likelihood == 1:
        current_data = data[0]
        other_data = data[1]
    if part_likelihood == 2:
        current_data = data[1]
        other_data = data[0]

    tn = max(current_data[-1], other_data[-1])

    summation_1 = [(1 - math.exp(- beta * (tn - event))) for event in current_data]
    summation_2 = [(1 - math.exp(- beta * (tn - event))) for event in other_data]

    summation_3 = 0
    for event in current_data[1:]:
        summation_same = [math.exp(- beta * (event - previous_event))
                          for previous_event in current_data if previous_event < event]
        summation_diff = [math.exp(- beta * (event - previous_event))
                          for previous_event in other_data if previous_event < event]
        summation_3 += math.log(mu + alpha_same * sum(summation_same) + alpha_diff * sum(summation_diff))

    return - mu * tn - (alpha_same / beta) * sum(summation_1) - (alpha_diff / beta) * sum(summation_2) + summation_3


"""
Computational methods to estimate the maximised likelihood.
"""


def compute_likelihood_exp_bivariate_partial(data, coefficients, part):
    """
    Computes the likelihood given the coefficients for only one of the two processes of the
    bivariate Hawkes.
    :param part: Part of the likelihood function, can take either values of 1 or 2.
    :param coefficients: It is list of the form
    [mu, beta, alpha_same, alpha_diff]
    Note that in this computation it is assumed that B11 = B12 and B21 = B22
    :param data:
    :return:
    """
    lhd = likelihood(mu=coefficients[0],
                     beta=coefficients[1],
                     alpha_same=coefficients[2],
                     alpha_diff=coefficients[3],
                     data=data,
                     part_likelihood=part
                     )

    return lhd


def maximise_likelihood_scipy_bivariate_partial(data, initial_guess, part_likelihood):
    """
    Maximises the likelihood function for the bivariate hawkes given the initial guess.
    :param part_likelihood: Part of the likelihood function, can take either values of 1 or 2.
    :param data: Data of the events of two bivariate Hawkes processes.
    :param initial_guess: Initial guess of the form
    [mu, beta, alpha_same, alpha_diff]
    :return:
    """
    assert len(initial_guess) == 4, "Incorrect form of the initial guess."

    def negative_likelihood_bivariate_scipy(x_scipy):
        return - compute_likelihood_exp_bivariate_partial(data, x_scipy, part_likelihood)

    def negative_gradient_mu_scipy(x_scipy):
        return - gradient_mu(data=data,
                             mu=x_scipy[0],
                             beta=x_scipy[1],
                             alpha_same=x_scipy[2],
                             alpha_diff=x_scipy[3],
                             part_likelihood=part_likelihood
                             )

    def negative_gradient_beta_scipy(x_scipy):
        return - gradient_beta(data=data,
                               mu=x_scipy[0],
                               beta=x_scipy[1],
                               alpha_same=x_scipy[2],
                               alpha_diff=x_scipy[3],
                               part_likelihood=part_likelihood
                               )

    def negative_gradient_alpha_same_scipy(x_scipy):
        return - gradient_alpha_same(data=data,
                                     mu=x_scipy[0],
                                     beta=x_scipy[1],
                                     alpha_same=x_scipy[2],
                                     alpha_diff=x_scipy[3],
                                     part_likelihood=part_likelihood
                                     )

    def negative_gradient_alpha_diff_scipy(x_scipy):
        return - gradient_alpha_diff(data=data,
                                     mu=x_scipy[0],
                                     beta=x_scipy[1],
                                     alpha_same=x_scipy[2],
                                     alpha_diff=x_scipy[3],
                                     part_likelihood=part_likelihood
                                     )

    results = minimize(fun=negative_likelihood_bivariate_scipy,
                       x0=initial_guess,
                       method='BFGS',
                       jac=[negative_gradient_mu_scipy,
                            negative_gradient_beta_scipy,
                            negative_gradient_alpha_same_scipy,
                            negative_gradient_alpha_diff_scipy])
    return results


"""
This section implements the specification tests for Hawkes processes.
"""


def compute_integrated_residual_uni(data, coefficients, tn):
    """
    Computes the integrated residuals of a univariate Hawkes process.
    :param data: [mu, beta, alpha]
    :param coefficients:
    :param tn:
    :return:
    """
    return coefficients[0] * tn - sum([coefficients[2] *
                                       math.exp(- coefficients[1] * (tn - event_time)) / coefficients[1]
                                       for event_time in data if tn >= event_time])


def compute_integrated_residual_biv(data, coefficients, number_process, t1, t2):
    """
    Computes the integrated residuals of a bivariate Hawkes process.
    :param data: [mu, beta, alpha_same, alpha_diff]
    :param coefficients:
    :param tn:
    :return:
    """
    residual_final = 0
    if number_process == 1:
        current_process = data[0]
        other_process = data[1]
    if number_process == 2:
        current_process = data[1]
        other_process = data[0]
    # All events that appeared within the timespan in current process
    restricted_current_process = [i for i in current_process if i > t1 and i <= t2]
    # All events that appeared within the timespan in other process
    restricted_other_process = [i for i in other_process if i > t1 and i < t2]
    assert len(coefficients) == 4, "Coefficients for kernel are given incorrectly."
    t01 = t1
    while restricted_current_process != []:
        if restricted_other_process != []:
            if restricted_current_process[0] < restricted_other_process[0]:
                t02 = restricted_current_process[0]
                restricted_current_process.pop(0)
                adjustment_factor = coefficients[2] / coefficients[1]
            if restricted_current_process[0] > restricted_other_process[0]:
                t02 = restricted_other_process[0]
                restricted_other_process.pop(0)
                adjustment_factor = coefficients[3] / coefficients[1]
        else:
            t02 = restricted_current_process[0]
            restricted_current_process.pop(0)
            adjustment_factor = coefficients[2] / coefficients[1]

        residual01 = coefficients[0] * t01 - (1 / coefficients[1]) * sum([coefficients[2] *
                                                                          math.exp(- coefficients[1] *
                                                                                   (t01 - event_time)) for event_time in
                                                                          current_process
                                                                          if t01 >= event_time])
        residual01 += (- 1 / coefficients[1]) * sum([coefficients[3] *
                                                     math.exp(- coefficients[1] *
                                                              (t01 - event_time)) for event_time in other_process
                                                     if t01 >= event_time])

        residual02 = coefficients[0] * t02 - (1 / coefficients[1]) * sum([coefficients[2] *
                                                                          math.exp(- coefficients[1] *
                                                                                   (t02 - event_time)) for event_time
                                                                          in current_process if t02 >= event_time])
        residual02 += (- 1 / coefficients[1]) * sum([coefficients[3] *
                                                     math.exp(- coefficients[1] *
                                                              (t02 - event_time)) for event_time in other_process
                                                     if t02 >= event_time])
        t01 = t02

        residual_final += (residual02 - residual01 + adjustment_factor)

    return residual_final


def lb_test_univariate(data, coefficients, lags):
    """
    Performs the Ljung-Box test on the normalised to exp(1) time intervals
    from Hawkes process.
    Null hypothsesis: The distributions are independent
    Alternative hypothesis: The distributions are not independent
    :param coefficients:
    :param data: One series of event timestamps
    [mu, beta, alpha]
    :param lags: Number of lags involved in LB test
    :return:
    """
    integrated_residuals = [compute_integrated_residual_uni(data, coefficients, t) for t in data]
    adjusted_durations = [integrated_residuals[i + 1] - integrated_residuals[i]
                          + coefficients[2] / coefficients[1]
                          for i in range(len(integrated_residuals) - 1)]
    # Perform the LB test
    result = sm.stats.acorr_ljungbox(adjusted_durations, lags=[lags], return_df=True)
    return result['lb_pvalue'].iloc[0]


def lb_test_bivariate(data, coefficients, lags):
    """
    Performs the Ljung-Box test on the normalised to exp(1) time intervals
    from Hawkes process.
    Null hypothsesis: The distributions are independent
    Alternatice hypothesis: The distributions are not independent
    :param coefficients:
    [mu, beta, alpha_same, alpha_diff]
    :param data: One series of event timestamps
    :param lags: Number of lags involved in LB test
    :return:
    """
    # Compute the integrated residuals
    integrated_residuals = [compute_integrated_residual_biv(data, coefficients[:4], 1,
                                                            data[0][i], data[0][i + 1])
                            for i in range(len(data[0]) - 1)]

    integrated_residuals += [compute_integrated_residual_biv(data, coefficients[4:], 2,
                                                             data[1][i], data[1][i + 1])
                             for i in range(len(data[1]) - 1)]
    # Perform the LB test
    result = sm.stats.acorr_ljungbox(integrated_residuals, lags=[lags], return_df=True)
    return result['lb_pvalue'].iloc[0]


def ks_test_univariate(data, coefficients):
    """
    Performs the Kolmogorov-Smirnov test on the integrated residuals
    from Hawkes process to exp(1)
    Null hypothsesis: The data follows an exp(1) distribution
    Alternative hypothesis: The data does not follow the exp(1) distribution
    :param coefficients: [mu, beta, alpha]
    :param data: One series of event timestamps
    :return:
    """
    # Compute the time residuals between each timestamp
    integrated_residuals = [compute_integrated_residual_uni(data, coefficients, t) for t in data]
    adjusted_durations = [integrated_residuals[i + 1] - integrated_residuals[i]
                          + coefficients[2] / coefficients[1]
                          for i in range(len(integrated_residuals) - 1)]
    # Perform the KS test
    result = kstest(adjusted_durations, 'expon')
    return result[1]


def ks_test_bivariate(data, coefficients):
    """
    Performs the Kolmogorov-Smirnov test on the integrated residuals
    from Hawkes process to exp(1)
    Null hypothsesis: The data follows an exp(1) distribution
    Alternative hypothesis: The data does not follow the exp(1) distribution
    :param coefficients: [mu, beta, alpha_same, alpha_diff]
    :param data: Two series of event timestamps
    :return:
    """
    # Compute the integrated residuals
    integrated_residuals = [compute_integrated_residual_biv(data, coefficients[:4], 1,
                                                            data[0][i], data[0][i + 1])
                            for i in range(len(data[0]) - 1)]

    integrated_residuals += [compute_integrated_residual_biv(data, coefficients[4:], 2,
                                                             data[1][i], data[1][i + 1])
                             for i in range(len(data[1]) - 1)]

    # Perform the KS test
    result = kstest(integrated_residuals, 'expon')
    return result[1]


def compare_likelihoods_bivariate(data, coefficients_true, coefficients_estimated):
    """
    Compares the values of true (if known) and estimated likelihoods.
    :param data:
    :param coefficients_true:
    :param coefficients_estimated:
    :return:
    """
    true_likelihood1 = compute_likelihood_exp_bivariate_partial(data, coefficients_true, 1)
    true_likelihood2 = compute_likelihood_exp_bivariate_partial(data, coefficients_true, 2)
    true_likelihood = true_likelihood1 + true_likelihood2
    estimated_likelihood1 = compute_likelihood_exp_bivariate_partial(data, coefficients_estimated, 1)
    estimated_likelihood2 = compute_likelihood_exp_bivariate_partial(data, coefficients_estimated, 2)
    estimated_likelihood = estimated_likelihood1 + estimated_likelihood2
    comparison = {"True log-likelihood": true_likelihood, "Estimated log-likelihood": estimated_likelihood}
    return comparison


def stationarity_condition(coefficients):
    """
    Computes whether the bivariate Hawkes process is stationary
    :param coefficients: List of the form
    [mu1, beta1, alpha11, alpha12, mu2, beta2, alpha22, alpha21]
    :return: Boolean True if the process is stationary,
    false if non-stationary
    """
    stationarity_factor = .5 * (coefficients[2] / coefficients[1] + coefficients[6] / coefficients[5]
                                + ((coefficients[2] / coefficients[1] - coefficients[6] / coefficients[5]) ** 2
                                   + 4 * (coefficients[3] / coefficients[1]) * (
                                           coefficients[7] / coefficients[5])) ** .5)
    return tuple([stationarity_factor, stationarity_factor < 1])


"""
This section implements the simulation and estimation of the Univariate Hawkes process.
"""

if SIMULATE_UNIVARIATE:
    # Set the coefficients for the simulation
    MU = .3
    ALPHA = .7
    BETA = 1.7
    COEFFICIENTS_SIMULATED_UNIVARIATE = [MU, BETA, ALPHA]
    # Simulate the univariate Hawkes
    DATA_SIMULATED_UNIVARIATE = sim_univariate_hawkes_exp_kernel(mu=COEFFICIENTS_SIMULATED_UNIVARIATE[0],
                                                                 beta=COEFFICIENTS_SIMULATED_UNIVARIATE[1],
                                                                 alpha=COEFFICIENTS_SIMULATED_UNIVARIATE[2],
                                                                 tn=60 * 20)
    print("Univariate Hawkes: \n"
          "Length of the Univariate Hawkes process is {}".format(len(DATA_SIMULATED_UNIVARIATE)))

if ESTIMATE_UNIVARIATE:
    """
    Code to perform the estimation of the parameters of the Bivariate Hawkes.
    """
    # Set the initial guess for the parameters in the form [mu, beta, alpha].
    INITIAL_GUESS_UNIVARIATE = [.1 for i in range(3)]

    # Estimate the log-likelihood Univariate Hawkes and report the results.

    RESULTS_UNIVARIATE = maximise_likelihood_uni_hawkes(times=DATA_SIMULATED_UNIVARIATE,
                                                        initial_guess=INITIAL_GUESS_UNIVARIATE)

    VALUES_UNIVARIATE = {
        "MU": (RESULTS_UNIVARIATE.x[0], MU),
        "ALPHA": (RESULTS_UNIVARIATE.x[1], ALPHA),
        "BETA": (RESULTS_UNIVARIATE.x[2], BETA)
    }

"""
This section implements the simulation and estimation of the Bivariate Hawkes process.
"""

if BIVARIATE_SIMULATE:
    """
    Code to perform the simulation of the Bivariate Hawkes.
    """

    # Set the coefficients for the simulation
    MU1 = .4
    BETA1 = 8
    ALPHA_SAME1 = 1
    ALPHA_DIFF1 = .5
    MU2 = .3
    BETA2 = 8.5
    ALPHA_SAME2 = 1.3
    ALPHA_DIFF2 = .7
    COEFFICIENTS_SIMULATED_BIVARIATE = [MU1, BETA1, ALPHA_SAME1, ALPHA_DIFF1, MU2, BETA2, ALPHA_SAME2, ALPHA_DIFF2]

    # Simulate the bivariate Hawkes
    DATA_SIMULATED_BIVARIATE = sim_bivariate_hawkes(COEFFICIENTS_SIMULATED_BIVARIATE, tn=60 * 60)
    print("Bivariate Hawkes: \n"
          "Length of the first Hawkes process is {} \n"
          "Length of the second Hawkes process is {}.".format(len(DATA_SIMULATED_BIVARIATE[0]),
                                                              len(DATA_SIMULATED_BIVARIATE[1])))

if BIVARIATE_ESTIMATE:
    """
    Code to perform the estimation of the parameters of the Bivariate Hawkes.
    """

    # Set the initial guess for the parameters in the form:
    # [mu1, beta1, alpha11, alpha12, mu2, beta2, alpha22, alpha21]
    INITIAL_GUESS_BIVARIATE = 2 * [.1, .1, .1, .1]
    INITIAL_GUESS_FIRST_PART = INITIAL_GUESS_BIVARIATE[:4]
    INITIAL_GUESS_SECOND_PART = INITIAL_GUESS_BIVARIATE[4:]

    """
    Set which part of the likelihood you need to estimate.
    BOTH = 0 : Estimates both parts of the partial likelihoods.
    ONLY_FIRST = 1 : Estimates only the first part of the likelihood
    ONLY_SECOND = 2 : Estimates only the second part of the likelihood
    """
    LIKELIHOOD_ESTIMATION = 0

    # Estimate the log-likelihood Univariate Hawkes and report the results.
    if LIKELIHOOD_ESTIMATION == 0 or LIKELIHOOD_ESTIMATION == 1:
        """
        Estimate the parameters for the first part of the Bivariate Hawkes and report the results.
        """
        RESULTS_FIRST_PART = maximise_likelihood_scipy_bivariate_partial(data=DATA_SIMULATED_BIVARIATE,
                                                                         initial_guess=INITIAL_GUESS_FIRST_PART,
                                                                         part_likelihood=1)
        print("Estimation of the first part of the Hawkes process is completed.")
        VALUES_FIRST_PART = {
            "MU": (RESULTS_FIRST_PART.x[0], MU1),
            "BETA": (RESULTS_FIRST_PART.x[1], BETA1),
            "ALPHA_SAME": (RESULTS_FIRST_PART.x[2], ALPHA_SAME1),
            "ALPHA_DIFF": (RESULTS_FIRST_PART.x[3], ALPHA_DIFF1)
        }

    """
    Estimate the parameters for the second part of the Bivariate Hawkes and report the results.
    """
    if LIKELIHOOD_ESTIMATION == 0 or LIKELIHOOD_ESTIMATION == 2:
        RESULTS_SECOND_PART = maximise_likelihood_scipy_bivariate_partial(data=DATA_SIMULATED_BIVARIATE,
                                                                          initial_guess=INITIAL_GUESS_SECOND_PART,
                                                                          part_likelihood=2)
        print("Estimation of the second part of the Hawkes process is completed.")
        VALUES_FIRST_PART = {
            "MU": (RESULTS_SECOND_PART.x[0], MU2),
            "BETA": (RESULTS_SECOND_PART.x[1], BETA2),
            "ALPHA_SAME": (RESULTS_SECOND_PART.x[2], ALPHA_SAME2),
            "ALPHA_DIFF": (RESULTS_SECOND_PART.x[3], ALPHA_DIFF2)
        }

    if LIKELIHOOD_ESTIMATION == 0:
        VALUES_BOTH_PARTS = {
            "MU1": (RESULTS_FIRST_PART.x[0], MU1),
            "BETA1": (RESULTS_FIRST_PART.x[1], BETA1),
            "ALPHA_SAME1": (RESULTS_FIRST_PART.x[2], ALPHA_SAME1),
            "ALPHA_DIFF1": (RESULTS_FIRST_PART.x[3], ALPHA_DIFF1),
            "MU2": (RESULTS_SECOND_PART.x[0], MU2),
            "BETA2": (RESULTS_SECOND_PART.x[1], BETA2),
            "ALPHA_SAME2": (RESULTS_SECOND_PART.x[2], ALPHA_SAME2),
            "ALPHA_DIFF2": (RESULTS_SECOND_PART.x[3], ALPHA_DIFF2)
        }
        COEFFICIENTS_ESTIMATED = [item[1][0] for item in VALUES_BOTH_PARTS.items()]
        likelihoods = compare_likelihoods_bivariate(DATA_SIMULATED_BIVARIATE,
                                                    COEFFICIENTS_SIMULATED_BIVARIATE,
                                                    COEFFICIENTS_ESTIMATED)
        print("True log-likelihood is {}. \n"
              "Estimated log-likelihood is {}. \n".format(likelihoods["True log-likelihood"],
                                                          likelihoods["Estimated log-likelihood"]))

"""
This section implements the specification testing of the Hawkes processes.
"""

if UNIVARIATE_TESTS:
    TEST_DATA_UNIVARIATE = DATA_SIMULATED_UNIVARIATE
    # Calculates the p_value of Ljung-Box test data from the TEST_DATA_UNIVARIATE
    COEFFICIENTS_TESTS_UNIVARIATE = COEFFICIENTS_SIMULATED_UNIVARIATE
    NUMBER_OF_LAGS_LB_UNIVARIATE = 20
    univariate_lb_test_p_value = lb_test_univariate(TEST_DATA_UNIVARIATE,
                                                    COEFFICIENTS_TESTS_UNIVARIATE,
                                                    NUMBER_OF_LAGS_LB_UNIVARIATE)
    # Calculates the Kolmogorov-Smirnov test from the TEST_DATA_UNIVARIATE
    univariate_ks_test_p_value = ks_test_univariate(TEST_DATA_UNIVARIATE,
                                                    COEFFICIENTS_TESTS_UNIVARIATE)

if BIVARIATE_TESTS:
    TEST_DATA_BIVARIATE = DATA_SIMULATED_BIVARIATE
    # Calculates the p_value of Ljung-Box test data from the TEST_DATA_BIVARIATE
    NUMBER_OF_LAGS_LB_BIVARIATE = 20
    bivariate_lb_test_p_value = lb_test_bivariate(TEST_DATA_BIVARIATE,
                                                  COEFFICIENTS_ESTIMATED,
                                                  NUMBER_OF_LAGS_LB_BIVARIATE)
    print("P-value of the Ljung-Box test is {}.".format(bivariate_lb_test_p_value))
    # Calculates the Kolmogorov-Smirnov test from the TEST_DATA_BIVARIATE
    bivariate_ks_test_p_value = ks_test_bivariate(TEST_DATA_BIVARIATE,
                                                  COEFFICIENTS_ESTIMATED)
    print("P-value of the Kolmogorov-Smirnov test is {}.".format(bivariate_ks_test_p_value))
    # Calculates the stationarity condition and reports whether the bivariate Hawkes is stationary
    bivariate_stationarity = stationarity_condition(COEFFICIENTS_ESTIMATED)
    print("Stationarity condition of the bivariate Hawkes: {}".format(bivariate_stationarity[1]))

"""
This section adjust the data for the Python readable format
"""

if CONVERT_DATA:
    SAMPLE_FILES = {"Apple": "20210816_AAPL.csv",
                    "Amazon": "20210816_AMZN.csv",
                    "Boeing": "20210816_BA.csv",
                    "Bank of America": "20210816_BAC.csv",
                    "ConocoPhillips": "20210816_COP.csv",
                    "Credit Suisse": "20210816_CS.csv",
                    "Facebook": "20210816_FB.csv",
                    "Goldman Sachs": "20210816_GS.csv",
                    "IBM": "20210816_IBM.csv",
                    "JP Morgan": "20210816_JPM.csv",
                    "Coca-Cola": "20210816_KO.csv",
                    "McDonalds": "20210816_MCD.csv",
                    "Morgan Stanley": "20210816_MS.csv",
                    "Netflix": "20210816_NFLX.csv",
                    "Pfizer": "20210816_PFE.csv",
                    "Starbucks": "20210816_SBUX.csv",
                    "Tesla": "20210816_TSLA.csv",
                    "Twitter": "20210816_TWTR.csv",
                    "Walmart": "20210816_WMT.csv"}
    SAMPLE_DATA = {}
    SAMPLE_DATA_LENGTH = {}
    for sample in SAMPLE_FILES.items():
        path = "/Users/vladargunov/PycharmProjects/Contemp_LOB/data/" + sample[1]
        with open(path, newline='') as f:
            reader = csv.reader(f)
            values = list(reader)
        ticks = [[], []]  # First list contains the buy orders, second list contains sell orders.
        starting_time = int(values[1][0])
        for item in values[1:4000]:
            if item:
                if item[1] == "B":
                    ticks[0].append((int(item[0]) - starting_time) / 1000)
                if item[1] == "S":
                    ticks[1].append((int(item[0]) - starting_time) / 1000)
        for sequence in ticks:
            if sequence[0] < .1:
                sequence.pop(0)
        SAMPLE_DATA[sample[0]] = ticks
        SAMPLE_DATA_LENGTH[sample[0]] = (len(ticks[0]), len(ticks[1]))
        with open('data/tick_data.pkl', 'wb') as f:
            pickle.dump(SAMPLE_DATA, f, pickle.HIGHEST_PROTOCOL)
        with open('data/tick_data_length.pkl', 'wb') as f:
            pickle.dump(SAMPLE_DATA_LENGTH, f, pickle.HIGHEST_PROTOCOL)

if RETRIEVE_DATA:
    with open('data/tick_data.pkl', 'rb') as f:
        SAMPLE_DATA = pickle.load(f)

if RETRIEVE_ESTIMATES_BIVARIATE:
    with open('data/results_estimated_bivariate.pkl', 'rb') as f:
        SAMPLE_DATA_RESULTS_BIVARIATE = pickle.load(f)

if RETRIEVE_ESTIMATES_UNIVARIATE:
    with open('data/results_estimated_univariate.pkl', 'rb') as f:
        SAMPLE_DATA_RESULTS_UNIVARIATE = pickle.load(f)

if ESTIMATE_DATA_BIVARIATE:
    SAMPLE_DATA_RESULTS_BIVARIATE = {}
    for company in SAMPLE_DATA.items():
        # Set the initial guess for the parameters in the form:
        # [mu1, beta1, alpha11, alpha12, mu2, beta2, alpha22, alpha21]
        INITIAL_GUESS_BIVARIATE = 2 * [.1, .1, .1, .1]
        INITIAL_GUESS_FIRST_PART = INITIAL_GUESS_BIVARIATE[:4]
        INITIAL_GUESS_SECOND_PART = INITIAL_GUESS_BIVARIATE[4:]
        print("Number of buy orders of the company {} is {}."
              .format(company[0], len(company[1][0])))
        print("Number of sell orders of the company {} is {}."
              .format(company[0], len(company[1][1])))
        # Estimate the parameters for the first part of the Bivariate Hawkes and report the results.
        RESULTS_FIRST_PART = maximise_likelihood_scipy_bivariate_partial(data=company[1],
                                                                         initial_guess=INITIAL_GUESS_FIRST_PART,
                                                                         part_likelihood=1)
        print("Estimation of the first part of the Hawkes process for company {} is completed."
              .format(company[0]))

        # Estimate the parameters for the second part of the Bivariate Hawkes and report the results.
        RESULTS_SECOND_PART = maximise_likelihood_scipy_bivariate_partial(data=company[1],
                                                                          initial_guess=INITIAL_GUESS_SECOND_PART,
                                                                          part_likelihood=2)
        print("Estimation of the second part of the Hawkes process for company {} is completed."
              .format(company[0]))

        VALUES_REAL_DATA = {
            "MU1": RESULTS_FIRST_PART.x[0],
            "BETA1": RESULTS_FIRST_PART.x[1],
            "ALPHA_SAME1": RESULTS_FIRST_PART.x[2],
            "ALPHA_DIFF1": RESULTS_FIRST_PART.x[3],
            "MU2": RESULTS_SECOND_PART.x[0],
            "BETA2": RESULTS_SECOND_PART.x[1],
            "ALPHA_SAME2": RESULTS_SECOND_PART.x[2],
            "ALPHA_DIFF2": RESULTS_SECOND_PART.x[3]}

        SAMPLE_DATA_RESULTS_BIVARIATE[company[0]] = VALUES_REAL_DATA
    with open('data/results_estimated_bivariate.pkl', 'wb') as f:
        pickle.dump(SAMPLE_DATA_RESULTS_BIVARIATE, f, pickle.HIGHEST_PROTOCOL)

if TESTS_BIVARIATE:
    TESTS_REAL_DATA = {}
    for company in SAMPLE_DATA_RESULTS_BIVARIATE.items():
        COEFFICIENTS_ESTIMATED_REAL_DATA = [item[1] for item in company[1].items()]
        TEST_DATA_BIVARIATE = SAMPLE_DATA[company[0]]
        # Calculates the p_value of Ljung-Box test data from the TEST_DATA_BIVARIATE
        NUMBER_OF_LAGS_LB_BIVARIATE = 20
        bivariate_lb_test_p_value = lb_test_bivariate(TEST_DATA_BIVARIATE,
                                                      COEFFICIENTS_ESTIMATED_REAL_DATA,
                                                      NUMBER_OF_LAGS_LB_BIVARIATE)
        print("P-value of the Ljung-Box test for company {} is {}."
              .format(company[0], bivariate_lb_test_p_value))
        # Calculates the Kolmogorov-Smirnov test from the TEST_DATA_BIVARIATE
        bivariate_ks_test_p_value = ks_test_bivariate(TEST_DATA_BIVARIATE,
                                                      COEFFICIENTS_ESTIMATED_REAL_DATA)
        print("P-value of the Kolmogorov-Smirnov test for company {} is {}."
              .format(company[0], bivariate_ks_test_p_value))
        # Calculates the stationarity condition and reports whether the bivariate Hawkes is stationary
        bivariate_stationarity = stationarity_condition(COEFFICIENTS_ESTIMATED_REAL_DATA)
        print("Stationarity condition of the bivariate Hawkes for company {}: {}"
              .format(company[0], bivariate_stationarity[1]))
        TESTS_REAL_DATA[company[0]] = (bivariate_lb_test_p_value,
                                       bivariate_ks_test_p_value,
                                       bivariate_stationarity)
    with open('data/tests_real_data_bivariate.pkl', 'wb') as f:
        pickle.dump(TESTS_REAL_DATA, f, pickle.HIGHEST_PROTOCOL)

if ESTIMATE_DATA_UNIVARIATE_BUY or ESTIMATE_DATA_UNIVARIATE_SELL:
    """
    Estimates the parameters of the Univariate Hawkes from real data.
    """
    if ESTIMATE_DATA_UNIVARIATE_BUY:
        process = 0
    else:
        process = 1
    SAMPLE_DATA_RESULTS_UNIVARIATE = {}
    # Set the initial guess for the parameters in the form [mu, beta, alpha].
    INITIAL_GUESS_UNIVARIATE = [.1 for i in range(3)]
    for company in SAMPLE_DATA.items():
        # Estimate the parameters of the Univariate Hawkes and report the results.
        RESULTS_UNIVARIATE = maximise_likelihood_uni_hawkes(times=company[1][process],
                                                            initial_guess=INITIAL_GUESS_UNIVARIATE)
        print("The estimation of the Univariate Hawkes for company {} is completed."
              .format(company[0]))

        VALUES_UNIVARIATE = {
            "MU": RESULTS_UNIVARIATE.x[0],
            "ALPHA": RESULTS_UNIVARIATE.x[1],
            "BETA": RESULTS_UNIVARIATE.x[2]
        }
        SAMPLE_DATA_RESULTS_UNIVARIATE[company[0]] = VALUES_UNIVARIATE
    if ESTIMATE_DATA_UNIVARIATE_BUY:
        with open('data/results_estimated_univariate_buy.pkl', 'wb') as f:
            pickle.dump(SAMPLE_DATA_RESULTS_UNIVARIATE, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('data/results_estimated_univariate_sell.pkl', 'wb') as f:
            pickle.dump(SAMPLE_DATA_RESULTS_UNIVARIATE, f, pickle.HIGHEST_PROTOCOL)

if TEST_UNIVARIATE_DATA_BUY or TEST_UNIVARIATE_DATA_SELL:
    """
    Performs the tests on real data for univariate Hawkes.
    """
    TESTS_REAL_DATA_UNIVARIATE = {}
    if ESTIMATE_DATA_UNIVARIATE_BUY:
        process = 0
    else:
        process = 1
    for company in SAMPLE_DATA.items():
        TEST_DATA_UNIVARIATE = company[1][process]
        # Adjust the coefficients so they fit the tests.
        # They should be in the form [MU, BETA, ALPHA].
        COEFFICIENTS_TESTS_UNIVARIATE = [item[1] for item
                                         in SAMPLE_DATA_RESULTS_UNIVARIATE[company[0]].items()]
        COEFFICIENTS_TESTS_UNIVARIATE = [
            COEFFICIENTS_TESTS_UNIVARIATE[0], COEFFICIENTS_TESTS_UNIVARIATE[2], COEFFICIENTS_TESTS_UNIVARIATE[1]
        ]
        # Calculates the p_value of Ljung-Box test data from the TEST_DATA_UNIVARIATE
        NUMBER_OF_LAGS_LB_UNIVARIATE = 20
        univariate_lb_test_p_value = lb_test_univariate(TEST_DATA_UNIVARIATE,
                                                        COEFFICIENTS_TESTS_UNIVARIATE,
                                                        NUMBER_OF_LAGS_LB_UNIVARIATE)
        print("The p-value of the LB test for company {} is {}."
              .format(company[0], univariate_lb_test_p_value))
        # Calculates the Kolmogorov-Smirnov test from the TEST_DATA_UNIVARIATE
        univariate_ks_test_p_value = ks_test_univariate(TEST_DATA_UNIVARIATE,
                                                        COEFFICIENTS_TESTS_UNIVARIATE)
        print("The p-value of the KS test for company {} is {}."
              .format(company[0], univariate_ks_test_p_value))
        # Calculates the univariate stationarity
        univariate_stationarity = COEFFICIENTS_TESTS_UNIVARIATE[2] < COEFFICIENTS_TESTS_UNIVARIATE[1]
        print("The stationarity condition for company {} is {}."
              .format(company[0], univariate_stationarity))
        TESTS_REAL_DATA_UNIVARIATE[company[0]] = (univariate_lb_test_p_value,
                                                  univariate_ks_test_p_value,
                                                  univariate_stationarity)
    if TEST_UNIVARIATE_DATA_BUY:
        with open('data/tests_univariate_buy.pkl', 'wb') as f:
            pickle.dump(TESTS_REAL_DATA_UNIVARIATE, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('data/tests_univariate_sell.pkl', 'wb') as f:
            pickle.dump(TESTS_REAL_DATA_UNIVARIATE, f, pickle.HIGHEST_PROTOCOL)
