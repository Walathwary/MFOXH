import numpy as np
import mpmath as mp
from scipy.special import gamma, psi, erf, comb
from MFOX import MultiFoxH_main
import time
import matplotlib.pyplot as plt


'''
This example illustrates how to use the MFOX module to compute the multivariate Fox H function. 
Specifically, it demonstrates the computation of the hybrid FSO/RF system capacity as described 
in equation (29) of [1], where the capacities of the FSO and RF links are provided in 
equations (32) and (33), and the main integral solution is presented in equation (8). 
Additionally, the asymptotic capacity in equation (39) is also included in this example. 
The example computes the hybrid system capacity using heterodyne detection (HD) 
(r=1 in the system parameters) and can be easily modified to r=2r=2 to compute the capacity 
with subcarrier intensity modulation detection (SIM). This example was used to obtain 
the results shown in Fig. 2(a) of [1].


[1] W. A. Alathwary and E. S. Altubaishi,"An Integral of Fox’s H-functions with
Application to the Performance of Hybrid FSO/RF Systems Over Generalized Fading 
Channels," 2024.
'''

class SystemParameters:
    """Class to store and compute system constants."""
    def __init__(self):
        self.alpha = 4.2  # Malaga parameter alpha
        self.beta = 3  # Malaga parameter beta
        self.b0 = 0.1079
        self.ro = 0.596
        self.Omega = 1.3265
        self.theta = np.pi / 2
        self.SNRT_FSO = 10 ** (5 / 10)  # SNR threshold for FSO
        self.SNRT_RF = 10 ** (5 / 10)  # SNR threshold for RF
        self.SNR_RF = 10 ** (15 / 10)  # Average RF SNR
        self.Wz = 2.5  # Beam radius
        self.Ra = 0.1  # Aperture radius
        self.Sigmas = 0.3  # Jitter standard deviation
        self.no_bits = int(1e6)  # Number of Monte Carlo bits
        self.no_terms = 2  # Number of asymptotic terms
        self.r = 1  # Type of FSO detection (HD: r=1, SIM: r=2)
        self.alpha_r = 2  # alpha_mu parameter alpha_r
        self.mu = 1  # alpha_mu parameter mu

        # Computed parameters
        self.v = np.sqrt(np.pi / 2) * (self.Ra / self.Wz)
        self.Ao = (erf(self.v)) ** 2
        self.Wez = np.sqrt((self.Wz ** 2) * np.sqrt(np.pi) * erf(self.v) / (2 * self.v * np.exp(-self.v ** 2)))
        self.xi = self.Wez / (2 * self.Sigmas)
        self.h = (self.xi ** 2) / ((self.xi ** 2) + 1)
        self.g = 2 * self.b0 * (1 - self.ro)
        self.Omega0 = self.Omega + 2 * self.b0 * self.ro + 2 * np.sqrt(2 * self.b0 * self.ro * self.Omega) * \
                      np.cos(self.theta)
        self.A = ((2 * self.alpha ** (self.alpha / 2)) / ((self.g ** (1 + self.alpha / 2)) * gamma(self.alpha))) * \
            (((self.g * self.beta) / (self.g * self.beta + self.Omega0)) ** (self.beta + self.alpha / 2))
        self.B = (self.alpha * self.beta * self.h * (self.g + self.Omega0)) / ((self.beta * self.g) + self.Omega0)

        # SNR values
        self.SNR_d_FSO = np.arange(0, 40.1, 2.5)  # Average SNR in dB
        self.SNR_FSO = 10 ** (self.SNR_d_FSO / 10)  # SNR in linear scale
        self.SNR_d_FSO_S = np.arange(5, 35.1, 5)  # SNR values for simulation
        self.SNR_FSO_S = 10 ** (self.SNR_d_FSO_S / 10)  # Convert to linear scale

        # Compute Mu2 for FSO
        Num = self.alpha * self.xi ** 2 * (self.xi ** 2 + 1) ** (-2) * (self.xi ** 2 + 2) * (self.g + self.Omega0)
        Den = (1 + self.alpha) * (2 * self.g * (self.g + 2 * self.Omega0) + self.Omega0 ** 2 * (1 + 1 / self.beta))
        if self.r == 1:
            Num, Den = 1, 1
        self.Mu2_FSO = (Num / Den) * self.SNR_FSO
        self.Mu2_FSO_S = (Num / Den) * self.SNR_FSO_S


def FSO_system():
    """
    Compute the capacity, outage, and asymptotic capacity of the FSO link through the sum of beta.
    """
    # Initialize accumulators
    cap_fso_sum = 0
    op_fso_sum = 0
    asymp_cap_fso_sum = 0
    loop_time_sum = 0
    total_time_sum = 0

    # Loop through each value of m (1 to beta)
    for m in range(1, params.beta + 1):
        # Compute am and bm of Malaga fading
        am_temp = ((params.g * params.beta + params.Omega0) / (params.alpha * params.beta)) ** ((params.alpha + m) / 2)

        am = comb(params.beta - 1, m - 1) * \
             (((params.g * params.beta + params.Omega0) ** (1 - m / 2)) / gamma(m)) * \
             ((params.Omega0 / params.g) ** (m - 1)) * \
             ((params.alpha / params.beta) ** (m / 2))

        bm = am_temp * am  # bm of malaga fading

        # Compute capacity, outage, and asymptotic capacity for m
        cap_fso, loop_time, total_time = FSO_capacity(params, m, bm)  # the capacity of the FSO link
        cap_fso_sum += np.array(cap_fso)
        op_fso_sum += np.array(FSO_outage(params, m, bm))  # the outage of the FSO link
        asymp_cap_fso_sum += np.array(asymp_capacity(params, m, bm))  # the asymptotic capacity of the FSO link

        # Record execution times
        loop_time_sum += np.array(loop_time)
        total_time_sum += np.array(total_time)


    # # Log execution times for each loop and the total
    for i in range(len(params.SNR_d_FSO)):
        print(f"Executing loop {i+1} took: {loop_time_sum[i]} seconds")
    print(f"Executing the whole loops took: {total_time_sum} seconds")

    return cap_fso_sum, op_fso_sum, asymp_cap_fso_sum


def FSO_capacity(params, m, bm):
    """
    Compute the capacity of the FSO link.
    """

    # Parameters for FSO capacity
    mn = [(1, 0), (1, 2), (3, 0)]
    pq = [(1, 1), (2, 2), (1, 3)]
    a = [tuple([1] + [1] + [1 / params.r])]
    b = [tuple([0] + [1] + [1 / params.r])]
    c = [[(1, 1), (1, 1)], [(params.xi ** 2 + 1, 1)]]
    d = [[(1, 1), (0, 1)], [(params.xi ** 2, 1), (params.alpha, 1), (m, 1)]]

    k = (params.xi ** 2 * params.A) / (2 * params.r * np.log(2))

    capacities = []
    inst_time = []
    # i = 0
    total_time = 0
    for SNR in params.Mu2_FSO:
        start = time.time()
        lambda_1 = params.B * (params.SNRT_FSO / SNR) ** (1 / params.r)
        z1 = [params.SNRT_FSO, lambda_1]
        params1 = z1, mn, pq, a, b, c, d
        H1 = np.real(MultiFoxH_main(params1, no_divisions=70))
        capacities.append(k * bm * H1)
        # i += 1
        inst_time.append(time.time() - start)
        # print(f"Executing loop {i} took: {time.time() - start:.9f} seconds")
        total_time += time.time() - start
    # print(f"Executing the whole loops took: {total_time:.9f} seconds")
    return capacities, inst_time, total_time


def FSO_outage(params, m, bm):
    """
    Compute the outage probability of the FSO link.
    """
    k = (params.xi ** 2 * params.A) / 2
    z = params.B * (params.SNRT_FSO / params.Mu2_FSO) ** (1 / params.r)
    op_fso = [k * bm * mp.meijerg([[1], [params.xi ** 2 + 1]], [[params.xi ** 2, params.alpha, m], [0]], zi)
              for zi in z]
    return op_fso


def RF_capacity(params):
    """
    Compute the capacity of the RF link.
    """

    # Parameters for RF capacity
    mn = [(1, 0)] + [(1, 2)] + [(1, 0)]
    pq = [(1, 1)] + [(2, 2)] + [(0, 1)]
    a = [tuple([1] + [1] + [params.alpha_r / 2])]
    b = [tuple([0] + [1] + [params.alpha_r / 2])]
    c = [[(1, 1), (1, 1)]] + [[]]
    d = [[(1, 1), (0, 1)]] + [[(params.mu, 1)]]

    k = params.alpha_r / (2 * np.log(2) * gamma(params.mu))

    start = time.time()
    lambda_1 = params.mu * (params.SNRT_RF / params.SNR_RF) ** (params.alpha_r / 2)
    z2 = [params.SNRT_RF, lambda_1]
    params2 = z2, mn, pq, a, b, c, d
    H2 = np.real(MultiFoxH_main(params2, no_divisions=70))
    print(f"Executing took: {time.time() - start:.9f} seconds")
    return k * H2


def asymp_capacity(params, m, bm):
    """
    Compute the asymptotic capacity for the FSO link at high SNR.
    """
    I1 = (1 / (2 * np.log(2))) * params.A * gamma(params.alpha) * bm * gamma(m) * (params.r * (psi(params.alpha) + psi(m) - \
          np.log(params.B) - 1 / params.xi ** 2) + np.log(params.Mu2_FSO))

    # Summation for asymptotic terms
    z2 = params.B / (params.Mu2_FSO) ** (1 / params.r)
    sum_terms = 0
    for i in range(params.no_terms):
        current_param = sorted([params.xi, params.alpha, m])[i]
        if current_param == params.xi:
            term = (z2 ** current_param ** 2) * gamma(params.alpha - current_param ** 2) * \
                   gamma(m - current_param  ** 2) / gamma(1 + params.xi ** 2 - current_param ** 2)
            term1 = (((current_param ** 2 / params.r) * np.log(params.SNRT_FSO) - 1) *
                     params.SNRT_FSO ** (current_param ** 2 / params.r)) / (current_param ** 2 / params.r) ** 2
        elif current_param == params.alpha:
            term = (z2 ** current_param) * gamma(params.xi ** 2 - current_param) * gamma(m - current_param) \
                   / gamma(1 + params.xi ** 2 - current_param)
            term1 = (((current_param / params.r) * np.log(params.SNRT_FSO) - 1) *
                     params.SNRT_FSO ** (current_param / params.r)) / (current_param / params.r) ** 2
        elif current_param == m:
            term = (z2 ** current_param) * gamma(params.xi ** 2 - current_param) * gamma(params.alpha - current_param) \
                   / gamma(1 + params.xi ** 2 - current_param)
            term1 = (((current_param / params.r) * np.log(params.SNRT_FSO) - 1) *
                     params.SNRT_FSO ** (current_param / params.r)) / (current_param / params.r) ** 2

        else:
            continue
        sum_terms += term * term1

    I2 = params.xi ** 2 * params.A * bm * sum_terms / (2 * params.r * np.log(2))
    asymp_capacity = I1 - I2
    return np.where((asymp_capacity < 0) | (asymp_capacity > 30), np.nan, asymp_capacity)


if __name__ == '__main__':
    params = SystemParameters()

    # Compute capacities
    tic = time.time()
    fso_capacity, fso_outage, asymp_cap = FSO_system()
    rf_capacity = RF_capacity(params)
    hybrid_capacity = fso_capacity + np.array(fso_outage) * rf_capacity
    toc = time.time()

    print(f"Total execution time: {toc - tic:.9f} seconds")

    # Plot results
    plt.plot(params.SNR_d_FSO, hybrid_capacity, '-.rs', label="Hybrid Capacity")
    plt.plot(params.SNR_d_FSO, asymp_cap, '--kd', label="Asymptotic Capacity")
    plt.xlabel('Average SNR of the FSO Link (dB)')
    plt.ylabel('Normalized Capacity (bps/Hz)')
    plt.title('Hybrid FSO/RF System Capacity for Malaga Model')
    plt.legend()
    plt.grid()
    # plt.ylim([0, np.max(fso_capacity)])
    plt.ylim([3, 13])
    plt.xlim([0, params.SNR_d_FSO[-1]])
    plt.show()
