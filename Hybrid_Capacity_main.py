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
(r=1 in the system parameters) and can be easily modified to r=2 to compute the capacity 
with subcarrier intensity modulation detection (SIM). This example was used to obtain 
the results shown in Fig. 2(a) of [1].


[1] W. A. Alathwary and E. S. Altubaishi,"An Integral of Fox’s H-functions with
Application to the Performance of Hybrid FSO/RF Systems Over Generalized Fading 
Channels," 2025.
'''

class SystemParameters:
    """Class to store and compute system constants."""
    def __init__(self):

        # FSO link parameters
        self.alpha = 4.2  # Malaga parameter alpha of the FSO link
        self.beta = 3  # Malaga parameter beta of the FSO link
        self.b0 = 0.1079
        self.ro = 0.596
        self.Omega = 1.3265
        self.theta = np.pi / 2
        self.no_terms = 2  # Number of asymptotic terms
        self.r = 1  # Type of FSO detection (HD: r=1, SIM: r=2)
        self.SNRT_FSO = 10 ** (5 / 10)  # SNR threshold for FSO

        # Pointing error parameters
        self.Wz = 2.5  # Beam radius
        self.Ra = 0.1  # Aperture radius
        self.Sigmas = 0.3  # Jitter standard deviation

        # Computed parameters of FSO link with pointing error
        self.v = np.sqrt(np.pi / 2) * (self.Ra / self.Wz)
        self.Ao = (erf(self.v)) ** 2
        self.Wez = np.sqrt((self.Wz ** 2) * np.sqrt(np.pi) * erf(self.v) / (2 * self.v * np.exp(-self.v ** 2)))
        self.xi = self.Wez / (2 * self.Sigmas)
        self.h = (self.xi ** 2) / ((self.xi ** 2) + 1)

        # Parameters in Table 2
        self.g = 2 * self.b0 * (1 - self.ro)
        self.Omega0 = self.Omega + 2 * self.b0 * self.ro + 2 * np.sqrt(2 * self.b0 * self.ro * self.Omega) * \
                      np.cos(self.theta)
        self.A = ((2 * self.alpha ** (self.alpha / 2)) / ((self.g ** (1 + self.alpha / 2)) * gamma(self.alpha))) * \
            (((self.g * self.beta) / (self.g * self.beta + self.Omega0)) ** (self.beta + self.alpha / 2))
        self.varphi = self.A / 2
        self.vartheta = self.g + self.Omega0
        self.Upsilon = self.alpha * self.vartheta * (self.alpha + 1) ** (-1)
        self.Q = 2 * self.g * (self.g + 2 * self.Omega0) + self.Omega0 ** 2 * (1 + 1 / self.beta)
        self.chi_tau = (self.alpha * self.beta) / ((self.beta * self.g) + self.Omega0)
        self.B = self.chi_tau * self.h * self.vartheta

        # Average SNR values of FSO
        self.ASNR_d_FSO = np.arange(0, 40.1, 2.5)  # Average SNR in dB
        self.ASNR_FSO = 10 ** (self.ASNR_d_FSO / 10)  # SNR in linear scale

        # Compute Mu_r for FSO as in eq (17)
        Num = self.Upsilon * self.h * (self.xi ** 2 + 2)
        Den = self.Q * (self.xi ** 2 + 1)
        if self.r == 1:
            Num, Den = 1, 1
        self.Mu_r_FSO = (Num / Den) * self.ASNR_FSO

        # RF link parameters
        self.alpha_r = 2  # alpha_mu parameter alpha_r of the RF link
        self.mu = 1  # alpha_mu parameter mu of the RF link
        self.C = self.alpha_r / 2
        self.SNRT_RF = 10 ** (5 / 10)  # SNR threshold for RF
        self.ASNR_RF = 10 ** (15 / 10)  # Average RF SNR in linear scale


def FSO_capacity(params):
    """
    Compute the capacity of the FSO link as in eq (32).
    """

    # initial values for timers
    i = 0
    total_time = 0
    # container for the FSO capacities
    cap_fso = []

    for SNR in params.Mu_r_FSO:
        start = time.time()
        cap_fso_sum = 0  # initial value of the summation container for FSO capacity

        for m in range(1, params.beta + 1):
            # Compute am and bm of Malaga fading
            am = comb(params.beta - 1, m - 1) * \
                 (((params.g * params.beta + params.Omega0) ** (1 - m / 2)) / gamma(m)) * \
                 ((params.Omega0 / params.g) ** (m - 1)) * \
                 ((params.alpha / params.beta) ** (m / 2))

            # bm of malaga fading
            bm = am * (((params.g * params.beta + params.Omega0) / (params.alpha * params.beta)) ** ((params.alpha + m) / 2))

            # Parameters for FSO capacity
            mn = [(1, 0), (1, 2), (3, 0)]  # sequence of m, n in eq (32)
            pq = [(1, 1), (2, 2), (1, 3)]  # sequence of p, q in eq (32)
            a = [tuple([1] + [1] + [1 / params.r])]  # (1-sigma, h1,..., hN) in eq (32)
            b = [tuple([0] + [1] + [1 / params.r])]  # (-sigma, h1,..., hN) in eq (32)
            c = [[(1, 1), (1, 1)], [(params.xi ** 2 + 1, 1)]]  # (a_p, alpha_p) from 1 to N in eq (32)
            d = [[(1, 1), (0, 1)],
                 [(params.xi ** 2, 1), (params.alpha, 1), (m, 1)]]  # (b_q, beta_q) from 1 to N in eq (32)

            z1 = [params.SNRT_FSO, params.B * (params.SNRT_FSO / SNR) ** (1 / params.r)]
            params1 = z1, mn, pq, a, b, c, d
            H_FSO = np.real(MultiFoxH_main(params1, no_divisions=70))  # multivariate Fox’s H-function
            cap_fso_sum += bm * H_FSO  # summation in eq(32)

        # The capacity of the FSO link as describe in eq (32)
        cap_fso.append((params.xi ** 2 * params.varphi * cap_fso_sum) / (params.r * np.log(2)))

        # Log execution times for each loop and the total
        i += 1
        print(f"Executing loop {i} took: {time.time() - start:.9f} seconds")
        total_time += time.time() - start
    print(f"Executing the whole loops took: {total_time:.9f} seconds")

    return cap_fso


def FSO_outage(params):
    """
    Compute the outage probability of the FSO link as in eq (19) with gamma_th.
    """

    op_fso_sum = 0  # initial value of the summation container for FSO outage

    for m in range(1, params.beta + 1):
        # Compute am and bm of Malaga fading
        am = comb(params.beta - 1, m - 1) * \
             (((params.g * params.beta + params.Omega0) ** (1 - m / 2)) / gamma(m)) * \
             ((params.Omega0 / params.g) ** (m - 1)) * \
             ((params.alpha / params.beta) ** (m / 2))

        # bm of malaga fading
        bm = am * (((params.g * params.beta + params.Omega0) / (params.alpha * params.beta)) ** ((params.alpha + m) / 2))

        z = params.B * (params.SNRT_FSO / params.Mu_r_FSO) ** (1 / params.r)
        op_fso_sum += np.array([bm * mp.meijerg([[1], [params.xi ** 2 + 1]], [[params.xi ** 2, params.alpha, m], [0]], zi)
                  for zi in z])  # summation in eq(19)

    # The outage of the FSO link as describe in eq (19)
    op_fso = params.xi ** 2 * params.varphi * op_fso_sum
    return op_fso


def RF_capacity(params):
    """
    Compute the capacity of the RF link as in eq (33).
    """

    # Parameters for RF capacity
    mn = [(1, 0)] + [(1, 2)] + [(1, 0)]  # sequence of m, n in eq (33)
    pq = [(1, 1)] + [(2, 2)] + [(0, 1)]  # sequence of p, q in eq (33)
    a = [tuple([1] + [1] + [params.alpha_r / 2])]  # (1-sigma, h1,..., hN) in eq (33)
    b = [tuple([0] + [1] + [params.alpha_r / 2])]  # (-sigma, h1,..., hN) in eq (33)
    c = [[(1, 1), (1, 1)]] + [[]]  # (a_p, alpha_p) from 1 to N in eq (33)
    d = [[(1, 1), (0, 1)]] + [[(params.mu, 1)]]  # (b_q, beta_q) from 1 to N in eq (33)

    start = time.time()

    z2 = [params.SNRT_RF, params.mu * (params.SNRT_RF / params.ASNR_RF) ** (params.alpha_r / 2)]
    params2 = z2, mn, pq, a, b, c, d
    H_RF = np.real(MultiFoxH_main(params2, no_divisions=70))  # multivariate Fox’s H-function

    # Executing time of capacity of the RF link
    print(f"Executing took: {time.time() - start:.9f} seconds")

    # The capacity of the RF link as describe in eq (33)
    cap_rf = (params.C * H_RF) / (np.log(2) * gamma(params.mu))
    return cap_rf


def asymp_capacity(params):
    """
    Compute the asymptotic capacity for the FSO link at high SNR as in (39).
    """

    # Compute I_1 as in eq (40)
    # Note: psi(params.xi ** 2) -psi(1-params.xi ** 2) = - 1 / params.xi ** 2
    # and gamma (1- params.xi ** 2) = params.xi ** 2 * gamma (params.xi ** 2)

    I1_sum = 0  # initial value of the summation container for I_1

    for m in range(1, params.beta + 1):
        # Compute am and bm of Malaga fading
        am = comb(params.beta - 1, m - 1) * \
             (((params.g * params.beta + params.Omega0) ** (1 - m / 2)) / gamma(m)) * \
             ((params.Omega0 / params.g) ** (m - 1)) * \
             ((params.alpha / params.beta) ** (m / 2))

        # bm of malaga fading
        bm = am * (((params.g * params.beta + params.Omega0) / (params.alpha * params.beta)) ** ((params.alpha + m) / 2))

        # summation in eq (40)
        I1_sum += gamma(params.alpha) * bm * gamma(m) * (params.r * (psi(params.alpha) + psi(m) - \
                                                                     np.log(params.B) - 1 / params.xi ** 2) + np.log(params.Mu_r_FSO))
    # The identity I_1 as describe in eq (40)
    I1 = (params.varphi / np.log(2)) * I1_sum

    # Compute I_2 as in eq (42)
    I2_sum1_terms = 0  # initial value of the first summation container for I_2

    for m in range(1, params.beta + 1):
        # Compute am and bm of Malaga fading
        am = comb(params.beta - 1, m - 1) * \
             (((params.g * params.beta + params.Omega0) ** (1 - m / 2)) / gamma(m)) * \
             ((params.Omega0 / params.g) ** (m - 1)) * \
             ((params.alpha / params.beta) ** (m / 2))

        # bm of malaga fading
        bm = am * (((params.g * params.beta + params.Omega0) / (params.alpha * params.beta)) ** ((params.alpha + m) / 2))

        sum2_terms = 0
        z2 = params.B / (params.Mu_r_FSO) ** (1 / params.r)
        for j in range(params.no_terms):
            current_param = sorted([params.xi, params.alpha, m])[j]
            if current_param == params.xi:
                term1 = (z2 ** current_param ** 2) * gamma(params.alpha - current_param ** 2) * \
                       gamma(m - current_param  ** 2) / gamma(1 + params.xi ** 2 - current_param ** 2)
                term2 = (((current_param ** 2 / params.r) * np.log(params.SNRT_FSO) - 1) *
                         params.SNRT_FSO ** (current_param ** 2 / params.r)) / (current_param ** 2 / params.r) ** 2
            elif current_param == params.alpha:
                term1 = (z2 ** current_param) * gamma(params.xi ** 2 - current_param) * gamma(m - current_param) \
                       / gamma(1 + params.xi ** 2 - current_param)
                term2 = (((current_param / params.r) * np.log(params.SNRT_FSO) - 1) *
                         params.SNRT_FSO ** (current_param / params.r)) / (current_param / params.r) ** 2
            elif current_param == m:
                term1 = (z2 ** current_param) * gamma(params.xi ** 2 - current_param) * gamma(params.alpha - current_param) \
                       / gamma(1 + params.xi ** 2 - current_param)
                term2 = (((current_param / params.r) * np.log(params.SNRT_FSO) - 1) *
                         params.SNRT_FSO ** (current_param / params.r)) / (current_param / params.r) ** 2

            else:
                continue
            sum2_terms += term1 * term2

        I2_sum1_terms += bm * sum2_terms

    # The identity I_2 as describe in eq (42)
    I2 = params.xi ** 2 * params.varphi * I2_sum1_terms / (params.r * np.log(2))

    # The asymptotic capacity for the FSO link at high SNR as describe in (39)
    asymp_capacity = I1 - I2

    return np.where((asymp_capacity < 0) | (asymp_capacity > 30), np.nan, asymp_capacity)


if __name__ == '__main__':
    params = SystemParameters()

    tic = time.time()

    # Compute capacities
    fso_capacity = FSO_capacity(params)
    fso_outage = FSO_outage(params)
    rf_capacity = RF_capacity(params)

    # Capacity of Hybrid FSO/RF system
    hybrid_capacity = fso_capacity + np.array(fso_outage) * rf_capacity

    # Asymptotic capacity of the system
    asymp_cap = asymp_capacity(params)

    toc = time.time()

    print(f"Total execution time: {toc - tic:.9f} seconds")

    # Plot results
    plt.plot(params.ASNR_d_FSO, hybrid_capacity, '-.rs', label="Hybrid Capacity")
    plt.plot(params.ASNR_d_FSO, asymp_cap, '--kd', label="Asymptotic Capacity")
    plt.xlabel('Average SNR of the FSO Link (dB)')
    plt.ylabel('Normalized Capacity (bps/Hz)')
    plt.title('Hybrid FSO/RF System Capacity for Malaga Model')
    plt.legend()
    plt.grid()
    # plt.ylim([0, np.max(fso_capacity)])
    plt.ylim([3, 13])
    plt.xlim([0, params.ASNR_d_FSO[-1]])
    plt.show()
