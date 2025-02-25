import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.signal import savgol_filter
from lmfit import Minimizer, Parameters, create_params, report_fit, Model
import os
import sys

from SQ_KAN import *
QD = (np.arange(100)+1)*0.2

## define scattering functions
def hardsphere(q, sigma=1):
    R = sigma / 2
    P = (3 * (np.sin(q * R) - q * R * np.cos(q * R)) / (q * R) ** 3) ** 2
    return P

def fuzzysphere(q, sigma=1, sigma_f=0.1):
    R = sigma / 2
    P = (3 * (np.sin(q * R) - q * R * np.cos(q * R)) / (q * R) ** 3) ** 2 * np.exp(-(sigma_f * sigma * q) ** 2 / 2)
    return P

def log_normal_pdf(mu, sigma, x):
    return np.exp(-(np.log(x) - mu) ** 2 / 2 / sigma ** 2) / x / sigma

def P_HS_eff(q, sigma=1, d_sigma=0.05, return_f=False):
    '''
    sigma: average particle size
    d_sigma: polydispersity
    return_f: toggle whether to return the particle size distribution
    '''
    # List of particle diameter
    n_sample = 101
    sigma_list = (1 + np.linspace(-5, 5, n_sample) * d_sigma) * sigma
    sigma_list = sigma_list[sigma_list > 0]

    # Size distribution
    f_sigma = log_normal_pdf(0, d_sigma, sigma_list / sigma)
    p_sigma = f_sigma * (sigma_list / sigma) ** 6

    # Calculate effective P(Q)
    P_eff = np.zeros_like(q)
    for i in range(len(sigma_list)):
        P_i = hardsphere(q, sigma_list[i]) * p_sigma[i]
        P_eff = P_eff + P_i

    P_eff = P_eff / np.sum(p_sigma)

    if return_f:
        return P_eff, sigma_list, f_sigma
    else:
        return P_eff

def IQ_th(params, Q):
    v = params.valuesdict()
    fp = [v['phi'],v['kappa'],np.log(v['A'])]

    # structure factor
    S = SQ_KAN(fp, Q, device=device)
    S = savgol_filter(S,7,2)

    # form factor
    P = P_HS_eff(Q,sigma=v['sigma'], d_sigma=v['d_sigma'])

    I = v['C']*P*S + v['I_inc']
    return I

## load fitting data
def load_IQ_file(file_name):
    data = np.genfromtxt(file_name,delimiter=',')
    QD_exp = data[:,0]
    IQ_exp = data[:,1]
    return QD_exp, IQ_exp

## perform curve fitting
def fit(initial_guess, file_name):
    # load data
    QD_exp, IQ_exp = load_IQ_file(file_name)

    # initialize parameters
    def fmt_string(string):
        fmtstr = np.fromstring(string, dtype=float, sep=',')
        return fmtstr

    # params_names = ['C', 'I_inc', 'sigma', 'd_sigma', 'phi', 'kappa', 'A']

    # params = Parameters()
    # for i_names, p_names in enumerate(params_names):
    #     param_i = fmt_string(initial_guess[i_names])
    #     print(param_i)
    #     params.add(params_names[i_names], value=param_i[0], min=param_i[1], max=param_i[2])
    
    params_names = ['C', 'I_inc', 'sigma', 'd_sigma', 'phi', 'kappa', 'A']    
    params = Parameters()
    for i_names, p_names in enumerate(params_names):
        param_i = [float(initial_guess[i_names*3+j]) for j in range(3)]
        print(param_i)
        params.add(params_names[i_names], value=param_i[0], min=param_i[1], max=param_i[2])

    def lmbda(params, Q, IQ_exp, index_Q):
        IQ = IQ_th(params, Q)
        # minimizer_target = lambda x, y, z: (np.log(x/y))/np.log(1+z/y)
        minimizer_target = lambda x, y, z: (x-y)**2
        # return minimizer_target(IQ[index_Q],IQ_exp[index_Q],np.ones_like(IQ_exp))
        return minimizer_target(IQ[index_Q],IQ_exp[index_Q],np.ones_like(IQ_exp))

    # do fit, here with the nelder algorithm
    minner = Minimizer(lmbda, params, fcn_args=(QD_exp, IQ_exp, np.arange(len(QD_exp))))
    result = minner.minimize('powell')
    # write error report
    # report_fit(result)
    
    for param in result.params.values():
        if param.stderr is None: 
            string = "%s:  %f +/- undetermined (init = %f)" % (param.name, param.value, param.init_value)
        else:
            string = "%s:  %f +/- %f (init = %f)" % (param.name, param.value, param.stderr, param.init_value)
        print(string)

    # plot the result
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)

    ax.plot(QD_exp,IQ_exp,'.b',label='SANS data')
    ax.plot(QD_exp,IQ_th(result.params, QD),'-r',label='NN')

    ax.set_yscale('log')
    ax.set_xscale('log')

    # ax.set_xlim([2,40])
    # ax.set_ylim([0.2,4000])
    ax.set_xlabel(r'$QD$',fontsize=20)
    ax.set_ylabel(r'$I(QD)$',fontsize=20)
    ax.tick_params(direction='in', axis='both', which='both', labelsize=18, pad=10)
    ax.legend(fontsize=16,frameon=False)
    plt.tight_layout()
    plt.savefig('output.png')
    # plt.show()

    # print the result
    print('Printing results to output.txt')
    with open('output.txt', 'a') as f:
        for param in result.params.values():
            if param.stderr is None: 
                string = "%s:  %f +/- undetermined (init = %f)\n" % (param.name, param.value, param.init_value)
            else:
                string = "%s:  %f +/- %f (init = %f)\n" % (param.name, param.value, param.stderr, param.init_value)
            f.write(string)


if __name__ == '__main__':
    print(sys.argv)
    print(len(sys.argv))
    
    if len(sys.argv) == 24:
        filename = sys.argv[2]
        initial_guess = sys.argv[3:24]
        # file_name = '/src/example_IQ.csv' # testing
        fit(initial_guess, filename)

    else:
        print("Usage: python3 /src/curvefit.py [-i] filename C I_inc sigma d_sigma phi kappa A")
        # python3 src/curvefit.py -i /src/example_IQ.csv 4100,4000,5000 0.5,0.1,2 0.975,0.9,1.1 0.07,0.05,0.1 0.08,0.05,0.5 0.15,0.1,0.18 10,0.5,20 
        # python3 /src/curvefit.py -i '/src/example_IQ.csv' '5000' '1000' '8000' '0.2' '0.1' '0.5' '1' '0.9' '1.1' '0.05' '0.01' '0.1' '0.2' '0.1' '0.5' '0.1' '0.01' '0.5' '10' '1' '20'
        sys.exit(1)