#!/usr/bin/env python
import argparse
from lyaforecast.cosmoCAMB import CosmoCamb
import numpy as np
import matplotlib.pyplot as plt

def main():
    args = get_args()

    cosmo = CosmoCamb(args.ini)

    k_range = np.linspace(1e-2,0.3,1000)
    pk_full = cosmo.get_pk_lin(k_range)
    pk_smooth = cosmo.get_pk_lin_smooth(k_range)
    pk_peak = cosmo.get_pk_lin_peak(k_range)

    fig,ax = plt.subplots(1,2,figsize=(16,8))
    ax[0].semilogx(k_range,k_range*pk_full,label='Full')
    ax[0].semilogx(k_range,k_range*pk_smooth,label='Smooth')
    ax[0].semilogx(k_range,k_range*pk_peak,label='Peak')

    ax[1].semilogx(k_range,pk_peak/pk_full)

    ax[0].set_xlabel('k [h/Mpc]',fontsize=24)
    ax[1].set_xlabel('k [h/Mpc]',fontsize=24)
    ax[0].set_ylabel(r'$k P(k) [h/Mpc]^3$',fontsize=24)
    ax[1].set_ylabel('ratio peak/full',fontsize=24)

    #ax[0].set_xlim(0.1,0.3)
    #ax[1].set_xlim(0.1,0.3)

    fig.savefig(args.output)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='Run Forecast.')

    parser.add_argument('--ini', '-i',
                        type=str, 
                        default=None, 
                        help='CAMB config file')
    
    parser.add_argument('--output', '-o',
                        type=str, 
                        default='/Users/calum/Documents/Work/pk_lin.png', 
                        help='path to output')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
