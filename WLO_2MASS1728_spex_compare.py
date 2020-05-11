#
#
#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from astropy.io import fits
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel


def bandnorm(wl, flux, band):
    #normalize a spectrum according to which band is given
    if band.upper() == 'J':
        wlLow, wlHi = 1.2700, 1.2900
    elif band.upper() == 'H':
        wlLow, wlHi = 1.700,1.7200
    elif band.upper() == 'K':
        wlLow, wlHi = 2.2000, 2.2200
    else:
        print('Invalid Band')
        wlLow, wlHi =100, 200
        sys.exit()
    peaks = flux[(np.around(wl, decimals=4)>=wlLow) \
    & (np.around(wl, decimals=4)<=wlHi)]
    norm_val = np.nanmean(peaks)
    normflux = flux/norm_val
    return normflux

def medianNpix(wl, spec, low, hi,clipspec=False):
    clip = spec[(np.around(wl, decimals=4)>=low)&(np.around(wl, decimals=4)<=hi)]
    wlclip = wl[(np.around(wl, decimals=4)>=low)&(np.around(wl, decimals=4)<=hi)]
    wlperpix = np.array([])
    for a in range(len(clip)):
        if a == len(clip)-1:
            wlperpix=np.append(wlperpix,(wlclip[a]-wlclip[0]))
        else:
            wlperpix=np.append(wlperpix,(wlclip[a]-wlclip[a+1]))
    specpix = np.nanmedian(np.abs(wlperpix))
    #print(low, hi, specpix)
    if clipspec==True:
        return wlclip, clip, specpix
    else:
        return specpix

def pixConvolveSPLAT(wldata, specdata, wlstd, specstd, lam, Rdat, Rstd,band):

    dellamdata = lam/Rdat
    dellamstd = lam/Rstd
    low, hi = wldata[0], wldata[len(wldata)-1]
    if low == 1.18:
        hi = 1.35
    medpixdata = medianNpix(wldata, specdata, wldata[0], wldata[len(wldata)-1])
    clipwlstd, specstd, medpixstd = medianNpix(wlstd, specstd, low, hi,clipspec=True)

    dellamK = np.sqrt(np.abs(dellamstd**2-dellamdata**2))
    NpixK = dellamK/medpixdata
    Kernel = Gaussian1DKernel(NpixK/2.355) #switch to sigma
    smoothed = convolve(specdata, Kernel, boundary='extend')

    interpsm = interpolate.interp1d(wldata, smoothed, fill_value='extrapolate')
    intsmoothed = interpsm(clipwlstd)
    return clipwlstd, intsmoothed

SpexTable = pd.read_csv('SPEX-PRISM/spectral_data.txt')
SourceTable = pd.read_csv('SPEX-PRISM/source_data.txt')

# wl, Aspec, Aerr = np.loadtxt('2MASS1728A_fullspec_nocoeff(K3-26-07).dat', unpack=True, skiprows = 3)
# wl, Bspec, Berr = np.loadtxt('2MASS1728B_fullspec_nocoeff(K3-26-07).dat', unpack=True, skiprows = 3)
#
# ABspec = Aspec+Bspec
# location = SourceTable.SOURCE_KEY.loc[SourceTable.NAME=='2MASSW J1728114+394859'].reset_index()
# filename = SpexTable.DATA_FILE[SpexTable.loc[SpexTable.SOURCE_KEY==location.SOURCE_KEY[0]].index[1]]
# spex = fits.getdata('SPEX-PRISM/'+str(filename))
# #spex = spex.transpose()
# wlstd, stdspec = spex[0], spex[1]
# ABspec = bandnorm(wl, ABspec, 'H')
# wl, ABspec = pixConvolveSPLAT(wl, ABspec, wlstd, stdspec, 1.6, 3800, 200, 'H')
#
# plt.figure(1)
# plt.plot(wlstd, bandnorm(wlstd,stdspec, 'H'), label='SPEX')
# plt.plot(wl, ABspec, label='K3-26')
# plt.legend(loc=0)
# plt.xlabel(r'Wavelength($\mu$m)')
# plt.ylabel(r'Normalized Flux')
# plt.ylim(0.3, 1.1)
# plt.xlim(1.4, 2.4)

wl, Aspec, Aerr = np.loadtxt('2MASS1728A_fullspec_nocoeff(K4-23-07)4.3.20.dat', unpack=True, skiprows = 3)
wl, Bspec, Berr = np.loadtxt('2MASS1728B_fullspec_nocoeff(K4-23-07)4.3.20.dat', unpack=True, skiprows = 3)

ABspec = Aspec+Bspec
location = SourceTable.SOURCE_KEY.loc[SourceTable.NAME=='2MASSW J1728114+394859'].reset_index()
filename = SpexTable.DATA_FILE[SpexTable.loc[SpexTable.SOURCE_KEY==location.SOURCE_KEY[0]].index[1]]
spex = fits.getdata('SPEX-PRISM/'+str(filename))
#spex = spex.transpose()
wlstd, stdspec = spex[0], spex[1]
ABspec = bandnorm(wl, ABspec, 'H')
wl, ABspec = pixConvolveSPLAT(wl, ABspec, wlstd, stdspec, 1.6, 3800, 250, 'H')

plt.figure(2)
plt.plot(wlstd, bandnorm(wlstd,stdspec, 'H'), label='SPEX')
plt.plot(wl, ABspec, label='New')
plt.legend(loc=0)
plt.xlabel(r'Wavelength($\mu$m)')
plt.ylabel(r'Normalized Flux')
plt.ylim(0.3, 1.1)
plt.xlim(1.4, 2.4)

wl, Aspec, Aerr = np.loadtxt('2MASS1728A_fullspec_nocoeff(K4-23-07).dat', unpack=True, skiprows = 3)
wl, Bspec, Berr = np.loadtxt('2MASS1728B_fullspec_nocoeff(K4-23-07).dat', unpack=True, skiprows = 3)

ABspec = Aspec+Bspec
location = SourceTable.SOURCE_KEY.loc[SourceTable.NAME=='2MASSW J1728114+394859'].reset_index()
filename = SpexTable.DATA_FILE[SpexTable.loc[SpexTable.SOURCE_KEY==location.SOURCE_KEY[0]].index[1]]
spex = fits.getdata('SPEX-PRISM/'+str(filename))
#spex = spex.transpose()
wlstd, stdspec = spex[0], spex[1]
ABspec = bandnorm(wl, ABspec, 'H')
wl, ABspec = pixConvolveSPLAT(wl, ABspec, wlstd, stdspec, 1.6, 3800, 250, 'H')

plt.figure(3)
plt.plot(wlstd, bandnorm(wlstd,stdspec, 'H'), label='SPEX')
plt.plot(wl, ABspec, label='Old')
plt.legend(loc=0)
plt.xlabel(r'Wavelength($\mu$m)')
plt.ylabel(r'Normalized Flux')
plt.ylim(0.3, 1.1)
plt.xlim(1.4, 2.4)
plt.show()
