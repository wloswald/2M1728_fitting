#
#
#
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def MCBC(mag, spt, band, num):
    lbol = np.array([])
    if band == 'K':
        rms = 0.08
    elif band =='H':
        rms = 0.07
    for a in range(num):
        magval = mag+np.random.randn(1)*rms
        lbol = np.hstack((lbol, BC(magval, spt, band)))
    # plt.clf()
    # plt.figure(1)
    # plt.hist(lbol)
    # plt.show()
    print("Lbol:" +str(np.mean(lbol))+"+/-"+str(np.std(lbol)))

def BC(mag, spt, band):
    if band.upper() == 'K':
        bc = [3.159780E-7, -3.177629E-5, 1.282657E-3, -2.647477E-2, 2.868014E-1, -1.471358, 5.795845]#Liu+ 2010
    elif band.upper() == 'H':
        bc = [1.148133E-6, -1.171595E-4, 4.733874E-3, -9.618535E-2, 1.027185, -5.426683, 13.66709]
    else:
        return
    Mbol = mag + np.polyval(bc, spt)
    logl = (Mbol-4.77)/-2.5
    return logl

def AbsMag(flux, dist, band, f_vega):
    app = -2.5*np.log10(flux/f_vega)
    Abs = app + 5 - 5*np.log10(dist)
    if band.upper() == 'K':
        cf = [0.000454547, -0.03068824, 0.8162709, -10.671188, 68.11147, -172.188]
    elif band.upper() =='H':
        cf = [0.0106200, -0.351721, 3.46876, -13.282]
    logL = np.polyval(cf, Abs)
    #MCBC (Abs, float(input('SpT:')), band, 1000)
    return Abs, logL

def specdiff(wl, B, C, band):
    B = np.nan_to_num(B)
    C = np.nan_to_num(C)
    filt = filter_int(wl, band)
    B_Fbol = np.trapz(B*filt, wl)
    C_Fbol = np.trapz(C*filt, wl)
    Vwl, Vflux = np.loadtxt('Vega99.dat', unpack=True, skiprows=9)
    Vflux = Vflux*10 #convert units
    Vfilt = filter_int(Vwl, band)
    VF_Bol = np.trapz(Vflux*Vfilt, Vwl)
    delK = -2.5*np.log10(B_Fbol/C_Fbol)
    print(band, AbsMag(B_Fbol, 27.4, band, VF_Bol))
    print(band, AbsMag(C_Fbol, 27.4, band, VF_Bol))
    return delK

def printHist(arr):
    plt.clf()
    plt.figure(1)
    plt.hist(arr, bins=25)
    plt.show()

def distMC(par, err, num):
    par_arr = par + np.random.randn(num)*err
    return 1/np.mean(par_arr)

def VegaMC(wl, flux, filt, err):
    V_Fbol = np.trapz(flux*filt, wl)
    return np.random.normal(loc=1, scale=err)*V_Fbol

def specdiffMC(wl, B, C, Berr, Cerr, band, num):
    B = np.nan_to_num(B)
    C = np.nan_to_num(C)
    Berr = np.nan_to_num(Berr)
    Cerr = np.nan_to_num(Cerr)
    filt = filter_int(wl, band)
    Vwl, Vflux = np.loadtxt('Vega99.dat', unpack=True, skiprows=9)
    Vflux = Vflux*10 #convert units
    Vfilt = filter_int(Vwl, band)
    VF_Bol = np.trapz(Vflux*Vfilt, Vwl)
    B_Abs, C_Abs = np.array([]), np.array([])
    B_lbol, C_lbol = np.array([]), np.array([])
    delK = np.array([])
    for a in range(num):
        tmpB = B + np.random.randn(len(Berr))*Berr
        tmpC = C + np.random.randn(len(Cerr))*Cerr
        B_Fbol = np.trapz(tmpB*filt, wl)
        C_Fbol = np.trapz(tmpC*filt, wl)
        dist = distMC(0.0364, 0.0006, 10000)
        VF_Bol = VegaMC(Vwl, Vflux, Vfilt, 0.02)
        delK = np.hstack((delK, -2.5*np.log10(B_Fbol/C_Fbol)))
        tmpB_Abs, tmpB_lbol = AbsMag(B_Fbol, dist, band, VF_Bol)
        tmpC_Abs, tmpC_lbol = AbsMag(C_Fbol, dist, band, VF_Bol)
        B_Abs, C_Abs = np.hstack((B_Abs, tmpB_Abs)), np.hstack((C_Abs, tmpC_Abs))
        B_lbol, C_lbol = np.hstack((B_lbol, tmpB_lbol)), np.hstack((C_lbol, tmpC_lbol))

    if band.upper()=='H':
        uncert = 0.023
    elif band.upper()=='K':
        uncert = 0.04
    B_lbol_arr = B_lbol+np.random.randn(num)*uncert
    C_lbol_arr = C_lbol+np.random.randn(num)*uncert
    # printHist(B_Abs)
    # printHist(C_Abs)
    # printHist(B_lbol)
    # printHist(C_lbol_arr)
    print(band)
    print(f'\delta m: {np.mean(delK):.4f} +/- {np.sqrt(np.std(B_Abs)**2+np.std(C_Abs)**2):.4f}')
    print(f'M B: {np.mean(B_Abs):.4f} +/- {np.std(B_Abs):.4f}')
    print(f'log(lbol/lsun) B: {np.mean(B_lbol_arr):.4f} +/- + {np.std(B_lbol_arr):.4f}')
    print(f'M C: {np.mean(C_Abs):.4f} +/- {np.std(C_Abs):.4f}')
    print(f'log(lbol/lsun) C: {np.mean(C_lbol_arr):.4f} +/- {np.std(C_lbol_arr):.4f}')
    return delK

def filter_int(wl, band):
    if band.upper() == 'K':
        fwl, filt = np.loadtxt('atMKO_K.dat', skiprows=12, unpack=True)
    elif band.upper() == 'KS':
        fwl, filt = np.loadtxt('atMKO_Ks.dat', skiprows=12, unpack=True)
    elif band.upper() == 'KCONT':
        fwl, filt = np.loadtxt('Keck_NIRC2.Kcont.dat', unpack=True)
        fwl = fwl*1e-4
    elif band.upper() =='H':
        fwl, filt = np.loadtxt('atMKO_H.dat', skiprows=12, unpack=True)

    filterCut = filt[(np.around(fwl, 4)>=np.min(wl)) & (np.around(fwl, 4)<=np.max(wl))]
    filterCutwl = fwl[(np.around(fwl, 4)>=np.min(wl)) & (np.around(fwl, 4)<=np.max(wl))]
    interpfilter = interpolate.interp1d(filterCutwl, filterCut, bounds_error=False, fill_value=0)
    filterInt = interpfilter(wl)
    return filterInt

wl, HspecA, Aerr = np.loadtxt('2MASS1728A_H4.3.20tellcor.dat', unpack=True, skiprows=11)
wl, HspecB, Berr = np.loadtxt('2MASS1728B_H4.3.20tellcor.dat', unpack=True, skiprows=11)

specdiffMC(wl, HspecA, HspecB, Aerr, Berr, 'H', 1000)

wl, KspecA, Aerr = np.loadtxt('2MASS1728A_4.23.07_K3.20.20tellcor.dat', unpack=True, skiprows=11)
wl, KspecB, Berr = np.loadtxt('2MASS1728B_4.23.07_K3.20.20tellcor.dat', unpack=True, skiprows=11)

specdiffMC(wl, KspecA, KspecB, Aerr, Berr, 'K', 1000)
