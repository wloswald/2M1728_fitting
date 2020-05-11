#Wayne L. Oswald
#Python 3.7
#last modified 5/16/19

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
import pandas as pd
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import sys

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

def loadspec(path, name, band,skiprow=0):
    #load the spectrum
    wl, spec, err = np.loadtxt(path+name, unpack=True, skiprows=skiprow)
    #convert to microns from nm if necessary
    if (wl>10).any():
        wl = wl/1000
    normspec = bandnorm(wl, spec, band)
    return wl, normspec

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

def chisq(spec, std):
    if len(spec) == len(std):
        diff = spec-std
        diff = diff[np.isfinite(diff)]
        chi = np.sum((diff)**2)
        return chi
    else:
        print("ya messed up, lengths of spec and std not identical")

def pixConvolve(wldata, specdata, pathstd, namestd, lam, Rdat, Rstd, headCount,band, wlstd = None):
    if type(wlstd) != 'numpy.ndarray':
        wlstd, specstd = loadspec(pathstd, namestd,band,headCount)
    else:
        a, specstd = loadspec(pathstd, namestd,band,headCount)
    dellamdata = lam/Rdat
    dellamstd = lam/Rstd

    medpixdata = medianNpix(wldata, specdata, wldata[0], wldata[len(wldata)-1])
    clipwlstd, specstd,medpixstd = medianNpix(wlstd, specstd, wldata[0], wldata[len(wldata)-1],clipspec=True)

    dellamK = np.sqrt(np.abs(dellamstd**2-dellamdata**2))
    NpixK = dellamK/medpixdata
    Kernel = Gaussian1DKernel(NpixK/2.355) #switch to sigma
    smoothed = convolve(specdata, Kernel, boundary='extend')
    #print(NpixK)

    interpsm = interpolate.interp1d(wldata, smoothed, fill_value='extrapolate')
    intsmoothed = interpsm(clipwlstd)
    return clipwlstd, intsmoothed, specstd

def fullspecConvolve(pathstd, namestd, wldat, specdat, headCount, res, lam, wlstd = None):
    if type(wlstd) != 'numpy.ndarray':
        wlstd, specstd = loadspec(pathstd, namestd,'H',headCount)
    else:
        a, specstd = loadspec(pathstd, namestd,'H',headCount)
    dellamdata = lam/3800
    dellamstd = lam/res

    medpixdata = medianNpix(wldat, specdat, wldat[0], wldat[len(wldat)-1])

    dellamK = np.sqrt(np.abs(dellamstd**2-dellamdata**2))
    NpixK = dellamK/medpixdata
    Kernel = Gaussian1DKernel(NpixK/2.355)#switch to sigma
    smoothed = convolve(specdat, Kernel, boundary='extend')
    #print(NpixK)
    low, hi = wldat[0], wldat[len(wldat)-1]
    if low == 1.18:
        hi = 1.35
    clipstd = specstd[(np.around(wlstd, decimals=4)>=low)&(np.around(wlstd, decimals=4)<=hi)]
    wlclip = wlstd[(np.around(wlstd, decimals=4)>=low)&(np.around(wlstd, decimals=4)<=hi)]
    interpsm = interpolate.interp1d(wldat, smoothed, fill_value='extrapolate')
    intsmoothed = interpsm(wlclip)
    return intsmoothed, clipstd, wlclip

def fullspecChisq(pathstd, namestd, wldat, specdat, res, headCount):
    nanloc = np.where(np.isnan(wldat))[0]

    Hwl = wldat[0: nanloc[0]]
    Kwl = wldat[nanloc[0]+1:]
    Hspecdat = specdat[0:nanloc[0]]
    Kspecdat = specdat[nanloc[0]+1:]
    if res == 120 or res == 100 or res == 150 or res == 132:
        Hsmooth, Hstd, Hwlclip = fullspecConvolve(pathstd, namestd, Hwl, Hspecdat, headCount, 129, 1.6)
        Ksmooth, Kstd, Kwlclip = fullspecConvolve(pathstd, namestd, Kwl, Kspecdat, headCount, 213, 2.2)
    elif res == 75 or res == 37 or res == 93 or res == 82:
        Hsmooth, Hstd, Hwlclip = fullspecConvolve(pathstd, namestd, Hwl, Hspecdat, headCount, 83, 1.6)
        Ksmooth, Kstd, Kwlclip = fullspecConvolve(pathstd, namestd, Kwl, Kspecdat, headCount, 139, 2.2)
    elif res == 200 or res == 220 or res == 250:
        Hsmooth, Hstd, Hwlclip = fullspecConvolve(pathstd, namestd, Hwl, Hspecdat, headCount, 207, 1.6)
        Ksmooth, Kstd, Kwlclip = fullspecConvolve(pathstd, namestd, Kwl, Kspecdat, headCount, 348, 2.2)
    else:
        print(res, namestd)
    smooth = np.hstack((Hsmooth, np.nan, Ksmooth))
    std = np.hstack((Hstd, np.nan, Kstd))
    wlclip = np.hstack((Hwlclip, np.nan, Kwlclip))
    return chisq(smooth, std), std, wlclip, smooth

def fullspec(pathdat, namedat, pathstd, lam, obj):
    plt.clf()
    if obj == 'B':
        #Kwlstd, Ksmooth = spectype('', '2MASS1728A_combined_Ktellcor3.20.20.dat',pathstd, 'K', 2.2)
        Kwlstd, Ksmooth = spectype('', '2MASS1728A_4.23.07_K3.20.20tellcor.dat',pathstd, 'K', 2.2)
        print('K')
        Hwlstd, Hsmooth = spectype('', '2MASS1728A_H4.3.20tellcor.dat',pathstd, 'H', 1.6)
        print('H')
    elif obj == 'C':
        #Kwlstd, Ksmooth = spectype('', '2MASS1728B_combined_Ktellcor3.20.20.dat',pathstd, 'K', 2.2)
        Kwlstd, Ksmooth = spectype('', '2MASS1728B_4.23.07_K3.20.20tellcor.dat',pathstd, 'K', 2.2)
        print('K')
        Hwlstd, Hsmooth = spectype('', '2MASS1728B_H4.3.20tellcor.dat',pathstd, 'H', 1.6)
        print('H')

    wldata, specdata = loadspec(pathdat, namedat,'H',3)
    dwarflist = open(pathstd+'spex-files.txt').read().split()

    minChi = np.array([1E6])
    minFiles = np.array(['test'])

    for a in range(len(dwarflist)):
        f = open(pathstd+dwarflist[a])
        findres = f.readlines()[:20]
        headCount = 0
        for b in findres:
            if 'Average resolution' in b:
                res = b.split()[-1]
            if '#' in b:
                headCount +=1
        chi, specstd, wlstd, intsmoothed = fullspecChisq(pathstd, dwarflist[a], wldata, specdata, float(res), headCount)
        if chi <= max(minChi):
            minChi = np.append(minChi, chi)
            minFiles = np.append(minFiles, dwarflist[a])
            if len(minChi) >= 11:
                minFiles = np.delete(minFiles, np.where(minChi==minChi.max()), None)
                minChi = np.delete(minChi,np.where(minChi==minChi.max()),None)
        if chi == min(minChi):
            minspec = specstd
            namespec = dwarflist[a]
            minwl = wlstd
            minsmooth = intsmoothed
    for a in range(len(minChi)):
        print(minFiles[a],minChi[a])
    print('Full')

    specplot(minwl, minsmooth, minspec, namedat, minFiles[np.where(minChi==minChi.min())][0], 'full')
    pdf_page.savefig()

def pixConvolve(wldata, specdata, pathstd, namestd, lam, Rdat, Rstd, headCount,band, wlstd = None):
    if type(wlstd) != 'numpy.ndarray':
        wlstd, specstd = loadspec(pathstd, namestd,band,headCount)
    else:
        a, specstd = loadspec(pathstd, namestd,band,headCount)
    dellamdata = lam/Rdat
    dellamstd = lam/Rstd
    low, hi = wldata[0], wldata[len(wldata)-1]
    if low ==1.18:
        hi=1.35
    medpixdata = medianNpix(wldata, specdata, low, hi)
    clipwlstd, specstd,medpixstd = medianNpix(wlstd, specstd, low, hi, clipspec=True)

    dellamK = np.sqrt(np.abs(dellamstd**2-dellamdata**2))
    NpixK = dellamK/medpixdata
    Kernel = Gaussian1DKernel(NpixK/2.355) #switch to sigma
    smoothed = convolve(specdata, Kernel, boundary='extend')
    #print(NpixK)

    interpsm = interpolate.interp1d(wldata, smoothed, fill_value='extrapolate')
    intsmoothed = interpsm(clipwlstd)
    return clipwlstd, intsmoothed, specstd

def spexres(res, band):
    #there are only so many configurations for spex prism, so these are grouped together depending on what people reported in the SPLAT megafile
    if res == 120 or res == 100 or res == 150 or res == 132:
        if band == 'J':
            res = 96
        elif band == 'H':
            res = 129
        else:
            res = 213
    elif res == 75 or res == 37 or res == 93 or res == 82:
        if band == 'J':
            res = 60
        elif band == 'H':
            res = 83
        else:
            res = 139
    elif res == 200 or res==220 or res == 250:
        if band == 'J':
            res = 150
        elif band == 'H':
            res = 207
        else:
            res = 348
    return res

def spectype(pathdat, namedat, pathstd, band, lam):
    wldata, specdata = loadspec(pathdat, namedat,band,11)
    dwarflist = open(pathstd+'spex-files.txt').read().split()

    minChi = np.array([1E6])
    minFiles = np.array(['test'])
    minspec = np.array([])
    for a in range(len(dwarflist)):
        f = open(pathstd+dwarflist[a])
        findres = f.readlines()[:20]
        headCount = 0
        for b in findres:
            if 'Average resolution' in b:
                res = b.split()[-1]
            if '#' in b:
                headCount +=1
        res = spexres(res, band)
        wlstd, intsmoothed, specstd = pixConvolve(wldata, specdata, pathstd,\
        dwarflist[a],lam, 3800, float(res), headCount, band)
        # if band == 'J':
        #     chi = chisq(intsmoothed[np.around(wlstd, 2)<1.34], specstd[np.around(wlstd, 2)<1.34])
        # else:
        #     chi = chisq(intsmoothed, specstd)
        chi = chisq(intsmoothed, specstd)
        #print(ldwarflist[a], chi)
        if chi <= max(minChi):
            minChi = np.append(minChi, chi)
            minFiles = np.append(minFiles, dwarflist[a])

            if len(minChi) >= 11:
                minFiles = np.delete(minFiles, np.where(minChi==minChi.max()), None)
                minChi = np.delete(minChi,np.where(minChi==minChi.max()),None)
        if chi == min(minChi):
            minspec = specstd
    for a in range(len(minChi)):
        print(minFiles[a],minChi[a])
    wlstd, intsmoothed, specstd = pixConvolve(wldata, specdata, pathstd,\
    minFiles[np.where(minChi==minChi.min())][0],lam, 3800, float(res), headCount, band)
    specplot(wlstd, intsmoothed, specstd, namedat, minFiles[np.where(minChi==minChi.min())][0], band)
    return wlstd, intsmoothed

def specplot(wl, spec, std, objname, stdname, band):
    if band == 'H':
        m = 311
    elif band == 'K':
        m = 312
    else:
        m = 313
    plt.figure(1)
    plt.subplot(m)
    plt.plot(wl, spec, 'k', label=objname, drawstyle='steps-mid')
    plt.plot(wl, std, 'r', label=stdname, drawstyle='steps-mid')
    plt.ylim(.9*min(spec), 1.1*max(spec))
    plt.xlim(.99*min(wl), 1.01*max(wl))
    plt.legend(loc=0, fontsize=6)
    plt.tick_params(axis='both', bottom = True, top=True, left=True, right=True)
    plt.minorticks_on()


def pixConvolveSPLAT(wldata, specdata, pathstd, namestd, lam, Rdat, Rstd,band, wlstd = None):
    std = fits.getdata(pathstd+namestd)
    if std.shape[0] != 3:
        if std.shape[0:1] == (2,3):
            std = np.reshape(std[0], (3,std.shape[2]))
        elif std.shape[0] == 4:
            std = std[0:2]
        else:
            std = std.transpose()
    wlstd, specstd = std[0], std[1]
    specstd = bandnorm(wlstd, specstd, band)
    dellamdata = lam/Rdat
    dellamstd = lam/Rstd
    low, hi = wldata[0], wldata[len(wldata)-1]
    if low == 1.18:
        hi = 1.35
    medpixdata = medianNpix(wldata, specdata, wldata[0], wldata[len(wldata)-1])
    clipwlstd, specstd,medpixstd = medianNpix(wlstd, specstd, low, hi,clipspec=True)

    dellamK = np.sqrt(np.abs(dellamstd**2-dellamdata**2))
    NpixK = dellamK/medpixdata
    Kernel = Gaussian1DKernel(NpixK/2.355) #switch to sigma
    smoothed = convolve(specdata, Kernel, boundary='extend')

    interpsm = interpolate.interp1d(wldata, smoothed, fill_value='extrapolate')
    intsmoothed = interpsm(clipwlstd)
    return clipwlstd, intsmoothed, specstd

def spectypeSPLAT(pathdat, namedat, stdpath, band, lam):
    #load in SPLAT tables and data file
    SpexTable = pd.read_csv(stdpath+'spectral_data.txt')
    wldata, specdata = loadspec(pathdat, namedat,band,11)
    SourceTable = pd.read_csv(stdpath+'source_data.txt')
    #initialize chi-sq pars
    minChi = np.array([1E6])
    minFiles = np.array(['test'])
    minspec = np.array([])
    #select the subset of SPLAT that is L-type, OK quality, and published
    #SpexTable = SpexTable[SpexTable.SPEX_TYPE.str.contains('L', na=False) or SpexTable.SPEX_TYPE.str.contains('T', na=False)]
    SpexTable = SpexTable[(SpexTable.QUALITY_FLAG == 'OK') & (SpexTable.PUBLISHED =='Y')].reset_index()
    #iterate through all SPLAT files
    for a in range(len(SpexTable.DATA_KEY)):
        res = SpexTable.RESOLUTION[a]
        res = spexres(res, band)
        wlstd, intsmoothed, specstd = pixConvolveSPLAT(wldata, specdata, stdpath,\
        SpexTable.DATA_FILE[a],lam, 3800, float(res), band)
        chi = chisq(intsmoothed, specstd)
        #print(ldwarflist[a], chi)
        if chi <= max(minChi) and chi !=0:
            minChi = np.append(minChi, chi)
            minFiles = np.append(minFiles, SpexTable.DATA_FILE[a])

            if len(minChi) >= 11:
                minFiles = np.delete(minFiles, np.where(minChi==minChi.max()), None)
                minChi = np.delete(minChi,np.where(minChi==minChi.max()),None)
        if chi == min(minChi):
            minspec = specstd
            namespec = SpexTable.DATA_FILE[a]
            minwl = wlstd
            minsmooth = intsmoothed
            minname = SpexTable.SOURCE_KEY[a]
            minspectype = SourceTable.LIT_TYPE[SourceTable.loc[SourceTable.SOURCE_KEY==minname].index[0]]
    for a in range(len(minChi)):
        print(minFiles[a],minChi[a])

    obj = SourceTable.NAME[SourceTable.loc[SourceTable.SOURCE_KEY==minname].index[0]]+' '+minspectype
    #obj = minname + ' '+ minspectype
    specplot(minwl, minsmooth, minspec, namedat, obj, band)
    return minwl, minsmooth

def fullspecConvolveSPLAT(pathstd, namestd, wldat, specdat, res, lam, wlstd = None):
    std = fits.getdata(pathstd+namestd)
    if std.shape[0] != 3:
        if std.shape[0:1] == (2,3):
            std = np.reshape(std[0], (3,std.shape[2]))
        elif std.shape[0] == 4:
            std = std[0:2]
        else:
            std = std.transpose()
    wlstd, specstd = std[0], std[1]
    specstd = bandnorm(wlstd, specstd, 'J')
    dellamdata = lam/3800
    dellamstd = lam/res

    medpixdata = medianNpix(wldat, specdat, wldat[0], wldat[len(wldat)-1])

    dellamK = np.sqrt(np.abs(dellamstd**2-dellamdata**2))
    NpixK = dellamK/medpixdata
    Kernel = Gaussian1DKernel(NpixK/2.355)#switch to sigma
    smoothed = convolve(specdat, Kernel, boundary='extend')
    #print(NpixK)
    low, hi = wldat[0], wldat[len(wldat)-1]
    if low == 1.18:
        hi = 1.35
    clipstd = specstd[(np.around(wlstd, decimals=4)>=low)&(np.around(wlstd, decimals=4)<=hi)]
    wlclip = wlstd[(np.around(wlstd, decimals=4)>=low)&(np.around(wlstd, decimals=4)<=hi)]
    interpsm = interpolate.interp1d(wldat, smoothed, fill_value='extrapolate')
    intsmoothed = interpsm(wlclip)
    return intsmoothed, clipstd, wlclip

def fullspecChisqSPLAT(pathstd, namestd, wldat, specdat, res):
    nanloc = np.where(np.isnan(wldat))[0]

    Hwl = wldat[0:nanloc[0]]
    Kwl = wldat[nanloc[0]+1:]

    Hspecdat = specdat[0:nanloc[0]]
    Kspecdat = specdat[nanloc[0]+1:]
    if res == 120 or res == 100 or res == 150 or res == 132:
        Hsmooth, Hstd, Hwlclip = fullspecConvolveSPLAT(pathstd, namestd, Hwl, Hspecdat, 129, 1.6)
        Ksmooth, Kstd, Kwlclip = fullspecConvolveSPLAT(pathstd, namestd, Kwl, Kspecdat, 213, 2.2)
    elif res == 75 or res == 37 or res == 93 or res == 82:
        Hsmooth, Hstd, Hwlclip = fullspecConvolveSPLAT(pathstd, namestd, Hwl, Hspecdat, 83, 1.6)
        Ksmooth, Kstd, Kwlclip = fullspecConvolveSPLAT(pathstd, namestd, Kwl, Kspecdat, 139, 2.2)
    elif res == 200 or res == 220 or res == 250:
        Hsmooth, Hstd, Hwlclip = fullspecConvolveSPLAT(pathstd, namestd, Hwl, Hspecdat, 207, 1.6)
        Ksmooth, Kstd, Kwlclip = fullspecConvolveSPLAT(pathstd, namestd, Kwl, Kspecdat, 348, 2.2)
    else:
        print(res, namestd)
    smooth = np.hstack((Hsmooth, np.nan, Ksmooth))
    std = np.hstack((Hstd, np.nan, Kstd))
    wlclip = np.hstack((Hwlclip, np.nan, Kwlclip))
    return chisq(smooth, std), std, wlclip, smooth

def fullspecSPLAT(pathdat, namedat, pathstd, lam, obj):
    plt.clf()
    if obj == 'B':
        # Kwlstd, Ksmooth = spectypeSPLAT('', '2MASS1728A_combined_Ktellcor3.20.20.dat',pathstd, 'K', 2.2)
        Kwlstd, Ksmooth = spectypeSPLAT('', '2MASS1728A_4.23.07_K3.20.20tellcor.dat',pathstd, 'K', 2.2)
        print('K')
        Hwlstd, Hsmooth = spectypeSPLAT('', '2MASS1728A_H4.3.20tellcor.dat',pathstd, 'H', 1.6)
        print('H')
    elif obj == 'C':
        # Kwlstd, Ksmooth = spectypeSPLAT('', '2MASS1728B_combined_Ktellcor3.20.20.dat',pathstd, 'K', 2.2)
        Kwlstd, Ksmooth = spectypeSPLAT('', '2MASS1728B_4.23.07_K3.20.20tellcor.dat',pathstd, 'K', 2.2)
        print('K')
        Hwlstd, Hsmooth = spectypeSPLAT('', '2MASS1728B_H4.3.20tellcor.dat',pathstd, 'H', 1.6)
        print('H')
    SpexTable = pd.read_csv(pathstd+'spectral_data.txt')
    wldata, specdata = loadspec(pathdat, namedat,'H',3)
    SourceTable = pd.read_csv(pathstd+'source_data.txt')
    minChi = np.array([1E6])
    minFiles = np.array(['test'])
    minspec = np.array([])
    #SpexTable = SpexTable[(SpexTable.SPEX_TYPE.str.contains('L', na=False)) or (SpexTable.SPEX_TYPE.str.contains('T', na=False))]
    SpexTable = SpexTable[(SpexTable.QUALITY_FLAG == 'OK') & (SpexTable.PUBLISHED =='Y')].reset_index()
    for a in range(len(SpexTable.DATA_KEY)):
        res = SpexTable.RESOLUTION[a]
        chi, specstd, wlstd, intsmoothed = fullspecChisqSPLAT(pathstd, SpexTable.DATA_FILE[a], wldata, specdata, float(res))
        if chi <= max(minChi) and chi !=0:
            minChi = np.append(minChi, chi)
            minFiles = np.append(minFiles, SpexTable.DATA_FILE[a])
            if len(minChi) >= 11:
                minFiles = np.delete(minFiles, np.where(minChi==minChi.max()), None)
                minChi = np.delete(minChi,np.where(minChi==minChi.max()),None)
        if chi == min(minChi):
            minspec = specstd
            namespec = SpexTable.DATA_FILE[a]
            minwl = wlstd
            minsmooth = intsmoothed
            minname = SpexTable.SOURCE_KEY[a]
            minspectype = SourceTable.LIT_TYPE[SourceTable.loc[SourceTable.SOURCE_KEY==minname].index[0]]
    for a in range(len(minChi)):
        print(minFiles[a],minChi[a])
    print('Full')

    obj = SourceTable.NAME[SourceTable.loc[SourceTable.SOURCE_KEY==minname].index[0]]+' '+minspectype
    #obj = minname + ' '+ minspectype
    specplot(minwl, minsmooth, minspec, namedat, obj, 'full')
    pdf_page.savefig()

# fullspec('', '2MASS1728A_fullspec_nocoeff.dat', 'nirstd/', 1.6,'B')
# fullspec('', '2MASS1728B_fullspec_nocoeff.dat', 'nirstd/', 1.6,'C')
#
#
# fullspecSPLAT('', '2MASS1728A_fullspec_nocoeff.dat', 'SPEX-PRISM/', 1.6,'B')
# fullspecSPLAT('', '2MASS1728B_fullspec_nocoeff.dat', 'SPEX-PRISM/', 1.6,'C')
pdf_page = PdfPages('SpT_plots_2M1728AB(4.3.20).pdf')

fullspec('', '2MASS1728A_fullspec_nocoeff(K4-23-07)4.3.20.dat', 'nirstd/', 1.6,'B')
fullspec('', '2MASS1728B_fullspec_nocoeff(K4-23-07)4.3.20.dat', 'nirstd/', 1.6,'C')


fullspecSPLAT('', '2MASS1728A_fullspec_nocoeff(K4-23-07)4.3.20.dat', 'SPEX-PRISM/', 1.6,'B')
fullspecSPLAT('', '2MASS1728B_fullspec_nocoeff(K4-23-07)4.3.20.dat', 'SPEX-PRISM/', 1.6,'C')

pdf_page.close()
def getindexFlux(index,wl, flux):
    clipFlux = flux[np.where((np.around(wl, decimals=3) >= (index-0.002))&(np.around(wl,decimals=3) <=(index+0.002)))]
    index = np.nanmedian(clipFlux)
    return index

def specIndices(pathdat, namedat, Indicies):
    wl, spec = loadspec(pathdat, namedat,'H',3)
    #indices defined in McLean et al 2003
    index = []
    if 'H2OA' in Indicies:
        H2OA = getindexFlux(1.343,wl,spec)/getindexFlux(1.313,wl,spec)
        print('H2OA',H2OA)
        index.append(H2OA)
    if 'H2OB' in Indicies:
        H2OB = getindexFlux(1.456,wl,spec)/getindexFlux(1.570,wl,spec)
        print('H2OB', H2OB)
        index.append(H2OB)
    if 'H2OC' in Indicies:
        H2OC = getindexFlux(1.788,wl,spec)/getindexFlux(1.722,wl,spec)
        print('H2OC',H2OC)
        index.append(H2OC)
    if 'H2OD' in Indicies:
        H2OD = getindexFlux(1.964,wl,spec)/getindexFlux(2.075,wl,spec)
        print('H2OD',H2OD)
        index.append(H2OD)
    if 'CO' in Indicies:
        CO = getindexFlux(2.300,wl,spec)/getindexFlux(2.285,wl,spec)
        print('CO',CO)
        index.append(CO)
    if 'JFeH' in Indicies:
        JFeH = getindexFlux(1.200,wl,spec)/getindexFlux(1.185,wl,spec)
        print('JFeH',JFeH)
        index.append(JFeH)
    return index



print('*Spectral Indicies*')
indicies = specIndices('','2MASS1728A_fullspec_nocoeff(K4-23-07)4.3.20.dat', ['H2OC', 'CO'])
print('')
indicies = specIndices('','2MASS1728B_fullspec_nocoeff(K4-23-07)4.3.20.dat', ['H2OC', 'CO'])

sys.exit()

def smoothSpec(wl, spec):
    dellamdata = 1.3/3800
    dellamBDSS = 1.3/2000
    wlBDSS, specBDSS = np.loadtxt('BDSSlowres_compositeset/2mass0015.dat', skiprows=6, unpack=True)
    medpixdata = medianNpix(wl, spec, wl[0], wl[len(wl)-1])

    dellamK = np.sqrt(np.abs(dellamBDSS**2-dellamdata**2))
    NpixK = dellamK/medpixdata
    Kernel = Gaussian1DKernel(NpixK/2.355)#switch to sigma
    smoothed = convolve(spec, Kernel, boundary='extend')
    #print(NpixK)
    low, hi = wl[0], wl[len(wl)-1]
    clipstd = specBDSS[(np.around(wlBDSS, decimals=4)>=low)&(np.around(wlBDSS, decimals=4)<=hi)]
    wlclip = wlBDSS[(np.around(wlBDSS, decimals=4)>=low)&(np.around(wlBDSS, decimals=4)<=hi)]
    interpsm = interpolate.interp1d(wl, smoothed, fill_value='extrapolate')
    intsmoothed = interpsm(wlclip)
    plt.figure(1)
    plt.plot(wlBDSS, specBDSS)
    plt.plot(wlclip, intsmoothed)
    #plt.show()
    return wlclip, intsmoothed

def equivWidthMCBDSS(pathdat, namedat, equiv_wl, num_runs, headcount, smooth=False):
    #not all of the BDSS sample has an error spectrum, but I need to load them either way
    try:
        wl, spec = np.loadtxt(pathdat+namedat, unpack=True, skiprows=headcount)
    except ValueError:
        wl, spec, err = np.loadtxt(pathdat+namedat, unpack=True, skiprows=headcount)
    #convert nmto microns if necessary
    if wl.any()> 10:
        wl=wl/1000
    if smooth == True:
        wl, spec = smoothSpec(wl, spec)

    a = 0
    wlKI = wl[np.where(np.abs(wl-equiv_wl) <=.02)]
    specKI = spec[np.where(np.abs(wl-equiv_wl) <=.02)]
    widths = np.array([])
    #hi and low are the continuum points, while hilim and lowlim are the edges of the band
    while a < num_runs:
        if equiv_wl == 1.244:
            hi, low = 1.248,1.233
            lowlim, hilim = 1.2415, 1.246
        elif equiv_wl == 1.253:
            hi,low = 1.260,1.248
            lowlim, hilim = 1.2498,1.2558
        #estimate local continuum by finding median in .002 micron window on either side of line
        hi_wid, low_wid = 0.001, 0.001
        #vary the location of the continuum points by up to 10% of the sample bandwidth
        hi = hi+(hilim-lowlim)*0.1*np.random.uniform(-1, 1)
        if hi < hilim:
            hi = hilim
        low = low+(hilim-lowlim)*0.1*np.random.uniform(-1, 1)
        if low > lowlim:
            low= lowlim
        #vary the width of the continuum window by up to 50%
        hi_wid = hi_wid*(1+.5*np.random.uniform(-1, 1))
        low_wid = low_wid*(1+.5*np.random.uniform(-1, 1))
        #create a sample wl and spec without the line center
        sampwl = wl[np.where(wl < lowlim)]
        sampwl = np.append(sampwl, wl[np.where(wl > hilim)])
        sampspec = spec[np.where(wl < lowlim)]
        sampspec = np.append(sampspec, spec[np.where(wl > hilim)])
        #create a sample of wl and spec in the window around the continuum point
        window_wl = sampwl[np.where(np.abs(sampwl-hi)<=hi_wid)]
        window_wl = np.append(window_wl, np.nan)
        window_wl = np.append(window_wl, sampwl[np.where(np.abs(sampwl-low)<=low_wid)])
        window_spec = sampspec[np.where(np.abs(sampwl-hi)<=hi_wid)]
        window_spec = np.append(window_spec, np.nan)
        window_spec = np.append(window_spec, sampspec[np.where(np.abs(sampwl-low)<=low_wid)])
        #take the median value of each continuum point
        hi_cont = np.median(sampspec[np.where(np.abs(sampwl-hi)<=hi_wid)])
        low_cont = np.median(sampspec[np.where(np.abs(sampwl-low)<=low_wid)])
        cont = interpolate.interp1d([hi,low], [hi_cont,low_cont], fill_value = 'extrapolate')
        wlinter = wl[(lowlim < wl) & (wl< hilim)]

        continuum = cont(wlinter)
        #EW measured by summing interpolated continuum minus line between certain limits and multiplying by resolution element in angstroms
        specEW = spec[np.where((wl > lowlim)&(wl<hilim))]
        # plt.figure(1)
        # plt.plot(wlKI, specKI, 'k')
        # plt.plot(wlinter, continuum, 'g')
        # plt.plot(window_wl, window_spec, 'r--')
        # plt.minorticks_on()
        # plt.grid(which='both')
        # plt.show()
        specNorm = specEW/continuum
        specint = 1-specNorm #this seems more like what I understand this process to be
        widths = np.append(widths, integrate.trapz(specint,wlinter*1e4))
        a+=1
    #return the mean and standard deviation of all of the runs of eq_width MC distribution
    return np.mean(widths), np.std(widths)

dwarflist = open('BDSSlowres_compositeset/files.txt').read().split()
df = pd.DataFrame(columns=['name', 'KI_1243', "err_1243", "KI_1254", "err_1254"])
#data from McLean 2003 - the BDSS
MCK1243 =np.array([3.5,3.3,4.3,4.6,5.5,5.1,6.1,5.9,6.9,6.8,7.6,7.3,6.8,8.2,7.6,7.3,7.7,7.1,6.9,6.7,8.4,\
8.3,8.5,7.3,7.5,7.2,7.0,4.3,4.2,1.8,4.3,5.6,3.9,3.3,3.2,3.2,3.0,4.4,np.nan,4.7,4.7,4.9,4.7,5.1,5.1,4.9,5.5,\
1.4,0.2,2.3,2.2,1.7,1.0])
MCK1254 = np.array([3.2,3.0,3.6,4.6,5.2,4.6,5.2,5.2,6.4,6.1,6.9,6.5,6.6,8.8,7.7,6.6,7.4,6.6,6.8,4.8,\
8.9,7.8,9.3,7.9,7.5,7.3,8.2,5.3,5.4,2.3,6.0,7.1,5.3,5.3,4.8,5.6,3.5,6.2,6.0,7.7,7.7,7.7,8.2,8.4,7.7,\
8.7,9.0,5.1,2.0,5.2,4.8,2.6,1.8])
MCObj = ['wolf359.dat', 'gl283b.dat', 'lhs2351.dat', 'vb8.dat', 'lp412-31.dat', 'vb10.dat', \
'lhs2065.dat', '2mass1239.dat', '2mass0345.dat', 'hd89744b.dat', '2mass0746.dat', '2mass0208.dat', \
'2mass1035.dat', '2mass1300.dat', '2mass1439.dat', '2mass1658.dat', '2mass2130.dat', '2mass0015.dat',\
'kelu1.dat', '2mass1726.dat', '2mass1506.dat', '2mass1615.dat', '2mass0036.dat', 'gd165b.dat', \
'2mass2158.dat', 'denis1228.dat', '2mass1507.dat', '2mass0103.dat', '2mass0850.dat', '2mass2244.dat',\
'denis0205.dat', '2mass1728.dat', '2mass0310.dat', '2mass0328.dat', 'gl337c.dat', 'gl584c.dat',\
'2mass1632.dat', 'sdss0423.dat', 'sdss0151.dat', 'sdss0837.dat', 'sdss1254.dat' ,'sdss1021.dat',\
'sdss1750.dat', 'sdss0926.dat', '2mass2254.dat', '2mass0559.dat', '2mass2356.dat', 'sdss1624.dat',\
'2mass1237.dat', '2mass0727.dat', '2mass1553.dat', 'gl570d.dat', '2mass0415.dat']
MCspectype = [6,6,7,7,8,8,9,9,10,10,10.5,11,11,11,11,11,11,12,12,12,13,13,13.5,14,14,15,15,16,16,\
16.5,17,17,18,18,18,18,18,20,21,21,22,23,23.5,24.5,25,25,26,26,26.5,27,27,28,28]
d = {'MCKI1243':pd.Series(MCK1243, index=MCObj), 'MCKI1254':pd.Series(MCK1254, index=MCObj), \
'SpecType':pd.Series(MCspectype, index=MCObj)}
df2 = pd.DataFrame(d)
for a in range(len(dwarflist)):
    f = open('BDSSlowres_compositeset/'+dwarflist[a])
    findhead = f.readlines()[:20]
    headCount = 0
    for b in findhead:
        if '#' in b:
            headCount +=1
    #print(dwarflist[a])
    K1244, err1244 = equivWidthMCBDSS('BDSSlowres_compositeset/', dwarflist[a], 1.244, 1000, headCount)
    K1253, err1253 = equivWidthMCBDSS('BDSSlowres_compositeset/', dwarflist[a], 1.253, 1000, headCount)
    row = pd.DataFrame([[dwarflist[a], K1244, err1244, K1253, err1253]], columns=['name', 'KI_1243', "err_1243","KI_1254","err_1254"])
    df = df.append(row)

b = 0
ind = {}
for a in df.name:
    ind.update({b:a})
    b+=1
df = df.reset_index()
df = df.rename(index=ind)
df = df.drop(labels=['name'], axis=1)
df = df.drop(labels=['index'], axis=1)
df['MCKI_1243'] = pd.Series(df2.MCKI1243, index=df2.index)
df['MCKI_1254'] = pd.Series(df2.MCKI1254, index=df2.index)
df['diff1243'] = pd.Series(df.MCKI_1243-df.KI_1243, index=df2.index)
df['diff1254'] = pd.Series(df.MCKI_1254-df.KI_1254, index=df2.index)
df['SPTYPE'] = pd.Series(df2.SpecType, index=df2.index)
#print(df.describe())
print('*Equivalent Width KI 1.244 microns*')
print(equivWidthMCBDSS('','HD130948B_fullspec_nocoeff12.9.19_2015.dat',1.244, 1000, 3, smooth=True))
print(equivWidthMCBDSS('','HD130948C_fullspec_nocoeff12.9.19_2015.dat',1.244, 1000, 3, smooth=True))

print('*Equivalent Width KI 1.253 microns*')
print(equivWidthMCBDSS('','HD130948B_fullspec_nocoeff12.9.19_2015.dat',1.253, 1000, 3, smooth=True))
print(equivWidthMCBDSS('','HD130948C_fullspec_nocoeff12.9.19_2015.dat',1.253, 1000, 3, smooth=True))

#df.to_csv('McLean_KI_EQ.csv')

    #for long wavelength pair the continuum points were centered at 1.233, 1.248, and 1.260 lm. From these points, a linear interpolation to determine continuum across line

    #limits were 1.167–1.171, 1.175–1.180, 1.2415–1.246, and 1.2498–1.2558 lm.

    # To determine an uncertainty for the equivalent width, the location of the continuum reference points was allowed to vary by about 10% of the sample bandwidth and EW recalculated
    #from McLean et al. 2003
