#written by Wayne L. Oswald
#last modified 08/22/2019
#Python 3.7.0

'''This code is designed to compare the combined aperture photometry of HD130948BC to our model fits. Ultimately, this should hopefully show that our fits don't suck.'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import stats
import photutils as pu
import pandas as pd
import sys
import astropy

def bandnorm(wl, flux, band):
    if band.upper() == 'J':
        wlLow, wlHi = 1.2700, 1.2900
    elif band.upper() == 'H':
        wlLow, wlHi = 1.700,1.7200
    elif band.upper() == 'K':
        wlLow, wlHi = 2.2000, 2.2200
    else:
        print('Invalid Band')
        wlLow, wlHi =100, 200
    peaks = flux[(np.around(wl, decimals=4)>=wlLow) \
    & (np.around(wl, decimals=4)<=wlHi)]
    peaks = stats.sigma_clip(peaks, sigma=5)
    norm_val = np.nanmean(peaks)
    normflux = flux/norm_val
    return normflux

def bg(data, cx, cy, parbg):
    arr_size = data.shape
    mask = np.ones(arr_size)
    mask[data==0] = 0
    #mask[np.abs(data)>np.abs(parbg*3)] = 0
    data = data*mask
    annular = pu.CircularAnnulus((cx, cy), r_in=12, r_out=20)
    area = pu.aperture_photometry(mask, annular)
    light = pu.aperture_photometry(data, annular)
    summ = light[0]['aperture_sum']
    size = area[0]['aperture_sum']
    avgbg = summ/size
    y, x = np.linspace(0,arr_size[0]-1, arr_size[0]), np.linspace(0, arr_size[1]-1, arr_size[1])
    xg, yg = np.meshgrid(x, y)
    xg -= cx
    yg -= cy
    dist = np.sqrt(xg**2+yg**2)
    loc = data[(dist >=12.) & (data!=0)]
    if len(loc)!=0:
        medbg = np.nanmean(loc)
    else:
        medbg = 0
    if size == 0:
        avgbg = 0
    #mean, median, std = stats.sigma_clipped_stats(data, sigma=3, mask_value=0, maxiters=None)
    #mean, median, std = stats.sigma_clipped_stats(loc, sigma=3, mask_value=0)
    #print(mean, avgbg, median, medbg, parbg)
    #sys.exit()
    return avgbg
    #return parbg

def phot(data, cxA, cyA):
    phot = pu.CircularAperture((cxA, cyA), r=10.5) #r=10 or 11 seems good with fitted bg
    table = pu.aperture_photometry(data, phot)
    summ = table[0]['aperture_sum']

    return summ

#make three gaussian single fit to object
def appphot(imgpath, files, cx, cy, parbg):

    data = fits.getdata(imgpath+files)
    photspec = np.array([])
    a = 0
    size = data.shape
    mask = np.ones(size)
    mask[data==0] = 0
    back_arr = np.array([])
    while a< size[0]:

        frame = data[a,:,:]
        mask = np.ones(frame.shape)
        mask[frame==0] = 0
        backg = bg(frame, cx[a], cy[a], parbg[a])
        back_arr = np.append(back_arr,backg)
        bgsub = frame - backg*mask
        #bgsub = frame - parbg[a]*mask
        photspec = np.append(photspec,phot(bgsub, cx[a], cy[a]))
        a+=1
    return photspec, back_arr

def loadspec(path,files):
    specs = pd.read_csv(path+files)
    return specs.SpecB+specs.SpecC


def loadimgs(path, img, cx, cy, backg):
    appspecs = appphot(path,img, cx, cy, backg)
    return appspecs

def smoothing(flux, wl, width):
    a=0
    smooth, smoothwl = np.array([]), np.array([])
    while a < len(flux):
        n, m = a - width, a+width
        if n < 0:
            n=0
        if m > len(flux)-1:
            m = len(flux)-1
        smooth = np.append(smooth, np.median(flux[n:m], axis = 0))
        smoothwl = np.append(smoothwl, np.median(wl[n:m], axis=0))

        a+=width*2
    return smooth, smoothwl

def parsLoad(path, parsfile):
    parsfile = parsfile.replace('specs', 'pars')
    frame = pd.read_csv(path+parsfile)
    return frame

def make_that_Plot(path, files, imgs, band):

    Jbbwl = 1.1800 + .00015*np.linspace(0, 1574, 1574)
    Hbbwl = 1.473+.0002*np.linspace(0, 1651, 1651)
    Kbbwl = 1.965+.00025*np.linspace(0, 1665, 1665)
    imgpath = path+'img_split/'
    specpath = path+'Python_fitted/'
    parspath = path+'Python_fitted/pars/'
    if band == "H":
        splt = 211
        wl = Hbbwl
    else:
        splt = 212
        wl = Kbbwl

    f = plt.figure(8, figsize=(3.5, 3.5))
    f2 = plt.figure(2, figsize=(3.5, 3.5))
    ax = f.add_subplot(splt)
    ax2 = f2.add_subplot(splt)
    print(band)
    for a in range(len(files)):
        pars = parsLoad(parspath, files[a])
        cx, cy = (2*pars.xcenA1 + pars.sepx)/2, (2*pars.ycenA1 + pars.sepy)/2
        phot_arr, back_arr = loadimgs(specpath, imgs[a],cx, cy, pars.bg)
        # plt.figure(1)
        smooth, smoothwl = smoothing(back_arr/pars.bg, wl, 12)
        ax2.plot(smoothwl, smooth)
        ax2.set_ylim(-0.5, 1.5)
        ax2.hlines(1, min(wl), max(wl), 'k', '--')
        ax2.text(0.7, 0.1, band+'bb', transform=ax2.transAxes, fontsize=10)
        ax2.tick_params(direction="in", which='both')
        ax2.minorticks_on()
        # plt.show()
        #sys.exit()
        model = bandnorm(wl, loadspec(specpath, files[a]), band)
        phot = bandnorm(wl, phot_arr, band)
        comp = model/phot
        #print(np.median(comp))
        smooth, smoothwl = smoothing(comp, wl, 12)
        ax.plot(smoothwl, smooth)
        ax.set_ylim(0.7,1.3)
        ax.hlines(1, min(wl), max(wl), 'k', '--')
        ax.text(0.7, 0.1, band+'bb', transform=ax.transAxes, fontsize=10)
        ax.tick_params(direction="in", which='both')
        ax.minorticks_on()
    if splt == 212:
        ax.set_ylabel(r'                              f$_\lambda$(Model)/f$_\lambda$(Model $\lambda_x$))/f$_\lambda$(Phot)/f$_\lambda$(Phot $\lambda_x$)', fontsize=12)
        ax2.set_ylabel(r'                         Aperture Background / Model Background', fontsize=12)
        ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=12)
        ax2.set_xlabel(r'Wavelength ($\mu$m)', fontsize=12)

plt.clf()
path = '/home/woswald/Desktop/Programming/IDL/Benchmarks/Data/2MASSJ1728/'

hdwarffiles = ['s070422_a006001_Hbb_035neg.fits', 's070422_a006001_Hbb_035pos.fits', 's070422_a011001_Hbb_035neg.fits']
hdwarfimgs = list(hdwarffiles)
for a in range(len(hdwarffiles)):
    hdwarffiles[a] = hdwarffiles[a].replace('.fits','LM2gaussspecs.csv')
    hdwarfimgs[a] = hdwarfimgs[a].replace('.fits', 'LM2gaussdata.fits')
make_that_Plot(path, hdwarffiles, hdwarfimgs, 'H')

kdwarfimgs = ['s070423_a051001_Kbb_035neg.fits', 's070423_a051001_Kbb_035pos.fits', 's070423_a052001_Kbb_035neg.fits', 's070423_a052001_Kbb_035pos.fits']
kdwarffiles=list(kdwarfimgs)
for a in range(len(kdwarffiles)):
    kdwarffiles[a] = kdwarfimgs[a].replace('.fits','LM2gaussspecs.csv')
    kdwarfimgs[a] = kdwarfimgs[a].replace('.fits', 'LM2gaussdata.fits')

make_that_Plot(path, kdwarffiles, kdwarfimgs, 'K')
plt.show()

plt.clf()
#stdfilesJ = ['26Ser_Jbb02_spec_016001Pos0_threegaussfit.fits', '26Ser_Jbb02_spec_016001Neg0_threegaussfit.fits', '26Ser_Jbb02_spec_017001Pos0_threegaussfit.fits', '26Ser_Jbb02_spec_017001Neg0_threegaussfit.fits', '26Ser_Jbb02_spec_018001Pos0_threegaussfit.fits', '26Ser_Jbb02_spec_018001Neg0_threegaussfit.fits', '26Ser_Jbb02_spec_019001Pos0_threegaussfit.fits', '26Ser_Jbb02_spec_019001Neg0_threegaussfit.fits']
stdfilesJ = ['s080426_a016001_Jbb_020posLM2gaussspecs.csv', 's080426_a016001_Jbb_020negLM2gaussspecs.csv', 's080426_a017001_Jbb_020posLM2gaussspecs.csv', 's080426_a017001_Jbb_020negLM2gaussspecs.csv', 's080426_a018001_Jbb_020posLM2gaussspecs.csv', 's080426_a018001_Jbb_020negLM2gaussspecs.csv', 's080426_a019001_Jbb_020posLM2gaussspecs.csv', 's080426_a019001_Jbb_020negLM2gaussspecs.csv']
stdimgsJ = ['s080426_a016001_Jbb_020pos.fits',  's080426_a016001_Jbb_020neg.fits', 's080426_a017001_Jbb_020pos.fits', 's080426_a017001_Jbb_020neg.fits', 's080426_a018001_Jbb_020pos.fits', 's080426_a018001_Jbb_020neg.fits', 's080426_a019001_Jbb_020pos.fits', 's080426_a019001_Jbb_020neg.fits']
scxJ, scyJ = [17.,16.,17.,16.,17.,19.,16.,15.],[10.,11.,10.,11.,10.,12.,9.,12.]
#make_that_Plot(path, stdfilesJ, stdimgsJ, scxJ, scyJ, 'J')

#stdfilesH = ['26Ser_Hbb02_spec_020001Pos0_threegaussfit.fits', '26Ser_Hbb02_spec_020001Neg0_threegaussfit.fits', '26Ser_Hbb02_spec_021001Pos0_threegaussfit.fits', '26Ser_Hbb02_spec_021001Neg0_threegaussfit.fits', '26Ser_Hbb02_spec_022001Pos0_threegaussfit.fits', '26Ser_Hbb02_spec_022001Neg0_threegaussfit.fits', '26Ser_Hbb02_spec_023001Pos0_threegaussfit.fits', '26Ser_Hbb02_spec_023001Neg0_threegaussfit.fits']
stdfilesH = ['s080426_a020001_Hbb_020posLM2gaussspecs.csv','s080426_a020001_Hbb_020negLM2gaussspecs.csv', 's080426_a021001_Hbb_020posLM2gaussspecs.csv','s080426_a021001_Hbb_020negLM2gaussspecs.csv', 's080426_a022001_Hbb_020posLM2gaussspecs.csv','s080426_a022001_Hbb_020negLM2gaussspecs.csv', 's080426_a023001_Hbb_020posLM2gaussspecs.csv','s080426_a023001_Hbb_020negLM2gaussspecs.csv']
stdimgsH = ['s080426_a020001_Hbb_020pos.fits', 's080426_a020001_Hbb_020neg.fits',  's080426_a021001_Hbb_020pos.fits', 's080426_a021001_Hbb_020neg.fits', 's080426_a022001_Hbb_020pos.fits', 's080426_a022001_Hbb_020neg.fits', 's080426_a023001_Hbb_020pos.fits', 's080426_a023001_Hbb_020neg.fits']
scxH, scyH = [15.,15.,15.,15.,15.,15.,15.,14.], [10.,12.,8.,11.,8.,11.,8.,11.]
#make_that_Plot(path, stdfilesH, stdimgsH, scxH, scyH, 'H')

#stdfilesK = ['7Ser_Kbb02_spec_031001Neg0_threegaussfit.fits', '7Ser_Kbb02_spec_031001Pos0_threegaussfit.fits', '7Ser_Kbb02_spec_032001Neg0_threegaussfit.fits', '7Ser_Kbb02_spec_032001Pos0_threegaussfit.fits', '7Ser_Kbb02_spec_033001Neg0_threegaussfit.fits', '7Ser_Kbb02_spec_033001Pos0_threegaussfit.fits', '7Ser_Kbb02_spec_034001Neg0_threegaussfit.fits', '7Ser_Kbb02_spec_034001Pos0_threegaussfit.fits', '7Ser_Kbb02_spec_035001Neg0_threegaussfit.fits', '7Ser_Kbb02_spec_035001Pos0_threegaussfit.fits']
stdfilesK = ['s080426_a031001_Kbb_020posLM2gaussspecs.csv', 's080426_a031001_Kbb_020negLM2gaussspecs.csv', 's080426_a032001_Kbb_020posLM2gaussspecs.csv', 's080426_a032001_Kbb_020negLM2gaussspecs.csv', 's080426_a033001_Kbb_020posLM2gaussspecs.csv', 's080426_a033001_Kbb_020negLM2gaussspecs.csv', 's080426_a034001_Kbb_020posLM2gaussspecs.csv', 's080426_a034001_Kbb_020negLM2gaussspecs.csv', 's080426_a035001_Kbb_020posLM2gaussspecs.csv', 's080426_a035001_Kbb_020negLM2gaussspecs.csv']
stdimgsK = ['s080426_a031001_Kbb_020pos.fits', 's080426_a031001_Kbb_020neg.fits', 's080426_a032001_Kbb_020pos.fits', 's080426_a032001_Kbb_020neg.fits', 's080426_a033001_Kbb_020pos.fits', 's080426_a033001_Kbb_020neg.fits', 's080426_a034001_Kbb_020pos.fits', 's080426_a034001_Kbb_020neg.fits', 's080426_a035001_Kbb_020pos.fits', 's080426_a035001_Kbb_020neg.fits']
scxK, scyK = [14.,14.,13.,13.,14.,12.,13.,13.,13.,13.],[7.,9.,8.,10.,8.,10.,8.,11.,7.,10.]
#make_that_Plot(path, stdfilesK, stdimgsK, scxK, scyK, 'K')
#plt.show()
