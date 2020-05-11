#written by Wayne L. Oswald
#last modified 6/2/19
#Python 3.6.0

#The idea is to reproduce the images from Bowler 2016 with my own data

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def medianFinder(path,files):

    for a in range(len(files)):
        b = fits.getdata(path+files[a])
        s = b.shape
        N = s[1]*s[2]
        for wl in range(s[0]):
            c = np.sqrt(1/N*np.sum(b[wl,:,:]*b[wl,:,:]))
            if wl == 0:
                rms = c
            else:
                rms = np.append(rms, c)

        if a ==0:
            rmsresid=rms
        else:
            rmsresid=np.vstack((rmsresid, rms))

    return np.median(rmsresid)

def AvgPeak(path, files):

    for a in range(len(files)):

        b = fits.getdata(path+files[a])
        n = np.array([])
        for a in b:
            n = np.append(n, np.max(a))

    return np.median(n)

#loads a series of data to compute the median image of each datacube
def loadMedCube(path,files):
    modelspecs = np.array([])
    for a in range(len(files)):
        b = fits.getdata(path+files[a])
        b = np.median(b, axis=0)

        if a ==0:
            cube=b
        else:
            cube=np.dstack((cube, b))
    medcube = np.median(cube, axis = 2)
    return medcube


def mediantestplot(files, band):

    path = '/home/woswald/Desktop/Programming/IDL/Benchmarks/Data/2MASSJ1728/Python_fitted/'

    medDatafiles = list(files)
    medModelfiles = list(files)
    medResidfiles = list(files)

    for a in range(len(files)):
        medDatafiles[a] = files[a]+'data.fits'
        medModelfiles[a] = files[a]+'modelBC.fits'
        #medModelfiles[a] = files[a]+'model.fits'
        medResidfiles[a] = files[a]+'resid.fits'


    medData = loadMedCube(path, medDatafiles)

    medModel = loadMedCube(path, medModelfiles)

    medResid = loadMedCube(path, medResidfiles)

    medResidval = medianFinder(path, medResidfiles)

    avgPeakval = AvgPeak(path, medDatafiles)

    print('Median RMS Residuals = %(value)s; Avg Peak Value = %(peak)s'%{'value':medResidval, 'peak':avgPeakval})

    #plt.clf()
    m=0
    if band == 'h':
        color = 'Greens'
        k=0
    elif band == 'k':
        color = 'Reds'
        k=1
    med = np.hstack((medData, medModel))
    mn, mx = round(med.min(),3), round(med.max(),3)
    md = round((mx+mn)/2, 3)
    fig = plt.figure(2, figsize=(8,4), constrained_layout=True)
    spec = fig.add_gridspec(ncols = 3, nrows=2, width_ratios=[.8, 1.1, 1.1], height_ratios=[1.,1.])
    ax = fig.add_subplot(spec[k,m])
    if k==0:
        plt.title('Data', fontsize=16)
    im = plt.imshow(medData, interpolation='nearest',cmap=color, origin='lower', vmin = mn, vmax=mx)
    plt.xticks([], [])
    plt.yticks([], [])
    m+=1
    ax = fig.add_subplot(spec[k,m])
    if k==0:
        plt.title('Model', fontsize=16)
    im = plt.imshow(medModel, interpolation='nearest',cmap=color,origin='lower')
    plt.xticks([], [])
    plt.yticks([], [])
    cb = add_colorbar(im)
    cb.set_ticks([mn,md,mx])
    cb.set_ticklabels([mn,md,mx])
    plt.clim(mn*.99, mx*1.01)
    m+=1
    ax = fig.add_subplot(spec[k,m])
    if k==0:
        plt.title('Residuals', fontsize=16)
    im = plt.imshow(medResid, interpolation='nearest',cmap=color,origin='lower')
    plt.xticks([], [])
    plt.yticks([], [])
    mn, mx = round(medResid.min(),4), round(medResid.max(),4)
    md = round((mx+mn)/2, 4)
    print([mn,md,mx])
    cb = add_colorbar(im)
    plt.clim(mn*1.001, mx*1.001)
    cb.set_label('Counts (DN)', rotation=270, fontsize=12, labelpad=12)
    cb.set_ticks([mn,md,mx])
    cb.set_ticklabels([mn,md,mx])


dwarffilesHA = ['s070422_a006001_Hbb_035negLM2gauss', 's070422_a006001_Hbb_035posLM2gauss', 's070422_a011001_Hbb_035negLM2gauss']
dwarffilesHA = ['s070422_a006001_Hbb_035negLM2gauss', 's070422_a006001_Hbb_035posLM2gauss']

dwarffilesKA = ['s070423_a051001_Kbb_035negLM2gauss', 's070423_a051001_Kbb_035posLM2gauss', 's070423_a052001_Kbb_035negLM2gauss', 's070423_a052001_Kbb_035posLM2gauss']


stdfilesJ = ['s080426_a016001_Jbb_020posLM2gauss', 's080426_a016001_Jbb_020negLM2gauss','s080426_a017001_Jbb_020posLM2gauss', 's080426_a017001_Jbb_020negLM2gauss', 's080426_a018001_Jbb_020posLM2gauss', 's080426_a018001_Jbb_020negLM2gauss', 's080426_a019001_Jbb_020posLM2gauss', 's080426_a019001_Jbb_020negLM2gauss']
stdimgsJ = ['s080426_a016001_Jbb_020pos.fits', 's080426_a016001_Jbb_020neg.fits','s080426_a017001_Jbb_020pos.fits', 's080426_a017001_Jbb_020neg.fits', 's080426_a018001_Jbb_020pos.fits', 's080426_a018001_Jbb_020neg.fits', 's080426_a019001_Jbb_020pos.fits', 's080426_a019001_Jbb_020neg.fits']
scxJ, scyJ = [17.,16.,17.,16.,17.,19.,16.,15.],[10.,11.,10.,11.,10.,12.,9.,12.]

stdfilesH = ['s080426_a020001_Hbb_020posLM2gauss', 's080426_a020001_Hbb_020negLM2gauss', 's080426_a021001_Hbb_020posLM2gauss', 's080426_a021001_Hbb_020negLM2gauss', 's080426_a022001_Hbb_020posLM2gauss', 's080426_a022001_Hbb_020negLM2gauss', 's080426_a023001_Hbb_020posLM2gauss', 's080426_a023001_Hbb_020negLM2gauss']

stdimgsH = ['s080426_a020001_Hbb_020pos.fits', 's080426_a020001_Hbb_020neg.fits', 's080426_a021001_Hbb_020pos.fits', 's080426_a021001_Hbb_020neg.fits', 's080426_a022001_Hbb_020pos.fits', 's080426_a022001_Hbb_020neg.fits', 's080426_a023001_Hbb_020pos.fits', 's080426_a023001_Hbb_020neg.fits']
scxH, scyH = [15.,15.,15.,15.,15.,14.,15.,14.],[10.,12.,8.,11.,8.,11.,8.,11.]

stdfilesK = ['s080426_a031001_Kbb_020posLM2gauss', 's080426_a031001_Kbb_020negLM2gauss', 's080426_a032001_Kbb_020posLM2gauss', 's080426_a032001_Kbb_020negLM2gauss', 's080426_a033001_Kbb_020posLM2gauss', 's080426_a033001_Kbb_020negLM2gauss', 's080426_a034001_Kbb_020posLM2gauss', 's080426_a034001_Kbb_020negLM2gauss', 's080426_a035001_Kbb_020posLM2gauss', 's080426_a035001_Kbb_020negLM2gauss']
stdimgsK = ['s080426_a031001_Kbb_020pos.fits', 's080426_a031001_Kbb_020neg.fits', 's080426_a032001_Kbb_020pos.fits', 's080426_a032001_Kbb_020neg.fits', 's080426_a033001_Kbb_020pos.fits', 's080426_a033001_Kbb_020neg.fits', 's080426_a034001_Kbb_020pos.fits', 's080426_a034001_Kbb_020neg.fits', 's080426_a035001_Kbb_020pos.fits', 's080426_a035001_Kbb_020neg.fits']
scxK, scyK = [14.,14.,13.,13.,14.,12.,13.,13.,13.,13.],[7.,9.,8.,10.,8.,10.,8.,11.,7.,10.]

mediantestplot(dwarffilesHA,'h')
mediantestplot(dwarffilesKA, 'k')
plt.show()
# plt.clf()
# mediantestplot(stdfilesJ, 'j')
# mediantestplot(stdfilesH,'h')
# mediantestplot(stdfilesK, 'k')
# plt.show()
