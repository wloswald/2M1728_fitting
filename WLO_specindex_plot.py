#Wayne L. Oswald
#Python 3.6
#last modified 5/25/19

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
import pandas as pd

def loadspec(path, name, band,skiprow=0):
    try:
        wl, spec, err = np.loadtxt(path+name, unpack=True, skiprows=skiprow)
    except ValueError:
        wl, spec = np.loadtxt(path+name, unpack=True, skiprows=skiprow)
    if (wl>10).any():
        wl = wl/1000
    return wl, spec
    #normspec = bandnorm(wl, spec, band)
def loadspecMC(path, name, band,skiprow=0):
    try:
        wl, spec, err = np.loadtxt(path+name, unpack=True, skiprows=skiprow)
    except ValueError:
        wl, spec = np.loadtxt(path+name, unpack=True, skiprows=skiprow)
    if (wl>10).any():
        wl = wl/1000
    return wl, spec, err

def getindexFlux(index,wl, flux):
    clipFlux = flux[np.where((np.around(wl, decimals=3) >= (index-0.002))&(np.around(wl,decimals=3) <=(index+0.002)))]
    index = np.nanmedian(clipFlux)
    return index

def specIndices(pathdat, namedat, Indicies, headcount):
    wl, spec = loadspec(pathdat, namedat,'J',headcount)
    #indices defined in McLean et al 2003
    index = []
    if 'H2OA' in Indicies:
        H2OA = getindexFlux(1.343,wl,spec)/getindexFlux(1.313,wl,spec)
        #print('H2OA',H2OA)
        index.append(H2OA)
    if 'H2OB' in Indicies:
        H2OB = getindexFlux(1.456,wl,spec)/getindexFlux(1.570,wl,spec)
        #print('H2OB', H2OB)
        index.append(H2OB)
    if 'H2OC' in Indicies:
        H2OC = getindexFlux(1.788,wl,spec)/getindexFlux(1.722,wl,spec)
        #print('H2OC',H2OC)
        index.append(H2OC)
    if 'H2OD' in Indicies:
        H2OD = getindexFlux(1.964,wl,spec)/getindexFlux(2.075,wl,spec)
        #print('H2OD',H2OD)
        index.append(H2OD)
    if 'CO' in Indicies:
        CO = getindexFlux(2.300,wl,spec)/getindexFlux(2.285,wl,spec)
        #print('CO',CO)
        index.append(CO)
    if 'JFeH' in Indicies:
        JFeH = getindexFlux(1.200,wl,spec)/getindexFlux(1.185,wl,spec)
        #print('JFeH',JFeH)
        index.append(JFeH)
    return index

def specIndicesMC(pathdat, namedat, Indicies, headcount):
    wl, spec, err = loadspecMC(pathdat, namedat,'J',headcount)
    mcspec = spec + np.random.randn(len(err))*err
    #indices defined in McLean et al 2003
    index = []
    if 'H2OA' in Indicies:
        H2OA = getindexFlux(1.343,wl,mcspec)/getindexFlux(1.313,wl,mcspec)
        #print('H2OA',H2OA)
        index.append(H2OA)
    if 'H2OB' in Indicies:
        H2OB = getindexFlux(1.456,wl,mcspec)/getindexFlux(1.570,wl,mcspec)
        #print('H2OB', H2OB)
        index.append(H2OB)
    if 'H2OC' in Indicies:
        H2OC = getindexFlux(1.788,wl,mcspec)/getindexFlux(1.722,wl,mcspec)
        #print('H2OC',H2OC)
        index.append(H2OC)
    if 'H2OD' in Indicies:
        H2OD = getindexFlux(1.964,wl,mcspec)/getindexFlux(2.075,wl,mcspec)
        #print('H2OD',H2OD)
        index.append(H2OD)
    if 'CO' in Indicies:
        CO = getindexFlux(2.300,wl,mcspec)/getindexFlux(2.285,wl,mcspec)
        #print('CO',CO)
        index.append(CO)
    if 'JFeH' in Indicies:
        JFeH = getindexFlux(1.200,wl,mcspec)/getindexFlux(1.185,wl,mcspec)
        #print('JFeH',JFeH)
        index.append(JFeH)
    return index


def specFrame(pathstd, savename):
    dwarflist = open(pathstd+'files.txt').read().split()
    specIndex = pd.DataFrame(data = None, columns = ['SPTYPE','H2OA', 'H2OC', 'H2OD', 'CO','JFeH'])
    for a in range(len(dwarflist)):
        f = open(pathstd+dwarflist[a])
        findtype = f.readlines()[:100]
        headCount = 0
        sptype = None
        for b in findtype:
            if 'SPTYPE' in b:
                sptype = b.split()[3]
            if '#' in b:
                headCount +=1
        if sptype == None:
            sptype = input('Spec type for %s' %dwarflist[a])
        row =  [sptype]
        row = row + specIndices(pathstd, dwarflist[a], ['H2OA', 'H2OC', 'H2OD', 'CO','JFeH'], headCount)
        rowFrame = pd.DataFrame(data=[row], columns = ['SPTYPE','H2OA', 'H2OC', 'H2OD', 'CO','JFeH'])
        specIndex = specIndex.append(rowFrame, ignore_index=True)
    print(specIndex.head())
    specIndex.to_csv(savename)
#measure spec indicies
#specFrame('L_text_091201/', 'IRTF_spec_indicies.csv')
#specFrame('BDSSlowres_compositeset/', 'BDSS_spec_indices.csv')

#IRTF = pd.read_csv('IRTF_spec_indicies.csv')
BDSS = pd.read_csv('BDSS_spec_indices.csv')
a = 0
Bindex, Cindex = np.array([]), np.array([])
while a <1000:
    Bindicies = specIndicesMC('','2MASS1728A_fullspec_nocoeff(K4-23-07)4.3.20.dat', ['H2OC', 'CO'], 3)
    Cindicies = specIndicesMC('','2MASS1728B_fullspec_nocoeff(K4-23-07)4.3.20.dat', ['H2OC', 'CO'], 3)
    if a == 0:
        Bindex = np.array(Bindicies)
        Cindex = np.array(Cindicies)
    else:
        Bindex = np.vstack((Bindex, Bindicies))
        Cindex = np.vstack((Cindex, Cindicies))
    a+=1
Bmean = np.mean(Bindex, axis = 0)
Cmean = np.mean(Cindex, axis = 0)
Bstd = np.std(Bindex, axis=0)
Cstd = np.std(Cindex, axis=0)
print(Bmean, Cmean)


def sptypereplace(frame):
    for a in range(len(frame.SPTYPE)):
        if "T" in frame.SPTYPE[a]:
            frame.SPTYPE[a] = frame.SPTYPE[a].replace("T", "2")
            frame.SPTYPE[a] = float(frame.SPTYPE[a])
        elif "'L" in frame.SPTYPE[a]:
            frame.SPTYPE[a] = frame.SPTYPE[a].replace("'L", "1")
            frame.SPTYPE[a] = float(frame.SPTYPE[a])
        elif "L" in frame.SPTYPE[a]:
            frame.SPTYPE[a] = frame.SPTYPE[a].replace("L", "1")
            frame.SPTYPE[a] = float(frame.SPTYPE[a])
        elif "M" in frame.SPTYPE[a]:
            frame.SPTYPE[a] = frame.SPTYPE[a].replace("M", "0")
            frame.SPTYPE[a] = float(frame.SPTYPE[a])
    return frame

#IRTF = sptypereplace(IRTF)
BDSS = sptypereplace(BDSS)

val = np.arange(0,1.5,.1)
def linefit(SpT, index):
    coeff = np.polyfit(list(index[index.notnull()]), list(SpT[index.notnull()]),1)
    val = np.arange(0,1.5,.1)
    rel = np.polyval(coeff, val)
    arg = SpT - np.polyval(coeff, index)
    std = np.std(arg)
    return coeff, rel, std

H2OCcoeff, H2OCrel, H2OCstd = linefit(BDSS.SPTYPE, BDSS.H2OC)

print("B"+str(np.polyval(H2OCcoeff, Bmean[0]))+'+/-'+str(H2OCstd))
print("C"+str(np.polyval(H2OCcoeff, Cmean[0]))+'+/-'+str(H2OCstd))
print(H2OCcoeff)


plt.clf()
plt.figure(2, figsize=(3.4, 4))
plt.subplot(211)
plt.ylabel('H2OC', fontsize=12)
#plt.plot(IRTF.SPTYPE, IRTF.H2OA, '.k', label='IRTF')
plt.plot(BDSS.SPTYPE, BDSS.H2OC, '.k', label='BDSS')
plt.plot(H2OCrel, val, 'k')
plt.hlines(Bmean[0], 5, 25, 'm', label='2M1728A')
plt.text(20,0.70,'2M1728A', color='m')
plt.text(20, 0.58, '2M1728B', color='g')
plt.xlim(4,29)
plt.ylim(.3, .85)
#plt.hlines(Bstd[0]+Bmean[0], 5, 25, 'm',linestyle = 'dashed', label='HD130948B stdev')
#plt.hlines(Bmean[0]-Bstd[0], 5, 25, 'm', linestyle = 'dashed')
plt.hlines(Cmean[0], 5, 25, 'g', label='2M1728B')
#plt.hlines(Cstd[0]+Cmean[0], 5, 25, 'g', linestyle = 'dashed', label='HD130948C stdev')
#plt.hlines(Cmean[0]-Cstd[0], 5, 25, 'g', linestyle = 'dashed')
plt.xticks([5,10,15,20,25],['M5', 'L0', 'L5', 'T0', "T5"])
plt.tick_params(direction="in", which='both')
plt.minorticks_on()
#plt.legend(fontsize=10, bbox_to_anchor=(0,0.6, 1,1), ncol=2)
plt.subplot(212)
plt.ylabel('CO', fontsize=12)
#plt.plot(IRTF.SPTYPE, IRTF.H2OC, '.k', label='IRTF')
plt.plot(BDSS.SPTYPE, BDSS.CO, '.k', label='BDSS')
plt.hlines(Bmean[1], 5, 25, 'm', label='2M1728A')
#plt.hlines(Bstd[1]+Bmean[1], 5, 25, 'm',linestyle = 'dashed', label='HD130948B stdev')
#plt.hlines(Bmean[1]-Bstd[1], 5, 25, 'm', linestyle = 'dashed')
plt.hlines(Cmean[1], 5, 25, 'g', label='2M1728B')
#plt.hlines(Cstd[1]+Cmean[1], 5, 25, 'g', linestyle = 'dashed', label='HD130948C stdev')
#plt.hlines(Cmean[1]-Cstd[1], 5, 25, 'g', linestyle = 'dashed')
plt.xticks([5,10,15,20,25],['M5', 'L0', 'L5', 'T0', "T5"])
plt.tick_params(direction="in", which='both')
plt.minorticks_on()
plt.ylim(0.63,0.9)
plt.xlim(4,29)

plt.show()
#make this into a nice plot
