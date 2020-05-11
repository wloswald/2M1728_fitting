#Written by Wayne L. Oswald
#Python3.7
#last modified
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, Parameter, minimize, Minimizer, report_fit, Model
import math
import matplotlib
import scipy.optimize as opt
from astropy.io import fits
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib.backends.backend_pdf import PdfPages
import photutils as pu
import sys
import pandas as pd

def progress(frame, total):
    percent = frame/total*100

#copy of gaussian2d.pro
def gauss2d(x_size, y_size, x_c, y_c, x_w, y_w, angle):

    s = [round(x_size), round(y_size)]
    c = [x_c, y_c]
    sigma = np.array([x_w, y_w])
    if angle == None:
        a = 0
    else:
        a=angle
    min_sigma = 1E-2
    if min(sigma) < min_sigma:
        gaussian = np.zeros(s[0], s[1])
        gaussian[round(c[0]), round(c[1])] = 1
        return gaussian
    w = math.sqrt(2)*sigma
    if w[0] == w[1]:
        a = 0
        #define array of x- & y-centered coords
    x, y = np.arange(0,s[0],1)-c[0], np.arange(0,s[1],1)-c[1]
    xx, yy = np.meshgrid(x, y)
    #compute x-gauss
    if a !=0:
        z = xx*math.cos(a)+yy*math.sin(a)
    else:
        z=xx
    gaussian = np.exp(-(z/w[0])**2)
    #multiply by y-gauss
    if a !=0:
        z = -xx*math.sin(a)+yy*math.cos(a)
    else:
        z=yy
    gaussian = gaussian*np.exp(-(z/w[1])**2)
    return gaussian


def twothreegauss(x,y,bg, ampA1, x_size, y_size, xcenA1, ycenA1,xwid1, ywid1,rot1,\
ampA2, xcenA2, ycenA2, xwid2, ywid2,rot2,ampA3, xcenA3, ycenA3, xwid3, \
ywid3, rot3, ampB1, sepx, sepy):
    x_size=float(x_size)
    y_size=float(y_size)
    xcenA1, ycenA1, xcenA2, ycenA2, xcenA3, ycenA3 = float(xcenA1), float(ycenA1), float(xcenA2), float(ycenA2), float(xcenA3), float(ycenA3)
    sepx, sepy = float(sepx), float(sepy)
    g1 = bg + ampA1*gauss2d(x_size,y_size,xcenA1,ycenA1,xwid1,ywid1,rot1)
    g2 = ampA2*gauss2d(x_size,y_size,xcenA2,ycenA2,xwid2,ywid2,rot2)
    g3 = ampA3*gauss2d(x_size,y_size,xcenA3,ycenA3,xwid3,ywid3,rot3)

    g4 = ampB1*gauss2d(x_size,y_size,xcenA1 +sepx,ycenA1 +sepy,xwid1,ywid1,rot1)
    g5 =  ampB1*ampA2/ampA1*gauss2d(x_size,y_size,xcenA2+sepx,ycenA2+sepy,xwid2,ywid2,rot2)
    g6 = ampB1*ampA3/ampA1*gauss2d(x_size,y_size,xcenA3+sepx,ycenA3+sepy,xwid3,ywid3,rot3)

    return np.transpose(g1+g2+g3+g4+g5+g6)

def twotwogauss(x,y,bg, ampA1, x_size, y_size, xcenA1, ycenA1,xwid1, ywid1,rot1,\
ampA2, xcenA2, ycenA2, xwid2, ywid2,rot2, ampB1, sepx, sepy):
    x_size=float(x_size)
    y_size=float(y_size)
    xcenA1, ycenA1, xcenA2, ycenA2 = float(xcenA1), float(ycenA1), float(xcenA2), float(ycenA2)
    sepx, sepy = float(sepx), float(sepy)
    g1 = bg + ampA1*gauss2d(x_size,y_size,xcenA1,ycenA1,xwid1,ywid1,rot1)
    g2 = ampA2*gauss2d(x_size,y_size,xcenA2,ycenA2,xwid2,ywid2,rot2)

    g3 = ampB1*gauss2d(x_size,y_size,xcenA1 +sepx,ycenA1 +sepy,xwid1,ywid1,rot1)
    g4 =  ampB1*ampA2/ampA1*gauss2d(x_size,y_size,xcenA2+sepx,ycenA2+sepy,xwid2,ywid2,rot2)

    return np.transpose(g1+g2+g3+g4)

def stdtwothreegauss(x,y, bg, ampA1, x_size, y_size, xcenA1, ycenA1,xwid1, ywid1,rot1,\
ampA2, xcenA2, ycenA2, xwid2, ywid2,rot2,ampA3, xcenA3, ycenA3, xwid3, \
ywid3, rot3):

    g1 = bg + ampA1*gauss2d(x_size,y_size,xcenA1,ycenA1,xwid1,ywid1,rot1)
    g2 = ampA2*gauss2d(x_size,y_size,xcenA2,ycenA2,xwid2,ywid2,rot2)
    g3 = ampA3*gauss2d(x_size,y_size,xcenA3,ycenA3,xwid3,ywid3,rot3)

    return np.transpose(g1+g2+g3)

def stdtwotwogauss(x,y, bg, ampA1, x_size, y_size, xcenA1, ycenA1,xwid1, ywid1,rot1,\
ampA2, xcenA2, ycenA2, xwid2, ywid2,rot2):

    g1 = bg + ampA1*gauss2d(x_size,y_size,xcenA1,ycenA1,xwid1,ywid1,rot1)
    g2 = ampA2*gauss2d(x_size,y_size,xcenA2,ycenA2,xwid2,ywid2,rot2)

    return np.transpose(g1+g2)

def bg(data, cx, cy):
    arr_size = data.shape
    mask = np.ones(arr_size)
    mask[data==0] = 0
    #mask[np.abs(data)>np.abs(parbg*3)] = 0
    # data = data*mask
    # annular = pu.CircularAnnulus((cx, cy), r_in=11, r_out=20)
    # area = pu.aperture_photometry(mask, annular)
    # light = pu.aperture_photometry(data, annular)
    # summ = light[0]['aperture_sum']
    # size = area[0]['aperture_sum']
    # avgbg = summ/size
    y, x = np.linspace(0,arr_size[0]-1, arr_size[0]), np.linspace(0, arr_size[1]-1, arr_size[1])
    xg, yg = np.meshgrid(x, y)
    xg -= cx
    yg -= cy
    dist = np.sqrt(xg**2+yg**2)
    loc = data[(dist >=12.) & (data!=0)]
    if len(loc)!=0:
        medbg = np.nanmedian(loc)
    else:
        medbg = 0
    # if size == 0:
    #     avgbg = 0
    return medbg

def get_fit2stdLM(fname, xcen, ycen, band):#for the standard star
    print(fname)
    data = fits.getdata(fname)
    fname = fname.replace('.fits', 'LM2gausspars.csv')
    pars = Parameters()
    parnames = ['bg', 'ampA1', 'x_size', "y_size", 'xcenA1', 'ycenA1','xwid1', 'ywid1','rot1',\
    'ampA2', 'xcenA2', 'ycenA2', 'xwid2', 'ywid2','rot2']
#build model for standard star and make spectrum
#normalize and compare to brendan for binary - compare standard to aperture photometry
    size = np.shape(data)
    Yin, Xin = np.mgrid[0:int(size[1]), 0:int(size[2])]
    fail = np.array([])
    #yl, yu = ycen-11, ycen+11
    #data = data[:,:,int(yl):int(yu)
    medimg = np.nanmedian(data, axis=0)
    mask = np.ones_like(medimg)
    mask[medimg==0] = 0.
    medimg[medimg==0] = np.median(medimg)
                   #name, Value, Vary, min, max
    pars.add_many(('bg', 0, True, -1., None),
                  ('ampA1',300., True, 0., None),
                  ('x_size', int(size[1]), False, None, None),
                  ('y_size', int(size[2]), False, None, None),
                  ('xcenA1',xcen, True, 0., 20.),
                  ('ycenA1',ycen, True, 0., 32.),
                  ('xwid1',1, True, 0.1, 5.),
                  ('ywid1',1, True, 0.1, 5.),
                  ('rot1', 0., True, None, None),
                  ('ampA2',100, True, 0., None),
                  ('xcenA2',xcen, True, 0., 20.),
                  ('ycenA2',ycen, True, 0., 32.),
                  ('xwid2',1.5, True, 0.1, 5.),
                  ('ywid2',1.5, True, 0.1, 5.),
                  ('rot2', 0, True, None, None))

    fxn = Model(stdtwotwogauss, independent_vars=['x', 'y'])
    results = fxn.fit(medimg, x=Xin, y=Yin, params=pars, weights = mask)
    plt.subplot(211)
    plt.imshow(medimg)
    plt.subplot(212)
    plt.imshow(results.best_fit)
    plt.show()
    # print(results.fit_report())
    # sys.exit()
    savedata = []
    pars2 = Parameters()
                   #name, Value, Vary, min, max
    pars2.add_many(('bg', results.params[parnames[0]].value, True, -1., None),
                   ('ampA1',results.params[parnames[1]].value, True, 0., None),
                   ('x_size',results.params[parnames[2]].value, False, None, None),
                   ('y_size',results.params[parnames[3]].value, False, None, None),
                   ('xcenA1',results.params[parnames[4]].value, False, 0., 20.),
                   ('ycenA1',results.params[parnames[5]].value, False, 0., 32.),
                   ('xwid1',results.params[parnames[6]].value, True, 0.1, 5.),
                   ('ywid1',results.params[parnames[7]].value, True, 0.1, 5.),
                   ('rot1', results.params[parnames[8]].value, True, None, None),
                   ('ampA2',results.params[parnames[9]].value, True, 0., None),
                   ('xcenA2',results.params[parnames[10]].value, False, 0., 20.),
                   ('ycenA2',results.params[parnames[11]].value, False, 0., 32.),
                   ('xwid2',results.params[parnames[12]].value, True, 0.1, 5.),
                   ('ywid2',results.params[parnames[13]].value, True, 0.1, 5.),
                   ('rot2', results.params[parnames[14]].value, True, None, None))


    spec1 = []
    model1 = []
    Hbbwl = 1.473+.0002*np.linspace(0, 1651, 1651)
    Kbbwl = 1.965+.00025*np.linspace(0, 1665, 1665)
    Jbbwl = 1.180+.00015*np.linspace(0,1574, 1574)
    if band.upper() == 'K':
        wl = Kbbwl
    elif band.upper() == 'H':
        wl = Hbbwl
    elif band.upper() == 'J':
        wl=Jbbwl

    for a in range(size[0]):
        if a%50==0:
            print(round(a/size[0]*100, 2),'%')
        frame = data[a,:,:]
        mask = np.ones_like(frame)
        mask[frame==0] = 0.

        results = fxn.fit(frame, x=Xin, y=Yin, params=pars2, weights=mask)
        #want to save pars as dataframe
        savecut = [results.params[parnames[0]].value, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value]
        savecut[8] = savecut[8]%(2*math.pi)
        savecut[14] = savecut[14]%(2*math.pi)
        if a == 0:
            savedata = savecut
        else:
            savedata = np.vstack((savedata, savecut))

        out1 = stdtwotwogauss(Xin,Yin,0, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value)
        spec = np.sum(out1)

        fullmodel = stdtwotwogauss(Xin,Yin, results.params[parnames[0]].value, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value)

        if a ==0:
            spec1 = spec
            model1 = out1
        else:
            spec1 = np.vstack((spec1, spec))
            model1 = np.vstack((model1, out1))

    #savedata = np.transpose(savedata)
    saveframe = pd.DataFrame(savedata, columns=parnames)
    fname = fname.replace('img_split/', '')

    saveframe.to_csv('Python_fitted/pars/'+fname)

    specs = pd.DataFrame({'wl':wl, 'Spec':np.reshape(spec1, len(wl))})
    fname = fname.replace('pars','specs')

    specs.to_csv('Python_fitted/'+fname)
    fname = fname.replace('specs.csv', 'model.dat')
    np.savetxt('Python_fitted/'+fname, model1)
    printSpec(specs.wl, specs.Spec,specs.Spec)

def get_fit2LM(fname, xcen, ycen, band):#for the brown dwarfs
    print(fname)
    data = fits.getdata(fname)
    fname = fname.replace('.fits', 'TESTLM2gausspars.csv')
    pars = Parameters()
    parnames = ['bg', 'ampA1', 'x_size', "y_size", 'xcenA1', 'ycenA1','xwid1', 'ywid1','rot1',\
    'ampA2', 'xcenA2', 'ycenA2', 'xwid2', 'ywid2','rot2', 'ampB1', 'sepx', 'sepy']
#build model for standard star and make spectrum
#normalize and compare to brendan for binary - compare standard to aperture photometry
    offset = 11
    if ycen[0]-offset < 0:
        yl, yu = ycen[0]-offset+5, ycen[0]+offset+5
    else:
        yl, yu = ycen[0]-offset, ycen[0]+offset
    print(yl, yu)
    data = data[:,:,int(yl):int(yu)]
    size = np.shape(data)
    #bound = (minPar, maxPar)

    Yin, Xin = np.mgrid[0:int(size[1]), 0:int(size[2])]
    fail = np.array([])

    #data[data == 0] = np.nan
    medimg = np.nanmedian(data, axis=0)
    mask = np.ones_like(medimg)
    #mask[medimg < 0] = 0
    mask[medimg==0] = 0.
                   #name, Value, Vary, min, max
    pars.add_many(('bg', bg(medimg, (xcen[0]+xcen[1])/2, (ycen[0]+ycen[1])/2), True, -1., None),
                  ('ampA1',.025, True, 0., None),
                  ('x_size', int(size[1]), False, None, None),
                  ('y_size', int(size[2]), False, None, None),
                  ('xcenA1',xcen[0], True, 0., int(size[1])),
                  ('ycenA1',ycen[0], True, 0.,int(size[2])),
                  ('xwid1',0.5, True, 0.1, 5.),
                  ('ywid1',0.5, True, 0.1, 5.),
                  ('rot1', 0., True, None, None),
                  ('ampA2',.005, True, 0., None),
                  ('xcenA2',xcen[0], True, 0., int(size[1])),
                  ('ycenA2',11, True, 0.,int(size[1])),
                  ('xwid2',1., True, 0.1, 5.),
                  ('ywid2',1., True, 0.1, 5.),
                  ('rot2', 0, True, None, None),
                  ('ampB1',0.03, True, 0., None),
                  ('sepx', xcen[1]-xcen[0], False, None, None),
                  ('sepy', ycen[1]-ycen[0], False, None, None))

    fxn = Model(twotwogauss, independent_vars=['x', 'y'])
    results = fxn.fit(medimg, x=Xin, y=Yin, params=pars, weights=mask)
    #print(xcen, ycen, results.params['xcenA2'].value, results.params['ycenA1'].value)
    plt.subplot(211)
    plt.imshow(medimg)
    plt.subplot(212)
    plt.imshow(results.best_fit)
    plt.show()
    #print(results.fit_report())
    # sys.exit()
    savedata = []
    pars2 = Parameters()
                   #name, Value, Vary, min, max
    pars2.add_many(('bg', results.params[parnames[0]].value, False, -1., None),
                   ('ampA1',results.params[parnames[1]].value, True, 0., .5),
                   ('x_size',results.params[parnames[2]].value, False, None, None),
                   ('y_size',results.params[parnames[3]].value, False, None, None),
                   ('xcenA1',results.params[parnames[4]].value, False, 0., int(size[1])),
                   ('ycenA1',results.params[parnames[5]].value, False, 0., int(size[2])),
                   ('xwid1',results.params[parnames[6]].value, True, 0.1, 5.),
                   ('ywid1',results.params[parnames[7]].value, True, 0.1, 5.),
                   ('rot1', results.params[parnames[8]].value, True, None, None),
                   ('ampA2',results.params[parnames[9]].value, True, 0., None),
                   ('xcenA2',results.params[parnames[10]].value, False, 0., int(size[1])),
                   ('ycenA2',results.params[parnames[11]].value, False, 0., int(size[2])),
                   ('xwid2',results.params[parnames[12]].value, True, 0.1, 5.),
                   ('ywid2',results.params[parnames[13]].value, True, 0.1, 5.),
                   ('rot2', results.params[parnames[14]].value, True, None, None),
                   ('ampB1',results.params[parnames[15]].value, True, 0., None),
                   ('sepx', results.params[parnames[16]].value, False, None, None),
                   ('sepy', results.params[parnames[17]].value, False, None, None))


    spec1, spec2 = [],[]
    model1, model2 = [],[]
    Hbbwl = 1.473+.0002*np.linspace(0, 1651, 1651)
    Kbbwl = 1.965+.00025*np.linspace(0, 1665, 1665)
    Jbbwl = 1.180+.00015*np.linspace(0,1574, 1574)
    if band.upper() == 'K':
        wl = Kbbwl
    elif band.upper() == 'H':
        wl = Hbbwl
    elif band.upper() == 'J':
        wl=Jbbwl

    for a in range(size[0]):
        if a%50==0:
            print(round(a/size[0]*100, 2),'%')
        frame = data[a,:,:]
        mask = np.ones_like(frame)
        mask[frame==0] = 0.
        cx, cy = (2*results.params[parnames[4]].value+results.params[parnames[16]].value)/2, (2*results.params[parnames[5]].value+results.params[parnames[17]].value)/2
        pars2['bg'].value = bg(frame, cx, cy)
        results = fxn.fit(frame, x=Xin, y=Yin, params=pars2, weights = mask)
        #want to save pars as dataframe
        savecut = [results.params[parnames[0]].value, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value%(2*math.pi), results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value%(2*math.pi), results.params[parnames[15]].value, results.params[parnames[16]].value, results.params[parnames[17]].value]

        if a == 0:
            savedata = savecut
        else:
            savedata = np.vstack((savedata, savecut))

        fullmodel = twotwogauss(Xin,Yin, results.params[parnames[0]].value, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value, results.params[parnames[15]].value, results.params[parnames[16]].value, results.params[parnames[17]].value)

        out12 = twotwogauss(Xin,Yin,0, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value, results.params[parnames[15]].value, results.params[parnames[16]].value, results.params[parnames[17]].value)

        out1 = twotwogauss(Xin, Yin,0, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value, 0, results.params[parnames[16]].value, results.params[parnames[17]].value)
        out2 = out12-out1
        specA, specB = np.sum(out1), np.sum(out2)
        out1, out2 = out1.reshape(1,int(size[1]),int(size[2])), out2.reshape(1,int(size[1]),int(size[2]))
        fullmodel = fullmodel.reshape(1,int(size[1]),int(size[2]))
        out12 = out12.reshape(1,int(size[1]),int(size[2]))
        if a ==0:
            spec1, spec2 = specA, specB
            model1, model2 = out1, out2
            model12 = out12
            fullmodels = fullmodel
        else:
            spec1,spec2 = np.vstack((spec1, specA)), np.vstack((spec2, specB))
            model1, model2 = np.vstack((model1, out1)), np.vstack((model2, out2))
            model12 = np.vstack((model12, out12))
            fullmodels = np.vstack((fullmodels, fullmodel))

    resid = data - fullmodels
    #savedata = np.transpose(savedata)
    saveframe = pd.DataFrame(savedata, columns=parnames)
    fname = fname.replace('img_split/', '')
    saveframe.to_csv('Python_fitted/TEST/pars/'+fname)

    specs = pd.DataFrame({'wl':wl, 'SpecB':np.reshape(spec1, len(wl)), 'SpecC':np.reshape(spec2,len(wl))})
    fname = fname.replace('pars','specs')

    specs.to_csv('Python_fitted/TEST/'+fname)
    fname = fname.replace('specs.csv', 'modelBC.fits')
    fits.writeto('Python_fitted/TEST/'+fname, model12, overwrite=True)
    fname = fname.replace('modelBC', 'modelB')
    fits.writeto('Python_fitted/TEST/'+fname, model1, overwrite=True)
    fname = fname.replace('modelB', 'modelC')
    fits.writeto('Python_fitted/TEST/'+fname, model2, overwrite=True)
    fname = fname.replace('modelC', 'resid')
    fits.writeto('Python_fitted/TEST/'+fname, resid, overwrite=True)
    fname = fname.replace('resid', 'data')
    fits.writeto('Python_fitted/TEST/'+fname, data, overwrite=True)
    printSpec(specs.wl, specs.SpecB, specs.SpecC)

def get_fitLM(fname, xcen, ycen, band):#for the brown dwarfs
    print(fname)
    data = fits.getdata(fname)
    fname = fname.replace('.fits', 'LMgausspars.csv')
    pars = Parameters()
    parnames = ['bg', 'ampA1', 'x_size', "y_size", 'xcenA1', 'ycenA1','xwid1', 'ywid1','rot1',\
    'ampA2', 'xcenA2', 'ycenA2', 'xwid2', 'ywid2','rot2','ampA3', 'xcenA3', 'ycenA3', 'xwid3', \
    'ywid3', 'rot3', 'ampB1', 'sepx', 'sepy']
    #build model for standard star and make spectrum
    #normalize and compare to brendan for binary - compare standard to aperture photometry
    size = np.shape(data)
    #bound = (minPar, maxPar)
    Yin, Xin = np.mgrid[0:int(size[0]), 0:int(size[1])]
    offset = 11
    fail = np.array([])
    if ycen[0]-offset < 0:
        yl, yu = ycen[0]-offset+5, ycen[0]+offset-5
    else:
        yl, yu = ycen[0]-offset, ycen[0]+offset
    data = data[:,:,int(yl):int(yu)]
    #data[data == 0] = np.nan
    medimg = np.nanmedian(data, axis=0)
    mask = np.ones_like(medimg)
    #mask[medimg < 0] = 0
    mask[medimg==0] = 0.
                   #name, Value, Vary, min, max
    pars.add_many(('bg', 0, True, -1., None),
                  ('ampA1',1., True, 0., None),
                  ('x_size', int(size[0]), False, None, None),
                  ('y_size', int(size[1]), False, None, None),
                  ('xcenA1',xcen[0], True, 0., 20.),
                  ('ycenA1',offset, True, 0.,22.),
                  ('xwid1',0.5, True, 0.1, 5.),
                  ('ywid1',0.5, True, 0.1, 5.),
                  ('rot1', 0., True, None, None),
                  ('ampA2',.1, True, 0., None),
                  ('xcenA2',xcen[0], True, 0., 20.),
                  ('ycenA2',11, True, 0.,22.),
                  ('xwid2',1., True, 0.1, 5.),
                  ('ywid2',1., True, 0.1, 5.),
                  ('rot2', 0, True, None, None),
                  ('ampA3',.01, True, 0., None),
                  ('xcenA3',xcen[0], False, 0., 20.),
                  ('ycenA3',11, False, 0., 22.),
                  ('xwid3',1.5, True, 0.1, 5.),
                  ('ywid3',1.5, True, 0.1, 5.),
                  ('rot3', 0, True, None, None),
                  ('ampB1',0.5, True, 0., None),
                  ('sepx', xcen[1]-xcen[0], False, None, None),
                  ('sepy', ycen[1]-ycen[0], False, None, None))

    fxn = Model(twothreegauss, independent_vars=['x', 'y'])
    results = fxn.fit(medimg, x=Xin, y=Yin, params=pars, weights=mask, method='emcee')
    #print(xcen, ycen, results.params['xcenA2'].value, results.params['ycenA1'].value)
    plt.subplot(211)
    plt.imshow(medimg)
    plt.subplot(212)
    plt.imshow(results.best_fit)
    plt.show()
    # print(results.fit_report())
    # sys.exit()
    savedata = []
    pars2 = Parameters()
                   #name, Value, Vary, min, max
    pars2.add_many(('bg', results.params[parnames[0]].value, True, -1., None),
                   ('ampA1',results.params[parnames[1]].value, True, 0., None),
                   ('x_size',results.params[parnames[2]].value, False, None, None),
                   ('y_size',results.params[parnames[3]].value, False, None, None),
                   ('xcenA1',results.params[parnames[4]].value, False, 0., 22.),
                   ('ycenA1',results.params[parnames[5]].value, False, 0., 20.),
                   ('xwid1',results.params[parnames[6]].value, True, 0.1, 5.),
                   ('ywid1',results.params[parnames[7]].value, True, 0.1, 5.),
                   ('rot1', results.params[parnames[8]].value, True, None, None),
                   ('ampA2',results.params[parnames[9]].value, True, 0., None),
                   ('xcenA2',results.params[parnames[10]].value, False, 0., 22.),
                   ('ycenA2',results.params[parnames[11]].value, False, 0., 20.),
                   ('xwid2',results.params[parnames[12]].value, True, 0.1, 5.),
                   ('ywid2',results.params[parnames[13]].value, True, 0.1, 5.),
                   ('rot2', results.params[parnames[14]].value, True, None, None),
                   ('ampA3',results.params[parnames[15]].value, True, 0., None),
                   ('xcenA3',results.params[parnames[16]].value, False, 0., 22.),
                   ('ycenA3',results.params[parnames[17]].value, False, 0., 20),
                   ('xwid3',results.params[parnames[18]].value, True, 0.1, 5.),
                   ('ywid3',results.params[parnames[19]].value, True, 0.1, 5),
                   ('rot3', results.params[parnames[20]].value, True, None, None),
                   ('ampB1',results.params[parnames[21]].value, True, 0., None),
                   ('sepx', results.params[parnames[22]].value, False, None, None),
                   ('sepy', results.params[parnames[23]].value, False, None, None))


    spec1, spec2 = [],[]
    model1, model2 = [],[]
    Hbbwl = 1.473+.0002*np.linspace(0, 1651, 1651)
    Kbbwl = 1.965+.00025*np.linspace(0, 1665, 1665)
    Jbbwl = 1.180+.00015*np.linspace(0,1574, 1574)
    if band.upper() == 'K':
        wl = Kbbwl
    elif band.upper() == 'H':
        wl = Hbbwl
    elif band.upper() == 'J':
        wl=Jbbwl

    for a in range(size[0]):
        if a%50==0:
            print(round(a/size[0]*100, 2),'%')
        frame = data[a,:,:]
        mask = np.ones_like(frame)
        mask[frame==0] = 0.
        results = fxn.fit(frame, x=Xin, y=Yin, params=pars2, weights = mask)
        #want to save pars as dataframe
        savecut = [results.params[parnames[0]].value, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value%(2*math.pi), results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value%(2*math.pi), results.params[parnames[15]].value, results.params[parnames[16]].value, results.params[parnames[17]].value, results.params[parnames[18]].value, results.params[parnames[19]].value, results.params[parnames[20]].value%(2*math.pi), results.params[parnames[21]].value, results.params[parnames[22]].value, results.params[parnames[23]].value]

        if a == 0:
            savedata = savecut
        else:
            savedata = np.vstack((savedata, savecut))

        savecut[0] = 0
        out12 = twothreegauss(Xin,Yin,0, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value, results.params[parnames[15]].value, results.params[parnames[16]].value, results.params[parnames[17]].value, results.params[parnames[18]].value, results.params[parnames[19]].value, results.params[parnames[20]].value, results.params[parnames[21]].value, results.params[parnames[22]].value, results.params[parnames[23]].value)
        savecut[21] = 0
        out1 = twothreegauss(Xin, Yin,0, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value, results.params[parnames[15]].value, results.params[parnames[16]].value, results.params[parnames[17]].value, results.params[parnames[18]].value, results.params[parnames[19]].value, results.params[parnames[20]].value, 0, results.params[parnames[22]].value, results.params[parnames[23]].value)
        out2 = out12-out1
        specA, specB = np.sum(out1), np.sum(out2)
        out1, out2 = out1.reshape(1,int(size[0]),int(size[1])), out2.reshape(1,int(size[0]),int(size[1]))
        out12 = out12.reshape(1,int(size[0]),int(size[1]))
        if a ==0:
            spec1, spec2 = specA, specB
            model1, model2 = out1, out2
            model12 = out12
        else:
            spec1,spec2 = np.vstack((spec1, specA)), np.vstack((spec2, specB))
            model1, model2 = np.vstack((model1, out1)), np.vstack((model2, out2))
            model12 = np.vstack((model12, out12))

    resid = data - model12
    #savedata = np.transpose(savedata)
    saveframe = pd.DataFrame(savedata, columns=parnames)
    fname = fname.replace('split_img_wcray/', '')
    saveframe.to_csv('Python_fitted/pars/'+fname)

    specs = pd.DataFrame({'wl':wl, 'SpecB':np.reshape(spec1, len(wl)), 'SpecC':np.reshape(spec2,len(wl))})
    fname = fname.replace('pars','specs')

    specs.to_csv('Python_fitted/'+fname)
    fname = fname.replace('specs.csv', 'modelBC.fits')
    fits.writeto('Python_fitted/'+fname, model12)
    fname = fname.replace('modelBC', 'modelB')
    fits.writeto('Python_fitted/'+fname, model1)
    fname = fname.replace('modelB', 'modelC')
    fits.writeto('Python_fitted/'+fname, model2)
    fname = fname.replace('modelC', 'resid')
    fits.writeto('Python_fitted/'+fname, resid)
    fname = fname.replace('resid', 'data')
    fits.writeto('Python_fitted/'+fname, data)
    printSpec(specs.wl, specs.SpecB, specs.SpecC)


def get_fitstdLM(fname, xcen, ycen, band):#for the standard star
    print(fname)
    data = fits.getdata(fname)
    fname = fname.replace('.fits', 'LMgausspars.csv')
    pars = Parameters()
    parnames = ['bg', 'ampA1', 'x_size', "y_size", 'xcenA1', 'ycenA1','xwid1', 'ywid1','rot1',\
    'ampA2', 'xcenA2', 'ycenA2', 'xwid2', 'ywid2','rot2','ampA3', 'xcenA3', 'ycenA3', 'xwid3', \
    'ywid3', 'rot3']
    #build model for standard star and make spectrum
    #normalize and compare to brendan for binary - compare standard to aperture photometry
    size = np.shape(data)
    Yin, Xin = np.mgrid[0:20, 0:32]
    fail = np.array([])
    #yl, yu = ycen-11, ycen+11
    #data = data[:,:,int(yl):int(yu)
    medimg = np.nanmedian(data, axis=0)
    mask = np.ones_like(medimg)
    mask[medimg==0] = 0.
    medimg[medimg==0] = np.median(medimg)
    # pars.add_many(('x_c', xcen, True, 0, 20),
    #               ('y_c', ycen, True, 0, 32),
    #               ('x_w', 2, True, 0.1, 5),
    #               ('y_w', 2, True, 0.1, 5),
    #               ('angle', 0, True, None, None))
    # fxn = Model(gauss2d, independent_vars=['x_size','y_size'])
    # single_result = fxn.fit(medimg, x_size=32, y_size=20, params=pars, weights=mask)
    # #for fitting gauss2d - noticed that x and y are swapped, so swapped them on the guesses
    # pars=Parameters()
                   #name, Value, Vary, min, max
    pars.add_many(('bg', 0, True, -1., None),
                  ('ampA1',150., True, 0., None),
                  ('x_size', 20, False, None, None),
                  ('y_size', 32, False, None, None),
                  ('xcenA1',xcen, True, 0., 20.),
                  ('ycenA1',ycen, True, 0., 32.),
                  ('xwid1',1, True, 0.1, 5.),
                  ('ywid1',1, True, 0.1, 5.),
                  ('rot1', 0., True, None, None),
                  ('ampA2',75, True, 0., None),
                  ('xcenA2',xcen, True, 0., 20.),
                  ('ycenA2',ycen, True, 0., 32.),
                  ('xwid2',1.5, True, 0.1, 5.),
                  ('ywid2',1.5, True, 0.1, 5.),
                  ('rot2', 0, True, None, None),
                  ('ampA3',25, True, 0., None),
                  ('xcenA3',xcen, True, 0., 20.),
                  ('ycenA3',ycen, True, 0., 32.),
                  ('xwid3',3, True, 0.1, 5.),
                  ('ywid3',3, True, 0.1, 5.),
                  ('rot3', 0., True, None, None))

    fxn = Model(stdtwothreegauss, independent_vars=['x', 'y'])
    results = fxn.fit(medimg, x=Xin, y=Yin, params=pars, weights = mask)
    plt.subplot(211)
    plt.imshow(medimg)
    plt.subplot(212)
    plt.imshow(results.best_fit)
    plt.show()
    # print(results.fit_report())
    # sys.exit()
    savedata = []
    pars2 = Parameters()
                   #name, Value, Vary, min, max
    pars2.add_many(('bg', results.params[parnames[0]].value, True, -1., None),
                   ('ampA1',results.params[parnames[1]].value, True, 0., None),
                   ('x_size',results.params[parnames[2]].value, False, None, None),
                   ('y_size',results.params[parnames[3]].value, False, None, None),
                   ('xcenA1',results.params[parnames[4]].value, False, 0., 20.),
                   ('ycenA1',results.params[parnames[5]].value, False, 0., 32.),
                   ('xwid1',results.params[parnames[6]].value, True, 0.1, 5.),
                   ('ywid1',results.params[parnames[7]].value, True, 0.1, 5.),
                   ('rot1', results.params[parnames[8]].value, True, None, None),
                   ('ampA2',results.params[parnames[9]].value, True, 0., None),
                   ('xcenA2',results.params[parnames[10]].value, False, 0., 20.),
                   ('ycenA2',results.params[parnames[11]].value, False, 0., 32.),
                   ('xwid2',results.params[parnames[12]].value, True, 0.1, 5.),
                   ('ywid2',results.params[parnames[13]].value, True, 0.1, 5.),
                   ('rot2', results.params[parnames[14]].value, True, None, None),
                   ('ampA3',results.params[parnames[15]].value, True, 0., 100.),
                   ('xcenA3',results.params[parnames[16]].value, False, 0., 20.),
                   ('ycenA3',results.params[parnames[17]].value, False, 0., 32),
                   ('xwid3',results.params[parnames[18]].value, True, 0.1, 5.),
                   ('ywid3',results.params[parnames[19]].value, True, 0.1, 5),
                   ('rot3', results.params[parnames[20]].value, True, None, None))


    spec1 = []
    model1 = []
    Hbbwl = 1.473+.0002*np.linspace(0, 1651, 1651)
    Kbbwl = 1.965+.00025*np.linspace(0, 1665, 1665)
    Jbbwl = 1.180+.00015*np.linspace(0,1574, 1574)
    if band.upper() == 'K':
        wl = Kbbwl
    elif band.upper() == 'H':
        wl = Hbbwl
    elif band.upper() == 'J':
        wl=Jbbwl

    for a in range(size[0]):
        if a%50==0:
            print(round(a/size[0]*100, 2),'%')
        frame = data[a,:,:]
        mask = np.ones_like(frame)
        mask[frame==0] = 0.

        results = fxn.fit(frame, x=Xin, y=Yin, params=pars2, weights=mask)
        #want to save pars as dataframe
        savecut = [results.params[parnames[0]].value, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value, results.params[parnames[15]].value, results.params[parnames[16]].value, results.params[parnames[17]].value, results.params[parnames[18]].value, results.params[parnames[19]].value, results.params[parnames[20]].value]
        savecut[8] = savecut[8]%(2*math.pi)
        savecut[14] = savecut[14]%(2*math.pi)
        savecut[20] = savecut[20]%(2*math.pi)
        if a == 0:
            savedata = savecut
        else:
            savedata = np.vstack((savedata, savecut))

        out1 = stdtwothreegauss(Xin,Yin,0, results.params[parnames[1]].value, results.params[parnames[2]].value, results.params[parnames[3]].value, results.params[parnames[4]].value, results.params[parnames[5]].value, results.params[parnames[6]].value, results.params[parnames[7]].value, results.params[parnames[8]].value, results.params[parnames[9]].value, results.params[parnames[10]].value, results.params[parnames[11]].value, results.params[parnames[12]].value, results.params[parnames[13]].value, results.params[parnames[14]].value, results.params[parnames[15]].value, results.params[parnames[16]].value, results.params[parnames[17]].value, results.params[parnames[18]].value, results.params[parnames[19]].value, results.params[parnames[20]].value)
        spec = np.sum(out1)
        if a ==0:
            spec1 = spec
            model1 = out1
        else:
            spec1 = np.vstack((spec1, spec))
            model1 = np.vstack((model1, out1))

    #savedata = np.transpose(savedata)
    saveframe = pd.DataFrame(savedata, columns=parnames)
    fname = fname.replace('split_img_wcray/', '')
    saveframe.to_csv('Python_fitted/pars/'+fname)

    specs = pd.DataFrame({'wl':wl, 'Spec':np.reshape(spec1, len(wl))})
    fname = fname.replace('pars','specs')

    specs.to_csv('Python_fitted/'+fname)
    fname = fname.replace('specs.csv', 'model.dat')
    np.savetxt('Python_fitted/'+fname, model1)
    printSpec(specs.wl, specs.Spec,specs.Spec)

def printSpec(wl, specA, specB):
    plt.clf()
    plt.figure(1)
    plt.subplot(211)
    plt.plot(wl, specA)
    plt.subplot(212)
    plt.plot(wl, specB)
    plt.show()

# jdwarffiles = ['s070422_a012001_Jbb_035neg.fits', 's070422_a012001_Jbb_035pos.fits']
# xcAJ, ycAJ = [8,10], [6,8]
# xcBJ, ycBJ = [12,15], [9,11]
# jdwarffiles = ['s070422_a012001_Jbb_035sum.fits']
# xcAJ, ycAJ = [8], [6]
# xcBJ, ycBJ = [12], [9]
# for a in range(len(xcAJ)):
#     get_fit2LM('img_split/'+jdwarffiles[a], [ycAJ[a], ycBJ[a]], [xcAJ[a], xcBJ[a]], 'J')

hdwarffiles = ['s070422_a006001_Hbb_035neg.fits', 's070422_a006001_Hbb_035pos.fits', 's070422_a011001_Hbb_035neg.fits', 's070422_a011001_Hbb_035pos.fits']
xcAH, ycAH = [14,17,7,10],[7,9,7,9]
xcBH, ycBH = [19,21,12,15],[10,13,10,12]
hdwarffiles = ['s070422_a011001_Hbb_035pos.fits']
xcAH, ycAH = [10],[9]
xcBH, ycBH = [15],[12]
# for a in range(len(xcAH)):
#     get_fit2LM('img_split/'+hdwarffiles[a], [ycAH[a], ycBH[a]], [xcAH[a], xcBH[a]], 'H')

kdwarffiles = ['s070423_a051001_Kbb_035neg.fits', 's070423_a051001_Kbb_035pos.fits', 's070423_a052001_Kbb_035neg.fits', 's070423_a052001_Kbb_035pos.fits']
xcAK, ycAK = [15,17,13,16],[8,10,8,9]
xcBK, ycBK = [20,22,18,21],[11,13,11,12]
# kdwarffiles = ['s080426_a029001_Kbb_020pos.fits']
# xcAK, ycAK = [17],[9]
# xcBK, ycBK = [16],[7]
for a in range(len(xcAK)):
    get_fit2LM('img_split/'+kdwarffiles[a], [ycAK[a], ycBK[a]], [xcAK[a], xcBK[a]], 'K')

jstdfiles = ['s070422_a018001_Jbb_035neg.fits', 's070422_a018001_Jbb_035pos.fits']
xcJ, ycJ = [15,18],[9,11]
# #jstdfiles = ['s080426_a018001_Jbb_020neg.fits', 's080426_a019001_Jbb_020neg.fits']
# #xcJ, ycJ = [16.,15.],[11.,12.]
# for a in range(len(xcJ)):
#     get_fit2stdLM('img_split/'+jstdfiles[a], ycJ[a], xcJ[a], 'J')

hstdfiles = ['s070422_a014001_Hbb_035neg.fits', 's070422_a014001_Hbb_035pos.fits', 's070422_a015001_Hbb_035neg.fits', 's070422_a015001_Hbb_035pos.fits', 's070422_a016001_Hbb_035neg.fits', 's070422_a016001_Hbb_035pos.fits', 's070422_a017001_Hbb_035neg.fits', 's070422_a017001_Hbb_035pos.fits']
xcH, ycH = [17,19,15,17,15,17,14,17],[9,10,9,10,10,11,9,11]
# hstdfiles = ['s080426_a020001_Hbb_020pos.fits']
# xcH, ycH = [15.],[10.]
# for a in range(len(xcH)):
#     get_fit2stdLM('img_split/'+hstdfiles[a], ycH[a], xcH[a], 'H')

kstdfiles = ['s070326_a046001_Kbb_035pos.fits', 's070423_a055001_Kbb_035neg.fits', 's070423_a055001_Kbb_035pos.fits', 's070423_a056001_Kbb_035neg.fits', 's070423_a056001_Kbb_035pos.fits', 's070423_a057001_Kbb_035neg.fits', 's070423_a057001_Kbb_035pos.fits', 's070423_a058001_Kbb_035neg.fits', 's070423_a058001_Kbb_035pos.fits']
xcK, ycK = [13,19,22,15,18,15,17,15,17],[8,11,13,8,10,8,9,8,9]
# kstdfiles = ['s080426_a032001_Kbb_020pos.fits', 's080426_a034001_Kbb_020pos.fits', 's080426_a035001_Kbb_020neg.fits']
# xcK, ycK = [13.,13.,13.],[8.,8.,10.]
# for a in range(len(xcK)):
#    get_fit2stdLM('img_split/'+kstdfiles[a], ycK[a], xcK[a], 'K')
