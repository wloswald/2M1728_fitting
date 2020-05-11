#created by Wayne L. Oswald
#Python 3.7
#last edited 4/15/19

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import astropy.table as pytable
import sys

path = 'Data/'

def reshape(filepath):
	data, header = fits.getdata(filepath, header = True)
	header = fits.open(filepath)
	size = data.shape
	data = np.split(data, size[2], axis = 2)
	data = np.asarray(data)
	data = np.reshape(data, (size[2], size[0], size[1]))
	wl = (np.arange(len(data))*header[0].header['cdelt1']+header[0].header['crval1'])
	return data

jfiles = ['s070422_a012001_Jbb_035.fits', 's070807_a024001_Jbb_035.fits', 's070807_a025001_Jbb_035.fits']
hfiles = ['s070326_a036001_Hbb_035.fits', 's070422_a004001_Hbb_035.fits', 's070422_a005001_Hbb_035.fits', 's070422_a006001_Hbb_035.fits', 's070422_a007001_Hbb_035.fits', 's070422_a010001_Hbb_035.fits', 's070422_a011001_Hbb_035.fits']
kfiles = ['s070326_a033001_Kbb_035.fits', 's070326_a034001_Kbb_035.fits', 's070326_a035001_Kbb_035.fits', 's070423_a047001_Kbb_035.fits', 's070423_a050001_Kbb_035.fits', 's070423_a051001_Kbb_035.fits', 's070423_a052001_Kbb_035.fits']

std_jfiles = ['s070422_a018001_Jbb_035.fits', 's070807_a027001_Jbb_035.fits', 's070807_a027002_Jbb_035.fits', 's070807_a028001_Jbb_035.fits', 's070807_a028002_Jbb_035.fits']
std_hfiles = ['s070422_a014001_Hbb_035.fits', 's070422_a015001_Hbb_035.fits', 's070422_a016001_Hbb_035.fits', 's070422_a017001_Hbb_035.fits']
std_kfiles = ['s070326_a039001_Kbb_035.fits', 's070326_a045001_Kbb_035.fits', 's070326_a046001_Kbb_035.fits', 's070423_a055001_Kbb_035.fits', 's070423_a056001_Kbb_035.fits', 's070423_a057001_Kbb_035.fits', 's070423_a058001_Kbb_035.fits']

files = std_kfiles

pdf_page = PdfPages('Std_K_Facebook.pdf')

#pdf_page = PdfPages('2MASSJ1728AB_J_Facebook.pdf')

for a in files:
    data = reshape(path+a)
    medimg = np.median(data, axis=0)
    plt.clf()
    plt.figure()
    plt.title(a+r' L5$\pm$1 L7$\pm$1')
    plt.imshow(medimg, origin='lower')
    plt.colorbar()
    pdf_page.savefig()

pdf_page.close()
