import pyfits
import pyspeckit
import agpy

def cube_to_total(fitsfile, plotspec=True):
    """
    Create average spectrum from a spectral cube
    """

    ff = pyfits.open(fitsfile)
    data = ff[0].data

    for kwarg in ('CRPIX','CRVAL','CDELT','CUNIT','CTYPE'):
        ff[0].header.update(kwarg+'1',ff[0].header[kwarg+'3'])

    for ii,aperture in enumerate(((128,128,5), (128,128,10), (128,128,15), (128,128,25), (128,128,50))):
        spec = agpy.cubes.extract_aperture(data,aperture)
        ff[0].data = spec
        averagedfile = fitsfile.replace('.fits','_ap%i.fits' % ii)
        try:
            ff.writeto(averagedfile, clobber=True)
        except:
            print "Failed to write "+averagedfile
        if plotspec:
            sp = pyspeckit.Spectrum(averagedfile,doplot=True)
            sp.plotter.savefig(averagedfile.replace(".fits",".png"))


    return averagedfile 

if __name__ == "__main__":
    import glob
    for fn in glob.glob("*.fits"):
        if 'averaged' in fn or 'summed' in fn:
            continue
        else:
            averagedfile = cube_to_total(fn, plotspec=True)
            #sp = pyspeckit.Spectrum(averagedfile,doplot=True)
            #sp.plotter.savefig(averagedfile.replace(".fits",".png"))
