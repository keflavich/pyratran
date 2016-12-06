from __future__ import print_function
import numpy as np
from numpy import pi
from astropy.io import fits
from astropy import units as u
from astropy import constants
import os

pc = 3.086e16 # m  (RATRAN uses meters)
au = 1.496e13 # cm
msun = 2e33 # grams
mh = 1.67e-24
G = 6.67e-8 # CGS

RATRAN_PATH = os.getenv('RATRAN')
if RATRAN_PATH:
    AMC_PATH = os.path.join(RATRAN_PATH, 'bin/amc')
    SKY_PATH = os.path.join(RATRAN_PATH, 'bin/sky')
else:
    AMC_PATH = "/usr/local/bin/amc"
    SKY_PATH = "/usr/local/bin/sky"

def make_model(filename, rmax, rproffunc, abund, tk, vr, td=None, dv_fwhm=1.0,
               ne=None, te=None, vz=None, ncells=20, gastodust=100.,
               tcmb=2.725, mindens=0.0, **kwargs):
    """
    Generate a RATRAN source model with a density profile (density as a
    function of r) given by rproffunc.  kwargs are passed to the radial profile
    function.
    Abundance is the fixed abundance of the molecule relative to H2.  nH2(r) is
    given by rproffunc
    """

    outf = open(filename,'w')

    db = dv_fwhm/(np.sqrt(4*np.log(2)))
    if td is None:
        td=tk

    # make header
    print("rmax=%g" % rmax, file=outf)
    print("ncell=%i" % ncells, file=outf)
    print("tcmb=%f" % tcmb, file=outf)
    cols = "id,ra,rb,nh,nm,tk,td,db,vr"
    if ne is not None and te is not None:
        cols += ",ne,te"
    if vz is not None:
        cols += ",vz"
    #ncols = cols.count(",") + 1
    print("columns=%s" % (cols), file=outf)
    print("gas:dust=%f" % gastodust, file=outf)
    print("@", file=outf)

    radarr = np.linspace(0.0,rmax,ncells+1)
    midradarr = (radarr[1:]+radarr[:-1])/2.0
    nharr = rproffunc(midradarr,**kwargs)
    nharr[nharr < mindens] = mindens
    if not all(radarr == np.sort(radarr)):
        import pdb; pdb.set_trace()

    vardict = {"tk":tk,"td":td,"vr":vr,"abund":abund,"ne":ne,"te":te,"vz":vz,"db":db,}
    for varname,var in vardict.items():
        if var is None:
            if varname in ('vz','ne','te'):
                continue
            else:
                # variable not specified: not good!
                raise ValueError("Variable {0} not specified.".format(varname))
        elif np.isscalar(var) == 1:
            var = np.zeros(ncells)+var
            vardict[varname] = var
            exec("%s = np.zeros(ncells) + %s" % (varname,varname))
            locals()[varname] = var
            print("{0} = {1} = {2}".format(varname,var,locals()[varname]))
        elif len(var) == ncells:
            # all is OK
            continue
        else:
            raise ValueError("Variable has wrong size.")

    tk = vardict['tk']
    td = vardict['td']
    vr = vardict['vr']
    abund = vardict['abund']
    ne = vardict['ne']
    te = vardict['te']
    vz = vardict['vz']
    db = vardict['db']

    for ii in range(ncells):

        if radarr[ii+1] <= radarr[ii]:
            import pdb; pdb.set_trace()
        outline = "%10i %15g %15g %15g %15g " % (ii+1,radarr[ii],radarr[ii+1],nharr[ii],nharr[ii]*abund[ii])
        outline += "%15g %15g %15g %15g " % (tk[ii],td[ii],db[ii],vr[ii])
        if ne is not None and te is not None:
            outline += "%15g %15g " % (ne[ii],te[ii])
        if vz is not None:
            outline += "%15g " % (vz[ii])
        print(outline, file=outf)

    outf.close()

def mt03(radii, column=1e22*u.cm**-2, krho=1.5, mass=1.0*u.M_sun,
         rhoouter=1e3*u.cm**-3, zh2=2.8, mh=1*u.Da, r_inner=1*u.au):
    """
    Remember, radius is given in METERS
    r_inner is given in cm
    """

    raise ValueError("mt03 doesn't really make sense in this context. "
                     "You need to specify an external pressure, etc.")

    radii = u.Quantity(radii, u.m)
    r_inner = u.Quantity(r_inner, u.m)

    column = u.Quantity(column, u.cm**-2)
    mass = u.Quantity(mass, u.M_sun)

    # handle inner singularity if non-scalar
    try:
        radii[radii<r_inner] = r_inner
    except TypeError:
        pass

    #phi_rho = argH!!!

    # eqn 12
    phi_P = (3-kapparho)/3. * (radii / Rcl)**-kapparho

    # gaseous fraction fg: pg 853
    # (fg=0 means all stars, fg=1 means all gas)
    fg = 1

    # eqn 24
    rho = (phi_rho**-1 * (3*phi_P/(3-kapparho))**(3/4.) *
           fg**(1/4.) * (eps_core * mass / mstar)**0.5)


    r_outer = ((mass/(pi * column * mh * zh2))**0.5).to(u.m)
    AA = (4-krho)*mass / (4/3. * pi * (r_outer/r_inner)**(4-krho) * u.m**3)
    # equivalent to prev. line
    # A = mass*msun / (r_outer**(4-krho)) * (3*(4-krho)/(4.*pi))

    rho = ((AA * (radii/r_inner)**-krho) * (radii<r_outer) +
           (rhoouter*zh2*mh) * (radii>r_outer))

    nh2 = rho / zh2 / mh

    return nh2.to(u.cm**-3).value


def plummer(r, mtot=0, a=0, zh2=2.8, mh=1.66053886e-24, ismeters=True):
    """
    Return the density given a Plummer profile
    """

    scale = 1e-6 if ismeters else 1.0
    rho = 3 * mtot / (4*np.pi*a**3) * ( 1. + r**2/a**2 )**(-2.5)
    nh2 = rho / (mh*zh2)
    return nh2 * scale

def broken_powerlaw(r,rbreak=1.0,power=-2.0,n0=1e5):
    """
    Return the density of a broken power law density profile where
    the central density is flat
    """

    if np.isscalar(r):
        if r < rbreak:
            return n0
        else:
            # n(rbreak) = n0
            n = n0 * (r/rbreak)**power
            return n
    else:
        narr = np.zeros(r.shape)
        narr[r<rbreak] = n0
        # this was previously
        # narr[r>=rbreak] = n0 * (r/rbreak)**power
        # which is wrong on two levels - first, the math is wrong, but second,
        # the assignment is supposed to be disallowed because of shape mismatch
        # numpy has failed me.
        narr[r>=rbreak] = n0 * (r[r>=rbreak]/rbreak)**power
        return narr

def bonnorebert(r,ncenter=1e5,ximax=6.9,viso=0.24,zh2=2.8):
    """
    approximation to a bonnor-ebert sphere using broken power law
    6.9 taken from Alves, Lada, Lada B68 Nature paper 2001
    viso is for zh2=2.8, T=20K
    """
    # cm
    rbreak = ximax*(viso*1e5) / np.sqrt(4*np.pi*G*ncenter*mh*zh2)
    # to m
    rbreak /= 100.
    print("bonnorebert rbreak: %g " % rbreak)

    return broken_powerlaw(r,rbreak=rbreak,n0=ncenter)

def run_model(mdlname,amc=AMC_PATH,sky=SKY_PATH,**kwargs):

    make_amc(mdlname,**kwargs)

    print("AMC command: ","%s %s.amc" % (amc,mdlname))
    proc1 = os.system("%s %s.amc" % (amc,mdlname))
    if proc1 != 0:
        import pdb; pdb.set_trace()

    make_sky(mdlname,**kwargs)

    print("SKY command: ","%s %s.sky" % (sky,mdlname))
    proc2 = os.system("%s %s.sky" % (sky,mdlname))
    if proc2 != 0:
        import pdb; pdb.set_trace()


def make_amc(mdlname,molfile="o-h2co-scaled1.6",**kwargs):

    outf = open(mdlname+".amc",'w')

    print("source=%s.mdl" % mdlname, file=outf)
    print("outfile=%s.pop" % mdlname, file=outf)
    print("molfile=%s.dat" % molfile, file=outf)
    print("snr=20", file=outf)
    print("nphot=1000", file=outf)
    print("tnorm=2.735", file=outf)
    print("velo=grid", file=outf)
    print("kappa=jena,thin,e5", file=outf)
    print("seed=1971", file=outf)
    print("go", file=outf)
    print("q", file=outf)

    outf.close()

def make_sky(mdlname, distance=100., central=None, trans="1,3,6", npix=128,
             pixsize=1, centralregion=32, nsightlines=2, **kwargs):

    outf = open(mdlname+".sky",'w')

    print("source=%s.pop" % mdlname, file=outf)
    print("format=fits", file=outf)
    print("outfile=%s" % mdlname, file=outf)
    print("trans=%s" % trans, file=outf)
    # will get the crash "Fortran runtime error: Bad integer for item 1 in list input" if these are floats
    print("pix=%i,%i,%i,%i" % (npix,pixsize,centralregion,nsightlines), file=outf)
    print("chan=200,0.1", file=outf)
    if central is not None:
        print("central=%s" % central, file=outf)
    print("distance=%g" % distance, file=outf)
    print("units=K", file=outf)
    print("go", file=outf)
    print("q", file=outf)

    outf.close()

def make_averagespec(mdlname,trans="1,3,6",clobber=True):

    for tnum in trans.split(","):
        fn = mdlname+"_%03i.fits" % tnum
        pf = fits.open(fn)
        #meanspec = pf[0].data.mean(axis=1).mean(axis=0)
        for k in "CTYPE1","CTYPE2","CDELT1","CDELT2","CRVAL1","CRVAL2","CRPIX1","CRPIX2":
            del pf[0].header[k]
        for k in "CTYPE1","CDELT1","CRVAL1","CRPIX1":
            pf[0].header.update(k,pf[0].header.get(k.replace("1","3")))
            del pf[0].header[k.replace("1","3")]
        pf[0].header.update("CUNIT1","m/s")

        pf.writeto(fn.replace(".fits","_onedspec.fits"),clobber=clobber)


if __name__ == "__main__":

    doplummer_incloud=False
    doplummer=False
    dobp=False
    doEvans=False
    domt03=False

    if doplummer_incloud:
        make_model("plummer_incloud_nomotion_20K_10msun_r0.01.mdl",1.0*pc,plummer,1e-9,20,0.0,mtot=10*msun,a=0.01*pc,mindens=100,ncells=50)
        run_model("plummer_incloud_nomotion_20K_10msun_r0.01",central="1000,2.73")
        make_model("plummer_incloud_nomotion_20K_1msun_r0.01.mdl",1.0*pc,plummer,1e-9,20,0.0,mtot=1*msun,a=0.01*pc,mindens=100,ncells=50)
        run_model("plummer_incloud_nomotion_20K_1msun_r0.01",central="1000,2.73")
        make_model("plummer_incloud_nomotion_20K_0.1msun_r0.01.mdl",1.0*pc,plummer,1e-9,20,0.0,mtot=0.1*msun,a=0.01*pc,mindens=100,ncells=50)
        run_model("plummer_incloud_nomotion_20K_0.1msun_r0.01",central="1000,2.73")
        make_model("plummer_incloud_nomotion_20K.mdl",1.0*pc,plummer,1e-9,20,0.0,mtot=msun,a=0.1*pc,mindens=100,ncells=50)
        run_model("plummer_incloud_nomotion_20K",central="1000,2.73")
        make_model("plummer_incloud_nomotion_20K_0.1msun.mdl",1.0*pc,plummer,1e-9,20,0.0,mtot=0.1*msun,a=0.1*pc,mindens=100,ncells=50)
        run_model("plummer_incloud_nomotion_20K_0.1msun",central="1000,2.73")
        make_model("plummer_incloud_nomotion_20K_10msun.mdl",1.0*pc,plummer,1e-9,20,0.0,mtot=10*msun,a=0.1*pc,mindens=100,ncells=50)
        run_model("plummer_incloud_nomotion_20K_10msun",central="1000,2.73")

    if doplummer:
        make_model("plummer_nomotion_20K_10msun_r0.01.mdl",0.3*pc,plummer,1e-9,20,0.0,mtot=10*msun,a=0.01*pc)
        run_model("plummer_nomotion_20K_10msun_r0.01",central="1000,2.73")
        make_model("plummer_nomotion_20K_1msun_r0.01.mdl",0.3*pc,plummer,1e-9,20,0.0,mtot=1*msun,a=0.01*pc)
        run_model("plummer_nomotion_20K_1msun_r0.01",central="1000,2.73")
        make_model("plummer_nomotion_20K_0.1msun_r0.01.mdl",0.3*pc,plummer,1e-9,20,0.0,mtot=0.1*msun,a=0.01*pc)
        run_model("plummer_nomotion_20K_0.1msun_r0.01",central="1000,2.73")
        make_model("plummer_nomotion_20K.mdl",0.3*pc,plummer,1e-9,20,0.0,mtot=msun,a=0.1*pc)
        run_model("plummer_nomotion_20K",central="1000,2.73")
        make_model("plummer_nomotion_20K_0.1msun.mdl",0.3*pc,plummer,1e-9,20,0.0,mtot=0.1*msun,a=0.1*pc)
        run_model("plummer_nomotion_20K_0.1msun",central="1000,2.73")
        make_model("plummer_nomotion_20K_10msun.mdl",0.3*pc,plummer,1e-9,20,0.0,mtot=10*msun,a=0.1*pc)
        run_model("plummer_nomotion_20K_10msun",central="1000,2.73")


    if dobp:
        make_model("broken_powerlaw_r0.3_n1e4_pow-2.0.mdl",1.0*pc,broken_powerlaw,1e-9,20,0.0,rbreak=0.3*pc,n0=1e4,power=-2.0)
        run_model("broken_powerlaw_r0.3_n1e4_pow-2.0",central="1000,2.73",pixsize=2.0)
        make_model("broken_powerlaw_r0.3_n1e5_pow-2.0.mdl",1.0*pc,broken_powerlaw,1e-9,20,0.0,rbreak=0.3*pc,n0=1e5,power=-2.0)
        run_model("broken_powerlaw_r0.3_n1e5_pow-2.0",central="1000,2.73",pixsize=2.0)
        make_model("broken_powerlaw_r0.3_n1e6_pow-2.0.mdl",1.0*pc,broken_powerlaw,1e-9,20,0.0,rbreak=0.3*pc,n0=1e6,power=-2.0)
        run_model("broken_powerlaw_r0.3_n1e6_pow-2.0",central="1000,2.73",pixsize=2.0)

        make_model("broken_powerlaw_r0.3_n1e4_pow-1.5.mdl",1.0*pc,broken_powerlaw,1e-9,20,0.0,rbreak=0.3*pc,n0=1e4,power=-1.5)
        run_model("broken_powerlaw_r0.3_n1e4_pow-1.5",central="1000,2.73",pixsize=2.0)
        make_model("broken_powerlaw_r0.3_n1e5_pow-1.5.mdl",1.0*pc,broken_powerlaw,1e-9,20,0.0,rbreak=0.3*pc,n0=1e5,power=-1.5)
        run_model("broken_powerlaw_r0.3_n1e5_pow-1.5",central="1000,2.73",pixsize=2.0)
        make_model("broken_powerlaw_r0.3_n1e6_pow-1.5.mdl",1.0*pc,broken_powerlaw,1e-9,20,0.0,rbreak=0.3*pc,n0=1e6,power=-1.5)
        run_model("broken_powerlaw_r0.3_n1e6_pow-1.5",central="1000,2.73",pixsize=2.0)

    if doEvans:
        make_model("l1544BE5.mdl",0.3*pc,bonnorebert,1e-9,20,0.0,viso=0.17,ncenter=1e6)
        run_model("l1544BE5",central="1000,2.73",pixsize=1.0,distance=140)

    # w51 case

    ncells = 100
    make_model("pl_m200_kappa1.5.mdl",
               rmax=10000*au,
               rproffunc=broken_powerlaw,
               abund=1e-10, # abundance
               tk=np.linspace(600,100,ncells), # tk
               vr=np.linspace(0,5.0,ncells), # vr
               ncells=ncells,
               rbreak=1000*au, # au
               n0=1e8, # cm**-3
               power=-1.5)
    run_model("pl_m200_kappa1.5",
              distance=5400, # pc
              central="0.2,200", # arcsec, kelvin
              pixsize=2.0,
              molfile='a-ch3oh',
              centralregion=5,
              trans='12,55,56,178,197,251,254,378,437,462,463,513,519,523,564,568',
             )

    #from astropy import units as u
    #print(mt03((5000*u.au).to(u.m).value,
    #           column=1e24, # cm**-2
    #           krho=1.5,
    #           mass=200, # msun
    #           rhoouter=1e3, # cm**-3
    #           r_inner=100*au,
    #          )
    #     )
    #make_model("mt03_rho1.5_mass200_col1e24.mdl",
    #           10000*au,
    #           mt03,
    #           1e-9, # abundance
    #           100, # tk
    #           0.0, # vr
    #           krho=1.5,
    #           mass=200,
    #           column=1e24)
    #run_model("mt03_rho1.5_mass200_col1e24",
    #          distance=5400, # pc
    #          central="0.04,300", # arcsec, kelvin
    #          pixsize=2.0,
    #          molfile='a-ch3oh',
    #          centralregion=5,
    #          trans='12,55,56,178,197,251,254,378,437,462,463,513,519,523,564,568',
    #         )

    # totally implausible
    #if domt03:
    #    for mass in (1.0,1e2,1e3):
    #        for krho in (1.0,1.5,2.0):
    #            for column in (1e21,1e22,1e23,1e24):
    #                if mt03(0.25*3.086e16,mass=mass,krho=krho,column=column) > 1e3:
    #                    make_model("mt03_rho%0.1f_mass%0.1e_col%0.1e.mdl" % (krho,mass,column),1.0*pc,mt03,1e-9,20,0.0, krho=krho, mass=mass, column=column)
    #                    run_model("mt03_rho%0.1f_mass%0.1e_col%0.1e" % (krho,mass,column),central="1000,2.73",pixsize=2.0)
    #                else:
    #                    print "Skipped parameters mass=%e  krho=%f column=%e" % (mass,krho,column)
