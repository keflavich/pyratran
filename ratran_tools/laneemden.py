import numpy as np
import pylab
import matplotlib.pyplot as pl

def rk4(x, y, z, a , dx):
    """
    Runge-Kutta Integrator
    """
    #y: initial theta
    #z: derivative of theta
    #z: acceleration function of theta g(x,y,z) 
    #dx: step
    x1 = x
    y1 = y
    z1 = z
    a1 = a(x1, y1, z1) #dzdx
 
    x2 = x + 0.5*dx
    y2 = y + 0.5*z1*dx
    z2 = z + 0.5*a1*dx
    a2 = a(x2, y2, z2)
 
    x3 = x + 0.5*dx
    y3 = y + 0.5*z2*dx
    z3 = z + 0.5*a2*dx
    a3 = a(x3, y3, z3)
 
    x4 = x + dx # not used
    y4 = y + z3*dx
    z4 = z + a3*dx
    a4 = a(x3, y4, z4)
 
    yf = y + (dx/6.0)*(z1 + 2.0*z2 + 2.0*z3 + z4)
    zf = z + (dx/6.0)*(a1 + 2.0*a2 + 2.0*a3 + a4)
    xf = z + dx # not used
 
    return yf, zf
 
def g(x,y,z): #dzdx or (d^2 y)/(dx^2) or theta''
    return  -(1.0/(x*x))*(2.0*x*z+x*x*np.power(y,n)) #dz dx 
 
if __name__ == "__main__":
    n=0.0 
    intersects=[]
    while (n<6):
       s1=[]
       s2=[]
       s1.append(0.0000001) #boundary values
       s2.append(1.0) 
       dx = 0.01 # step in x
       x=dx #taylor aprox pulled from literature for xo and yo
       y0=1.0-x*x/6.0 + n/120.0*x*x*x*x - n*(8.0*n-5.0)/15120.0*np.power(x,6.0) #yo???
       z0=-1.0/3.0*x+4.0*n/120.0*x*x*x-6.0*n*(8.0*n-5)/15120.0*np.power(x,5.0)
       state = y0,z0 # theta, theta'
       while x < 15:
          s1.append(x)    
          s2.append(state[0])  #y
          state=rk4(x,state[0], state[1], g, dx)
          x += dx

       i=0
       smin=666
       catch=666
       while (i<np.size(s1)): #find zero point intersects
          if (s2[i] < 0):
              catch=0
          if (catch==666 and np.abs(s2[i])<smin):
              smin=s2[i]
              imin=i
          i+=1
       print n, s1[imin]
     
       pl.plot(s1,s2,'k')
    #   pl.xlim(0,10)
    #   pl.ylim(-.1, 1.05)

       n+=1.0

    pl.plot(s1,np.zeros(np.size(s1)),'k') #line across at zero
    pl.xlim(0,15)
    pl.ylim(-0.4, 1.05)
    pl.ylabel(r'$\theta(\xi)$')
    pl.xlabel(r'$\xi$')
    #pl.annotate('n=1.5',xy=(3.5,.1))
    pl.annotate('n=0',xy=(1.0,.1))
    pl.annotate('n=1',xy=(4,-.29))
    pl.annotate('n=2',xy=(7.0,-.21))
    pl.annotate('n=3',xy=(6.0,.1))
    pl.annotate('n=4',xy=(10,.08))
    pl.annotate('n=5',xy=(12,.18))

    pl.show()

