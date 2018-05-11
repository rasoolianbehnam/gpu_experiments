import numpy as np
import gnumpy
from working_gnumpy import *
import time as Time

def printf(text, *args):
    print(text%args)
imax=10
jmax=10
kmax=16
tmax=50
upper_lim = 3
n1 = imax+upper_lim
n2 = jmax+upper_lim
n3 = kmax+upper_lim
qi=1.6E-19
qe=-1.6E-19
q=1.6E-19
pie=3.14159
Kb    = 1.38E-23;
B     = 1.0;
Te    = 2.5*11604.5;
Ti    = 0.025*11604.5;
me    = 9.109E-31;
mi    = 6.633E-26;
ki    = 0.0;
dt    = 1.0E-12;
h     = 1.0E-3;
eps0  = 8.854E-12;
si    = 0.0;
sf    =0.0;


nn=1.33/(Kb*Ti); #neutral density=p/(Kb.T)
nue=nn*1.0E-20*np.sqrt(Kb*Te/me); # electron collision frequency= neutral density * sigma_e*Vth_e
nui=nn*5.0E-19*np.sqrt(Kb*Ti/mi);

wce=q*B/me;
wci=q*B/mi;
mue=q/(me*nue);
mui=q/(mi*nui);
dife=Kb*Te/(me*nue);
difi=Kb*Ti/(mi*nui);
ki=0.00002/(nn*dt);

denominator_e= (1+wce*wce/(nue*nue));
denominator_i= (1+wci*wci/(nui*nui));
print("%f %f \n"%(wce,wci));

Ta=np.arccos((np.cos(pie/imax)+np.cos(pie/jmax)+\
        np.cos(pie/kmax))/3.0);# needs to be double checked

w=2.0/(1.0+np.sin(Ta));
print("%f \n"%w);


def density_initialization(ne, ni, x_position, y_position, z_position):
    for i in range(1, imax+1):
        for j in range(1, jmax+1):
            for k in range(1, kmax-1):
                ne[i, j, k]=\
                        1.0E14+1.0E14*np.exp(-((i-x_position)**2\
                        +(j-y_position)**2+\
                        (k-z_position)**2)/100.0);
                ni[i, j, k]=1.0E14+1.0E14*np.exp(-(((i-x_position)**2)\
                        +((j-y_position)**2)+\
                        ((k-z_position)**2))/100.0)

def BC_densities(ne, ni):
     # BC on densities
    ne[imax+1, 0:jmax+1, 0:kmax] = ne[1, 0:jmax+1, 0:kmax]
    ni[imax+1, 0:jmax+1, 0:kmax] = ni[1, 0:jmax+1, 0:kmax]

    ne[0:jmax+1, jmax+1, 0:kmax] = ne[0:jmax+1, 1, 0:kmax]
    ni[0:jmax+1, jmax+1, 0:kmax] = ni[0:jmax+1, 1, 0:kmax]

    ne[0, 0:jmax+1, 0:kmax] = ne[imax, 0:jmax+1, 0:kmax]
    ni[0, 0:jmax+1, 0:kmax] = ni[imax, 0:jmax+1, 0:kmax]

    ne[0:jmax+1, 0, 0:kmax] = ne[0:jmax+1, jmax, 0:kmax]
    ni[0:jmax+1, 0, 0:kmax] = ni[0:jmax+1, jmax, 0:kmax]


def poisson_brute(V, g, num_iterations, imax, jmax, kmax, h=h, w=w):
    for kk in range(num_iterations):
        for i in range(1, imax+1):
            for j in range(1, jmax+1):
                for k in range(1, kmax-1):
                    r = V[i+1, j, k] / 6. + V[i-1, j, k] / 6. + V[i, j+1, k] / 6. + V[i, j-1, k] / 6. + V[i ,j, k+1] / 6. + V[i, j, k-1] / 6. - V[i, j, k] - (h**2) * g[i, j, k] / 6.
                    r = w * r
                    V[i, j, k] += r
    return V

def plasma_sim_solve_poisson_equation_on_grid(V, g, ne, ni, pv):
    w = pv.w
    h = pv.h
    num_iterations = pv.num_iterations
   # Here we calculate the right hand side of the Poisson equation
    g[1:imax+1, 1:jmax+1, 1:kmax-1]=-(ne[1:imax+1, 1:jmax+1, 1:kmax-1]*qe\
            +ni[1:imax+1, 1:jmax+1, 1:kmax-1]*qi)/eps0;
    #print("Starting poisson")
    ################################333
    #V1 = pv.poisson_fast_no_loop(V.reshape(-1),  g.reshape(-1)).reshape(n1, n2, n3)
    V1 = pv.poisson_fast_no_loop_torch(V.reshape(-1, 1),  g.reshape(-1, 1)).reshape(n1, n2, n3)
    ################################333
    #V2 = poisson_brute(V*1., g*1., 40, pv.imax, pv.jmax, pv.kmax, pv.h, pv.w)
    #V3 = pv.poisson_brute_main(V*1., g*1.)
    #V[:, :, :] = pv.poisson_brute_main_flat(V.reshape(-1),  g.reshape(-1)).reshape(n1, n2, n3)
    #stat_diff(V1, V2, "fast no loop vs brute here")
    #stat_diff(V2, V3, "brute pv loop vs brute here")
    V = V1


    V[imax+1, 0:jmax+1, 0:kmax]=V[1, 0:jmax+1, 0:kmax];
    V[0:jmax+1, jmax+1, 0:kmax]=V[0:jmax+1, 1, 0:kmax];
    V[0, 0:jmax+1, 0:kmax]=V[imax, 0:jmax+1, 0:kmax];
    V[0:jmax+1, 0, 0:kmax]=V[0:jmax+1, jmax, 0:kmax];

    return V, g



def electric_field_elements(V, ne, ni, Ez, Ex, Ey, difxne, difxni, difyne, difyni, difzne, difzni):
    Ez[1:imax+1, 1:jmax+1, 0:kmax-1] = \
            (V[1:imax+1, 1:jmax+1, 0:kmax-1]
            - V[1:imax+1, 1:jmax+1, 1:kmax])/h

    difzne[1:imax+1, 1:jmax+1, 0:kmax-1]=\
        (ne[1:imax+1, 1:jmax+1, 1:kmax]\
        -ne[1:imax+1, 1:jmax+1, 0:kmax-1])/h;
    difzni[1:imax+1, 1:jmax+1, 0:kmax-1]=\
        (ni[1:imax+1, 1:jmax+1, 1:kmax]\
        -ni[1:imax+1, 1:jmax+1, 0:kmax-1])/h;

    Ex[1:imax+1, 1:jmax+1, 1:kmax-1] = (V[1:imax+1, 1:jmax+1, 1:kmax-1]-\
            V[1+1:imax+1+1, 1:jmax+1, 1:kmax-1]) / h
    Ey[1:imax+1, 1:jmax+1, 1:kmax-1] = (V[1:imax+1, 1:jmax+1, 1:kmax-1]-\
            V[1:imax+1, 1+1:jmax+1+1, 1:kmax-1]) / h

    difxne[1:imax+1, 1:jmax+1, 1:kmax-1] = \
            (ne[2:imax+1+1, 1:jmax+1, 1:kmax-1] - ne[1:imax+1, 1:jmax+1, 1:kmax-1]) / h
    difxni[1:imax+1, 1:jmax+1, 1:kmax-1] = \
            (ni[2:imax+1+1, 1:jmax+1, 1:kmax-1] - ni[1:imax+1, 1:jmax+1, 1:kmax-1]) / h
    difyne[1:imax+1, 1:jmax+1, 1:kmax-1] = \
            (ne[1:imax+1, 2:jmax+1+1, 1:kmax-1] - ne[1:imax+1, 1:jmax+1, 1:kmax-1]) / h
    difyni[1:imax+1, 1:jmax+1, 1:kmax-1] = \
            (ni[1:imax+1, 2:jmax+1+1, 1:kmax-1] - ne[1:imax+1, 1:jmax+1, 1:kmax-1]) / h

def average_x(ne, ni, Ex, Exy, difxne, difxni, difxyne, difxyni):
    Exy[2:imax+1, 1:jmax, 1:kmax-1] = .25*(Ex[2:imax+1, 1:jmax, 1:kmax-1] +\
            Ex[2:imax+1, 2:jmax+1, 1:kmax-1] + Ex[1:imax, 1:jmax, 1:kmax-1] + \
            Ex[1:imax, 2:jmax+1, 1:kmax-1])
    difxyne[2:imax+1, 1:jmax, 1:kmax-1] = .25*(difxne[2:imax+1, 1:jmax, 1:kmax-1] +\
            difxne[2:imax+1, 2:jmax+1, 1:kmax-1] + difxne[1:imax, 1:jmax, 1:kmax-1]) +\
            difxne[1:imax, 2:jmax+1, 1:kmax-1]
    difxyni[2:imax+1, 1:jmax, 1:kmax-1] = .25*(difxni[2:imax+1, 1:jmax, 1:kmax-1] +\
            difxni[2:imax+1, 2:jmax+1, 1:kmax-1] + difxni[1:imax, 1:jmax, 1:kmax-1]) +\
            difxni[1:imax, 2:jmax+1, 1:kmax-1]

    Exy[1, 1:jmax, 1:kmax-1] = .25*(Ex[1, 1:jmax, 1:kmax-1] +\
            Ex[1, 2:jmax+1, 1:kmax-1] \
            + Ex[imax, 1:jmax, 1:kmax-1]\
            + Ex[imax, 2:jmax+1, 1:kmax-1])


    difxyne[1, 1:jmax, 1:kmax-1]=0.25*(difxne[1, 1:jmax, 1:kmax-1]+difxne[1, 2:jmax+1, 1:kmax-1]+difxne[imax, 1:jmax, 1:kmax-1]+difxne[imax, 2:jmax+1, 1:kmax-1]);
    difxyni[1, 1:jmax, 1:kmax-1]=0.25*(difxni[1, 1:jmax, 1:kmax-1]+difxni[1, 2:jmax+1, 1:kmax-1]+difxni[imax, 1:jmax, 1:kmax-1]+difxni[imax, 2:jmax+1, 1:kmax-1]);

    Exy[1:imax, jmax, 1:kmax-1]= \
            0.25*(Ex[1:imax, jmax, 1:kmax-1]+Ex[1:imax, 1, 1:kmax-1]+Ex[0:imax-1, jmax, 1:kmax-1]+Ex[0:imax-1, 1, 1:kmax-1]) ;

    difxyne[1:imax, jmax, 1:kmax-1]=\
            0.25*(difxne[1:imax, jmax, 1:kmax-1]+difxne[1:imax, 1, 1:kmax-1]+difxne[0:imax-1, jmax, 1:kmax-1]+difxne[0:imax-1, 1, 1:kmax-1]);
    difxyni[1:imax, jmax, 1:kmax-1]=\
            0.25*(difxni[1:imax, jmax, 1:kmax-1]+difxni[1:imax, 1, 1:kmax-1]+difxni[0:imax-1, jmax, 1:kmax-1]+difxni[0:imax-1, 1, 1:kmax-1]);

    Exy[imax, jmax, 1:kmax-1]=(Ex[imax, jmax, 1:kmax-1]+Ex[imax-1, jmax, 1:kmax-1]+Ex[imax-1, 1, 1:kmax-1])/3.0;
    difxyne[imax, jmax, 1:kmax-1]=(difxne[imax, jmax, 1:kmax-1]+difxne[imax-1, jmax, 1:kmax-1]+difxne[imax-1, 1, 1:kmax-1])/3.0;
    difxyni[imax, jmax, 1:kmax-1]=(difxni[imax, jmax, 1:kmax-1]+difxni[imax-1, jmax, 1:kmax-1]+difxni[imax-1, 1, 1:kmax-1])/3.0;

    Exy[1, jmax, 1:kmax-1]=(Ex[1, jmax, 1:kmax-1]+Ex[imax, jmax, 1:kmax-1]+Ex[1, 1, 1:kmax-1])/3.0;
    difxyne[1, jmax, 1:kmax-1]=(difxne[1, jmax, 1:kmax-1]+difxne[imax, jmax, 1:kmax-1]+difxne[1, 1, 1:kmax-1])/3.0;
    difxyni[1, jmax, 1:kmax-1]=(difxni[1, jmax, 1:kmax-1]+difxni[imax, jmax, 1:kmax-1]+difxni[1, 1, 1:kmax-1])/3.0;



def flux_x(ne, ni, fex, fix, Ex, Exy, difxne, difxni, difxyne, difxyni):
    fex[1:imax+1, 1:jmax+1, 1:kmax-1]=\
            (-(ne[1:imax+1, 1:jmax+1, 1:kmax-1]*0.5\
            +ne[2:imax+1+1, 1:jmax+1, 1:kmax-1])*mue\
            *Ex[1:imax+1, 1:jmax+1, 1:kmax-1]\
            -difxne[1:imax+1, 1:jmax+1, 1:kmax-1]*dife
            +difxyne[1:imax+1, 1:jmax+1, 1:kmax-1]*wce*dife/nue\
                    +(ne[1:imax+1, 1:jmax+1, 1:kmax-1]\
                    +ne[2:imax+1+1, 1:jmax+1, 1:kmax-1])*wce*q*0.5/\
                    (me*nue*nue)*Exy[1:imax+1, 1:jmax+1, 1:kmax-1])/denominator_e;

    fix[1:imax+1, 1:jmax+1, 1:kmax-1]=\
            (0.5*(ni[1:imax+1, 1:jmax+1, 1:kmax-1]\
            +ni[2:imax+1+1, 1:jmax+1, 1:kmax-1])\
            *Ex[1:imax+1, 1:jmax+1, 1:kmax-1]*mui\
            -difxni[1:imax+1, 1:jmax+1, 1:kmax-1]*difi\
            -difxyni[1:imax+1, 1:jmax+1, 1:kmax-1]*wci*difi/nui\
            +(ni[1:imax+1, 1:jmax+1, 1:kmax-1]+ni[2:imax+1+1, 1:jmax+1, 1:kmax-1])*wci*q*0.5\
            *Exy[1:imax+1, 1:jmax+1, 1:kmax-1]/(mi*nui*nui))/denominator_i;

    fex[0, 1:jmax+1, 1:kmax-1] = fex[imax, 1:jmax+1, 1:kmax-1];
    fix[0, 1:jmax+1, 1:kmax-1] = fix[imax, 1:jmax+1, 1:kmax-1];


def average_y(ne, ni, Ey, Exy, difyne, difyni, difxyne, difxyni):
    Exy[1:imax, 2:jmax+1, 1:kmax-1]= 0.25*(Ey[1:imax, 2:jmax+1, 1:kmax-1]+Ey[1:imax, 1:jmax+1-1, 1:kmax-1]+Ey[2:imax+1, 2:jmax+1, 1:kmax-1]+Ey[2:imax+1, 1:jmax+1-1, 1:kmax-1]); 
    difxyne[1:imax, 2:jmax+1, 1:kmax-1]= 0.25*(difyne[1:imax, 2:jmax+1, 1:kmax-1]+difyne[1:imax, 1:jmax+1-1, 1:kmax-1]+difyne[2:imax+1, 2:jmax+1, 1:kmax-1]+difyne[2:imax+1, 1:jmax+1-1, 1:kmax-1]);
    difxyni[1:imax, 2:jmax+1, 1:kmax-1]= 0.25*(difyni[1:imax, 2:jmax+1, 1:kmax-1]+difyni[1:imax, 1:jmax+1-1, 1:kmax-1]+difyni[2:imax+1, 2:jmax+1, 1:kmax-1]+difyni[2:imax+1, 1:jmax+1-1, 1:kmax-1]);

    Exy[1:imax, 1, 1:kmax-1]= 0.25*(Ey[1:imax, 1, 1:kmax-1]+Ey[1:imax, jmax, 1:kmax-1]+Ey[2:imax+1, 1, 1:kmax-1]+Ey[2:imax+1, jmax, 1:kmax-1]);
    difxyne[1:imax, 1, 1:kmax-1]= 0.25*(difyne[1:imax, 1, 1:kmax-1]+difyne[1:imax, jmax, 1:kmax-1]+difyne[2:imax+1, 1, 1:kmax-1]+difyne[2:imax+1, jmax, 1:kmax-1]);
    difxyni[1:imax, 1, 1:kmax-1]= 0.25*(difyni[1:imax, 1, 1:kmax-1]+difyni[1:imax, jmax, 1:kmax-1]+difyni[2:imax+1, 1, 1:kmax-1]+difyni[2:imax+1, jmax, 1:kmax-1]);


    Exy[imax, 2:jmax+1, 1:kmax-1]= 0.25*(Ey[imax, 2:jmax+1, 1:kmax-1]+Ey[imax, 1:jmax+1-1, 1:kmax-1]+Ey[1, 2:jmax+1, 1:kmax-1]+Ey[1, 1:jmax+1-1, 1:kmax-1]); 
    difxyne[imax, 2:jmax+1, 1:kmax-1]= 0.25*(difyne[imax, 2:jmax+1, 1:kmax-1]+difyne[imax, 1:jmax+1-1, 1:kmax-1]+difyne[1, 2:jmax+1, 1:kmax-1]+difyne[1, 1:jmax+1-1, 1:kmax-1]);
    difxyni[imax, 2:jmax+1, 1:kmax-1]= 0.25*(difyni[imax, 2:jmax+1, 1:kmax-1]+difyni[imax, 1:jmax+1-1, 1:kmax-1]+difyni[1, 2:jmax+1, 1:kmax-1]+difyni[1, 1:jmax+1-1, 1:kmax-1]);


    Exy[imax, 1, 1:kmax-1]=(Ey[imax, 1, 1:kmax-1]+Ey[1, 1, 1:kmax-1]+Ey[imax, jmax, 1:kmax-1])/3.0;
    difxyne[imax, 1, 1:kmax-1]=(difyne[imax, 1, 1:kmax-1]+difyne[1, 1, 1:kmax-1]+difyne[imax, jmax, 1:kmax-1])/3.0;
    difxyni[imax, 1, 1:kmax-1]=(difyni[imax, 1, 1:kmax-1]+difyni[1, 1, 1:kmax-1]+difyni[imax, jmax, 1:kmax-1])/3.0; 
    Exy[imax, jmax, 1:kmax-1]=(Ey[imax, jmax-1, 1:kmax-1]+Ey[imax, jmax, 1:kmax-1]+Ey[1, jmax-1, 1:kmax-1])/3.0;
    difxyne[imax, jmax, 1:kmax-1]=(difyne[imax, jmax-1, 1:kmax-1]+difyne[imax, jmax, 1:kmax-1]+difyne[1, jmax-1, 1:kmax-1])/3.0;
    difxyni[imax, jmax, 1:kmax-1]=(difyni[imax, jmax-1, 1:kmax-1]+difyni[imax, jmax, 1:kmax-1]+difyni[1, jmax-1, 1:kmax-1])/3.0; 

def flux_y(ne, ni, fey, fiy, Ey, Ez, Exy, difyne, difyni, difxyne, difxyni):
    fey[1:imax+1, 1:jmax+1, 1:kmax-1]= \
            (-(ne[1:imax+1, 2:jmax+1+1, 1:kmax-1]+ne[1:imax+1, 1:jmax+1, 1:kmax-1])*0.5\
            *Ey[1:imax+1, 1:jmax+1, 1:kmax-1]*mue\
            -difyne[1:imax+1, 1:jmax+1, 1:kmax-1]*dife\
            -(ne[1:imax+1, 2:jmax+1+1, 1:kmax-1]+ne[1:imax+1, 1:jmax+1, 1:kmax-1])*wce*q*0.5\
            *Exy[1:imax+1, 1:jmax+1, 1:kmax-1]/(me*nue*nue)\
            -difxyne[1:imax+1, 1:jmax+1, 1:kmax-1]*wce*dife/nue)/denominator_e;

    fiy[1:imax+1, 1:jmax+1, 1:kmax-1]= \
            ((ni[1:imax+1, 2:jmax+1+1, 1:kmax-1]*0.5\
            +ni[1:imax+1, 1:jmax+1, 1:kmax-1])\
            *Ey[1:imax+1, 1:jmax+1, 1:kmax-1]*mui\
            -difyni[1:imax+1, 1:jmax+1, 1:kmax-1]*difi
    -(ni[1:imax+1, 2:jmax+1+1, 1:kmax-1]+ni[1:imax+1, 1:jmax+1, 1:kmax-1])*wci*q*0.5\
            *Exy[1:imax+1, 1:jmax+1, 1:kmax-1]/(mi*nui*nui)\
            +difxyni[1:imax+1, 1:jmax+1, 1:kmax-1]*wci*difi/nui)/denominator_i;


    fey[1:imax+1, 0, 1:kmax-1] = fey[1:imax+1, jmax, 1:kmax-1];
    fiy[1:imax+1, 0, 1:kmax-1] = fiy[1:imax+1, jmax, 1:kmax-1];

def flux_z(ne, ni, Ez, fez, fiz, difzne, difzni):
    fez[1:imax+1, 1:jmax+1, 1:kmax-1]=\
            -(ne[1:imax+1, 1:jmax+1, 1:kmax-1]\
            +ne[1:imax+1, 1:jmax+1, 2:kmax-1+1])*.5\
            *Ez[1:imax+1, 1:jmax+1, 1:kmax-1]*mue\
            -difzne[1:imax+1, 1:jmax+1, 1:kmax-1]*dife;

    fiz[1:imax+1, 1:jmax+1, 1:kmax-1]=\
            (ni[1:imax+1, 1:jmax+1, 1:kmax-1]\
            +ni[1:imax+1, 1:jmax+1, 2:kmax-1+1])*0.5\
            *Ez[1:imax+1, 1:jmax+1, 1:kmax-1]*mui\
            -difzni[1:imax+1, 1:jmax+1, 1:kmax-1]*difi;

    fez[:, :, 1] = fez[:, :, 1] * (fez[:, :, 1] <= 0)
    fiz[:, :, 1] = fiz[:, :, 1] * (fiz[:, :, 1] <= 0)
    fez[:, :, kmax-1] = fez[:, :, kmax-1] * (fez[:, :, kmax-1] <= 0)
    fiz[:, :, kmax-1] = fiz[:, :, kmax-1] * (fiz[:, :, kmax-1] <= 0)


def main(want_cuda=False):
    g       = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    R       = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    ne      = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    ni      = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    V       = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    Ex      = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    Ey      = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    Ez      = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    fex     = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    fey     = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    fez     = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    fix     = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    fiy     = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    fiz     = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    difxne  = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    difxni  = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    difyne  = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    difyni  = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    difzne  = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    difzni  = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    Exy     = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    difxyne = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))
    difxyni = gnumpy.zeros((imax+upper_lim, jmax+upper_lim, kmax+upper_lim))

    method = 'ndarray'
    pv = poisson_vectorized(imax+upper_lim, jmax+upper_lim, kmax+upper_lim,\
            w=w, h=h, num_iterations=40, method=method\
            , upper_lim=upper_lim, want_cuda=want_cuda)


    density_initialization(ne, ni, 15,15,15);

    si = gp.sum(ne[1:imax+1, 1:jmax+1, 1:kmax+1])
    #printf("si before loop: %f", si);
    #printf("ne[%d, %d, %d] = %e\n", 5, 6, 7, ne[5, 6, 7]);
    #printf("ni[%d, %d, %d] = %e\n", 5, 6, 7, ni[5, 6, 7]);
    BC_densities(ne, ni)
    
    start = Time.time()
    for time in range(1, tmax):
        V, g = plasma_sim_solve_poisson_equation_on_grid(V, g, ne, ni, pv)
        #printf("V[%d, %d, %d] = %f\n", 5, 6, 7, V[5, 6, 7]);
        electric_field_elements(V, ne, ni, Ez, Ex, Ey, difxne, difxni, difyne, difyni, difzne, difzni)
        average_x(ne, ni, Ex, Exy, difxne, difxni, difxyne, difxyni)
        flux_y(ne, ni, fey, fiy, Ey, Ez, Exy, difyne, difyni, difxyne, difxyni)
        average_y(ne, ni, Ey, Exy, difyne, difyni, difxyne, difxyni)
        flux_x(ne, ni, fex, fix, Ex, Exy, difxne, difxni, difxyne, difxyni)
        flux_z(ne, ni, Ez, fez, fiz, difzne, difzni)

        ne[1:imax+1, 1:jmax+1, 1:kmax] = \
            ne[1:imax+1, 1:jmax+1, 1:kmax] -\
            dt * (\
              fex[1:imax+1, 1:jmax+1, 1:kmax  ] \
            - fex[0:imax  , 1:jmax+1, 1:kmax  ] \
            + fey[1:imax+1, 1:jmax+1, 1:kmax  ] \
            - fey[1:imax+1, 0:jmax  , 1:kmax  ] \
            + fez[1:imax+1, 1:jmax+1, 1:kmax  ] \
            - fez[1:imax+1, 1:jmax+1, 0:kmax-1])/h
#                
        ni[1:imax+1, 1:jmax+1, 1:kmax] = \
            ni[1:imax+1, 1:jmax+1, 1:kmax] -\
            dt * (
              fix[1:imax+1, 1:jmax+1, 1:kmax] \
            - fix[0:imax, 1:jmax+1, 1:kmax] \
            + fiy[1:imax+1, 1:jmax+1, 1:kmax] \
            - fiy[1:imax+1, 0:jmax, 1:kmax] \
            + fiz[1:imax+1, 1:jmax+1, 1:kmax] \
            - fiz[1:imax+1, 1:jmax+1, 0:kmax-1])/h

        ne[1:imax+1, 1:jmax+1, 0] = -dt * fez[1:imax+1, 1:jmax+1, 0] / h
        ni[1:imax+1, 1:jmax+1, 0] = -dt * fiz[1:imax+1, 1:jmax+1, 0] / h

        BC_densities(ne, ni)

        sf = gp.sum(ne[1:imax+1, 1:jmax+1, 1:kmax+1])
        
        alpha = (si -sf) / sf;
        ne[1:imax+1, 1:jmax+1, 1:kmax-1] = \
                ne[1:imax+1, 1:jmax+1, 1:kmax-1] +\
                alpha * ne[1:imax+1, 1:jmax+1, 1:kmax-1]

        #printf("%d \n", time);
    time_taken2 = Time.time() - start
    print("Time taken: %f"%(time_taken2))
    return V, g, pv

if __name__=="__main__":
    main()
