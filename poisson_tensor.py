import numpy as np
from working_tensor import *
import time as Time
def slice_add(c, a, axes, N):
    if len(axes) == 1:
        n = N[0]
        xs = axes[0]
        print(n)
        mask = tf.ones_like(c)
        padding = tf.constant([[xs[0], n-xs[1]]])
        c_pad = tf.pad(c, padding, 'CONSTANT')
        print(c_pad)
        mask = tf.cast(tf.pad(mask, padding, 'CONSTANT'), tf.bool)
        return tf.where(mask, c_pad, a)    
    elif len(axes) == 2:
        n, m = N
        xs = axes[0]
        ys = axes[1]
        print(n, m)
        mask = tf.ones_like(c)
        padding = tf.constant([[xs[0], n-xs[1]], [ys[0], m-ys[1]]])
        c_pad = tf.pad(c, padding, 'CONSTANT')
        print(c_pad)
        mask = tf.cast(tf.pad(mask, padding, 'CONSTANT'), tf.bool)
        return tf.where(mask, c_pad, a)
    elif len(axes) == 3:
        n, m, s = N
        xs = axes[0]
        ys = axes[1]
        zs = axes[2]
        #print(n, m,)
        mask = tf.ones_like(c)
        padding = tf.constant([[xs[0], n-xs[1]], [ys[0], m-ys[1]], [zs[0], s-zs[1]]])
        c_pad = tf.pad(c, padding, 'CONSTANT')
        #print(c_pad)
        mask = tf.cast(tf.pad(mask, padding, 'CONSTANT'), tf.bool)

    return tf.where(mask, c_pad, a)    

def printf(text, *args):
    print(text%args)

imax=16
jmax=16
kmax=16
tmax=500
upper_lim = 3
n1 = imax+upper_lim
n2 = jmax+upper_lim
n3 = kmax+upper_lim
nnn = [n1, n2, n3]
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


def density_initialization(n1, n2, n3, x_position, y_position, z_position):
    ne = np.zeros((n1, n2, n3))
    ni = np.zeros((n1, n2, n3))
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
    return tf.constant(ne, dtype=tf.float32), tf.constant(ni, dtype=tf.float32)

def BC_densities(ne, ni):
     # BC on densities
    indices = [[imax+1, imax+2], [0, jmax+1],  [0, kmax]]
    ne_temp = ne[1:2, 0:jmax+1, 0:kmax]
    ni_temp = ni[1:2, 0:jmax+1, 0:kmax]
    ne = slice_add(ne_temp, ne, indices, nnn)
    ni = slice_add(ne_temp, ne, indices, nnn)

    indices = [[0, imax+1], [jmax+1, jmax+2],  [0, kmax]]
    ne_temp = ne[0:jmax+1, 1:2, 0:kmax]
    ni_temp = ni[0:jmax+1, 1:2, 0:kmax]
    ne = slice_add(ne_temp, ne, indices, nnn)
    ni = slice_add(ne_temp, ne, indices, nnn)

    indices = [[0, 1], [0, jmax+1],  [0, kmax]]
    ne_temp = ne[imax:imax+1, 0:jmax+1, 0:kmax]
    ni_temp = ni[imax:imax+1, 0:jmax+1, 0:kmax]
    ne = slice_add(ne_temp, ne, indices, nnn)
    ni = slice_add(ne_temp, ne, indices, nnn)

    indices = [[0,jmax+1], [ 0,1], [0,kmax]]
    ne_temp = ne[0:jmax+1, jmax:jmax+1, 0:kmax]
    ni_temp = ni[0:jmax+1, jmax:jmax+1, 0:kmax]
    ne = slice_add(ne_temp, ne, indices, nnn)
    ni = slice_add(ne_temp, ne, indices, nnn)



def poisson_brute(V, g, num_iterations, imax, jmax, kmax, h=h, w=w):
    for kk in range(num_iterations):
        for i in range(1, imax+1):
            for j in range(1, jmax+1):
                for k in range(1, kmax-1):
                    r = V[i+1, j, k] / 6. + V[i-1, j, k] / 6. + V[i, j+1, k] / 6. + V[i, j-1, k] / 6. + V[i ,j, k+1] / 6. + V[i, j, k-1] / 6. - V[i, j, k] - (h**2) * g[i, j, k] / 6.
                    r = w * r
                    V[i, j, k] += r
    return V

def plasma_sim_solve_poisson_equation_on_grid(V, g, ne, ni, pv, sess):
    w = pv.w
    h = pv.h
    num_iterations = pv.num_iterations
   # Here we calculate the right hand side of the Poisson equation
    indices = [[1, imax+1], [1, jmax+1], [1, kmax-1]]
    g_temp=-(ne[1:imax+1, 1:jmax+1, 1:kmax-1]*qe\
            +ni[1:imax+1, 1:jmax+1, 1:kmax-1]*qi)/eps0;
    g = slice_add(g_temp, g, indices, nnn)
    #print("Starting poisson")
    #start = Time.time()
    ################################333
    #V1 = pv.poisson_fast_no_loop(V.reshape(-1),  g.reshape(-1)).reshape(n1, n2, n3)
    V1 = tf.reshape(pv.poisson_fast_no_loop_tensor(tf.reshape(V, [-1, 1]), tf.reshape(g, [-1, 1]), sess), [n1, n2, n3])
    ################################333
    #V2 = poisson_brute(V*1., g*1., 40, pv.imax, pv.jmax, pv.kmax, pv.h, pv.w)
    #V3 = pv.poisson_brute_main(V*1., g*1.)
    #V[:, :, :] = pv.poisson_brute_main_flat(V.reshape(-1),  g.reshape(-1)).reshape(n1, n2, n3)
    #stat_diff(V1, V2, "fast no loop vs brute here")
    #stat_diff(V2, V3, "brute pv loop vs brute here")
    #time_taken2 = Time.time() - start
    #print("Time taken: %f"%(time_taken2))
    V = V1


    indices = [[imax+1, imax+2], [0,jmax+1], [0, kmax]]
    V_temp=V[1:2, 0:jmax+1, 0:kmax];
    V = slice_add(V_temp, V, indices, nnn)

    indices = [[0,jmax+1], [jmax+1,jmax+2], [0, kmax]]
    V_temp=V[0:jmax+1, 1:2, 0:kmax];
    V = slice_add(V_temp, V, indices, nnn)

    indices = [[0,1], [0, jmax+1], [0, kmax]]
    V_temp=V[imax:imax+1, 0:jmax+1, 0:kmax];
    V = slice_add(V_temp, V, indices, nnn)

    indices = [[0,jmax+1], [0, 1], [0, kmax]]
    V_temp=V[0:jmax+1, jmax:jmax+1, 0:kmax];
    V = slice_add(V_temp, V, indices, nnn)

    return V, g



def electric_field_elements(V, ne, ni, Ez, Ex, Ey, difxne, difxni, difyne, difyni, difzne, difzni):
    indices = [[1, imax+1], [1,jmax+1], [0,kmax-1]]
    Ez_temp = (V[1:imax+1, 1:jmax+1, 0:kmax-1]
            - V[1:imax+1, 1:jmax+1, 1:kmax])/h

    difzne_temp=\
        (ne[1:imax+1, 1:jmax+1, 1:kmax]\
        -ne[1:imax+1, 1:jmax+1, 0:kmax-1])/h;
    difzni_temp=\
        (ni[1:imax+1, 1:jmax+1, 1:kmax]\
        -ni[1:imax+1, 1:jmax+1, 0:kmax-1])/h;

    Ez = slice_add(Ez_temp, Ez, indices, nnn)
    difzne = slice_add(difzne_temp, difzne, indices, nnn)
    difzni = slice_add(difzni_temp, difzni, indices, nnn)

    indices = [[1, imax+1], [1,jmax+1], [1,kmax-1]]
    Ex_temp = (V[1:imax+1, 1:jmax+1, 1:kmax-1]-\
            V[1+1:imax+1+1, 1:jmax+1, 1:kmax-1]) / h
    Ey_temp = (V[1:imax+1, 1:jmax+1, 1:kmax-1]-\
            V[1:imax+1, 1+1:jmax+1+1, 1:kmax-1]) / h

    difxne_temp = \
            (ne[2:imax+1+1, 1:jmax+1, 1:kmax-1] - ne[1:imax+1, 1:jmax+1, 1:kmax-1]) / h
    difxni_temp = \
            (ni[2:imax+1+1, 1:jmax+1, 1:kmax-1] - ni[1:imax+1, 1:jmax+1, 1:kmax-1]) / h
    difyne_temp = \
            (ne[1:imax+1, 2:jmax+1+1, 1:kmax-1] - ne[1:imax+1, 1:jmax+1, 1:kmax-1]) / h
    difyni_temp = \
            (ni[1:imax+1, 2:jmax+1+1, 1:kmax-1] - ne[1:imax+1, 1:jmax+1, 1:kmax-1]) / h

    Ex = slice_add(Ex_temp, Ex, indices, nnn)
    Ey = slice_add(Ex_temp, Ex, indices, nnn)
    difxne = slice_add(difxne_temp, difxne, indices, nnn)
    difxni = slice_add(difxni_temp, difxni, indices, nnn)
    difyne = slice_add(difyne_temp, difyne, indices, nnn)
    difyni = slice_add(difyni_temp, difyni, indices, nnn)

def average_x(ne, ni, Ex, Exy, difxne, difxni, difxyne, difxyni):
    indices = [[2, imax+1], [1, jmax], [1, kmax-1]]
    Exy_temp = .25*(Ex[2:imax+1, 1:jmax, 1:kmax-1] +\
            Ex[2:imax+1, 2:jmax+1, 1:kmax-1] + Ex[1:imax, 1:jmax, 1:kmax-1] + \
            Ex[1:imax, 2:jmax+1, 1:kmax-1])
    difxyne_temp = .25*(difxne[2:imax+1, 1:jmax, 1:kmax-1] +\
            difxne[2:imax+1, 2:jmax+1, 1:kmax-1] + difxne[1:imax, 1:jmax, 1:kmax-1]) +\
            difxne[1:imax, 2:jmax+1, 1:kmax-1]
    difxyni_temp = .25*(difxni[2:imax+1, 1:jmax, 1:kmax-1] +\
            difxni[2:imax+1, 2:jmax+1, 1:kmax-1] + difxni[1:imax, 1:jmax, 1:kmax-1]) +\
            difxni[1:imax, 2:jmax+1, 1:kmax-1]
    Exy = slice_add(Exy_temp, Exy, slices, nnn)
    difxyne = slice_add(difxyne_temp, difxyne, slices, nnn)
    difxyni = slice_add(difxyni_temp, difxyni, slices, nnn)

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
    indices = [[1, imax+1], [1, jmax+1], [1, kmax-1]]
    fex_temp=\
            (-0.5*(ne[1:imax+1, 1:jmax+1, 1:kmax-1]\
            +ne[2:imax+1+1, 1:jmax+1, 1:kmax-1])*mue\
            *Ex[1:imax+1, 1:jmax+1, 1:kmax-1]\
            -dife*difxne[1:imax+1, 1:jmax+1, 1:kmax-1]
            +wce*dife*difxyne[1:imax+1, 1:jmax+1, 1:kmax-1]\
                    /nue+wce*q*0.5*(ne[1:imax+1, 1:jmax+1, 1:kmax-1]\
                    +ne[2:imax+1+1, 1:jmax+1, 1:kmax-1])/(me*nue*nue)*Exy[1:imax+1, 1:jmax+1, 1:kmax-1])/denominator_e;

    fix_temp=(0.5*(ni[1:imax+1, 1:jmax+1, 1:kmax-1]+ni[2:imax+1+1, 1:jmax+1, 1:kmax-1])*mui*Ex[1:imax+1, 1:jmax+1, 1:kmax-1]-difi*difxni[1:imax+1, 1:jmax+1, 1:kmax-1]
    -wci*difi*difxyni[1:imax+1, 1:jmax+1, 1:kmax-1]/nui+wci*q*0.5*(ni[1:imax+1, 1:jmax+1, 1:kmax-1]+ni[2:imax+1+1, 1:jmax+1, 1:kmax-1])*Exy[1:imax+1, 1:jmax+1, 1:kmax-1]/(mi*nui*nui))/denominator_i;

    fex = slice_add(fex_temp, fex, indices, nnn)
    fix = slice_add(fex_temp, fex, indices, nnn)

    indices = [[0, 1], [1, jmax+1], [1, kmax-1]]
    fex_temp = fex[imax:imax+1, 1:jmax+1, 1:kmax-1];
    fix_temp = fix[imax:imax+1, 1:jmax+1, 1:kmax-1];
    fex = slice_add(fex_temp, fex, indices, nnn)
    fix = slice_add(fex_temp, fex, indices, nnn)


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
    fey[1:imax+1, 1:jmax+1, 1:kmax-1]= (-0.5*(ne[1:imax+1, 2:jmax+1+1, 1:kmax-1]+ne[1:imax+1, 1:jmax+1, 1:kmax-1])*mue*Ey[1:imax+1, 1:jmax+1, 1:kmax-1]-dife*difyne[1:imax+1, 1:jmax+1, 1:kmax-1]
    -wce*q*0.5*(ne[1:imax+1, 2:jmax+1+1, 1:kmax-1]+ne[1:imax+1, 1:jmax+1, 1:kmax-1])*Exy[1:imax+1, 1:jmax+1, 1:kmax-1]/(me*nue*nue)-wce*dife*difxyne[1:imax+1, 1:jmax+1, 1:kmax-1]/nue)/denominator_e;

    fiy[1:imax+1, 1:jmax+1, 1:kmax-1]= (0.5*(ni[1:imax+1, 2:jmax+1+1, 1:kmax-1]+ni[1:imax+1, 1:jmax+1, 1:kmax-1])*mui*Ey[1:imax+1, 1:jmax+1, 1:kmax-1]-difi*difyni[1:imax+1, 1:jmax+1, 1:kmax-1]
    -wci*q*0.5*(ni[1:imax+1, 2:jmax+1+1, 1:kmax-1]+ni[1:imax+1, 1:jmax+1, 1:kmax-1])*Exy[1:imax+1, 1:jmax+1, 1:kmax-1]/(mi*nui*nui)+wci*difi*difxyni[1:imax+1, 1:jmax+1, 1:kmax-1]/nui)/denominator_i;


    fey[1:imax+1, 0, 1:kmax-1] = fey[1:imax+1, jmax, 1:kmax-1];
    fiy[1:imax+1, 0, 1:kmax-1] = fiy[1:imax+1, jmax, 1:kmax-1];

def flux_z(ne, ni, Ez, fez, fiz, difzne, difzni):
    fez[1:imax+1, 1:jmax+1, 1:kmax-1]=-0.5*(ne[1:imax+1, 1:jmax+1, 1:kmax-1]+ne[1:imax+1, 1:jmax+1, 2:kmax-1+1])*mue*Ez[1:imax+1, 1:jmax+1, 1:kmax-1]-dife*difzne[1:imax+1, 1:jmax+1, 1:kmax-1];

    fiz[1:imax+1, 1:jmax+1, 1:kmax-1]=0.5*(ni[1:imax+1, 1:jmax+1, 1:kmax-1]+ni[1:imax+1, 1:jmax+1, 2:kmax-1+1])*mui*Ez[1:imax+1, 1:jmax+1, 1:kmax-1]-difi*difzni[1:imax+1, 1:jmax+1, 1:kmax-1];

    fez[:, :, 1][np.where(fez[:, :, 1] > 0)] = 0
    fiz[:, :, 1][np.where(fiz[:, :, 1] > 0)] = 0
    fez[:, :, kmax-1][np.where(fez[:, :, kmax-1] > 0)] = 0
    fiz[:, :, kmax-1][np.where(fiz[:, :, kmax-1] > 0)] = 0
 

def main():
    g       = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    R       = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    ne      = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    ni      = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    V       = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    Ex      = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    Ey      = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    Ez      = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    fex     = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    fey     = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    fez     = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    fix     = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    fiy     = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    fiz     = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    difxne  = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    difxni  = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    difyne  = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    difyni  = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    difzne  = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    difzni  = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    Exy     = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    difxyne = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
    difxyni = tf.zeros([imax+upper_lim, jmax+upper_lim, kmax+upper_lim])
   
    method = 'ndarray'
    pv = poisson_vectorized(imax+upper_lim, jmax+upper_lim, kmax+upper_lim,\
            w=w, h=h, num_iterations=40, method=method\
            , upper_lim=upper_lim)


    ne, ni = density_initialization(n1, n2, n3, 15,15,15);

    si = tf.reduce_sum(ne[1:imax+1, 1:jmax+1, 1:kmax+1])
    BC_densities(ne, ni)
    start = Time.time()
    with tf.Session() as sess:
        for time in range(1, tmax):
            V, g = plasma_sim_solve_poisson_equation_on_grid(V, g, ne, ni, pv, sess)
            #printf("V[%d, %d, %d] = %f\n", 5, 6, 7, V[5, 6, 7]);
            electric_field_elements(V, ne, ni, Ez, Ex, Ey, difxne, difxni, difyne, difyni, difzne, difzni)
            #average_x(ne, ni, Ex, Exy, difxne, difxni, difxyne, difxyni)
            #flux_y(ne, ni, fey, fiy, Ey, Ez, Exy, difyne, difyni, difxyne, difxyni)
            #average_y(ne, ni, Ey, Exy, difyne, difyni, difxyne, difxyni)
            #flux_x(ne, ni, fex, fix, Ex, Exy, difxne, difxni, difxyne, difxyni)
            #flux_z(ne, ni, Ez, fez, fiz, difzne, difzni)

            #ne[1:imax+1, 1:jmax+1, 1:kmax] = \
            #    ne[1:imax+1, 1:jmax+1, 1:kmax] -\
            #    dt * (\
            #      fex[1:imax+1, 1:jmax+1, 1:kmax  ] \
            #    - fex[0:imax  , 1:jmax+1, 1:kmax  ] \
            #    + fey[1:imax+1, 1:jmax+1, 1:kmax  ] \
            #    - fey[1:imax+1, 0:jmax  , 1:kmax  ] \
            #    + fez[1:imax+1, 1:jmax+1, 1:kmax  ] \
            #    - fez[1:imax+1, 1:jmax+1, 0:kmax-1])/h
    #       #         
            #ni[1:imax+1, 1:jmax+1, 1:kmax] = \
            #    ni[1:imax+1, 1:jmax+1, 1:kmax] -\
            #    dt * (
            #      fix[1:imax+1, 1:jmax+1, 1:kmax] \
            #    - fix[0:imax, 1:jmax+1, 1:kmax] \
            #    + fiy[1:imax+1, 1:jmax+1, 1:kmax] \
            #    - fiy[1:imax+1, 0:jmax, 1:kmax] \
            #    + fiz[1:imax+1, 1:jmax+1, 1:kmax] \
            #    - fiz[1:imax+1, 1:jmax+1, 0:kmax-1])/h

            #ne[1:imax+1, 1:jmax+1, 0] = -dt * fez[1:imax+1, 1:jmax+1, 0] / h
            #ni[1:imax+1, 1:jmax+1, 0] = -dt * fiz[1:imax+1, 1:jmax+1, 0] / h

            #BC_densities(ne, ni)

            #sf = np.sum(ne[1:imax+1, 1:jmax+1, 1:kmax+1])
            #
            #alpha = (si -sf) / sf;
            #ne[1:imax+1, 1:jmax+1, 1:kmax-1] = \
            #        ne[1:imax+1, 1:jmax+1, 1:kmax-1] +\
            #        alpha * ne[1:imax+1, 1:jmax+1, 1:kmax-1]

            #printf("%d \n", time);
    time_taken2 = Time.time() - start
    print("Time taken: %f"%(time_taken2))
    print(V[5, 6, 7])
    with tf.Session() as sess:
        print("running...")
        start = Time.time()
        sess.run(V)
        time_taken2 = Time.time() - start
        print("Time taken: ", time_taken2)
    return V, g, pv

if __name__=="__main__":
    main()
