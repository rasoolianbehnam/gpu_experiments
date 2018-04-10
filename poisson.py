import numpy as np
def printf(text, *args):
    print(text%args)
imax=32
jmax=32
kmax=64
tmax=50
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

g   = np.zeros((imax+3, jmax+3, kmax+3))
R   = np.zeros((imax+3, jmax+3, kmax+3))
ne  = np.zeros((imax+3, jmax+3, kmax+3))
ni  = np.zeros((imax+3, jmax+3, kmax+3))
V   = np.zeros((imax+3, jmax+3, kmax+3))
Ex  = np.zeros((imax+3, jmax+3, kmax+3))
Ey  = np.zeros((imax+3, jmax+3, kmax+3))
Ez  = np.zeros((imax+3, jmax+3, kmax+3))
fex = np.zeros((imax+3, jmax+3, kmax+3))
fey = np.zeros((imax+3, jmax+3, kmax+3))
fez = np.zeros((imax+3, jmax+3, kmax+3))
fix = np.zeros((imax+3, jmax+3, kmax+3))
fiy = np.zeros((imax+3, jmax+3, kmax+3))
fiz = np.zeros((imax+3, jmax+3, kmax+3))
difxne = np.zeros((imax+3, jmax+3, kmax+3))
difxni = np.zeros((imax+3, jmax+3, kmax+3))
difyne = np.zeros((imax+3, jmax+3, kmax+3))
difyni = np.zeros((imax+3, jmax+3, kmax+3))
difzne = np.zeros((imax+3, jmax+3, kmax+3))
difzni = np.zeros((imax+3, jmax+3, kmax+3))
Exy = np.zeros((imax+3, jmax+3, kmax+3))
difxyne = np.zeros((imax+3, jmax+3, kmax+3))
difxyni = np.zeros((imax+3, jmax+3, kmax+3))

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

def density_initialization(x_position, y_position, z_position):
    for i in range(1, imax+1):
        for j in range(1, jmax+1):
            for k in range(1, kmax-1):
                ne[i][j][k]=\
                        1.0E14+1.0E14*np.exp(-((i-x_position)**2\
                        +(j-y_position)**2+\
                        (k-z_position)**2)/100.0);
                ni[i][j][k]=1.0E14+1.0E14*np.exp(-(((i-x_position)**2)\
                        +((j-y_position)**2)+\
                        ((k-z_position)**2))/100.0)

#def plasma_sim_solve_poisson_equation_on_grid():
#   # Here we calculate the right hand side of the Poisson equation
#   g[1:imax+1, 1:jmax+1, 1:kmax-1] = -(ne[1:imax+1, 1:jmax+1, 1:kmax-1]*qe +\
#           ni[1:imax+1, 1:jmax+1, 1:kmax-1] * qi) / eps0

def BC_densities():
     # BC on densities
    ne[imax+1, 0:jmax+1, 0:kmax] = ne[1, 0:jmax+1, 0:kmax]
    ni[imax+1, 0:jmax+1, 0:kmax] = ni[1, 0:jmax+1, 0:kmax]

    ne[0:jmax+1, jmax+1, 0:kmax] = ne[0:jmax+1, 1, 0:kmax]
    ni[0:jmax+1, jmax+1, 0:kmax] = ni[0:jmax+1, 1, 0:kmax]

    ne[0, 0:jmax+1, 0:kmax] = ne[imax, 0:jmax+1, 0:kmax]
    ni[0, 0:jmax+1, 0:kmax] = ni[imax, 0:jmax+1, 0:kmax]

    ne[0:jmax+1, 0, 0:kmax] = ne[0:jmax+1, jmax, 0:kmax]
    ni[0:jmax+1, 0, 0:kmax] = ni[0:jmax+1, jmax, 0:kmax]
#    for k in range(0, kmax):
#        for j in range(0, jmax+1):
#            ne[imax+1][j][k]=ne[1][j][k];
#            ni[imax+1][j][k]=ni[1][j][k];
#
#            ne[j][jmax+1][k]=ne[j][1][k];
#            ni[j][jmax+1][k]=ni[j][1][k];
#
#
#            ne[0][j][k]=ne[imax][j][k];
#            ni[0][j][k]=ni[imax][j][k];
#
#            ne[j][0][k]=ne[j][jmax][k];
#            ni[j][0][k]=ni[j][jmax][k];
#    for i in range(1, imax+1):
#        for j in range(j, jmax+1):
#            ne[i][j][kmax-1]=0.0;
#            ne[i][j][0]=0.0;
#
#            ni[i][j][kmax-1]=0.0;
#            ni[i][j][0]=0.0;


def poisson_brute(V, g, num_iterations, imax, jmax, kmax, h=h, w=w):
    for kk in range(num_iterations):
        temp = V * 1.
        for i in range(1, imax-1):
            for j in range(1, jmax+1):
                for k in range(1, kmax+1):
                    r = temp[i+1, j, k] / 6. + temp[i-1, j, k] / 6. + temp[i, j+1, k] / 6. + temp[i, j-1, k] / 6. + temp[i ,j, k+1] / 6. + temp[i, j, k-1] / 6. - temp[i, j, k] - (h**2) * g[i, j, k] / 6.
                    r = w * r
                    V[i, j, k] += r
    return V

def plasma_sim_solve_poisson_equation_on_grid():
   # Here we calculate the right hand side of the Poisson equation
   g[1:imax+1, 1:jmax+1, 1:kmax-1]=-(ne[1:imax+1, 1:jmax+1, 1:kmax-1]*qe\
            +ni[1:imax+1, 1:jmax+1, 1:kmax-1]*qi)/eps0;
   #for j in range(1, jmax+1):
   #    for i in range(1, imax+1):
   #        for k in range(1, kmax-1):
   #             g[i][j][k]=-(ne[i][j][k]*qe+ni[i][j][k]*qi)/eps0;

   for kk in range(0, 40):
       for k in range(1, kmax-1):
           for i in range(1, imax+1):
               for j in range(1, jmax+1):
                   R[i][j][k]= (V[i+1][j][k]+V[i-1][j][k]+V[i][j+1][k]+V[i][j-1][k]+V[i][j][k+1]+V[i][j][k-1])/6.0-V[i][j][k]-(h*h)*g[i][j][k]/6.0;
                   V[i][j][k]=V[i][j][k]+w*R[i][j][k];

   for k in range(0, kmax):
       for j in range(jmax+1):
           V[imax+1][j][k]=V[1][j][k];
           V[j][jmax+1][k]=V[j][1][k];
           V[0][j][k]=V[imax][j][k];
           V[j][0][k]=V[j][jmax][k];


def electric_field_elements():
#    for k in range(1, kmax-1):
#        for i in range(1, imax+1):
#            for j in range(1, jmax+1):
#                Ex[i][j][k]= (V[i][j][k]-V[i+1][j][k])/h;
#                Ey[i][j][k]= (V[i][j][k]-V[i][j+1][k])/h;
#
#                difxne[i][j][k]=(ne[i+1][j][k]-ne[i][j][k])/h;
#                difxni[i][j][k]=(ni[i+1][j][k]-ni[i][j][k])/h;
#                difyne[i][j][k]=(ne[i][j+1][k]-ne[i][j][k])/h;
#                difyni[i][j][k]=(ni[i][j+1][k]-ni[i][j][k])/h;

    for k in range(0, kmax-1):
        for i in range(1, imax+1):
            for j in range(1, jmax+1):
                Ez[i][j][k]= (V[i][j][k]-V[i][j][k+1])/h;

                difzne[i][j][k]=(ne[i][j][k+1]-ne[i][j][k])/h;
                difzni[i][j][k]=(ni[i][j][k+1]-ni[i][j][k])/h;

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

def average_x():
    pass
    Exy[2:imax+1, 1:jmax, 1:kmax-1] = .25*(Ex[2:imax+1, 1:jmax, 1:kmax-1] +\
            Ex[2:imax+1, 2:jmax+1, 1:kmax-1] + + Ex[1:imax, 1:jmax, 1:kmax-1] + \
            Ex[1:imax, 2:jmax+1, 1:kmax-1])
    difxyne[2:imax+1, 1:jmax, 1:kmax-1] = .25*(difxne[2:imax+1, 1:jmax, 1:kmax-1] +\
            difxne[2:imax+1, 2:jmax+1, 1:kmax-1] + difxne[1:imax, 1:jmax, 1:kmax-1]) +\
            difxne[1:imax, 2:jmax+1, 1:kmax-1]
    difxyni[2:imax+1, 1:jmax, 1:kmax-1] = .25*(difxni[2:imax+1, 1:jmax, 1:kmax-1] +\
            difxni[2:imax+1, 2:jmax+1, 1:kmax-1] + difxni[1:imax, 1:jmax, 1:kmax-1]) +\
            difxni[1:imax, 2:jmax+1, 1:kmax-1]
#    for k in range(1, kmax-1):
#        for i in range(2, imax+1):
#            for j in range(1, jmax):
#                Exy[i][j][k]= 0.25*(Ex[i][j][k]+Ex[i][j+1][k]+Ex[i-1][j][k]+Ex[i-1][j+1][k]) ; 
#                difxyne[i][j][k]=0.25*(difxne[i][j][k]+difxne[i][j+1][k]+difxne[i-1][j][k]+difxne[i-1][j+1][k]);
#                difxyni[i][j][k]=0.25*(difxni[i][j][k]+difxni[i][j+1][k]+difxni[i-1][j][k]+difxni[i-1][j+1][k]);
#

    for k in range(1, kmax-1):
        for j in range(1, jmax):
            Exy[1][j][k]= 0.25*(Ex[1][j][k]+Ex[1][j+1][k]+Ex[imax][j][k]+Ex[imax][j+1][k]) ;

            difxyne[1][j][k]=0.25*(difxne[1][j][k]+difxne[1][j+1][k]+difxne[imax][j][k]+difxne[imax][j+1][k]);
            difxyni[1][j][k]=0.25*(difxni[1][j][k]+difxni[1][j+1][k]+difxni[imax][j][k]+difxni[imax][j+1][k]);

    for k in range(1, kmax-1):
        for i in range(1, imax):
            Exy[i][jmax][k]= \
                    0.25*(Ex[i][jmax][k]+Ex[i][1][k]+Ex[i-1][jmax][k]+Ex[i-1][1][k]) ;

            difxyne[i][jmax][k]=\
                    0.25*(difxne[i][jmax][k]+difxne[i][1][k]+difxne[i-1][jmax][k]+difxne[i-1][1][k]);
            difxyni[i][jmax][k]=\
                    0.25*(difxni[i][jmax][k]+difxni[i][1][k]+difxni[i-1][jmax][k]+difxni[i-1][1][k]);

    for k in range(1, kmax-1):
        Exy[imax][jmax][k]=(Ex[imax][jmax][k]+Ex[imax-1][jmax][k]+Ex[imax-1][1][k])/3.0;
        difxyne[imax][jmax][k]=(difxne[imax][jmax][k]+difxne[imax-1][jmax][k]+difxne[imax-1][1][k])/3.0;
        difxyni[imax][jmax][k]=(difxni[imax][jmax][k]+difxni[imax-1][jmax][k]+difxni[imax-1][1][k])/3.0;

        Exy[1][jmax][k]=(Ex[1][jmax][k]+Ex[imax][jmax][k]+Ex[1][1][k])/3.0;
        difxyne[1][jmax][k]=(difxne[1][jmax][k]+difxne[imax][jmax][k]+difxne[1][1][k])/3.0;
        difxyni[1][jmax][k]=(difxni[1][jmax][k]+difxni[imax][jmax][k]+difxni[1][1][k])/3.0;



def flux_x():
    pass
    for k in range(1, kmax-1):
        for j in range(1, jmax+1):
            for i in range(1, imax+1):
                fex[i][j][k]=(-0.5*(ne[i][j][k]+ne[i+1][j][k])*mue*Ex[i][j][k]-dife*difxne[i][j][k]
                +wce*dife*difxyne[i][j][k]/nue+wce*q*0.5*(ne[i][j][k]+ne[i+1][j][k])/(me*nue*nue)*Exy[i][j][k])/denominator_e;

                fix[i][j][k]=(0.5*(ni[i][j][k]+ni[i+1][j][k])*mui*Ex[i][j][k]-difi*difxni[i][j][k]
                -wci*difi*difxyni[i][j][k]/nui+wci*q*0.5*(ni[i][j][k]+ni[i+1][j][k])*Exy[i][j][k]/(mi*nui*nui))/denominator_i;

    for k in range(1, kmax-1):
        for j in range(1, jmax+1):
            fex[0][j][k] = fex[imax][j][k];
            fix[0][j][k] = fix[imax][j][k];


def average_y():
    pass
    for k in range(1, kmax-1):
        for i in range(1, imax):
            for j in range(2, jmax+1):
                Exy[i][j][k]= 0.25*(Ey[i][j][k]+Ey[i][j-1][k]+Ey[i+1][j][k]+Ey[i+1][j-1][k]); 
                difxyne[i][j][k]= 0.25*(difyne[i][j][k]+difyne[i][j-1][k]+difyne[i+1][j][k]+difyne[i+1][j-1][k]);
                difxyni[i][j][k]= 0.25*(difyni[i][j][k]+difyni[i][j-1][k]+difyni[i+1][j][k]+difyni[i+1][j-1][k]);

        for k in range(1, kmax-1):
            for i in range(1, imax):
                Exy[i][1][k]= 0.25*(Ey[i][1][k]+Ey[i][jmax][k]+Ey[i+1][1][k]+Ey[i+1][jmax][k]);
                difxyne[i][1][k]= 0.25*(difyne[i][1][k]+difyne[i][jmax][k]+difyne[i+1][1][k]+difyne[i+1][jmax][k]);
                difxyni[i][1][k]= 0.25*(difyni[i][1][k]+difyni[i][jmax][k]+difyni[i+1][1][k]+difyni[i+1][jmax][k]);


        for k in range(1, kmax-1):
            for j in range(2, jmax+1):
                Exy[imax][j][k]= 0.25*(Ey[imax][j][k]+Ey[imax][j-1][k]+Ey[1][j][k]+Ey[1][j-1][k]); 
                difxyne[imax][j][k]= 0.25*(difyne[imax][j][k]+difyne[imax][j-1][k]+difyne[1][j][k]+difyne[1][j-1][k]);
                difxyni[imax][j][k]= 0.25*(difyni[imax][j][k]+difyni[imax][j-1][k]+difyni[1][j][k]+difyni[1][j-1][k]);


    for k in range(1, kmax-1):
        Exy[imax][1][k]=(Ey[imax][1][k]+Ey[1][1][k]+Ey[imax][jmax][k])/3.0;
        difxyne[imax][1][k]=(difyne[imax][1][k]+difyne[1][1][k]+difyne[imax][jmax][k])/3.0;
        difxyni[imax][1][k]=(difyni[imax][1][k]+difyni[1][1][k]+difyni[imax][jmax][k])/3.0; 
        Exy[imax][jmax][k]=(Ey[imax][jmax-1][k]+Ey[imax][jmax][k]+Ey[1][jmax-1][k])/3.0;
        difxyne[imax][jmax][k]=(difyne[imax][jmax-1][k]+difyne[imax][jmax][k]+difyne[1][jmax-1][k])/3.0;
        difxyni[imax][jmax][k]=(difyni[imax][jmax-1][k]+difyni[imax][jmax][k]+difyni[1][jmax-1][k])/3.0; 

def flux_y():
    pass
    for k in range(1, kmax-1):
        for i in range(1, imax+1):
            for j in range(1, jmax+1):
                fey[i][j][k]= (-0.5*(ne[i][j+1][k]+ne[i][j][k])*mue*Ey[i][j][k]-dife*difyne[i][j][k]
                -wce*q*0.5*(ne[i][j+1][k]+ne[i][j][k])*Exy[i][j][k]/(me*nue*nue)-wce*dife*difxyne[i][j][k]/nue)/denominator_e;

                fiy[i][j][k]= (0.5*(ni[i][j+1][k]+ni[i][j][k])*mui*Ey[i][j][k]-difi*difyni[i][j][k]
                -wci*q*0.5*(ni[i][j+1][k]+ni[i][j][k])*Exy[i][j][k]/(mi*nui*nui)+wci*difi*difxyni[i][j][k]/nui)/denominator_i;


    for k in range(1, kmax-1):
        for i in range(1, imax+1):
            fey[i][0][k] = fey[i][jmax][k];
            fiy[i][0][k] = fiy[i][jmax][k];

def flux_z():
    pass
    for k in range(1, kmax-1):
        for j in range(1, jmax+1):
            for i in range(1, imax+1):
                fez[i][j][k]=-0.5*(ne[i][j][k]+ne[i][j][k+1])*mue*Ez[i][j][k]-dife*difzne[i][j][k];

                fiz[i][j][k]=0.5*(ni[i][j][k]+ni[i][j][k+1])*mui*Ez[i][j][k]-difi*difzni[i][j][k];

    for i in range(1, imax+1):
        for j in range(1, jmax+1):
            if (fez[i][j][1]>0.0):
                fez[i][j][1]=0.0;
            if (fiz[i][j][1]>0.0):
                fiz[i][j][1]=0.0;
            if (fez[i][j][kmax-1]<0.0):
                fez[i][j][kmax-1]=0.0;
            if (fiz[i][j][kmax-1]<0.0):
                fiz[i][j][kmax-1]=0.0;
 

def main():
    density_initialization(15,15,15);

    si = np.sum(ne[1:imax+1, 1:jmax+1, 1:kmax+1])
    printf("si before loop: %f", si);
    printf("ne[%d, %d, %d] = %e\n", 15, 12, 12, ne[15, 12, 12]);
    printf("ni[%d, %d, %d] = %e\n", 15, 12, 12, ni[15, 12, 12]);
    BC_densities()
    
    for time in range(1, tmax):
        plasma_sim_solve_poisson_equation_on_grid()
        printf("V[%d, %d, %d] = %f\n", 10, 12, 24, V[1][12][24]);
        electric_field_elements()
        average_x()
        flux_y()
        average_y()
        flux_x()
        flux_z()
        for k in range(1, kmax):
            for j in range(1, jmax+1):
                for i in range(1, imax+1):
                    ne[i][j][k]=ne[i][j][k]-dt*(fex[i][j][k]-fex[i-1][j][k]+fey[i][j][k]-fey[i][j-1][k]+fez[i][j][k]-fez[i][j][k-1])/h ;
                    ni[i][j][k]=ni[i][j][k]-dt*(fix[i][j][k]-fix[i-1][j][k]+fiy[i][j][k]-fiy[i][j-1][k]+fiz[i][j][k]-fiz[i][j][k-1])/h ;

        for j in range(1, jmax+1):
            for i in range(1, imax+1):
                ne[i][j][0] = -dt*fez[i][j][0]/h ;
                ni[i][j][0] = -dt*fiz[i][j][0]/h ;

#        ne[1:imax+1, 1:jmax+1, 1:kmax] += -dt * (fex[1:imax+1, 1:jmax+1, 1:kmax] -\
#                fex[0:imax, 1:jmax+1, 1:kmax] + fey[1:imax+1, 1:jmax+1, 1:kmax] - \
#                fey[1:imax+1, 0:jmax, 1:kmax] + fez[1:imax+1, 1:jmax+1, 1:kmax] - \
#                fez[1:imax+1, 1:jmax+1, 0:kmax-1])
#                
#        ni[1:imax+1, 1:jmax+1, 1:kmax] += -dt * (fix[1:imax+1, 1:jmax+1, 1:kmax] -\
#                fix[0:imax, 1:jmax+1, 1:kmax] + fiy[1:imax+1, 1:jmax+1, 1:kmax] - \
#                fiy[1:imax+1, 0:jmax, 1:kmax] + fiz[1:imax+1, 1:jmax+1, 1:kmax] - \
#                fiz[1:imax+1, 1:jmax+1, 0:kmax-1])
#
#        ne[1:imax+1, 1:jmax+1, 0] = -dt * fez[1:imax+1, 1:jmax+1, 0] / h
#        ni[1:imax+1, 1:jmax+1, 0] = -dt * fiz[1:imax+1, 1:jmax+1, 0] / h

        BC_densities()

        #sf = np.sum(ne[1:imax+1, 1:jmax+1, 1:kmax+1])
        sf=0.0;
        for k in range(1, kmax+1):
            for j in range(1, jmax+1):
                for i in range(1, imax+1):
                     sf=sf+ne[i][j][k] ;


        alpha = (si -sf) / sf;
        #ne[1:imax+1, 1:jmax+1, 1:kmax-1] += alpha * ne[1:imax+1, 1:jmax+1, 1:kmax-1]
        for k in range(1, kmax-1):
            for j in range(1, jmax+1):
                for i in range(1, imax+1):
                    ne[i][j][k]=ne[i][j][k]+alpha*ne[i][j][k] ;
                    ni[i][j][k]=ni[i][j][k]+alpha*ne[i][j][k] ;

        printf("%d \n", time);


if __name__=="__main__":
    main()
