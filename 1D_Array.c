#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define my_index 3









void mardas(int imax, int jmax, int kmax, int tmax, double *ne, double *ni, double *difxne, double *difyne, double *difxni,
             double *difyni, double *difxyne, double *difxyni, double *Exy, double *fexy, double *fixy, double *g, double *R,
              double *Ex, double *Ey, double *fex, double *fey, double *fix, double *fiy, double *V, double *L, double *difzne,
               double *difzni, double *Ez, double *fez, double *fiz, double qi, double qe, double kr, double ki, double si,
                double sf, double alpha, double q, double pie, double Ta , double w , double eps0 , double Te, double Ti,
                 double B, double Kb, double me, double mi, double nue, double nui, double denominator_e, double denominator_i,
                  double nn, double dt, double h, double wce, double wci, double mue, double mui, double dife, double difi) {
    int  n1=imax+3, n2 = jmax+3, n3 = kmax+3,i,j,k,fuckingCount,myTime,kk,I,N,s1;

    N=n1*n2*n3;

for ( myTime=1; myTime<tmax; myTime++){  // This for loop takes care of myTime evolution
     fuckingCount = 0;


    //    printf("time %d_%d V: %f\n", myTime, fuckingCount, V[3 + n3 * (3 + n2 * (3))]);
    //    printf("time %d_%d g: %f\n", myTime, fuckingCount, g[3 + n3 * (3 + n2 * (3))]);
    //    printf("time %d_%d ne: %f\n", myTime, fuckingCount, ne[3 + n3 * (3 + n2 * (3))]);
// -----------------------------------------------------------------------------------------------


        printf("%d \n", myTime);

      // Solving Poisson's eq. to get the voltage everywhere
  // Here we calculate the right hand side of the Poisson equation



    for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        g[k + n3 * (j + n2 * (i))]=-(ne[k + n3 * (j + n2 * (i))]*qe+ni[k + n3 * (j + n2 * (i))]*qi)/eps0;
    }



/////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//// has problem at the edges
/////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// solving Poisson eq. using  successive over-relaxation method


      for ( kk=0; kk<40; kk++) {
        for ( I = 0; I < N; I ++) {
             k = I % n3;
             s1 = (I - k) / n3;
             j = s1 % n2;
             i = (s1 - j) / n2;
            if (i * j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
            R[k + n3 * (j + n2 * (i))]=
                (V[k + n3 * (j + n2 * (i+1))]+
                     V[k + n3 * (j + n2 * (i-1))]+
                     V[k + n3 * (j+1 + n2 * (i))]+
                     V[k + n3 * (j-1 + n2 * (i))]+
                     V[k+1 + n3 * (j + n2 * (i))]+
                     V[k-1 + n3 * (j + n2 * (i))]
                 ) / 6.0 - V[k + n3 * (j + n2 * (i))]- (h*h)*g[k + n3 * (j + n2 * (i))]/6.0;
            V[k + n3 * (j + n2 * (i))] += w*R[k + n3 * (j + n2 * (i))];
        }
    }



        ///printf("time %d_%d V: %f\n", myTime, fuckingCount, V[3 + n3 * (3 + n2 * (3))]);
        ///printf("time %d_%d g: %f\n", myTime, fuckingCount, g[3 + n3 * (3 + n2 * (3))]);
        ///printf("time %d_%d ne: %f\n", myTime, fuckingCount, ne[3 + n3 * (3 + n2 * (3))]);

        ///printf("qi = %f\n", qi);
        ///printf("qe = %f\n", qe);
        ///printf("kr = %f\n", kr);
        ///printf("ki = %f\n", ki);
        ///printf("si = %f\n", si);
        ///printf("sf = %f\n", sf);
        ///printf("alpha = %f\n", alpha);
        ///printf("q = %f\n", q);
        ///printf("pie = %f\n", pie);
        ///printf("Ta = %f\n", Ta);
        ///printf("w = %f\n", w);
        ///printf("eps0 = %f\n", eps0);
        ///printf("Te = %f\n", Te);
        ///printf("Ti = %f\n", Ti);
        ///printf("B = %f\n", B);
        ///printf("Kb = %f\n", Kb);
        ///printf("me = %f\n", me);
        ///printf("mi = %f\n", mi);
        ///printf("nue = %f\n", nue);
        ///printf("nui = %f\n", nui);
        ///printf("denominator_e = %f\n", denominator_e);
        ///printf("denominator_i = %f\n", denominator_i);
        ///printf("nn = %f\n", nn);
        ///printf("dt = %f\n", dt);
        ///printf("h = %f\n", h);
        ///printf("wce = %f\n", wce);
        ///printf("wci = %f\n", wci);
        ///printf("mue = %f\n", mue);
        ///printf("mui = %f\n", mui);
        ///printf("dife = %f\n", dife);
        ///printf("difi = %f\n", difi);
// -----------------------------------------------------------------------------------------------
      // Now Calculating electric field and density gradient which are calculated through function electric_field_elements

  for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i >= imax-1 || j >= jmax || k >= kmax) continue;
        Ex[k + n3 * (j + n2 * (i))]= (V[k + n3 * (j + n2 * (i))]-V[k + n3 * (j + n2 * (i+1))])/h;
        difxne[k + n3 * (j + n2 * (i))]=(ne[k + n3 * (j + n2 * (i+1))]-ne[k + n3 * (j + n2 * (i))])/h;
        difxni[k + n3 * (j + n2 * (i))]=(ni[k + n3 * (j + n2 * (i+1))]-ni[k + n3 * (j + n2 * (i))])/h;
        }


    for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i >= imax || j >= jmax-1 || k >= kmax) continue;
        Ey[k + n3 * (j + n2 * (i))]= (V[k + n3 * (j + n2 * (i))]-V[k + n3 * (j+1 + n2 * (i))])/h;
        difyne[k + n3 * (j + n2 * (i))]=(ne[k + n3 * (j+1 + n2 * (i))]-ne[k + n3 * (j + n2 * (i))])/h;
        difyni[k + n3 * (j + n2 * (i))]=(ni[k + n3 * (j+1 + n2 * (i))]-ni[k + n3 * (j + n2 * (i))])/h;
        }


    for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i >= imax || j >= jmax || k >= kmax-1) continue;
       Ez[k + n3 * (j + n2 * (i))]= (V[k + n3 * (j + n2 * (i))]-V[k+1 + n3 * (j + n2 * (i))])/h;
       difzne[k + n3 * (j + n2 * (i))]=(ne[k+1 + n3 * (j + n2 * (i))]-ne[k + n3 * (j + n2 * (i))])/h;
       difzni[k + n3 * (j + n2 * (i))]=(ni[k+1 + n3 * (j + n2 * (i))]-ni[k + n3 * (j + n2 * (i))])/h;
     }

// -----------------------------------------------------------------------------------------------
       /* Since I am using mid points for Calculating electric field and density gradient,
        to calculate Ex at any point that I don't have it directly, the average over
        the neighboring points is used instead. these average variables are, exy, fexy, fixy, ...*/
        // Calculating the average values of Ex and gradiant_x
   for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;

        Exy[k + n3 * (j + n2 * (i))]= 0.0 ;
        difxyne[k + n3 * (j + n2 * (i))]=0.0;
        difxyni[k + n3 * (j + n2 * (i))]=0.0;
    }

    for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * k == 0 ||  i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        Exy[k + n3 * (j + n2 * (i))]= 0.25*(Ex[k + n3 * (j + n2 * (i))]+Ex[k + n3 * (j+1 + n2 * (i))]+Ex[k + n3 * (j + n2 * (i-1))]+Ex[k + n3 * (j+1 + n2 * (i-1))]) ;
        difxyne[k + n3 * (j + n2 * (i))]=0.25*(difxne[k + n3 * (j + n2 * (i))]+difxne[k + n3 * (j+1 + n2 * (i))]+difxne[k + n3 * (j + n2 * (i-1))]+difxne[k + n3 * (j+1 + n2 * (i-1))]);
        difxyni[k + n3 * (j + n2 * (i))]=0.25*(difxni[k + n3 * (j + n2 * (i))]+difxni[k + n3 * (j+1 + n2 * (i))]+difxni[k + n3 * (j + n2 * (i-1))]+difxni[k + n3 * (j+1 + n2 * (i-1))]);
    }


// -----------------------------------------------------------------------------------------------
        // Here we calculate the fluxes in y direction

       for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * k == 0 ||  i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        fey[k + n3 * (j + n2 * (i))]= (-0.5*(ne[k + n3 * (j+1 + n2 * (i))]+ne[k + n3 * (j + n2 * (i))])*mue*Ey[k + n3 * (j + n2 * (i))]-dife*difyne[k + n3 * (j + n2 * (i))]
        -wce*q*0.5*(ne[k + n3 * (j+1 + n2 * (i))]+ne[k + n3 * (j + n2 * (i))])*Exy[k + n3 * (j + n2 * (i))]/(me*nue*nue)-wce*dife*difxyne[k + n3 * (j + n2 * (i))]/nue)/denominator_e;
        fiy[k + n3 * (j + n2 * (i))]= (0.5*(ni[k + n3 * (j+1 + n2 * (i))]+ni[k + n3 * (j + n2 * (i))])*mui*Ey[k + n3 * (j + n2 * (i))]-difi*difyni[k + n3 * (j + n2 * (i))]
        -wci*q*0.5*(ni[k + n3 * (j+1 + n2 * (i))]+ni[k + n3 * (j + n2 * (i))])*Exy[k + n3 * (j + n2 * (i))]/(mi*nui*nui)+wci*difi*difxyni[k + n3 * (j + n2 * (i))]/nui)/denominator_i;
    }


    for ( I =0; I < n1 * n3; I ++) {
         k = I % n3;
         i = (I - k) / n3;

        if (i * k == 0 ||  i >= imax-1 || k >= kmax-1) continue;

        if (fey[k + n3 * (0 + n2 * (i))] > 0.0){
                fey[k + n3 * (0 + n2 * (i))] = 0.0;
                }

        if (fiy[k + n3 * (0 + n2 * (i))] > 0.0){
                fiy[k + n3 * (0 + n2 * (i))] = 0.0;
                }

        if (fey[k + n3 * (jmax-2 + n2 * (i))] < 0.0){
                fey[k + n3 * (jmax-2 + n2 * (i))] = 0.0;
                }

        if (fiy[k + n3 * (jmax-2 + n2 * (i))] < 0.0){
                fiy[k + n3 * (jmax-2 + n2 * (i))] = 0.0;
                }

    }

// -----------------------------------------------------------------------------------------------
        // Calculating the average Exy and difxy to be used in x direction fluxes
       // Calculating the average values of Ey and gradiant_y


      for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;

        Exy[k + n3 * (j + n2 * (i))]= 0.0 ;
        difxyne[k + n3 * (j + n2 * (i))]=0.0;
        difxyni[k + n3 * (j + n2 * (i))]=0.0;
    }


      for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        Exy[k + n3 * (j + n2 * (i))]= 0.25*(Ey[k + n3 * (j + n2 * (i))]+Ey[k + n3 * (j-1 + n2 * (i))]+Ey[k + n3 * (j + n2 * (i+1))]+Ey[k + n3 * (j-1 + n2 * (i+1))]);
        difxyne[k + n3 * (j + n2 * (i))]= 0.25*(difyne[k + n3 * (j + n2 * (i))]+difyne[k + n3 * (j-1 + n2 * (i))]+difyne[k + n3 * (j + n2 * (i+1))]+difyne[k + n3 * (j-1 + n2 * (i+1))]);
        difxyni[k + n3 * (j + n2 * (i))]= 0.25*(difyni[k + n3 * (j + n2 * (i))]+difyni[k + n3 * (j-1 + n2 * (i))]+difyni[k + n3 * (j + n2 * (i+1))]+difyni[k + n3 * (j-1 + n2 * (i+1))]);

    }

// -----------------------------------------------------------------------------------------------
        // Now ready to calculate the fluxes in x direction

    for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        fex[k + n3 * (j + n2 * (i))]=(-0.5*(ne[k + n3 * (j + n2 * (i))]+ne[k + n3 * (j + n2 * (i+1))])*mue*Ex[k + n3 * (j + n2 * (i))]-dife*difxne[k + n3 * (j + n2 * (i))]
        +wce*dife*difxyne[k + n3 * (j + n2 * (i))]/nue+wce*q*0.5*(ne[k + n3 * (j + n2 * (i))]+ne[k + n3 * (j + n2 * (i+1))])/(me*nue*nue)*Exy[k + n3 * (j + n2 * (i))])/denominator_e;
        fix[k + n3 * (j + n2 * (i))]=(0.5*(ni[k + n3 * (j + n2 * (i))]+ni[k + n3 * (j + n2 * (i+1))])*mui*Ex[k + n3 * (j + n2 * (i))]-difi*difxni[k + n3 * (j + n2 * (i))]
        -wci*difi*difxyni[k + n3 * (j + n2 * (i))]/nui+wci*q*0.5*(ni[k + n3 * (j + n2 * (i))]+ni[k + n3 * (j + n2 * (i+1))])*Exy[k + n3 * (j + n2 * (i))]/(mi*nui*nui))/denominator_i;
    }


    for ( I = 0; I < n2 * n3; I ++) {
         k = I % n3;
         j = (I - k) / n3;

        if (j * k == 0 ||  j >= jmax-1 || k >= kmax-1) continue;

        if (fex[k + n3 * (j + n2 * (0))] > 0.0){
                fex[k + n3 * (j + n2 * (0))] = 0.0;
                }

        if (fix[k + n3 * (j + n2 * (0))] > 0.0){
                fix[k + n3 * (j + n2 * (0))] = 0.0;
                }

        if (fex[k + n3 * (j + n2 * (imax-2))] < 0.0){
                fex[k + n3 * (j + n2 * (imax-2))] = 0.0;
                }

        if (fix[k + n3 * (j + n2 * (imax-2))] < 0.0){
                fix[k + n3 * (j + n2 * (imax-2))] = 0.0;
                }

    }

// -----------------------------------------------------------------------------------------------

        // Now we calculate the fluxes in z direction
      for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        fez[k + n3 * (j + n2 * (i))]=-0.5*(ne[k + n3 * (j + n2 * (i))]+ne[k+1 + n3 * (j + n2 * (i))])*mue*Ez[k + n3 * (j + n2 * (i))]-dife*difzne[k + n3 * (j + n2 * (i))];
        fiz[k + n3 * (j + n2 * (i))]=0.5*(ni[k + n3 * (j + n2 * (i))]+ni[k+1 + n3 * (j + n2 * (i))])*mui*Ez[k + n3 * (j + n2 * (i))]-difi*difzni[k + n3 * (j + n2 * (i))];

    }
       // BC on fluxes

    for ( I = 0; I < n1 * n2; I ++) {
         j = I % n2;
         i = (I - j) / n2;
        if (i * j == 0 || i >= imax-1 || j >= jmax-1) continue;
        if (fez[0 + n3 * (j + n2 * (i))]>0.0){
            fez[0 + n3 * (j + n2 * (i))]=0.0;
        }
        if (fiz[0 + n3 * (j + n2 * (i))]>0.0){
            fiz[0 + n3 * (j + n2 * (i))]=0.0;
        }
        if (fez[kmax-2 + n3 * (j + n2 * (i))]<0.0){
            fez[kmax-2 + n3 * (j + n2 * (i))]=0.0;
        }
        if (fiz[kmax-2 + n3 * (j + n2 * (i))]<0.0){
            fiz[kmax-2 + n3 * (j + n2 * (i))]=0.0;
        }
    }
// -----------------------------------------------------------------------------------------------


       for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j * k == 0 || i >= imax || j >= jmax || k >= kmax) continue;
        ne[k + n3 * (j + n2 * (i))]=ne[k + n3 * (j + n2 * (i))]-dt*(fex[k + n3 * (j + n2 * (i))]-fex[k + n3 * (j + n2 * (i-1))]+fey[k + n3 * (j + n2 * (i))]-fey[k + n3 * (j-1 + n2 * (i))]+fez[k + n3 * (j + n2 * (i))]-fez[k-1 + n3 * (j + n2 * (i))])/h ;
        ni[k + n3 * (j + n2 * (i))]=ni[k + n3 * (j + n2 * (i))]-dt*(fix[k + n3 * (j + n2 * (i))]-fix[k + n3 * (j + n2 * (i-1))]+fiy[k + n3 * (j + n2 * (i))]-fiy[k + n3 * (j-1 + n2 * (i))]+fiz[k + n3 * (j + n2 * (i))]-fiz[k-1 + n3 * (j + n2 * (i))])/h ;
    }





        for ( I = 0; I < n1 * n2; I ++) {
         j = I % n2;
         i = (I - j) / n2;
        if (i * j == 0 || i >= imax || j >= jmax ) continue;

        ne[0 + n3 * (j + n2 * (i))] = -dt*fez[0 + n3 * (j + n2 * (i))]/h ;
        ni[0 + n3 * (j + n2 * (i))] = -dt*fiz[0 + n3 * (j + n2 * (i))]/h ;

    }

     for ( I =0; I < n1 * n3; I ++) {
         k = I % n3;
         i = (I - k) / n3;

        if (i * k == 0 ||  i >= imax || k >= kmax) continue;

        ne[k + n3 * (0 + n2 * (i))] = -dt*fey[k + n3 * (0 + n2 * (i))]/h ;
        ni[k + n3 * (0 + n2 * (i))] = -dt*fiy[k + n3 * (0 + n2 * (i))]/h ;

    }




      for ( I = 0; I < n2 * n3; I ++) {
         k = I % n3;
         j = (I - k) / n3;
        if (j * k == 0 ||  j >= jmax || k >= kmax) continue;

       ne[k + n3 * (j + n2 * (0))]= -dt*fex[k + n3 * (j + n2 * (0))]/h ;
       ni[k + n3 * (j + n2 * (0))]= -dt*fix[k + n3 * (j + n2 * (0))]/h ;


    }




        // BC on densities

     for ( I =0; I < n1 * n3; I ++) {
         k = I % n3;
         i = (I - k) / n3;

        if (i * k == 0 ||  i >= imax || k >= kmax) continue;

        ne[k + n3 * (0 + n2 * (i))] = 0.0 ;
        ni[k + n3 * (0 + n2 * (i))] = 0.0 ;

        ne[k + n3 * (jmax-1 + n2 * (i))] = 0.0 ;
        ni[k + n3 * (jmax-1 + n2 * (i))] = 0.0 ;

    }




      for ( I = 0; I < n2 * n3; I ++) {
         k = I % n3;
         j = (I - k) / n3;
        if (j * k == 0 ||  j >= jmax || k >= kmax) continue;

       ne[k + n3 * (j + n2 * (0))]= 0.0 ;
       ni[k + n3 * (j + n2 * (0))]= 0.0 ;

       ne[k + n3 * (j + n2 * (imax-1))]= 0.0 ;
       ni[k + n3 * (j + n2 * (imax-1))]= 0.0 ;


    }



    for ( I = 0; I < n1*n2; I ++) {
         j = I % n2;
         i = (I - j) / n2;
        if (i * j == 0 || i >= imax+1 || j >= jmax+1) continue;
        ne[kmax-1 + n3 * (j + n2 * (i))]=0.0;
        ne[0 + n3 * (j + n2 * (i))]=0.0;
        ni[kmax-1 + n3 * (j + n2 * (i))]=0.0;
        ni[0 + n3 * (j + n2 * (i))]=0.0;

    }




        // calculating the loss
         sf=0.0;
       for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
         if (i * j * k == 0 || i >= imax || j >= jmax || k >= kmax) continue;
        sf=sf+ne[k + n3 * (j + n2 * (i))] ;
    }

        alpha=(si-sf)/sf;

    for ( I = 0; I < N; I ++) {
         k = I % n3;
         s1 = (I - k) / n3;
         j = s1 % n2;
         i = (s1 - j) / n2;
        if (i * j * k == 0 || i >= imax-1 || j >= jmax-1 || k >= kmax-1) continue;
        ne[k + n3 * (j + n2 * (i))]=ne[k + n3 * (j + n2 * (i))]+alpha*ne[k + n3 * (j + n2 * (i))] ;
        ni[k + n3 * (j + n2 * (i))]=ni[k + n3 * (j + n2 * (i))]+alpha*ne[k + n3 * (j + n2 * (i))] ;
    }
      // if (myTime%100==0.0){
     //  }


     }

}

// -----------------------------------------------------------------------------------------------

int main()
{
int imax = 32, jmax = 32, kmax = 64,i,j,k;
int n1 = imax+3, n2 = jmax+3, n3 = kmax+3;
double qi=1.6E-19,qe=-1.6E-19, kr = 0,ki = 0,si = 0,sf = 0,alpha = 0, q=1.6E-19,pie=3.14159,Ta,w,eps0,Te,Ti,B,Kb,me,mi,nue,nui,denominator_e,denominator_i,nn,dt,h,wce,wci,mue,mui,dife,difi;
int tmax = 100;

double *ne;
double *ni;
double *difxne;
double *difyne;
double *difxni;
double *difyni;
double *difxyne;
double *difxyni;
double *Exy;
double *fexy;
double *fixy;
double *g;
double *R;
double *Ex;
double *Ey;
double *fex;
double *fey;
double *fix;
double *fiy;
double *V;
double *L;
double *difzne;
double *difzni;
double *Ez;
double *fez;
double *fiz;
    ne = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    ni = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    difxne = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    difyne = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    difxni = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    difyni = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    difxyne = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    difxyni = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    Exy = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    fexy = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    fixy = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    g = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    R = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    Ex = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    Ey = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    fex = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    fey = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    fix = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    fiy = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    V = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    L = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    difzne = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    difzni = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    Ez = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    fez = (double *) malloc(n1 * n2 * n3 * sizeof(double));
    fiz = (double *) malloc(n1 * n2 * n3 * sizeof(double));

    Kb    = 1.38E-23;
    B     = 0.0;
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



    for ( i=0; i<imax+3;i++){
        for ( j=0; j<jmax+3; j++){
            for ( k=0; k<kmax+3;k++){
                ne[k + n3 * (j + n2 * (i))] = 1e-9;
                ni[k + n3 * (j + n2 * (i))] = 1e-9;
                difxne[k + n3 * (j + n2 * (i))] = 1e-9;
                difyne[k + n3 * (j + n2 * (i))] = 1e-9;
                difxni[k + n3 * (j + n2 * (i))] = 1e-9;
                difyni[k + n3 * (j + n2 * (i))] = 1e-9;
                difxyne[k + n3 * (j + n2 * (i))] = 1e-9;
                difxyni[k + n3 * (j + n2 * (i))] = 1e-9;
                Exy[k + n3 * (j + n2 * (i))] = 1e-9;
                fexy[k + n3 * (j + n2 * (i))] = 1e-9;
                fixy[k + n3 * (j + n2 * (i))] = 1e-9;
                g[k + n3 * (j + n2 * (i))] = 1e-9;
                R[k + n3 * (j + n2 * (i))] = 1e-9;
                Ex[k + n3 * (j + n2 * (i))] = 1e-9;
                Ey[k + n3 * (j + n2 * (i))] = 1e-9;
                fex[k + n3 * (j + n2 * (i))] = 1e-9;
                fey[k + n3 * (j + n2 * (i))] = 1e-9;
                fix[k + n3 * (j + n2 * (i))] = 1e-9;
                fiy[k + n3 * (j + n2 * (i))] = 1e-9;
                V[k + n3 * (j + n2 * (i))] = 1e-9;
                L[k + n3 * (j + n2 * (i))] = 1e-9;
                difzne[k + n3 * (j + n2 * (i))] = 1e-9;
                difzni[k + n3 * (j + n2 * (i))] = 1e-9;
                Ez[k + n3 * (j + n2 * (i))] = 1e-9;
                fez[k + n3 * (j + n2 * (i))] = 1e-9;
                fiz[k + n3 * (j + n2 * (i))] = 1e-9;
             }
        }
    }


    nn=1.33/(Kb*Ti); //neutral density=p/(Kb.T)
    nue=nn*1.1E-19*sqrt(Kb*Te/me); // electron collision frequency= neutral density * sigma_e*Vth_e
    nui=nn*4.4E-19*sqrt(Kb*Ti/mi);
    wce=q*B/me;
    wci=q*B/mi;
    mue=q/(me*nue);
    mui=q/(mi*nui);
    dife=Kb*Te/(me*nue);
    difi=Kb*Ti/(mi*nui);
    ki=0.00002/(nn*dt);
    denominator_e= (1+wce*wce/(nue*nue));
    denominator_i= (1+wci*wci/(nui*nui));
    // Ta and W are just some constants needed for the iterative method that we have used to solve Poisson eq.
    Ta=acos((cos(pie/imax)+cos(pie/jmax)+cos(pie/kmax))/3.0);// needs to be double checked
    w=2.0/(1.0+sin(Ta));
// -----------------------------------------------------------------------------------------------
      //Density initialization
      // To add multiple Gaussian sources, just simply use the density_initialization function at the (x,y) points that you want
        int x_position = 15, y_position = 15, z_position = 15;
       for ( i=1; i<imax-1;i++){
        for ( j=1; j<jmax-1;j++){
            for ( k=1; k<kmax-1;k++){
                ne[k + n3 * (j + n2 * (i))]= 2.0E13;/*
                    1.0E14+1.0E14*exp(-(pow((i-x_position),2)+
                    pow((j-y_position),2)+pow((k-z_position),2))/100.0);*/
                ni[k + n3 * (j + n2 * (i))]=2.0E13;/*
                    1.0E14+1.0E14*exp(-(pow((i-x_position),2)+
                    pow((j-y_position),2)+pow((k-z_position),2))/100.0);*/
            }
        }
    }



        for ( k=1; k<kmax+1; k++) {
          for ( j=1; j<jmax+1; j++) {
           for ( i=1; i<imax+1;i++) {
         si=si+ne[k + n3 * (j + n2 * (i))] ;
          }
         }
        }
// -----------------------------------------------------------------------------------------------

mardas(imax, jmax, kmax, tmax, ne, ni, difxne, difyne, difxni, difyni, difxyne, difxyni, Exy, fexy, fixy, g, R, Ex, Ey, fex, fey, fix, fiy, V, L, difzne, difzni, Ez, fez, fiz, qi, qe, kr, ki, si, sf, alpha, q, pie,Ta ,w ,eps0 , Te, Ti, B, Kb, me, mi, nue, nui, denominator_e, denominator_i, nn, dt, h, wce, wci, mue, mui, dife, difi);




    printf("%f\n", V[25 + (kmax+3) * (13 + (jmax+3) * 13)]);
 //   printf("%f\n", V[my_index + (kmax+3) * (my_index + (jmax+3) * my_index)]);

}

