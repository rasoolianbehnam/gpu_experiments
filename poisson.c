// -----------------------------------------------------------------------------------------------
//
//                                         fluid plasma simulation V 0.01
//
//                                      (c) Mohammad Menati 2017
//
// -----------------------------------------------------------------------------------------------
// Version: 0.10
// -----------------------------------------------------------------------------------------------
// History:
// 2017-xx-xx: Start of coding
// July 25th 2017
// -----------------------------------------------------------------------------------------------
// Comments:
// This code solves the plasma fluid equations with dt as large as 5ns for 2D version
// This is a 3-D version
// The code is specially written in a way that x-dimension=y-dimension but z could be different!
// This version uses grounded walls in Z and PBC in X and Y
// In this version the walls absorb %100 of particles fluxes
// Gain and loss terms added in this version
// PBC in XY plane and walls in Z
// -----------------------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// -----------------------------------------------------------------------------------------------
// global variables
// -----------------------------------------------------------------------------------------------

int i,j,k,kk,imax=32,jmax=32,kmax=64,myTime,tmax=50;
double qi=1.6E-19,qe=-1.6E-19, kr,ki,si,sf,alpha;
double q=1.6E-19,pie=3.14159,Ta,w,eps0,Te,Ti,B,Kb,me,mi,nue,nui,denominator_e,denominator_i,nn,dt,h,s,wce,wci,mue,mui,dife,difi;
double  ne[35][35][67],ni[35][35][67],difxne[35][35][67],difyne[35][35][67],difxni[35][35][67],difyni[35][35][67];
double  difxyne[35][35][67],difxyni[35][35][67],Exy[35][35][67],fexy[35][35][67],fixy[35][35][67],g[35][35][67],R[35][35][67];
double  Ex[35][35][67],Ey[35][35][67],fex[35][35][67],fey[35][35][67],fix[35][35][67],fiy[35][35][67],V[35][35][67],L[35][35][67];
double  difzne[35][35][67],difzni[35][35][67],Ez[35][35][67],fez[35][35][67],fiz[35][35][67];

// -----------------------------------------------------------------------------------------------
// initialization of arrays and constants
// -----------------------------------------------------------------------------------------------

void plasma_sim_initialize_simualtion_constants()
{
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

   for (j=0; j<jmax+2; j++){
      for (i=0; i<imax+2;i++){
        for (k=0; k<kmax+2;k++){

         g[i][j][k]=0.0;
         R[i][j][k]=0.0;
         ne[i][j][k]=0.0;
         ni[i][j][k]=0.0;
         V[i][j][k]=0.0;
         Ex[i][j][k]=0.0;
         Ey[i][j][k]=0.0;
         Ez[i][j][k]=0.0;
         fex[i][j][k]=0.0;
         fey[i][j][k]=0.0;
         fez[i][j][k]=0.0;
         fix[i][j][k]=0.0;
         fiy[i][j][k]=0.0;
         fiz[i][j][k]=0.0;


        }
      }
    }
}




// -----------------------------------------------------------------------------------------------
// Density Initialization
// -----------------------------------------------------------------------------------------------



void density_initialization(int x_position,int y_position,int z_position)
{

     for (i=1; i<imax+1;i++){
       for (j=1; j<jmax+1;j++){
        for (k=1; k<kmax-1;k++){

         ne[i][j][k]=1.0E14+1.0E14*exp(-(pow((i-x_position),2)+pow((j-y_position),2)+pow((k-z_position),2))/100.0);
         ni[i][j][k]=1.0E14+1.0E14*exp(-(pow((i-x_position),2)+pow((j-y_position),2)+pow((k-z_position),2))/100.0);

        }
       }
      }




}



// -----------------------------------------------------------------------------------------------
// poisson equation
// -----------------------------------------------------------------------------------------------

void plasma_sim_solve_poisson_equation_on_grid()
{



   // Here we calculate the right hand side of the Poisson equation

   for (j=1; j<jmax+1; j++){
      for (i=1; i<imax+1;i++){
        for (k=1; k<kmax-1;k++){

         g[i][j][k]=-(ne[i][j][k]*qe+ni[i][j][k]*qi)/eps0;

        }
      }
    }






/////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//// has problem at the edges
/////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


// solving Poisson eq. using  successive over-relaxation method

  for (kk=0; kk<40; kk++) {
     for (k=1; k<kmax-1; k++) {
        for (i=1; i<imax+1;i++) {
          for (j=1; j<jmax+1;j++) {

            R[i][j][k]= (V[i+1][j][k]+V[i-1][j][k]+V[i][j+1][k]+V[i][j-1][k]+V[i][j][k+1]+V[i][j][k-1])/6.0-V[i][j][k]-(h*h)*g[i][j][k]/6.0;
            V[i][j][k]=V[i][j][k]+w*R[i][j][k];

          }
        }
      }
   }


        for (k=0; k<kmax; k++){
         for (j=0; j<jmax+1; j++){

         V[imax+1][j][k]=V[1][j][k];

         V[j][jmax+1][k]=V[j][1][k];

         V[0][j][k]=V[imax][j][k];

         V[j][0][k]=V[j][jmax][k];

        }
      }


}



// -----------------------------------------------------------------------------------------------
// Electric field components
// -----------------------------------------------------------------------------------------------



void electric_field_elements()
{


     for (k=1; k<kmax-1; k++) {
        for (i=1; i<imax+1;i++) {
          for (j=1; j<jmax+1;j++) {

           Ex[i][j][k]= (V[i][j][k]-V[i+1][j][k])/h;
           Ey[i][j][k]= (V[i][j][k]-V[i][j+1][k])/h;

           difxne[i][j][k]=(ne[i+1][j][k]-ne[i][j][k])/h;
           difxni[i][j][k]=(ni[i+1][j][k]-ni[i][j][k])/h;
           difyne[i][j][k]=(ne[i][j+1][k]-ne[i][j][k])/h;
           difyni[i][j][k]=(ni[i][j+1][k]-ni[i][j][k])/h;

          }
        }
     }


     for (k=0; k<kmax-1; k++) {
        for (i=1; i<imax+1;i++) {
          for (j=1; j<jmax+1;j++) {

           Ez[i][j][k]= (V[i][j][k]-V[i][j][k+1])/h;

           difzne[i][j][k]=(ne[i][j][k+1]-ne[i][j][k])/h;
           difzni[i][j][k]=(ni[i][j][k+1]-ni[i][j][k])/h;

          }
        }
     }




}



// -----------------------------------------------------------------------------------------------
// Ex and gradiant_x average
// -----------------------------------------------------------------------------------------------



void average_x()
{

        // Calculating the average values of Ex and gradiant_x

        for (k=1; k<kmax-1; k++) {
          for (i=2; i<imax+1;i++)   {
           for (j=1; j<jmax;j++) {

            Exy[i][j][k]= 0.25*(Ex[i][j][k]+Ex[i][j+1][k]+Ex[i-1][j][k]+Ex[i-1][j+1][k]) ;

            difxyne[i][j][k]=0.25*(difxne[i][j][k]+difxne[i][j+1][k]+difxne[i-1][j][k]+difxne[i-1][j+1][k]);
            difxyni[i][j][k]=0.25*(difxni[i][j][k]+difxni[i][j+1][k]+difxni[i-1][j][k]+difxni[i-1][j+1][k]);

           }
          }
        }

         for (k=1; k<kmax-1; k++) {
           for (j=1; j<jmax;j++) {

            Exy[1][j][k]= 0.25*(Ex[1][j][k]+Ex[1][j+1][k]+Ex[imax][j][k]+Ex[imax][j+1][k]) ;

            difxyne[1][j][k]=0.25*(difxne[1][j][k]+difxne[1][j+1][k]+difxne[imax][j][k]+difxne[imax][j+1][k]);
            difxyni[1][j][k]=0.25*(difxni[1][j][k]+difxni[1][j+1][k]+difxni[imax][j][k]+difxni[imax][j+1][k]);

           }
        }

        for (k=1; k<kmax-1; k++) {
          for (i=2; i<imax;i++) {

            Exy[i][jmax][k]= 0.25*(Ex[i][jmax][k]+Ex[i][1][k]+Ex[i-1][jmax][k]+Ex[i-1][1][k]) ;

            difxyne[i][jmax][k]=0.25*(difxne[i][jmax][k]+difxne[i][1][k]+difxne[i-1][jmax][k]+difxne[i-1][1][k]);
            difxyni[i][jmax][k]=0.25*(difxni[i][jmax][k]+difxni[i][1][k]+difxni[i-1][jmax][k]+difxni[i-1][1][k]);

           }
          }



        // Calculating Exy and difxy at the corners


      for (k=1; k<kmax-1;k++) {

        Exy[imax][jmax][k]=(Ex[imax][jmax][k]+Ex[imax-1][jmax][k]+Ex[imax-1][1][k])/3.0;
        difxyne[imax][jmax][k]=(difxne[imax][jmax][k]+difxne[imax-1][jmax][k]+difxne[imax-1][1][k])/3.0;
        difxyni[imax][jmax][k]=(difxni[imax][jmax][k]+difxni[imax-1][jmax][k]+difxni[imax-1][1][k])/3.0;

        Exy[1][jmax][k]=(Ex[1][jmax][k]+Ex[imax][jmax][k]+Ex[1][1][k])/3.0;
        difxyne[1][jmax][k]=(difxne[1][jmax][k]+difxne[imax][jmax][k]+difxne[1][1][k])/3.0;
        difxyni[1][jmax][k]=(difxni[1][jmax][k]+difxni[imax][jmax][k]+difxni[1][1][k])/3.0;

      }





}


// -----------------------------------------------------------------------------------------------
// Fluxes in Y direction
// -----------------------------------------------------------------------------------------------



void flux_y()
{

        for (k=1; k<kmax-1; k++) {
         for (i=1; i<imax+1;i++) {
          for (j=1; j<jmax+1;j++) {

            fey[i][j][k]= (-0.5*(ne[i][j+1][k]+ne[i][j][k])*mue*Ey[i][j][k]-dife*difyne[i][j][k]
                -wce*q*0.5*(ne[i][j+1][k]+ne[i][j][k])*Exy[i][j][k]/(me*nue*nue)-wce*dife*difxyne[i][j][k]/nue)/denominator_e;

            fiy[i][j][k]= (0.5*(ni[i][j+1][k]+ni[i][j][k])*mui*Ey[i][j][k]-difi*difyni[i][j][k]
                -wci*q*0.5*(ni[i][j+1][k]+ni[i][j][k])*Exy[i][j][k]/(mi*nui*nui)+wci*difi*difxyni[i][j][k]/nui)/denominator_i;
          }
         }
        }


        for (k=1; k<kmax-1; k++) {
        for (i=1; i<imax+1; i++) {

          fey[i][0][k] = fey[i][jmax][k];
          fiy[i][0][k] = fiy[i][jmax][k];

        }
       }






}





// -----------------------------------------------------------------------------------------------
// Ey and gradiant_y average
// -----------------------------------------------------------------------------------------------


void average_y()
{


       // Calculating the average values of Ey and gradiant_y

       for (k=1; k<kmax-1; k++) {
          for (i=1; i<imax;i++)   {
           for (j=2; j<jmax+1;j++) {

          Exy[i][j][k]= 0.25*(Ey[i][j][k]+Ey[i][j-1][k]+Ey[i+1][j][k]+Ey[i+1][j-1][k]);

          difxyne[i][j][k]= 0.25*(difyne[i][j][k]+difyne[i][j-1][k]+difyne[i+1][j][k]+difyne[i+1][j-1][k]);
          difxyni[i][j][k]= 0.25*(difyni[i][j][k]+difyni[i][j-1][k]+difyni[i+1][j][k]+difyni[i+1][j-1][k]);

         }
        }
       }


       for (k=1; k<kmax-1; k++) {
          for (i=1; i<imax;i++)   {

          Exy[i][1][k]= 0.25*(Ey[i][1][k]+Ey[i][jmax][k]+Ey[i+1][1][k]+Ey[i+1][jmax][k]);

          difxyne[i][1][k]= 0.25*(difyne[i][1][k]+difyne[i][jmax][k]+difyne[i+1][1][k]+difyne[i+1][jmax][k]);
          difxyni[i][1][k]= 0.25*(difyni[i][1][k]+difyni[i][jmax][k]+difyni[i+1][1][k]+difyni[i+1][jmax][k]);

         }
        }


        for (k=1; k<kmax-1; k++) {
           for (j=2; j<jmax+1;j++) {

          Exy[imax][j][k]= 0.25*(Ey[imax][j][k]+Ey[imax][j-1][k]+Ey[1][j][k]+Ey[1][j-1][k]);

          difxyne[imax][j][k]= 0.25*(difyne[imax][j][k]+difyne[imax][j-1][k]+difyne[1][j][k]+difyne[1][j-1][k]);
          difxyni[imax][j][k]= 0.25*(difyni[imax][j][k]+difyni[imax][j-1][k]+difyni[1][j][k]+difyni[1][j-1][k]);

         }
       }



        // Calculating Exy and difxy at the corners

       for (k=1; k<kmax-1; k++) {

        Exy[imax][1][k]=(Ey[imax][1][k]+Ey[1][1][k]+Ey[imax][jmax][k])/3.0;
        difxyne[imax][1][k]=(difyne[imax][1][k]+difyne[1][1][k]+difyne[imax][jmax][k])/3.0;
        difxyni[imax][1][k]=(difyni[imax][1][k]+difyni[1][1][k]+difyni[imax][jmax][k])/3.0;

        Exy[imax][jmax][k]=(Ey[imax][jmax-1][k]+Ey[imax][jmax][k]+Ey[1][jmax-1][k])/3.0;
        difxyne[imax][jmax][k]=(difyne[imax][jmax-1][k]+difyne[imax][jmax][k]+difyne[1][jmax-1][k])/3.0;
        difxyni[imax][jmax][k]=(difyni[imax][jmax-1][k]+difyni[imax][jmax][k]+difyni[1][jmax-1][k])/3.0;
       }



}




// -----------------------------------------------------------------------------------------------
// Fluxes in X direction
// -----------------------------------------------------------------------------------------------



void flux_x()
{


       for (k=1; k<kmax-1; k++) {
        for (j=1; j<jmax+1; j++) {
         for (i=1; i<imax+1;i++) {

          fex[i][j][k]=(-0.5*(ne[i][j][k]+ne[i+1][j][k])*mue*Ex[i][j][k]-dife*difxne[i][j][k]
          +wce*dife*difxyne[i][j][k]/nue+wce*q*0.5*(ne[i][j][k]+ne[i+1][j][k])/(me*nue*nue)*Exy[i][j][k])/denominator_e;

          fix[i][j][k]=(0.5*(ni[i][j][k]+ni[i+1][j][k])*mui*Ex[i][j][k]-difi*difxni[i][j][k]
          -wci*difi*difxyni[i][j][k]/nui+wci*q*0.5*(ni[i][j][k]+ni[i+1][j][k])*Exy[i][j][k]/(mi*nui*nui))/denominator_i;

         }
        }
       }


      for (k=1; k<kmax-1; k++) {
        for (j=1; j<jmax+1; j++) {

          fex[0][j][k] = fex[imax][j][k];
          fix[0][j][k] = fix[imax][j][k];

        }
       }




}
// -----------------------------------------------------------------------------------------------
// Fluxes in Z direction
// -----------------------------------------------------------------------------------------------



void flux_z()
{


       for (k=0; k<kmax-1; k++) {
        for (j=1; j<jmax+1; j++) {
         for (i=1; i<imax+1;i++) {

          fez[i][j][k]=-0.5*(ne[i][j][k]+ne[i][j][k+1])*mue*Ez[i][j][k]-dife*difzne[i][j][k];

          fiz[i][j][k]=0.5*(ni[i][j][k]+ni[i][j][k+1])*mui*Ez[i][j][k]-difi*difzni[i][j][k];

         }
        }
       }

       // BC on fluxes


       for (i=1; i<imax+1; i++) {
        for (j=1; j<jmax+1; j++) {



           if (fez[i][j][1]>0.0){

            fez[i][j][1]=0.0;
           }


           if (fiz[i][j][1]>0.0){


            fiz[i][j][1]=0.0;
           }


           if (fez[i][j][kmax-1]<0.0){

            fez[i][j][kmax-1]=0.0;
           }


           if (fiz[i][j][kmax-1]<0.0){


            fiz[i][j][kmax-1]=0.0;
           }


        }
      }




}


// -----------------------------------------------------------------------------------------------
// Periodic boundary condition on densities
// -----------------------------------------------------------------------------------------------


void BC_densities()
{

     // BC on densities

        for (k=0; k<kmax; k++){
         for (j=0; j<jmax+1; j++){

         ne[imax+1][j][k]=ne[1][j][k];
         ni[imax+1][j][k]=ni[1][j][k];

         ne[j][jmax+1][k]=ne[j][1][k];
         ni[j][jmax+1][k]=ni[j][1][k];


         ne[0][j][k]=ne[imax][j][k];
         ni[0][j][k]=ni[imax][j][k];

         ne[j][0][k]=ne[j][jmax][k];
         ni[j][0][k]=ni[j][jmax][k];

        }
      }





      for (i=1; i<imax+1; i++){
         for (j=1; j<jmax+1; j++){

         ne[i][j][kmax-1]=0.0;
         ne[i][j][0]=0.0;

         ni[i][j][kmax-1]=0.0;
         ni[i][j][0]=0.0;

        }
      }

}


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
// -----------------------------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{

    FILE*fp1;
    FILE*fp2;
    FILE*fp3;
    FILE*fp4;
    FILE*fp5;
    FILE*fp6;

    fp1=fopen("potential.txt","w");
    fp2=fopen("ne.txt","w");
    fp3=fopen("ni.txt","w");
    fp4=fopen("ni200.txt","w");
    fp5=fopen("ne400.txt","w");
    fp6=fopen("ni400.txt","w");


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    plasma_sim_initialize_simualtion_constants();

    nn=1.33/(Kb*Ti); //neutral density=p/(Kb.T)
    nue=nn*1.0E-20*sqrt(Kb*Te/me); // electron collision frequency= neutral density * sigma_e*Vth_e
    nui=nn*5.0E-19*sqrt(Kb*Ti/mi);

    wce=q*B/me;
    wci=q*B/mi;
    mue=q/(me*nue);
    mui=q/(mi*nui);
    dife=Kb*Te/(me*nue);
    difi=Kb*Ti/(mi*nui);
    ki=0.00002/(nn*dt);



    denominator_e= (1+wce*wce/(nue*nue));
    denominator_i= (1+wci*wci/(nui*nui));

    printf("%f %f \n", wce,wci);

    // Ta and W are just some constants needed for the iterative method that we have used to solve Poisson eq.

    Ta=acos((cos(pie/imax)+cos(pie/jmax)+cos(pie/kmax))/3.0);// needs to be double checked


    w=2.0/(1.0+sin(Ta));
        printf("w: %f \n",w);


// -----------------------------------------------------------------------------------------------


      //Density initialization
      // To add multiple Gaussian sources, just simply use the density_initialization function at the (x,y) points that you want


      density_initialization(15,15,15);

      //density_initialization(55,40,20);
    printf("ne[%d, %d, %d] = %e\n", 5, 6, 7, ne[5, 6, 7]);
    printf("ni[%d, %d, %d] = %e\n", 5, 6, 7, ni[5, 6, 7]);

      BC_densities();

        for (k=1; k<kmax+1; k++) {
          for (j=1; j<jmax+1; j++) {
           for (i=1; i<imax+1;i++) {

         si=si+ne[i][j][k] ;

          }
         }
        }

printf("si before loop: %f\n", si);



// -----------------------------------------------------------------------------------------------



    clock_t begin = clock();
    for (myTime=1; myTime<tmax; myTime++){  // This for loop takes care of myTime evolution



// -----------------------------------------------------------------------------------------------

      // Solving Poisson's eq. to get the voltage everywhere

      plasma_sim_solve_poisson_equation_on_grid();


// -----------------------------------------------------------------------------------------------

      // Now Calculating electric field and density gradient which are calculated through function electric_field_elements
    printf("V[%d, %d, %d] = %f\n", 5, 6, 7, V[5][6][7]);
      electric_field_elements();


// -----------------------------------------------------------------------------------------------

       /* Since I am using mid points for Calculating electric field and density gradient,
        to calculate Ex at any point that I don't have it directly, the average over
        the neighboring points is used instead. these average variables are, exy, fexy, fixy, ...*/

       average_x();


// -----------------------------------------------------------------------------------------------

        // Here we calculate the fluxes in y direction

        flux_y();



// -----------------------------------------------------------------------------------------------


        // Calculating the average Exy and difxy to be used in x direction fluxes

        average_y();


// -----------------------------------------------------------------------------------------------


        // Now ready to calculate the fluxes in x direction

        flux_x();


// -----------------------------------------------------------------------------------------------


        // Now we calculate the fluxes in z direction

        flux_z();


// -----------------------------------------------------------------------------------------------



         for (k=1; k<kmax; k++) {
          for (j=1; j<jmax+1; j++) {
           for (i=1; i<imax+1;i++) {

         ne[i][j][k]=ne[i][j][k]-dt*(fex[i][j][k]-fex[i-1][j][k]+fey[i][j][k]-fey[i][j-1][k]+fez[i][j][k]-fez[i][j][k-1])/h ;
         ni[i][j][k]=ni[i][j][k]-dt*(fix[i][j][k]-fix[i-1][j][k]+fiy[i][j][k]-fiy[i][j-1][k]+fiz[i][j][k]-fiz[i][j][k-1])/h ;

          }
         }
        }


          for (j=1; j<jmax+1; j++) {
           for (i=1; i<imax+1;i++) {

         ne[i][j][0] = -dt*fez[i][j][0]/h ;
         ni[i][j][0] = -dt*fiz[i][j][0]/h ;

          }
         }

        // BC on densities

        BC_densities();

        // calculating the loss

        sf=0.0;
        for (k=1; k<kmax+1; k++) {
          for (j=1; j<jmax+1; j++) {
           for (i=1; i<imax+1;i++) {

         sf=sf+ne[i][j][k] ;

          }
         }
        }


        alpha=(si-sf)/sf;

        for (k=1; k<kmax-1; k++) {
          for (j=1; j<jmax+1; j++) {
           for (i=1; i<imax+1;i++) {

         ne[i][j][k]=ne[i][j][k]+alpha*ne[i][j][k] ;
         ni[i][j][k]=ni[i][j][k]+alpha*ne[i][j][k] ;

          }
         }
        }







      // if (myTime%100==0.0){

        printf("%d \n", myTime);

     //  }




        if (myTime==50000000000){

         for (j=1; j<jmax+1; j++) {
           for (i=1; i<imax+1;i++) {

             fprintf(fp3,"%d %d %f \n", i,j,ne[i][j]);
             fprintf(fp4,"%d %d %f \n", i,j,ni[i][j]);

           }
         }
        }

        if (myTime==5000){

        for (k=1; k<kmax+1; k++) {
         for (j=1; j<jmax+1; j++) {
           for (i=1; i<imax+1;i++) {

             fprintf(fp5,"%d %d %f \n", i,j,k,ne[i][j][k]);
             fprintf(fp6,"%d %d %f \n", i,j,k,ni[i][j][k]);

           }
          }
         }


        }






     }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f\n", time_spent);


             printf("%f \n", V[15][15][15]);




       for (k=1; k<kmax+1; k++) {
        for (j=1; j<jmax+1; j++) {

         fprintf(fp2,"%d %d %f \n", j,k,V[j][20][k]);
         fprintf(fp3,"%d %d %f \n", j,k,ne[j][20][k]);

        }
      }


    return 0;
}
