function [i_kernel, g_kernel, A] = calc_kernels(n1, n2, n3, w, h, num_iterations)
    imax = n1 - 3;
    jmax = n2 - 3;
    kmax = n3 - 3;
    %i_kernel = zeros(n1*n2*n3, n1*n2*n3, 'float');
    %g_kernel = zeros(n1*n2*n3, n1*n2*n3, 'float');
    i_kernel = sparse(n1*n2*n3, n1*n2*n3)
    g_kernel = sparse(n1*n2*n3, n1*n2*n3)
    for I = 1:n1*n2*n3;
        k = mod(I, n3);
        s1 = (I - k) / n3;
        j = mod(s1, n2);
        i = (s1 - j) / n2;
        if (i >= 1 && i < imax+1 && 
        j >= 1 && j < jmax+1 &&
        k >= 1 && k < kmax-1)
            if (mod(I, 1000) == 0)
                fprintf('%d/%d\n', I, n1*n2*n3)
            end
            i_kernel(I, :) =w*1./6* \
            ( i_kernel(I-1, :) \
            + i_kernel(I-n3, :) \
            + i_kernel(I-n3*n2, :));
            i_kernel(I, I+1) += w*1./6;
            i_kernel(I, I+n3) += w*1./6;
            i_kernel(I, I+n2*n3) += w*1./6;
            i_kernel(I, I) += 1 - w;

            g_kernel(I, :) = w / 6. * \
            (g_kernel(I-1, :) + g_kernel(I-n3, :) + g_kernel(I-n2*n3, :));
            g_kernel(I, I) += 1;
        end
    end

    fprintf('Calculating powers...');
    A = i_kernel;
    for i =1:num_iterations-1
        if (mod(i, 10) == 0)
            fprintf('%d\n', i);
        end
        B = B + A;
        A = i_kernel * A;
    end
    fprintf('done\n');
    save('Amat', 'A');
    save('Bmat', 'B');

    %i_kernel = sparse(i_kernel); 
    %g_kernel = sparse(g_kernel); 
    %A = sparse(A); 
end

function v = poisson_brute(v, g, num_iterations, imax, jmax, kmax, h, w)
    for kk = 1:num_iterations
        for k =2:kmax-1
            for j = 2:jmax+1
                for i = 2:imax+1
                    r = v(i+1, j, k) / 6. + v(i-1, j, k) / 6. + v(i, j+1, k) / 6. + v(i, j-1, k) / 6. + v(i ,j, k+1) / 6. + v(i, j, k-1) / 6. - v(i, j, k) - (h**2) * g(i, j, k) / 6.;
                    r = w * r;
                    v(i, j, k) += r;
                end
            end
        end
    end
end
%
function [v, g] = plasma_sim_solve_poisson_equation_on_grid(v, g, ne, ni, imax, jmax, kmax, w, h, num_iterations, qe, qi, eps0)
%   # here we calculate the right hand side of the poisson equation
    fprintf('starting poisson\n')
    start = clock();
%    ################################333
%    v1 = pv.poisson_fast_no_loop(v.reshape(-1),  g.reshape(-1)).reshape(n1, n2, n3)
%    ################################333
    %v = poisson_brute(v, g, num_iterations, imax, jmax, kmax, h, w);

    elapsedtime = etime(clock(), start);
    fprintf('elapsed time: %f\n', elapsedtime);
%    #v3 = pv.poisson_brute_main(v*1., g*1.)
%    #v(:, :, :) = pv.poisson_brute_main_flat(v.reshape(-1),  g.reshape(-1)).reshape(n1, n2, n3)
%    #stat_diff(v1, v2, "fast no loop vs brute here")
%    #stat_diff(v2, v3, "brute pv loop vs brute here")
%    time_taken2 = time.time() - start
%    print("time taken: %f"%(time_taken2))
%    v = v1
%
%
%    v(imax+1, 0:jmax+1, 0:kmax)=v(1, 0:jmax+1, 0:kmax);
%    v(0:jmax+1, jmax+1, 0:kmax)=v(0:jmax+1, 1, 0:kmax);
%    v(0, 0:jmax+1, 0:kmax)=v(imax, 0:jmax+1, 0:kmax);
%    v(0:jmax+1, 0, 0:kmax)=v(0:jmax+1, jmax, 0:kmax);
end

imax=16
jmax=16
kmax=16
tmax=50
upper_lim = 3
n1 = imax+upper_lim
n2 = jmax+upper_lim
n3 = kmax+upper_lim
qi=1.6e-19
qe=-1.6e-19
q=1.6e-19
pie=3.14159
kb    = 1.38e-23;
b     = 1.0;
te    = 2.5*11604.5;
ti    = 0.025*11604.5;
me    = 9.109e-31;
mi    = 6.633e-26;
ki    = 0.0;
dt    = 1.0e-12;
h     = 1.0e-3;
eps0  = 8.854e-12;
si    = 0.0;
sf    =0.0;


nn=1.33/(kb*ti); #neutral density=p/(kb.t)
nue=nn*1.0e-20*sqrt(kb*te/me); # electron collision frequency= neutral density * sigma_e*vth_e
nui=nn*5.0e-19*sqrt(kb*ti/mi);

wce=q*b/me;
wci=q*b/mi;
mue=q/(me*nue);
mui=q/(mi*nui);
dife=kb*te/(me*nue);
difi=kb*ti/(mi*nui);
ki=0.00002/(nn*dt);
denominator_e= (1+wce*wce/(nue*nue));
denominator_i= (1+wci*wci/(nui*nui));
ta=acos((cos(pie/imax)+cos(pie/jmax)+\
       cos(pie/kmax))/3.0);# needs to be double checked;
w=2.0/(1.0+sin(ta));

g       = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
r       = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
ne      = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
ni      = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
v       = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
ex      = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
ey      = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
ez      = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
fex     = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
fey     = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
fez     = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
fix     = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
fiy     = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
fiz     = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
difxne  = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
difxni  = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
difyne  = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
difyni  = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
difzne  = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
difzni  = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
exy     = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
difxyne = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
difxyni = zeros(imax+upper_lim, jmax+upper_lim, kmax+upper_lim);
   
%    method = 'ndarray'
%    pv = poisson_vectorized(imax+upper_lim, jmax+upper_lim, kmax+upper_lim,\
%            w=w, h=h, num_iterations=40, method=method\
%            , upper_lim=upper_lim)
%
%
%    density_initialization(ne, ni, 15,15,15);
%
x_position = y_position = z_position = 15;
for i =1:imax+1
    for j =1:jmax+1
        for k=1:kmax-1
            ne(i, j, k)= 1.0e14+1.0e14*exp(-((i-x_position)^2
                    +(j-y_position)^2+
                    (k-z_position)^2)/100.0);
            ni(i, j, k)=1.0e14+1.0e14*exp(-(((i-x_position)^2)
                    +((j-y_position)^2)+
                    ((k-z_position)^2))/100.0);
        end
    end
end

si = sum(sum(sum(ne(1:imax+1, 1:jmax+1, 1:kmax+1))))
fprintf('si before loop: %f\n', si);
fprintf('ne(%d, %d, %d) = %e\n', 5, 6, 7, ne(5, 6, 7));
fprintf('ni(%d, %d, %d) = %e\n', 5, 6, 7, ni(5, 6, 7));
%        bc_densities(ne, ni)
ne(imax+1+1, 1:jmax+1, 1:kmax) = ne(1+1, 1:jmax+1, 1:kmax);
ni(imax+1+1, 1:jmax+1, 1:kmax) = ni(1+1, 1:jmax+1, 1:kmax);

ne(1:jmax+1, jmax+1+1, 1:kmax) = ne(1:jmax+1, 1+1, 1:kmax);
ni(1:jmax+1, jmax+1+1, 1:kmax) = ni(1:jmax+1, 1+1, 1:kmax);

ne(0+1, 1:jmax+1, 1:kmax) = ne(imax+1, 1:jmax+1, 1:kmax);
ni(0+1, 1:jmax+1, 1:kmax) = ni(imax+1, 1:jmax+1, 1:kmax);

ne(1:jmax+1, 0+1, 1:kmax) = ne(1:jmax+1, jmax+1, 1:kmax);
ni(1:jmax+1, 0+1, 1:kmax) = ni(1:jmax+1, jmax+1, 1:kmax);

[i_kernel, g_kernel, A] = calc_kernels(n1, n2, n3, w, h, 40);

tic
for time = 1:tmax
    g(2:imax+1, 2:jmax+1, 2:kmax-1)=-(ne(2:imax+1, 2:jmax+1, 2:kmax-1)*qe\
            +ni(2:imax+1, 2:jmax+1, 2:kmax-1)*qi)/eps0;
    g_temp = w * h^2 * g / 6.;
    g_temp = g_kernel * g_temp(:);
    v = A * v(:) - A * g_temp;
    v = reshape(v, [n1, n2, n3]);
%%        electric_field_elements(v, ne, ni, ez, ex, ey, difxne, difxni, difyne, difyni, difzne, difzni)
    printf('v(%d, %d, %d) = %f\n', 5, 6, 7, v(5, 6, 7));
    ez(2:imax+1, 2:jmax+1, 1:kmax-1) = (v(2:imax+1, 2:jmax+1, 1:kmax-1)
            - v(2:imax+1, 2:jmax+1, 2:kmax))/h;

    difzne(2:imax+1, 2:jmax+1, 1:kmax-1)= (ne(2:imax+1, 2:jmax+1, 2:kmax)
        -ne(2:imax+1, 2:jmax+1, 1:kmax-1))/h;
    difzni(2:imax+1, 2:jmax+1, 1:kmax-1)= (ni(2:imax+1, 2:jmax+1, 2:kmax)
        -ni(2:imax+1, 2:jmax+1, 1:kmax-1))/h;

    ex(2:imax+1, 2:jmax+1, 2:kmax-1) = (v(2:imax+1, 2:jmax+1, 2:kmax-1)-
            v(1+2:imax+1+1, 2:jmax+1, 2:kmax-1)) / h;
    ey(2:imax+1, 2:jmax+1, 2:kmax-1) = (v(2:imax+1, 2:jmax+1, 2:kmax-1)-
            v(2:imax+1, 1+2:jmax+1+1, 2:kmax-1)) / h;

    difxne(2:imax+1, 2:jmax+1, 2:kmax-1) = (ne(3:imax+1+1, 2:jmax+1, 2:kmax-1) - ne(2:imax+1, 2:jmax+1, 2:kmax-1)) / h;
    difxni(2:imax+1, 2:jmax+1, 2:kmax-1) = (ni(3:imax+1+1, 2:jmax+1, 2:kmax-1) - ni(2:imax+1, 2:jmax+1, 2:kmax-1)) / h;
    difyne(2:imax+1, 2:jmax+1, 2:kmax-1) = (ne(2:imax+1, 3:jmax+1+1, 2:kmax-1) - ne(2:imax+1, 2:jmax+1, 2:kmax-1)) / h;
    difyni(2:imax+1, 2:jmax+1, 2:kmax-1) = (ni(2:imax+1, 3:jmax+1+1, 2:kmax-1) - ne(2:imax+1, 2:jmax+1, 2:kmax-1)) / h;
        
%        average_x(ne, ni, ex, exy, difxne, difxni, difxyne, difxyni)
    exy(3:imax+1, 2:jmax, 2:kmax-1) = .25*(ex(3:imax+1, 2:jmax, 2:kmax-1) +\
            ex(3:imax+1, 3:jmax+1, 2:kmax-1) + ex(2:imax, 2:jmax, 2:kmax-1) + \
            ex(2:imax, 3:jmax+1, 2:kmax-1));
    difxyne(3:imax+1, 2:jmax, 2:kmax-1) = .25*(difxne(3:imax+1, 2:jmax, 2:kmax-1) +\
            difxne(3:imax+1, 3:jmax+1, 2:kmax-1) + difxne(2:imax, 2:jmax, 2:kmax-1)) +\
            difxne(2:imax, 3:jmax+1, 2:kmax-1);
    difxyni(3:imax+1, 2:jmax, 2:kmax-1) = .25*(difxni(3:imax+1, 2:jmax, 2:kmax-1) +\
            difxni(3:imax+1, 3:jmax+1, 2:kmax-1) + difxni(2:imax, 2:jmax, 2:kmax-1)) +\
            difxni(2:imax, 3:jmax+1, 2:kmax-1);

    exy(2, 2:jmax, 2:kmax-1) = .25*(ex(2, 2:jmax, 2:kmax-1) +\
            ex(2, 3:jmax+1, 2:kmax-1) \
            + ex(imax+1, 2:jmax, 2:kmax-1)\
            + ex(imax+1, 3:jmax+1, 2:kmax-1));


    difxyne(2, 2:jmax, 2:kmax-1)=0.25*(difxne(2, 2:jmax, 2:kmax-1)+difxne(2, 3:jmax+1, 2:kmax-1)+difxne(imax+1, 2:jmax, 2:kmax-1)+difxne(imax+1, 3:jmax+1, 2:kmax-1));
    difxyni(2, 2:jmax, 2:kmax-1)=0.25*(difxni(2, 2:jmax, 2:kmax-1)+difxni(2, 3:jmax+1, 2:kmax-1)+difxni(imax+1, 2:jmax, 2:kmax-1)+difxni(imax+1, 3:jmax+1, 2:kmax-1));

    exy(2:imax, jmax+1, 2:kmax-1)= \
            0.25*(ex(2:imax, jmax+1, 2:kmax-1)+ex(2:imax, 2, 2:kmax-1)+ex(1:imax-1, jmax+1, 2:kmax-1)+ex(1:imax-1, 2, 2:kmax-1)) ;

    difxyne(2:imax, jmax+1, 2:kmax-1)=\
            0.25*(difxne(2:imax, jmax+1, 2:kmax-1)+difxne(2:imax, 2, 2:kmax-1)+difxne(1:imax-1, jmax+1, 2:kmax-1)+difxne(1:imax-1, 2, 2:kmax-1));
    difxyni(2:imax, jmax+1, 2:kmax-1)=\
            0.25*(difxni(2:imax, jmax+1, 2:kmax-1)+difxni(2:imax, 2, 2:kmax-1)+difxni(1:imax-1, jmax+1, 2:kmax-1)+difxni(1:imax-1, 2, 2:kmax-1));

    exy(imax+1, jmax+1, 2:kmax-1)=(ex(imax+1, jmax+1, 2:kmax-1)+ex(imax-1, jmax+1, 2:kmax-1)+ex(imax-1, 1, 2:kmax-1))/3.0;
    difxyne(imax+1, jmax+1, 2:kmax-1)=(difxne(imax, jmax+1, 2:kmax-1)+difxne(imax-1, jmax+1, 2:kmax-1)+difxne(imax-1, 1+1, 2:kmax-1))/3.0;
    difxyni(imax+1, jmax+1, 2:kmax-1)=(difxni(imax, jmax+1, 2:kmax-1)+difxni(imax-1, jmax+1, 2:kmax-1)+difxni(imax-1, 1+1, 2:kmax-1))/3.0;

    exy(1+1, jmax+1, 2:kmax-1)=(ex(1+1, jmax+1, 2:kmax-1)+ex(imax+1, jmax+1, 2:kmax-1)+ex(1+1, 1+1, 2:kmax-1))/3.0;
    difxyne(1+1, jmax+1, 2:kmax-1)=(difxne(1+1, jmax+1, 2:kmax-1)+difxne(imax+1, jmax+1, 2:kmax-1)+difxne(1+1, 1+1, 2:kmax-1))/3.0;
    difxyni(1+1, jmax+1, 2:kmax-1)=(difxni(1+1, jmax+1, 2:kmax-1)+difxni(imax+1, jmax+1, 2:kmax-1)+difxni(1+1, 1+1, 2:kmax-1))/3.0;

%        flux_y(ne, ni, fey, fiy, ey, ez, exy, difyne, difyni, difxyne, difxyni)
    fey(2:imax+1, 2:jmax+1, 2:kmax-1)= \
        (-0.5*(ne(2:imax+1, 3:jmax+1+1, 2:kmax-1)\
        +ne(2:imax+1, 2:jmax+1, 2:kmax-1))*mue.*ey(2:imax+1, 2:jmax+1, 2:kmax-1)\
        -dife*difyne(2:imax+1, 2:jmax+1, 2:kmax-1)
        -wce*q*0.5*(\
            ne(2:imax+1, 3:jmax+1+1, 2:kmax-1)\
            +ne(2:imax+1, 2:jmax+1, 2:kmax-1))\
        .*exy(2:imax+1, 2:jmax+1, 2:kmax-1)\
        /(me*nue*nue)-wce*dife*difxyne(2:imax+1, 2:jmax+1, 2:kmax-1)/nue)/denominator_e;

    fiy(2:imax+1, 2:jmax+1, 2:kmax-1)= \
    (0.5*(ni(2:imax+1, 3:jmax+1+1, 2:kmax-1)+ni(2:imax+1, 2:jmax+1, 2:kmax-1))*mui.*ey(2:imax+1, 2:jmax+1, 2:kmax-1)-difi*difyni(2:imax+1, 2:jmax+1, 2:kmax-1)
    -wci*q*0.5.*(ni(2:imax+1, 3:jmax+1+1, 2:kmax-1)\
    +ni(2:imax+1, 2:jmax+1, 2:kmax-1)).*exy(2:imax+1, 2:jmax+1, 2:kmax-1)/(mi*nui*nui)+wci*difi*difxyni(2:imax+1, 2:jmax+1, 2:kmax-1)/nui)/denominator_i;


    fey(2:imax+1, 0+1, 2:kmax-1) = fey(2:imax+1, jmax+1, 2:kmax-1);
    fiy(2:imax+1, 0+1, 2:kmax-1) = fiy(2:imax+1, jmax+1, 2:kmax-1);


%        average_y(ne, ni, ey, exy, difyne, difyni, difxyne, difxyni)
    exy(2:imax, 3:jmax+1, 2:kmax-1)= 0.25*(ey(2:imax, 3:jmax+1, 2:kmax-1)+ey(2:imax, 2:jmax+1-1, 2:kmax-1)+ey(3:imax+1, 3:jmax+1, 2:kmax-1)+ey(3:imax+1, 2:jmax+1-1, 2:kmax-1)); 
    difxyne(2:imax, 3:jmax+1, 2:kmax-1)= 0.25*(difyne(2:imax, 3:jmax+1, 2:kmax-1)+difyne(2:imax, 2:jmax+1-1, 2:kmax-1)+difyne(3:imax+1, 3:jmax+1, 2:kmax-1)+difyne(3:imax+1, 2:jmax+1-1, 2:kmax-1));
    difxyni(2:imax, 3:jmax+1, 2:kmax-1)= 0.25*(difyni(2:imax, 3:jmax+1, 2:kmax-1)+difyni(2:imax, 2:jmax+1-1, 2:kmax-1)+difyni(3:imax+1, 3:jmax+1, 2:kmax-1)+difyni(3:imax+1, 2:jmax+1-1, 2:kmax-1));

    exy(2:imax, 1+1, 2:kmax-1)= 0.25*(ey(2:imax, 1+1, 2:kmax-1)+ey(2:imax, jmax+1, 2:kmax-1)+ey(3:imax+1, 1+1, 2:kmax-1)+ey(3:imax+1, jmax+1, 2:kmax-1));
    difxyne(2:imax, 1+1, 2:kmax-1)= 0.25*(difyne(2:imax, 1+1, 2:kmax-1)+difyne(2:imax, jmax+1, 2:kmax-1)+difyne(3:imax+1, 1+1, 2:kmax-1)+difyne(3:imax+1, jmax+1, 2:kmax-1));
    difxyni(2:imax, 1+1, 2:kmax-1)= 0.25*(difyni(2:imax, 1+1, 2:kmax-1)+difyni(2:imax, jmax+1, 2:kmax-1)+difyni(3:imax+1, 1+1, 2:kmax-1)+difyni(3:imax+1, jmax+1, 2:kmax-1));


    exy(imax+1, 3:jmax+1, 2:kmax-1)= 0.25*(ey(imax+1, 3:jmax+1, 2:kmax-1)+ey(imax+1, 2:jmax+1-1, 2:kmax-1)+ey(1+1, 3:jmax+1, 2:kmax-1)+ey(1+1, 2:jmax+1-1, 2:kmax-1)); 
    difxyne(imax+1, 3:jmax+1, 2:kmax-1)= 0.25*(difyne(imax+1, 3:jmax+1, 2:kmax-1)+difyne(imax+1, 2:jmax+1-1, 2:kmax-1)+difyne(1+1, 3:jmax+1, 2:kmax-1)+difyne(1+1, 2:jmax+1-1, 2:kmax-1));
    difxyni(imax+1, 3:jmax+1, 2:kmax-1)= 0.25*(difyni(imax+1, 3:jmax+1, 2:kmax-1)+difyni(imax+1, 2:jmax+1-1, 2:kmax-1)+difyni(1+1, 3:jmax+1, 2:kmax-1)+difyni(1+1, 2:jmax+1-1, 2:kmax-1));


    exy(imax+1, 1+1, 2:kmax-1)=(ey(imax+1, 1+1, 2:kmax-1)+ey(1+1, 1+1, 2:kmax-1)+ey(imax+1, jmax+1, 2:kmax-1))/3.0;
    difxyne(imax+1, 1+1, 2:kmax-1)=(difyne(imax+1, 1+1, 2:kmax-1)+difyne(1+1, 1+1, 2:kmax-1)+difyne(imax+1, jmax+1, 2:kmax-1))/3.0;
    difxyni(imax+1, 1+1, 2:kmax-1)=(difyni(imax+1, 1+1, 2:kmax-1)+difyni(1+1, 1+1, 2:kmax-1)+difyni(imax+1, jmax+1, 2:kmax-1))/3.0; 
    exy(imax+1, jmax+1, 2:kmax-1)=(ey(imax+1, jmax-1+1, 2:kmax-1)+ey(imax+1, jmax+1, 2:kmax-1)+ey(1+1, jmax-1+1, 2:kmax-1))/3.0;
    difxyne(imax+1, jmax+1, 2:kmax-1)=(difyne(imax+1, jmax-1+1, 2:kmax-1)+difyne(imax+1, jmax+1, 2:kmax-1)+difyne(1+1, jmax-1+1, 2:kmax-1))/3.0;
    difxyni(imax+1, jmax+1, 2:kmax-1)=(difyni(imax+1, jmax-1+1, 2:kmax-1)+difyni(imax+1, jmax+1, 2:kmax-1)+difyni(1+1, jmax-1+1, 2:kmax-1))/3.0; 

%        flux_x(ne, ni, fex, fix, ex, exy, difxne, difxni, difxyne, difxyni)
    fex(2:imax+1, 2:jmax+1, 2:kmax-1)=\
            (-0.5*(ne(2:imax+1, 2:jmax+1, 2:kmax-1)\
            +ne(3:imax+1+1, 2:jmax+1, 2:kmax-1))*mue\
            .*ex(2:imax+1, 2:jmax+1, 2:kmax-1)\
            -dife*difxne(2:imax+1, 2:jmax+1, 2:kmax-1)
            +wce*dife*difxyne(2:imax+1, 2:jmax+1, 2:kmax-1)\
                    /nue+wce*q*0.5*(ne(2:imax+1, 2:jmax+1, 2:kmax-1)\
                    +ne(3:imax+1+1, 2:jmax+1, 2:kmax-1))/(me*nue*nue).*exy(2:imax+1, 2:jmax+1, 2:kmax-1))/denominator_e;

    fix(2:imax+1, 2:jmax+1, 2:kmax-1)=(0.5*(ni(2:imax+1, 2:jmax+1, 2:kmax-1)+ni(3:imax+1+1, 2:jmax+1, 2:kmax-1))*mui.*ex(2:imax+1, 2:jmax+1, 2:kmax-1)-difi*difxni(2:imax+1, 2:jmax+1, 2:kmax-1)
    -wci*difi*difxyni(2:imax+1, 2:jmax+1, 2:kmax-1)/nui+wci*q*0.5*(ni(2:imax+1, 2:jmax+1, 2:kmax-1)+ni(3:imax+1+1, 2:jmax+1, 2:kmax-1)).*exy(2:imax+1, 2:jmax+1, 2:kmax-1)/(mi*nui*nui))/denominator_i;

    fex(0+1, 2:jmax+1, 2:kmax-1) = fex(imax+1, 2:jmax+1, 2:kmax-1);
    fix(0+1, 2:jmax+1, 2:kmax-1) = fix(imax+1, 2:jmax+1, 2:kmax-1);

%        flux_z(ne, ni, ez, fez, fiz, difzne, difzni)
    fez(2:imax+1, 2:jmax+1, 2:kmax-1)=-0.5*(ne(2:imax+1, 2:jmax+1, 2:kmax-1)+ne(2:imax+1, 2:jmax+1, 3:kmax-1+1))*mue.*ez(2:imax+1, 2:jmax+1, 2:kmax-1)-dife*difzne(2:imax+1, 2:jmax+1, 2:kmax-1);

    fiz(2:imax+1, 2:jmax+1, 2:kmax-1)=0.5*(ni(2:imax+1, 2:jmax+1, 2:kmax-1)+ni(2:imax+1, 2:jmax+1, 3:kmax-1+1))*mui.*ez(2:imax+1, 2:jmax+1, 2:kmax-1)-difi*difzni(2:imax+1, 2:jmax+1, 2:kmax-1);


    fez(:, :, 2) = fez(:, :, 2) .* (fez(:, :, 2) <= 0);
    fiz(:, :, 2) = fez(:, :, 2) .* (fiz(:, :, 2) <= 0);
    fez(:, :, 2) = fez(:, :, 2) .* (fez(:, :, 2) <= 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
%           functions finished!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
    ne(2:imax+1, 2:jmax+1, 2:kmax) = \
        ne(2:imax+1, 2:jmax+1, 2:kmax) -\
        dt * (\
          fex(2:imax+1, 2:jmax+1, 2:kmax  ) \
        - fex(1:imax  , 2:jmax+1, 2:kmax  ) \
        + fey(2:imax+1, 2:jmax+1, 2:kmax  ) \
        - fey(2:imax+1, 1:jmax  , 2:kmax  ) \
        + fez(2:imax+1, 2:jmax+1, 2:kmax  ) \
        - fez(2:imax+1, 2:jmax+1, 1:kmax-1))/h;
    ni(2:imax+1, 2:jmax+1, 2:kmax) = \
        ni(2:imax+1, 2:jmax+1, 2:kmax) -\
        dt * (
          fix(2:imax+1, 2:jmax+1, 2:kmax) \
        - fix(1:imax, 2:jmax+1, 2:kmax) \
        + fiy(2:imax+1, 2:jmax+1, 2:kmax) \
        - fiy(2:imax+1, 1:jmax, 2:kmax) \
        + fiz(2:imax+1, 2:jmax+1, 2:kmax) \
        - fiz(2:imax+1, 2:jmax+1, 1:kmax-1))/h;

    ne(2:imax+1, 2:jmax+1, 1) = -dt * fez(2:imax+1, 2:jmax+1, 1) / h;
    ni(2:imax+1, 2:jmax+1, 1) = -dt * fiz(2:imax+1, 2:jmax+1, 1) / h;
%        bc_densities(ne, ni)
    ne(imax+1+1, 1:jmax+1, 1:kmax) = ne(1+1, 1:jmax+1, 1:kmax);
    ni(imax+1+1, 1:jmax+1, 1:kmax) = ni(1+1, 1:jmax+1, 1:kmax);

    ne(1:jmax+1, jmax+1+1, 1:kmax) = ne(1:jmax+1, 1+1, 1:kmax);
    ni(1:jmax+1, jmax+1+1, 1:kmax) = ni(1:jmax+1, 1+1, 1:kmax);

    ne(0+1, 1:jmax+1, 1:kmax) = ne(imax+1, 1:jmax+1, 1:kmax);
    ni(0+1, 1:jmax+1, 1:kmax) = ni(imax+1, 1:jmax+1, 1:kmax);

    ne(1:jmax+1, 0+1, 1:kmax) = ne(1:jmax+1, jmax+1, 1:kmax);
    ni(1:jmax+1, 0+1, 1:kmax) = ni(1:jmax+1, jmax+1, 1:kmax);


    sf = sum(ne(1:imax+1, 1:jmax+1, 1:kmax+1)(:));
    alpha = (si -sf) / sf;
    ne(2:imax+1, 2:jmax+1, 2:kmax-1) = \
            ne(2:imax+1, 2:jmax+1, 2:kmax-1) +\
            alpha * ne(2:imax+1, 2:jmax+1, 2:kmax-1);
    printf('%d \n', time);
end
toc
