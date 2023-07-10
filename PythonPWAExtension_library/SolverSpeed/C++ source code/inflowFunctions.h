// #include <gsl/gsl_errno.h>
// #include <gsl/gsl_roots.h>
// #include <gsl/gsl_math.h>
// #include "params.h"


typedef double (*elstnc) (double, void*);  
typedef double* (*odesystem)(double, double*, void*, elstnc);
typedef double (*fun)(double, void*);

double Fint(double t, void* par);
double elastance(double t, params par);
double FindZero(double x_lo, fun F, params par);
double* rk4(double* x0, double t0, double h, odesystem f, int N, void* params, elstnc elastance);
double* model(double t, double* y0, void *params, elstnc elast);
double* model2(double t, double* y0, void *params, elstnc elast);

double FindZero(double x_lo, fun F, params par){
    // find the x_hi near x_lo, where F(x_hi) has the opposite sign 
    
    double x_hi = 0; 
    int iter = 0, max_iter = 100; 
    
    do {
        x_hi += 0.05;
    } while (F(x_hi, &par) * F(x_lo, &par) > 0 && iter < max_iter);
    

    const gsl_root_fsolver_type *T; 
    gsl_root_fsolver *s; 
    gsl_function G; 

    double r; 

    G.function = F; 
    G.params = &par;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T); 
    gsl_root_fsolver_set(s, &G, x_lo, x_hi);
    
    int status; 

    do {
        iter ++;        
        status = gsl_root_fsolver_iterate(s); 
        r = gsl_root_fsolver_root(s);
        x_lo = gsl_root_fsolver_x_lower(s); 
        x_hi = gsl_root_fsolver_x_upper(s); 
        status = gsl_root_test_interval(x_lo, x_hi, 0, 0.001);

        // if (status == GSL_SUCCESS)
        //     printf("Converged:\n");

        //printf ("%5d [%.7f, %.7f] %.7f\n",
        //iter, x_lo, x_hi, r);

    } while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free(s);

    return r; 
}


// one step of runge kutta method (ode45)
double * 
rk4(double* x0, double t0, double h, odesystem f, int N, void* params,
        elstnc elastance){

    class params *par 
        = (class params *) params;  
    //printf("parameters of rk45: x0: (%f, %f, %f), t0: %.2f, dt: %.2f\n", x0[0], x0[1], x0[2], t0, h); 
    //printf("parameters of param: par.V0: %f, par.pa: %f, par.R: %f\n", par->V0, par->pa, par->R);
    //printf("par.Llv: %f, par.tm: %f, par.Emin: %f, par.Emax: %f, par.a: %f\n", par->Llv, par->tm, par->Emin, par->Emax, par->a);
    
    double k1[N], k2[N], k3[N], k4[N]; 
    double arg2[N], arg3[N], arg4[N]; 
    double* fTemp; //= new double [N];
    double* x = new double [N]; 
    double plv; 
    double value; 
    
    fTemp = f(t0, x0, params, elastance);     
    for (int i = 0; i<N; i++){
        k1[i] = fTemp[i];
        arg2[i] = x0[i] + 0.5*h*k1[i];
    };
    delete [] fTemp;
    
    fTemp = f(t0 + 0.5*h, arg2, params, elastance);
    for (int i=0; i<N; i++){
        k2[i] = fTemp[i];
        arg3[i] = x0[i] + 0.5*h*k2[i];
    };
    delete [] fTemp;

    fTemp = f(t0 + 0.5*h, arg3, params, elastance); 
    for (int i=0; i<N; i++){
        k3[i] = fTemp[i];
        arg4[i] = x0[i] + h*k3[i]; 
    };
    delete [] fTemp;

    fTemp = f(t0 + h, arg3, params, elastance); 
    for (int i=0; i<N; i++){
        k4[i] = fTemp[i];
        x[i] = x0[i] + (h/6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    };
    delete [] fTemp; 
    //printf("t: %f, returning value x: (%f, %f, %f)\n", t0, k3[0], k3[1], k3[2]); 
     
    return x; 
};


double 
elastance(double t, void *params){     
    class params *par 
        = (class params *) params; 

    t = fmod(t, par->Time); 
    double phi; 
    if(t > par->tm){
        phi = 0; 
    }
    else {
        phi = par->a*sin(M_PI*t/par->tm)-par->b*sin(2*M_PI*t/par->tm);
    };

//     printf("elast in elast: %f\n", par->Emin * (1 - phi) + par->Emax * phi);
    double var; 
    double eps = 0.0; 
    var = par->Emin*(1 - phi) + par->Emax * phi; 

    if (var < eps){
        printf("\nE(t) = %.2f < 0, computation stopped", var);
    }
    
    return var; 
}


double 
FInt(double t, void *params){
    struct params *par 
        = (struct params *) params; 
    
    return (elastance(t, params) * (par->VlvInit - par->V0) - par->pa); 
}


double *
model(double t, double* y0, void *params, elstnc elast){
    double* y = new double [2]; 
    double plv; 
    
    struct params *par
        = (struct params *) params;

    plv = elast(t, params) *(y0[1] - par->V0);
    y[0] = 1/par->Llv * (plv - (par->pa + par->R*y0[0]));
    // y[0] = 1/par->Llv * (plv - par->pa);
    y[1] = - y0[0];
//     printf("model; plv: %f, par.Llv: %f, par.pa: %f, par.R: %f\n", plv, par->Llv, par->pa, par->R);   
    return y;
}


double *
model2(double t, double *y0, void *params, elstnc elast){
    double* y = new double [2];
    double plv; 

    struct params *par 
        = (struct params *) params;
    
    plv = elast(t, params) * (y0[1] - par->V0); 
    y[0] = 1/par->Lla * (par->pla - plv) - par->Rla*y0[0]/par->Lla; 
    y[1] = y0[0] - par->Qprev;
    //printf("model 2. plv = %f, par.La: %f, par.pla: %f, par.Rla = %f, par.Qprev = %f", plv, par->Lla, par->pla, par->Rla, par->Qprev);
    return y;
}


double * 
model3(double t, double *y0, void *params, elstnc elast){
    struct params *par 
        = (struct params *) params;
    
    double * y = new double [3]; 
    double plv = elast(t, params) * (y0[1] - par->V0); 
    double Deltapm = par->pla - plv;
    
    double AeoMax = par->Mst * par->A; 
    double AeoMin = par->Mrg * par->A; 

    double Aeo = (AeoMax - AeoMin) * y0[2] + AeoMin;   // eq. (6.12)
    double Aeff = 1/(1/Aeo - 1/par->A);  // eq. (6.7->
    
    double Lm = 4 * M_PI * par->rho * sqrt(1/Aeo - 1/par->A);  // eq. (6.9)
    double Bm = par->rho/(2*Aeff); // eq. (6.8)
    
    // system of equations (7.1a -- 7.1b)
    y[0] = 1/Lm * (par->pla - plv) - Bm * y0[0] * abs(y0[0]);  //Q_la
    y[1] = y0[0] - par->Qprev;  // V_la 

    if (Deltapm > par->plao){   // zeta
        y[2] = (1 - y[2]) * par->Kvo * (Deltapm - par->plao);
    }
    else if (Deltapm < par->plac){
        y[2] = y[2] * par->Kvc * (Deltapm - par->plac);
    }
    else {
        y[2] = 0; 
    }
    printf("Phase 3. t: %f, y = (%f, %f, %f)\n", t, y[0], y[1], y[2]);
    return y; 
} 