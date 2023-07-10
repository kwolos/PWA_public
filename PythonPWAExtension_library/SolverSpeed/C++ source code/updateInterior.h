#include "inflowFunctions.h"
// typedef double (*elstnc) (double, void*);
// typedef double* (*odesystem)(double, double*, void*, elstnc);


//-- DEFINITIONS OF RHS FUNCTIONS
double F(int i, vessel *P, params par) {  
   return P->Q[i]*P->Q[i]/P->A[i] + P->f[i+1]*sqrt(P->A[i]*P->A0[i+1]); //ok
}

double dBdr0(int i, vessel *P, params par) {
    return 2*sqrt(P->A[i])*(P->f[i+1]*sqrt(M_PI)+P->dfdr0[i+1]*sqrt(P->A0[i+1]))-P->dfdr0[i+1]*P->A[i]; //ok
}

double S(int i, vessel *P, params par) {
    return -8*M_PI*par.mu/(par.rho*P->A[i])*P->Q[i]*P->rc/P->qc + dBdr0(i, P, par)*P->dr0dx[i+1]; //ok
}

double Fh(int i, vessel *P, params par) {
    return P->Qnh[i]*P->Qnh[i]/P->Anh[i] + P->fh[i+1]*sqrt(P->Anh[i]*P->A0h[i+1]);//ok
}

double dBdr0h(int i, vessel *P, params par) {
    return 2*sqrt(P->Anh[i])*(P->fh[i+1]*sqrt(M_PI)+P->dfdr0h[i+1]*sqrt(P->A0h[i+1]))-P->dfdr0h[i+1]*P->Anh[i];//ok
}

double Sh(int i, vessel *P, params par) {
    return -8*M_PI*par.mu/(par.rho*P->Anh[i])*P->Qnh[i]*P->rc/P->qc + dBdr0h(i, P, par)*P->dr0dxh[i+1];//ok
}
//--
double consSin(double t){
    return (t < 0.2) ? std::sin(M_PI*t*2.5) : 1.0;
}

//-- INFLOW FUNCTION
double flow(double t, double rc, double qc, params par) {//ok
    double tm=fmod(t,par.T);
    return (par.q0*tm/pow(par.tau,2)*exp(-pow(tm,2)/(2.*pow(par.tau,2)))+par.qB)/qc;
}

//-- INFLOW FUNCTION - ELASTANCE
double* 
flow_new(double t, odesystem f1, odesystem f2, elstnc elastance, params & par, double PA){
    
    //printf("time: %f, phase: %i, isQgret: %i, isQzero: %i\n", t, par.phase, par.isQgret, par.isQZero);
    
    //printf("plv: %.2f ", plv);
    par.pa = PA; 

    double* ret = new double [4]; 

    // frist phase of isovolumic contraction
    // if the aortic and mitral valve are closed
    if (par.phase == 0){
        double x0[2] = {par.Qprev, par.Vprev};     
        // pressure in the left ventricle 
        double plv = elastance(t, &par) *(x0[1] - par.V0);
        // check if plv > pa - pressure in the aorta 
        if (plv > par.pa){ // if yes, aortic valve opens 
            // printf("t: %f, plv: %f\n", t, plv); 
            par.phase = 1; 
        }
        else { // else: nothing happens, return previous value
            ret[0] = x0[0];
            // ret[0] = (x0[0] > 0.0) ? x0[0] : 0.0;
            ret[1] = x0[1];
            ret[2] = elastance(t, &par); 
            ret[3] = plv; 
            return ret; 
        };
    };  

    if (par.phase == 1){
        // if we are here, this means that aortic valve is open 
        double* x; // = new double [2]; 
        double x0[2] = {par.Qprev, par.Vprev};          

        // we are solving system of ODE, which describes how 
        // blood fluids flux trought the heart
        x =  rk4(x0, t, par.dt, f1, 2, &par, elastance);
        
        // we have to control Q = x[0] value. If it is negative 
        // we have backflow 
        
        // backflow starts
        if(par.Qprev < -0.000001 && par.isQZero == 0){           
            par.isQZero = 1;
            par.Vfin = x[1];
        };
        
        // stop when backflow is done. If par.isQzero is not 0, 
        // we are checking if backflow > Vb. If not, nothing happens. 
        // If yes, the aortic valve closes 
        if (par.isQZero == 1){
            double val = par.Vfin - x[1]; 
            if (-val > par.Vb){
                par.isQZero = 2; 
                par.Qprev = x[0];
                par.Qparam1 = par.Qprev; 
                par.Qparam2 = t;

                ret[0] = par.Qprev;
                // ret[0] = (x[0] > 0.0) ? x[0] : 0.0;
                ret[1] = x[1];
                ret[2] = elastance(t, &par); 
                ret[3] = elastance(t, &par) *(x[1] - par.V0); 
            };
        };

        if (par.isQZero == 2){
            if(par.Qprev < -0.000001){
                
                // expotential 
                // par.Qprev = exp(100*(t - par.Qparam2)) + (par.Qparam1 - 1);           
                // hermite polynomial 
                // printf("par.l = %f\n", par.l); 
                par.Qprev = par.Qparam1*(1 - pow(t-par.Qparam2, 2)/pow(par.l, 2) + 2*pow(t-par.Qparam2,2)*(t-(par.Qparam2+par.l))/pow(par.l, 3));
                ret[0] = par.Qprev;
                // ret[0] = (x[0] > 0.0) ? x[0] : 0.0;
                ret[1] = x[1];
                ret[2] = elastance(t, &par); 
                ret[3] = elastance(t, &par) *(x[1] - par.V0);  
                delete [] x;            
                return ret;
            }
            else{
                par.phase = 2; 
                par.isQZero = 0; 
                ret[0] = 0.0;
                // ret[0] = (x[0] > 0.0) ? x[0] : 0.0;
                ret[1] = x[1];
                ret[2] = elastance(t, &par); 
                ret[3] = elastance(t, &par) *(x[1] - par.V0);    
                delete [] x;          
                return ret;
            };
        };

        par.Qprev = x[0];
        par.Vprev = x[1];

        ret[0] = par.Qprev;
        // ret[0] = (x[0] > 0.0) ? x[0] : 0.0;
        ret[1] = x[1];
        ret[2] = elastance(t, &par); 
        ret[3] = elastance(t, &par) *(x[1] - par.V0); 
        delete [] x; 
        return ret; 
        //return x[0];
    
    };
    
    if (par.phase == 2){
        double x0[2] = {par.Qprev, par.Vprev};   
        
        double plv = elastance(t, &par)*(x0[1] - par.V0);
        // printf("phase 2, plv: %f\n", plv);
        // isovolumetric relaxation is the phase, where plv decreasing. 
        // we assume at this moment, that artery pressure - pla - 
        // is constant. if the difference between pla and plv is grater 
        // than given value, mitral valve opens 
        //printf("difference pla - plv = %f\n", par.pla = plv);
        //printf("elastance: %f\n", elastance(t, &par));
        if (par.pla > plv){
            par.phase = 3; 
        }

        ret[0] = x0[0];
        // ret[0] = (x0[0] > 0.0) ? x0[0] : 0.0;
        ret[1] = x0[1];
        ret[2] = elastance(t, &par); 
        ret[3] = plv; 
        
        return ret; 
        //return x0[0];
        
    };

    if (par.phase == 3){
        // in the phase iv, the mitral valve is open. Volume of the
        // ventricle is growing. Using system of ODE we describe relation 
        // between the flux from left atrium and volume. If the Volume 
        // will be grater than limit volume + some volume related with 
        // flowback, we stop computations of the ODE. Mitral valve closes 
        // and the pase is returning to 0 i.e. phase i. 
        double* x; // = new double [2];
        double x0[2] = {par.Qlaprev, par.Vprev};

        x =  rk4(x0, t, par.dt, f2, 2, &par, elastance);
        //printf("phase 3, x = (%f, %f)\n", x[0], x[1]); 
        // if volume of the left ventricle will be grater than initial volume 
        // plus some additional volume (which is nedeed to close mitral 
        // valve), close mitral valve, and set par.phase to 0 
        // TODO: Im not sure if there will be something like backflow, 
        // but we can try with following assumption: 
        // x[0] < 0 && isQgret == 0
        if(x[1] > par.VlvInit && par.isQgret == 0){
            par.isQgret = 1;
            par.VfinUp = x[1];
        };


        // stop backflow if certain amount of backflow is reached
        if (par.isQgret == 1){
            double val = par.VfinUp - x[1];       
            if(-val > par.Vbup){
                x[1] = par.VlvInit;
                par.phase = 0;
                par.isQgret = 0; 
                //printf("heart beat finished at Vfin = %f, t = %f\n\n\n\n", par.Vfin, t);      
            };
        };

        par.Qlaprev = x[0];
        par.Vprev = x[1]; 
        //printf("Phase 3, Volume: %f, Q: %f\n", x[1], x[0]);

        // aortic valve is closed so we return par.Qprev wich should be 0 
        // or we can also return 0.0 for certanity
        ret[0] = par.Qprev;
        // ret[0] = (par.Qprev > 0.0) ? par.Qprev : 0.0;
        ret[1] = x[1];
        ret[2] = elastance(t, &par); 
        ret[3] = elastance(t, &par) *(x[1] - par.V0); 
        delete [] x; 
        return ret; 
        //return par.Qprev; 
    };


    printf("t: %f, We shouldn't be here! Phase: %i", t, par.phase);
    return 0; 
}

//-- FUNCTION THAT UPDATES INTERIOR POINTS
bool updateInterior (vessel* P, params & par, double t, bool last) {
    
    //calculate the intermediate points
    for(int i=0; i<P->N; i++) {
    		P->Anh[i] = (P->A[i]+P->A[i+1])/2.+P->dt/2.*(-(P->Q[i+1]-P->Q[i])/P->dx);
    		P->Qnh[i] = (P->Q[i]+P->Q[i+1])/2.+P->dt/2.*(-(F(i+1,P,par)-F(i,P,par))/P->dx+
                                                       (S(i+1,P,par)+S(i,P,par))/2);
    }
     
    //variables to be used in biffurcation and terminal conditions
    P->Flrh = Fh(0,P,par);
    P->Frlh = Fh(P->N-1,P,par);
    P->Slrh = Sh(0,P,par);
    P->Srlh = Sh(P->N-1,P,par); 
    
    //calculate next points
    //double Qprev = P->Q[0];
    for(int i=1; i<P->N; i++) {
        
        //if (!(Qprev==0 && P->Q[i]==0 && P->Q[i+1]==0)) {//update only if there is some flow in the stencil
            P->A[i] -= P->dt/(P->dx)*(P->Qnh[i]-P->Qnh[i-1]);
            //Qprev = P->Q[i];
            P->Q[i] -= P->dt/(P->dx)*(Fh(i,P,par)-Fh(i-1,P,par))-P->dt/2.*(Sh(i,P,par)+Sh(i-1,P,par));
        //} else {
        //   Qprev = P->Q[i]; 
        //}
    }
    
    bool terminate = false;
    if(P->isTerminal) {//updating outflow condition
        terminate = updateTerminalEnd(P,par, t);
        P->Anm_1 = P->A[P->N-1]; 
    }
    
    if(par.model == 1){
        if(P->isInitial) {//updating inflow conition
            P->Q[0] = flow(t+par.dt,P->rc,P->qc,par);
            P->A[0] -= 2.*P->dt/P->dx*(P->Qnh[0]-flow(t+par.dt/2.,P->rc,P->qc,par));

            // pressure on the inlet 
            double PV0 = (P->areaToSPressure(0, P->A[0])*pow(P->qc,2)*P->rho/pow(P->rc,4))/1333.322365; 
            if (par.message){
                printf("time: %f\r", t);
            };
        }
    }
    else if (par.model == 2) {
        if(P->isInitial){
            // pressure in ascending aorta from previous step 
            double PA = (P->areaToSPressure(1, P->A[0])*pow(P->qc,2)*P->rho/pow(P->rc,4))/1333.322365;
            
            double QprevTemp = par.Qprev; 
            double VprevTemp = par.Vprev; 

            double QprevTemp2; 
            double VprevTemp2; 
    
            // flow_new 
            double* r; // = new double [4]; 
            r = flow_new(t+par.dt/2, model, model2, elastance, par, PA);
            P->A[0] -= 2.*P->dt/P->dx*(P->Qnh[0]-r[0]/par.div); 
            delete [] r; 

            QprevTemp2 = par.Qprev;
            VprevTemp2 = par.Vprev; 

            par.Qprev = QprevTemp; 
            par.Vprev = VprevTemp;

            // flow_new
            r = flow_new(t+par.dt, model, model2, elastance, par, PA);
            P->Q[0] = r[0]/par.div;

            if (r[2] < 0.0){
                terminate = true; 
                if (par.message){
                    printf("termination at time: %f\n", t);
                };
                return terminate; 
            };

            // termination if pressure behaves not properly 
            // (huge or very negative values)
            if (PA < 0.0 || PA > 220.0){
                terminate = true; 
                if (par.message){
                    printf("termiantion at time: %f, Pressure PA = %f is < 10.0 or > 220.0\n", t, PA);
                };
                return terminate; 
            }

            // if after some time pressure is very small, also 
            // interrupt
            if (t > 2.0 && PA < 5.0){ 
                terminate = true;
                if (par.message){ 
                    printf("termiantion at time: %f, Pressure PA = %f is < 5.0\n", t, PA);
                };
                return terminate;
            }

            par.Qprev = QprevTemp2; // przywróć docelowe Qprev i Vprev
            par.Vprev = VprevTemp2;
            
            // pressure on the inlet 
            double PV0 = (P->areaToSPressure(0, P->A[0])*pow(P->qc,2)*P->rho/pow(P->rc,4))/1333.322365;

            if (par.message){
                printf("time: %f, phase: %i, par: Q = %f, V = %f, PA = %f\r", t, par.phase, par.Qprev, par.Vprev, PA);
            };

            delete [] r;
            }
    }
    else if (par.model == 3) {
        P->Q[0] = consSin(t)/P->qc; 
        P->A[0]-= 2.*P->dt/P->dx*(P->Qnh[0]-consSin(t+par.dt/2)/P->qc);

        double PV0 = (P->areaToSPressure(0, P->A[0])*pow(P->qc,2)*P->rho/pow(P->rc,4))/1333.322365;
    } 
    
    return terminate;
};
//--