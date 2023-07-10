struct rparamsT //structure with the parameters
{
    double thp;
    double gmp;
    
    double k1;
    double k3;
    double k5;
    double k7;
    
    double lp;
    double bp;
    double cp;
    double dp;
    
    double Qpn;
    double pC;
    double pT;
    double R1;
    double R2;
    double C;
    
    double tp1;
    double tp2;
};

//int rhs_terminal(const gsl_vector * x, void *pars,
//                 gsl_vector * f)
double rhs_terminal(double x, void *pars)
{
    //getting parameters
    //rewritning params
    double thp = ((struct rparamsT *) pars)->thp;
    double gmp = ((struct rparamsT *) pars)->gmp;
    
    double k1 = ((struct rparamsT *) pars)->k1;
    double k3 = ((struct rparamsT *) pars)->k3;
    double k5 = ((struct rparamsT *) pars)->k5;
    double k7 = ((struct rparamsT *) pars)->k7;
    
    double lp = ((struct rparamsT *) pars)->lp;
    double bp = ((struct rparamsT *) pars)->bp;
    double cp = ((struct rparamsT *) pars)->cp;
    double dp = ((struct rparamsT *) pars)->dp;
    
    double Qpn = ((struct rparamsT *) pars)->Qpn;
    double pC  = ((struct rparamsT *) pars)->pC;
    double pT  = ((struct rparamsT *) pars)->pT;
    double R1  = ((struct rparamsT *) pars)->R1;
    double R2  = ((struct rparamsT *) pars)->R2;
    double C   = ((struct rparamsT *) pars)->C;
    
    double tp1 = ((struct rparamsT *) pars)->tp1;
    double tp2 = ((struct rparamsT *) pars)->tp2;
    
    //getiing current values
    //const double x0  = gsl_vector_get (x, 0);
    //const double x1  = gsl_vector_get (x, 1);
    const double x5  = x;//gsl_vector_get (x, 0);
    //const double x3  = gsl_vector_get (x, 3);
    //const double x4  = gsl_vector_get (x, 4);
    //const double x5  = gsl_vector_get (x, 5);
    
    const double x4 =  k7 + x5/2.; //linear
    if (x4 < 0 )
        return 0;
    
    //I need to translate area to pressure
    double pCnh = pC+gmp/C*(Qpn - (pC-pT)/R2);
    const double x1 = (tp1-tp2/sqrt(x4)-pCnh)/R1;
    
    //const double f2 = -x1 + k5 + x2/2.; //linear
    const double x2 = (x1 - k5)*2.; //linear
    
    const double x3 = k3 -thp*x2; //linear
    
    double pCn1 = pC+2.*gmp/C*(x1 - (pCnh-pT)/R2);
    const double x0 = (tp1-tp2/sqrt(x3)-pCn1)/R1;
    if (x3<0 || x5<0)
        return 0;
    
    const double f0 = k1 -x0-thp*(x2*x2/x5+ lp*sqrt(x5))+gmp*(bp* x2/x5 +sqrt(x5)*cp-dp*x5);
    
    return f0;
}

double rhs_terminal_deriv(double x, void *pars)
{
    //getting parameters
    //rewritning params
    double thp = ((struct rparamsT *) pars)->thp;
    double gmp = ((struct rparamsT *) pars)->gmp;
    
    double k1 = ((struct rparamsT *) pars)->k1;
    double k3 = ((struct rparamsT *) pars)->k3;
    double k5 = ((struct rparamsT *) pars)->k5;
    double k7 = ((struct rparamsT *) pars)->k7;
    
    double lp = ((struct rparamsT *) pars)->lp;
    double bp = ((struct rparamsT *) pars)->bp;
    double cp = ((struct rparamsT *) pars)->cp;
    double dp = ((struct rparamsT *) pars)->dp;
    
    double Qpn = ((struct rparamsT *) pars)->Qpn;
    double pC  = ((struct rparamsT *) pars)->pC;
    double pT  = ((struct rparamsT *) pars)->pT;
    double R1  = ((struct rparamsT *) pars)->R1;
    double R2  = ((struct rparamsT *) pars)->R2;
    double C   = ((struct rparamsT *) pars)->C;
    
    double tp1 = ((struct rparamsT *) pars)->tp1;
    double tp2 = ((struct rparamsT *) pars)->tp2;
    
    //getiing current values
    const double x5  = x;//gsl_vector_get (x, 0);
    
    const double x4 =  k7 + x5/2.; //linear
    const double dx4dx5 = 0.5;
    
    //I need to translate area to pressure
    double pCnh = pC+gmp/C*(Qpn - (pC-pT)/R2);
    const double x1 = (tp1-tp2/sqrt(x4)-pCnh)/R1;
    const double dx1dx5 = (0.5*tp2/pow(sqrt(x4),3))/R1*dx4dx5;
    
    //const double f2 = -x1 + k5 + x2/2.; //linear
    const double x2 = (x1 - k5)*2.; //linear
    const double dx2dx5 = dx1dx5*2.; //linear
    
    const double x3 = k3 -thp*x2; //linear
    const double dx3dx5 = -thp*dx2dx5; //linear
    
    double pCn1 = pC+2.*gmp/C*(x1 - (pCnh-pT)/R2);
    double dpCn1dx5 = 2.*gmp/C*dx1dx5;
    
    const double x0 = (tp1-tp2/sqrt(x3)-pCn1)/R1;
    const double dx0dx5 = (0.5*tp2/pow(sqrt(x3),3)*dx3dx5-dpCn1dx5)/R1;
    
    const double f0 = k1 -x0-thp*(x2*x2/x5+ lp*sqrt(x5))+gmp*(bp* x2/x5 +sqrt(x5)*cp-dp*x5);
    const double df0dx5 = -dx0dx5-thp*(2.*x2*dx2dx5/x5-x2*x2/x5/x5+0.5*lp/sqrt(x5))+
                          gmp*(bp* dx2dx5/x5-bp*x2/x5/x5 +0.5/sqrt(x5)*cp-dp);
    
    return df0dx5;
}

bool updateTerminalEnd(vessel* P, params par, double t) {
    //this function updates the inflow sondition
    
    int Np = P->N;
    
    if (P->Q[Np-1] != 0 ) {//if there is any flow
        
        double thp = P->dt/P->dx;//theta
        double gmp = P->dt/2;//gamma;
        double k1   = P->Q[Np]+thp*P->Frlh + gmp*P->Srlh; //ok
        double k3   = P->A[Np]+thp*P->Qnh[Np-1]; //ok
        double k5   = P->Qnh[Np-1]/2; //ok
        double k7   = P->Anh[Np-1]/2; //ok
        
        double lp = P->fh[Np+1]*sqrt(P->A0h[Np+1]);
        double bp = -8*M_PI*par.mu/par.rho*P->rc/P->qc;
        double cp = 2*P->dr0dxh[Np+1]*(P->fh[Np+1]*sqrt(M_PI)+P->dfdr0h[Np+1]*sqrt(P->A0h[Np+1]));
        double dp = P->dr0dxh[Np+1]*P->dfdr0h[Np+1];
        
        double Qpn = P->Q[Np];
        double pC  = P->pC;
        double pT  = P->pT;
        double R1  = P->R1;
        double R2  = P->R2;
        double C   = P->C;
        
        double tp1 = P->f[Np+1];
        double tp2 = P->f[Np+1]*sqrt(P->A0[Np+1]);
        
        struct rparamsT p = {thp,gmp,k1,k3,k5,k7,lp,bp,cp,dp,Qpn,pC,pT,R1,R2,C,tp1,tp2};
        
        
        int status;
        size_t iter = 0;
        
        double xt, x = P->A[Np];
        
        
        double f = rhs_terminal(x,&p);
        double th = 1.;
        double df = rhs_terminal_deriv(x,&p);
        
        
        do
        {
            iter++;
            //printf("Iter:%d, x = %e, f = %e, df = %e\n",iter,x, f, df);
            xt = x;
            x -= th*f/df;
            
            f = rhs_terminal(x,&p);
            df = rhs_terminal_deriv(x,&p);
            
            if (f == 0) {
                x = xt;
                th/=2.;
                
                f = rhs_terminal(x,&p);
                df = rhs_terminal_deriv(x,&p);
            } else {
                
                if (std::abs(f) < 1e-7) {
                    //printf("Iter:fin, x = %e, f = %e, df = %e\n",x, f, df);
                    status = GSL_SUCCESS;
                    break;
                }
            }
            
        }
        while (iter < 20);
        
        if(status!=GSL_SUCCESS){
            //printf("Warning: lack of convergence at terminal end, ID = %d, time = %f, err = %f.\n", P->ID,t,std::abs(f));
            //return true;
        }
        
        //assigning the output
        const double x5  = x;
        
        const double x4 =  k7 + x5/2.; //linear
        
        //I need to translate area to pressure
        double pCnh = pC+gmp/C*(Qpn - (pC-pT)/R2);
        const double x1 = (tp1-tp2/sqrt(x4)-pCnh)/R1;
        
        //const double f2 = -x1 + k5 + x2/2.; //linear
        const double x2 = (x1 - k5)*2.; //linear
        
        const double x3 = k3 -thp*x2; //linear
        
        double pCn1 = pC+2.*gmp/C*(x1 - (pCnh-pT)/R2);
        const double x0 = (tp1-tp2/sqrt(x3)-pCn1)/R1;
        
        P->A[Np]  = x3;
        P->Q[Np]  = x0;
        P->pC +=2.*gmp/C*(x1 - (pCnh-pT)/R2);
        
    }
    
    return false;
}