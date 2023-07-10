struct rparams
{
    double thp;
    double thd1;
    double thd2;
    double gmp;
    double gmd1;
    double gmd2;
    
    double k1;
    double k2;
    double k2a;
    double k3;
    double k4;
    double k4a;
    double k5;
    double k6;
    double k6a;
    double k7;
    double k8;
    double k8a;
    
    double k9;
    double k9a;
    double k10;
    double k11;
    double k11a;
    
    double qcp;
    double qcd1;
    double qcd2;
    
    double lp;
    double bp;
    double cp;
    double dp;
    
    double ld1;
    double bd1;
    double cd1;
    double dd1;
    
    double ld2;
    double bd2;
    double cd2;
    double dd2;
};

int rhs(const gsl_vector * x, void *pars,
        gsl_vector * f)
{
    //rewritning params
    double thp = ((struct rparams *) pars)->thp;
    double thd1 = ((struct rparams *) pars)->thd1;
    double thd2 = ((struct rparams *) pars)->thd2;
    double gmp = ((struct rparams *) pars)->gmp;
    double gmd1 = ((struct rparams *) pars)->gmd1;
    double gmd2 = ((struct rparams *) pars)->gmd2;
    
    double k1 = ((struct rparams *) pars)->k1;
    double k2 = ((struct rparams *) pars)->k2;
    double k2a = ((struct rparams *) pars)->k2a;
    double k3 = ((struct rparams *) pars)->k3;
    double k4 = ((struct rparams *) pars)->k4;
    double k4a = ((struct rparams *) pars)->k4a;
    double k5 = ((struct rparams *) pars)->k5;
    double k6 = ((struct rparams *) pars)->k6;
    double k6a = ((struct rparams *) pars)->k6a;
    double k7 = ((struct rparams *) pars)->k7;
    double k8 = ((struct rparams *) pars)->k8;
    double k8a = ((struct rparams *) pars)->k8a;
    
    double k9 = ((struct rparams *) pars)->k9;
    double k9a = ((struct rparams *) pars)->k9a;
    double k10 = ((struct rparams *) pars)->k10;
    double k11 = ((struct rparams *) pars)->k11;
    double k11a = ((struct rparams *) pars)->k11a;
    
    double qcp = ((struct rparams *) pars)->qcp;
    double qcd1 = ((struct rparams *) pars)->qcd1;
    double qcd2 = ((struct rparams *) pars)->qcd2;
    
    double lp = ((struct rparams *) pars)->lp;
    double bp = ((struct rparams *) pars)->bp;
    double cp = ((struct rparams *) pars)->cp;
    double dp = ((struct rparams *) pars)->dp;
    
    double ld1 = ((struct rparams *) pars)->ld1;
    double bd1 = ((struct rparams *) pars)->bd1;
    double cd1 = ((struct rparams *) pars)->cd1;
    double dd1 = ((struct rparams *) pars)->dd1;
    
    double ld2 = ((struct rparams *) pars)->ld2;
    double bd2 = ((struct rparams *) pars)->bd2;
    double cd2 = ((struct rparams *) pars)->cd2;
    double dd2 = ((struct rparams *) pars)->dd2;
    
    
    //getiing current values
    //const double x0  = gsl_vector_get (x, 0);
    //const double x1  = gsl_vector_get (x, 1);
    const double x9  = gsl_vector_get (x, 0);
    //const double x3  = gsl_vector_get (x, 3);
    //const double x4  = gsl_vector_get (x, 4);
    //const double x5  = gsl_vector_get (x, 1);
    //const double x6  = gsl_vector_get (x, 6);
    //const double x7  = gsl_vector_get (x, 7);
    //const double x8  = gsl_vector_get (x, 2);
    //const double x9  = gsl_vector_get (x, 9);
    //const double x10 = gsl_vector_get (x, 10);
    const double x11 = gsl_vector_get (x, 1);
    //const double x12 = gsl_vector_get (x, 12);
    //const double x13 = gsl_vector_get (x, 13);
    //const double x14 = gsl_vector_get (x, 4);
    //const double x15 = gsl_vector_get (x, 15);
    //const double x16 = gsl_vector_get (x, 16);
    //const double x17 = gsl_vector_get (x, 5);
    
    
    //calculating RHSs

    
    //const double f3 = -x9 +k3 -thp*x2; //linear
    //const double x9 = k3 -thp*x2; //linear
    const double x2 = (-x9+k3)/thp;
    
    //const double f14 = k10/sqrt(x9)-k11/sqrt(x12) +k9; //k10*sqrt(x12)-k11*sqrt(x9)+k9*sqrt(x9*x12);//
    const double x12 = pow(k11/(k10/sqrt(x9)+k9),2);
    //const double f15 = k10/sqrt(x9)-k11a/sqrt(x15)+k9a; //k10*sqrt(x15)-k11a*sqrt(x9)+k9a*sqrt(x9*x15);//
    const double x15 = pow(k11a/(k10/sqrt(x9)+k9a),2);
    
    //const double f4 = -x12+k4 +thd1*x5; //linear
    //const double x12 = k4 +thd1*x5; //linear
    const double x5 = (x12-k4)/thd1;
    //const double f5 = -x15+k4a+thd2*x8; //linear
    //const double x15 = k4a+thd2*x8; //linear
    const double x8 = (x15-k4a)/thd2;
    
    //const double f6 = -x1+k5+ x2/2; //linear
    const double x1 = k5+ x2/2.; //linear
    //const double f7 = -x4+k6+ x5/2; //linear
    const double x4 = k6+ x5/2.; //linear
    //const double f8 = -x7+k6a+x8/2; //linear
    const double x7 = k6a+x8/2.; //linear
    
    //const double f9  = -x10+k7+ x11/2; //linear
    const double x10 = k7+ x11/2.; //linear
    //const double f16 = k10/sqrt(x10)-k11/sqrt(x13)+k9;  //k10*sqrt(x13)-k11*sqrt(x10)+k9*sqrt(x10*x13);//
    const double x13 = pow(k11/(k10/sqrt(x10) +k9),2);
    //const double f17 = k10/sqrt(x10)-k11a/sqrt(x16)+k9a; //k10*sqrt(x16)-k11a*sqrt(x10)+k9a*sqrt(x10*x16);//
    const double x16 = pow(k11a/(k10/sqrt(x10) +k9a),2);
    //const double f10 = -x13+k8+ x14/2; //linear
    //const double x13 = k8+ x14/2.; //linear
    const double x14 = (x13-k8)*2.;
    //const double f11 = -x16+k8a+x17/2; //linear
    //const double x16 = k8a+x17/2.; //linear
    const double x17 = (x16-k8a)*2.;
    
    
    //const double f0 = k1 -x0-thp*(x2*x2/x11+ lp*sqrt(x11))+gmp*(bp* x2/x11 +sqrt(x11)*cp-dp*x11);
    const double x0 = k1-thp*(x2*x2/x11+ lp*sqrt(x11))+gmp*(bp* x2/x11 +sqrt(x11)*cp-dp*x11);
    //const double f1 = k2 -x3+thd1*(x5*x5/x14+ld1*sqrt(x14))+gmd1*(bd1*x5/x14+sqrt(x14)*cd1-dd1*x14);
    const double x3 = k2+thd1*(x5*x5/x14+ld1*sqrt(x14))+gmd1*(bd1*x5/x14+sqrt(x14)*cd1-dd1*x14);
    //const double f2 = k2a-x6+thd2*(x8*x8/x17+ld2*sqrt(x17))+gmd2*(bd2*x8/x17+sqrt(x17)*cd2-dd2*x17);
    const double x6 = k2a+thd2*(x8*x8/x17+ld2*sqrt(x17))+gmd2*(bd2*x8/x17+sqrt(x17)*cd2-dd2*x17);
    

    const double f12  = x0*qcp-x3*qcd1-x6*qcd2; //linear
    const double f13  = x1*qcp-x4*qcd1-x7*qcd2; //linear
    //setting output vector
    /*gsl_vector_set (f, 0, f0);
    gsl_vector_set (f, 1, f1);
    gsl_vector_set (f, 2, f2);
    gsl_vector_set (f, 3, f3);
    gsl_vector_set (f, 4, f4);
    gsl_vector_set (f, 5, f5);
    gsl_vector_set (f, 6, f6);
    gsl_vector_set (f, 7, f7);
    gsl_vector_set (f, 8, f8);
    gsl_vector_set (f, 9, f9);
    gsl_vector_set (f, 10, f10);
    gsl_vector_set (f, 11, f11);*/
    gsl_vector_set (f, 0, f12);
    gsl_vector_set (f, 1, f13);
    //gsl_vector_set (f, 2, f14);
    //gsl_vector_set (f, 3, f15);
    //gsl_vector_set (f, 4, f16);
    //gsl_vector_set (f, 5, f17);
    
    return GSL_SUCCESS;
}

int
print_state (size_t iter, gsl_multiroot_fsolver * s)
{
    printf ("iter = %3u x = % .3f % .3f\n"
            " f(x) = % .3e % .3e\n",
            iter,
            gsl_vector_get (s->x, 0),
            gsl_vector_get (s->x, 1),
            gsl_vector_get (s->f, 0),
            gsl_vector_get (s->f, 1));
    return 0;
}

bool updateBranchingPoint(vessel* P, vessel* D1, vessel*D2, params par, double t, double c) {
    //this function updates the inflow sondition
    int Np = P->N;
    int Nd2 = D2->N;

    int D2idx = (c > 0.5) ? Nd2 : 0;
    int D2idxPlus0 = (c > 0.5) ? Nd2+1 : 0;
    int D2idxPlus  = (c > 0.5) ? Nd2+1 : 1; 
    int D2idxMinus = (c > 0.5) ? Nd2-1 : 0;
    int D2idxMinus1 = (c > 0.5) ? Nd2-1 : 1;
    double sign = (c > 0.5) ? -1 : 1;
    double D2_Fxh = (c > 0.5) ? D2->Frlh : D2->Flrh;
    double D2_Sxh = (c > 0.5) ? D2->Srlh : D2->Slrh;
    
    //if (!(P->Q[Np-1]==0 && D1->Q[1]==0 && D2->Q[1]==0)) { //check if there is any flow

        double thp = P->dt/P->dx;//theta
        double thd1 = D1->dt/D1->dx;//theta
        double thd2 = D2->dt/D2->dx;//theta
        double gmp = P->dt/2;//gamma;
        double gmd1 = D1->dt/2;//gamma;
        double gmd2 = D2->dt/2;//gamma;
        
        //calculating equation parameters
        double k1   = P->Q[Np]+thp*P->Frlh + gmp*P->Srlh; //ok
        double k2   = D1->Q[0]-thd1*D1->Flrh + gmd1*D1->Slrh; //ok
        double k2a  = sign*D2->Q[D2idxMinus] - thd2*D2_Fxh + gmd2*D2_Sxh; //ok

        double k3   = P->A[Np]+thp*P->Qnh[Np-1]; //ok
        double k4   = D1->A[0]-thd1*D1->Qnh[0];  //ok
        double k4a  = D2->A[D2idx] - sign*thd2*D2->Qnh[D2idxMinus];  //ok

        double k5   = P->Qnh[Np-1]/2; //ok
        double k6   = D1->Qnh[0]/2; //ok
        double k6a  = sign*D2->Qnh[D2idxMinus]/2; //ok

        double k7   = P->Anh[Np-1]/2; //ok
        double k8   = D1->Anh[0]/2; //ok
        double k8a  = D2->Anh[D2idxMinus]/2; //ok

        double k9   = -P->f[Np+1]*pow(P->qc,2)/pow(P->rc,4)*P->rho+D1->f[1]*pow(D1->qc,2)/pow(D1->rc,4)*D1->rho; //ok
        double k9a  = -P->f[Np+1]*pow(P->qc,2)/pow(P->rc,4)*P->rho+D2->f[D2idxPlus]*pow(D2->qc,2)/pow(D2->rc,4)*D2->rho; //ok

        double k10  = P->f[Np+1]*sqrt(P->A0[Np+1])*pow(P->qc,2)/pow(P->rc,4)*P->rho; //ok
        double k11  = D1->f[1]*sqrt(D1->A0[1])*pow(D1->qc,2)/pow(D1->rc,4)*D1->rho; //ok
        double k11a = D2->f[D2idxPlus]*sqrt(D2->A0[D2idxPlus])*pow(D2->qc,2)/pow(D2->rc,4)*D2->rho; //ok

        double qcp  = P->qc; //ok
        double qcd1 = D1->qc; //ok
        double qcd2 = D2->qc; //ok
        
        double lp = P->fh[Np+1]*sqrt(P->A0h[Np+1]);
        double bp = -8*M_PI*par.mu/par.rho*P->rc/P->qc;
        double cp = 2*P->dr0dxh[Np+1]*(P->fh[Np+1]*sqrt(M_PI)+P->dfdr0h[Np+1]*sqrt(P->A0h[Np+1]));
        double dp = P->dr0dxh[Np+1]*P->dfdr0h[Np+1];
        
        double ld1 = D1->fh[0]*sqrt(D1->A0h[0]);
        double bd1 = -8*M_PI*par.mu/par.rho*D1->rc/D1->qc;
        double cd1 = 2*D1->dr0dxh[0]*(D1->fh[0]*sqrt(M_PI)+D1->dfdr0h[0]*sqrt(D1->A0h[0]));
        double dd1 = D1->dr0dxh[0]*D1->dfdr0h[0];
        
        double ld2 = D2->fh[D2idxPlus0]*sqrt(D2->A0h[D2idxPlus0]);
        double bd2 = -8*M_PI*par.mu/par.rho*D2->rc/D2->qc;
        double cd2 = 2*D2->dr0dxh[D2idxPlus0]*(D2->fh[D2idxPlus0]*sqrt(M_PI)+D2->dfdr0h[D2idxPlus0]*sqrt(D2->A0h[D2idxPlus0]));
        double dd2 = D2->dr0dxh[D2idxPlus0]*D2->dfdr0h[D2idxPlus0];
        
        
        struct rparams p = {thp,thd1,thd2, gmp, gmd1, gmd2, k1, k2, k2a, k3, k4, k4a, k5, k6, k6a, k7, k8, k8a,k9,k9a,k10,k11,k11a,
            qcp,qcd1,qcd2,
            lp,bp,cp,dp,ld1,bd1,cd1,dd1,ld2,bd2,cd2,dd2};
        
        
        const gsl_multiroot_fsolver_type *T;
        gsl_multiroot_fsolver *s;
        
        int status;
        size_t iter = 0;
        
        const size_t n = 2;
        
        gsl_multiroot_function f = {&rhs, n, &p};
        
        //--- defining initial condition
        gsl_vector *x = gsl_vector_alloc (n);
        
        //gsl_vector_set (x, 0, P->Q[P->N]);
        //gsl_vector_set (x, 1, P->Q[P->N]);
        //gsl_vector_set (x, 0, P->Q[P->N-1]);
        
        //gsl_vector_set (x, 3,  D1->Q[0]);
        //gsl_vector_set (x, 4,  D1->Q[0]);
        //gsl_vector_set (x, 5,  D1->Q[0]);
        
        //gsl_vector_set (x, 6,  D2->Q[0]);
        //gsl_vector_set (x, 7,  D2->Q[0]);
        //gsl_vector_set (x, 8,  D2->Q[0]);
        
        
        gsl_vector_set (x, 0,  P->A[P->N]);
        //gsl_vector_set (x, 10, P->A[P->N]);
        gsl_vector_set (x, 1, P->A[P->N]);
        
        //gsl_vector_set (x, 12, D1->A[0]);
        //gsl_vector_set (x, 13, D1->A[0]);
        //gsl_vector_set (x, 14, D1->A[0]);
        
        //gsl_vector_set (x, 15, D2->A[0]);
        //gsl_vector_set (x, 16, D2->A[0]);
        //gsl_vector_set (x, 17, D2->A[0]);
        //--
        
        T = gsl_multiroot_fsolver_hybrids;
        s = gsl_multiroot_fsolver_alloc (T, n);
        gsl_multiroot_fsolver_set (s, &f, x);
        
        
        do
        {
            iter++;
            
            status = gsl_multiroot_fsolver_iterate (s);
            
            
            if (status)
                break;
            
            status = gsl_multiroot_test_residual (s->f, 1e-10);
        }
        while (status == GSL_CONTINUE && iter < 20);
        if(status!=GSL_SUCCESS) {
            double err = 0;
            for(int i = 0; i<2; i++)
                err+=std::abs(gsl_vector_get (s->f, i));
            
            //printf("Warning: lack of convergence at biffurcation, ID = %d, time = %f, err = %e\n", P->ID,t,err);
            //return true;
            
        }
        
        
        //getiing current values
        //const double x0  = gsl_vector_get (x, 0);
        //const double x1  = gsl_vector_get (x, 1);
        const double x9  = gsl_vector_get (s->x, 0);
        //const double x3  = gsl_vector_get (x, 3);
        //const double x4  = gsl_vector_get (x, 4);
        //const double x5  = gsl_vector_get (s->x, 1);
        //const double x6  = gsl_vector_get (x, 6);
        //const double x7  = gsl_vector_get (x, 7);
        //const double x8  = gsl_vector_get (s->x, 2);
        //const double x9  = gsl_vector_get (x, 9);
        //const double x10 = gsl_vector_get (x, 10);
        const double x11 = gsl_vector_get (s->x, 1);
        //const double x12 = gsl_vector_get (x, 12);
        //const double x13 = gsl_vector_get (x, 13);
        //const double x14 = gsl_vector_get (s->x, 4);
        //const double x15 = gsl_vector_get (x, 15);
        //const double x16 = gsl_vector_get (x, 16);
        //const double x17 = gsl_vector_get (s->x, 5);
        
        //const double f3 = -x9 +k3 -thp*x2; //linear
        //const double x9 = k3 -thp*x2; //linear
        const double x2 = (-x9+k3)/thp;
        
        //const double f14 = k10/sqrt(x9)-k11/sqrt(x12) +k9; //k10*sqrt(x12)-k11*sqrt(x9)+k9*sqrt(x9*x12);//
        const double x12 = pow(k11/(k10/sqrt(x9)+k9),2);
        //const double f15 = k10/sqrt(x9)-k11a/sqrt(x15)+k9a; //k10*sqrt(x15)-k11a*sqrt(x9)+k9a*sqrt(x9*x15);//
        const double x15 = pow(k11a/(k10/sqrt(x9)+k9a),2);
        
        //const double f4 = -x12+k4 +thd1*x5; //linear
        //const double x12 = k4 +thd1*x5; //linear
        const double x5 = (x12-k4)/thd1;
        //const double f5 = -x15+k4a+thd2*x8; //linear
        //const double x15 = k4a+thd2*x8; //linear
        const double x8 = (x15-k4a)/thd2;
        
        //const double f6 = -x1+k5+ x2/2; //linear
        const double x1 = k5+ x2/2.; //linear
        //const double f7 = -x4+k6+ x5/2; //linear
        const double x4 = k6+ x5/2.; //linear
        //const double f8 = -x7+k6a+x8/2; //linear
        const double x7 = k6a+x8/2.; //linear

        //const double f9  = -x10+k7+ x11/2; //linear
        const double x10 = k7+ x11/2.; //linear
        //const double f16 = k10/sqrt(x10)-k11/sqrt(x13)+k9;  //k10*sqrt(x13)-k11*sqrt(x10)+k9*sqrt(x10*x13);//
        const double x13 = pow(k11/(k10/sqrt(x10) +k9),2);
        //const double f17 = k10/sqrt(x10)-k11a/sqrt(x16)+k9a; //k10*sqrt(x16)-k11a*sqrt(x10)+k9a*sqrt(x10*x16);//
        const double x16 = pow(k11a/(k10/sqrt(x10) +k9a),2);
        //const double f10 = -x13+k8+ x14/2; //linear
        //const double x13 = k8+ x14/2.; //linear
        const double x14 = (x13-k8)*2.;
        //const double f11 = -x16+k8a+x17/2; //linear
        //const double x16 = k8a+x17/2.; //linear
        const double x17 = (x16-k8a)*2.;
        
        
        //const double f0 = k1 -x0-thp*(x2*x2/x11+ lp*sqrt(x11))+gmp*(bp* x2/x11 +sqrt(x11)*cp-dp*x11);
        const double x0 = k1-thp*(x2*x2/x11+ lp*sqrt(x11))+gmp*(bp* x2/x11 +sqrt(x11)*cp-dp*x11);
        //const double f1 = k2 -x3+thd1*(x5*x5/x14+ld1*sqrt(x14))+gmd1*(bd1*x5/x14+sqrt(x14)*cd1-dd1*x14);
        const double x3 = k2+thd1*(x5*x5/x14+ld1*sqrt(x14))+gmd1*(bd1*x5/x14+sqrt(x14)*cd1-dd1*x14);
        //const double f2 = k2a-x6+thd2*(x8*x8/x17+ld2*sqrt(x17))+gmd2*(bd2*x8/x17+sqrt(x17)*cd2-dd2*x17);
        const double x6 = k2a+thd2*(x8*x8/x17+ld2*sqrt(x17))+gmd2*(bd2*x8/x17+sqrt(x17)*cd2-dd2*x17);
        // if (D1->ID == 55){
        //     printf("x5: %f\n", x5);
        // }
        //assigning the output
        P->A[P->N]  = x9;
        P->Q[P->N]  = x0;
        D1->A[0]    = x12;
        D1->Q[0]    = x3;
        D2->A[D2idx]    = x15;    
        D2->Q[D2idx]    = sign*x6;

        gsl_multiroot_fsolver_free (s);
        gsl_vector_free (x);
    //}
    
    return false;
}