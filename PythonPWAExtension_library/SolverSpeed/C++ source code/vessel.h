class vessel{
public:
	
    int ID; //vessel ID
    bool isTerminal, isInitial;
    
    //characterisitc quantities for the vessel
    double qc;
    double rc;
    
    //additional parameters
    double rho; //blood density
    
    int N; //number of segments
    double dx, dt;
    double rIn, rOut; //proximal and distal diameters
    double L; //length of the vessel
    
    int Nx; //number of segments in x output mesh
    
    double *A, *Q; //current state
    double *Anh, *Qnh; //middle points
    
    double *A0, *f; //reference radius, area, and radius derivative at dx multiplications
    double *A0h, *fh; //reference radius, etc. at the middle points
    
    double *dr0dx, *dfdr0;
    double *dr0dxh, *dfdr0h;
    
    //variables kept for biffuraction points
    double  Flrh, Frlh, Slrh, Srlh;
    
    //resistance parameters
    double R1, R2, C;
    double p0, pT;
    
	double whereMeasure;

    //is autoregulation
    int Autoregulation; 
    
    vessel *d1, *d2; //pointers to daughter vessels
    
    std::vector<double> Pt,Qt;
    std::vector<std::vector<double > > Px, Qx;

    double Anm_1;
    
    //FUNCTIONS
    
    double pC;
    
    vessel(); //default constructor
    void initializeVessel(bool, bool, double, double, double, params, double, double, double, int, double, int, double, double, int);
    ~vessel(); //default destructor

    void   saveCurrentVals();
    void   saveSpaceState();
    std::vector<double> interpSpline(double*);
    
    double pressureToArea(int, double);
    double pressureToAreah(int, double);
    double areaToSPressure(int, double);
    double areaToSPressureh(int, double);
    
};


vessel::vessel(){}; //default constructor

vessel::~vessel(){ //default destructor
    delete[] A0;
    delete[] f;
    
    delete[] A0h;
    delete[] fh;
    
    delete[] A;
    delete[] Q;
    delete[] Anh;
    delete[] Qnh;
    
    delete[] dr0dx;
    delete[] dfdr0;
    delete[] dr0dxh;
    delete[] dfdr0h;
};

std::vector<double> vessel::interpSpline(double *s) {
    //function to evaluate on a given xmesh
    std::vector<double> output; //definingn output variable
    double dNx = L/(double)Nx; //defininig distance between the points on the output mesh
    double st = 0; //initial point (0 by default)
    
    gsl_interp_accel  *acc = gsl_interp_accel_alloc (); //preallocating interpolating core
    gsl_spline        *spline = gsl_spline_alloc (gsl_interp_cspline, N+1); //preallocating intorpolation kernel
    
    double *xmesh;
	xmesh = new double[N + 1];//defining mesh that was set for the input to interpolate


    xmesh[0] = 0;
    for (int i = 1; i<=N; i++)
        xmesh[i] = xmesh[i-1]+dx;

    gsl_spline_init (spline, xmesh, s, N+1); //initializing interpolation function
    while (st <= L) {
        output.push_back(gsl_spline_eval(spline, st, acc));
        st+=dNx;
    }
    
    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);
    
	delete[] xmesh;
    return output;
}

void vessel::initializeVessel(bool isInit, bool isTerm, double rIN, double rOUT, double Lt, params par, double R1a, double R2a, double Ca, int IDa, double qci, int NxP, double whereToMeasure, 
                              double dxD, int autoReg) {
    

	whereMeasure = whereToMeasure;

    ID = IDa;
    
    Nx = NxP; //number of points on x mesh
    
    isTerminal = isTerm;
    isInitial  = isInit;
    
    rho = par.rho;
    
    rIn  = rIN;
    rOut = rOUT;
    
    rc = 1; //characteristic radius of the vessel
    qc = 10; //characteristic flow through the vessel, could be given by the user
    
    
    rIn  = rIn/rc;  //scaling
    rOut = rOut/rc; //scaling
    
    dt = par.dt*qc/pow(rc,3); //scaling
    
    L = Lt/rc;     //scaling
	whereMeasure = whereMeasure / rc;
    // dx = par.dx/rc;// scaling FIXME
    dx = dxD/rc;    // FIXME 
    Autoregulation = autoReg;
    
    N = (int)(L/dx)+1; //number of vessel segments
    
    R1 = R1a*pow(rc,4)/qc/rho; //scaling resistance
    R2 = R2a*pow(rc,4)/qc/rho; //scaling resistance
    C  = Ca*rho*pow(qc,2)/pow(rc,7); //scaling compliance
    
    //allocating memory
    A0    = new double[N+3];
    f     = new double[N+3];
    
    dr0dx    = new double[N+3];
    dfdr0    = new double[N+3];
    
    dr0dxh    = new double[N+3];
    dfdr0h    = new double[N+3];
    
    A0h    = new double[N+3];
    fh     = new double[N+3];
    
    A     = new double[N+1];
    Q     = new double[N+1];
    Anh   = new double[N];
    Qnh   = new double[N];
    
    
    p0 = par.p0*pow(rc,4)/pow(qc,2)/rho;//scaling reference pressure
    pT = par.pT*pow(rc,4)/pow(qc,2)/rho-p0;//scaling terminal pressure
    pC = pT; //initialize pressure in small arteries
    
    //pg = pT;
    
    //mesh specification
    //     0 ------------------- L
    //     0 -- 1 -- 2 -- ... -- N                (N+1) entries for A,Q
    //0 -- 1 -- 2 -- 3 -- ... -- N+1 -- N+2       (N+3) entries for A0, f
    //  0 ---1----2----3------N ----N+1 -- N+2    (N+3) entries for A0h, fh
    //       0 -- 1 -- 2 --- N-1                  (N) entries for Anh and Qnh
    
    //initializing values
    double r0, r0h;
    for (int i=0; i<=N+2; i++) {
        
        r0       = rIn*pow(rOut/rIn,((double)(i-1))*dx/L); //ok
        r0h 	 = rIn*pow(rOut/rIn,(dx/2.+((double)(i-1))*dx)/L); //ok
        
        A0[i]    = M_PI*r0*r0;
        A0h[i]   = M_PI*r0h*r0h;
        
        f[i]     = 4./3.*(par.k1*exp(par.k2*rc*rIn)+par.k3)*pow(rc,4)/pow(qc,2)/rho;  //ok
        fh[i]    = 4./3.*(par.k1*exp(par.k2*rc*rIn)+par.k3)*pow(rc,4)/pow(qc,2)/rho; //ok

        dr0dx[i] = r0*log(rOut/rIn)/L;
		dfdr0[i] = 0;//4./3.*(par.k1*par.k2*rc*exp(par.k2*rc*r0))*pow(rc,4)/pow(qc,2)/rho;  //ok
        
        dr0dxh[i] = r0h*log(rOut/rIn)/L;
		dfdr0h[i] = 0;//4./3.*(par.k1*par.k2*rc*exp(par.k2*rc*r0h))*pow(rc,4)/pow(qc,2)/rho; //ok
    }
    
    for(int i=0; i<=N; i++) {
        Q[i]  = 0;
        A[i]  = pressureToArea(i,pT);
    }

    Anm_1 = A[N-1];
}


void vessel::saveCurrentVals() {
    //int place = (int)((double)N/2.); //return the value from the middle of the vessel
	int place = (int)((double)N * whereMeasure / L);
	Pt.push_back(areaToSPressure(place, A[place])*pow(qc,2)*rho/pow(rc,4));
    Qt.push_back(Q[place]*qc);
}

void vessel::saveSpaceState() {
    //vector<double > stateP, stateQ;
    //for(int i=0; i<N+1; i++) {
	double *Pr;
	double *Qr;
	Pr = new double [N+1];
    Qr = new double [N+1];
    for (int i = 0; i<=N; i++) {
        Pr[i] = areaToSPressure(i, A[i])*pow(qc,2)*rho/pow(rc,4);
        Qr[i] = Q[i]*qc;
    }
    Px.push_back(interpSpline(Pr));
    Qx.push_back(interpSpline(Qr));

	delete[] Pr;
	delete[] Qr;
}

double vessel::pressureToArea(int i, double P){
    return A0[i+1]/pow(1.-(P)/f[i+1],2);
};

double vessel::pressureToAreah(int i, double P){
    return A0h[i]/pow(1.-(P)/fh[i],2);
};

double vessel::areaToSPressure(int i, double A){
    return p0+f[i+1]*(1.-sqrt(A0[i+1]/A));
};

double vessel::areaToSPressureh(int i, double A){
    return p0+fh[i]*(1.-sqrt(A0h[i]/A));
};