class params{
public:
    
    double pT; //terminal pressure
    double p0; //reference pressure
    
    double dx;
    double dt;
    
    double rho;
    double mu;
    
    //parameters assoctited with inflow condition
    double q0, T, tau, qB;
    
    //parameters associated with elasticity
    double k1, k2, k3;
    
    bool terminate;

    //parameters associated with elastance model 
    double tm;      // time to the oneset constant elastance 

    double a, b;    // parameters in Phi function 
    double Emin, Emax; 

    double V0;      
    double VlvInit, plvInit;    
    double pa;      
    double Vb, Vbup;      
    double R;       
    double Llv; 
    double Time;
    double xF0; 

    double Qprev, Qlaprev, Vprev;
    int isQZero;
    int isQgret; 
    double Vfin, VfinUp; 
    
    double value;
    double valuePrev;
    bool aorticValve;
    int phase; 
    double pla; 
    double constPhase3; 
    double Lla; 
    double Rla;
    double div;
    double Ela; 
    double V0a; 
    double Qparam1, Qparam2; 
    double l;   // length of the jump

    int iterator; 

    // parameters connected with opening/closing valve 
    double Kvo; 
    double Kvc; 
    double plao, plac; // pressure close, open
    double Zprev; 
    double A; 
    double Mst, Mrg; 

    double model;
    bool message; 
  
    params(); //default constructor
};

params::params(){}; //default constructor
