#include <Python.h>  
#include <numpy/arrayobject.h>
#include <vector>

#include <stdlib.h>
#include <stdio.h>

#include <cmath>
#include <math.h>
#include <fstream>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_linalg.h>

#include "params.h"
#include "vessel.h"
#include "terminalEnd.h"
#include "updateInterior.h"
#include "biffurcationPoint.h"
// #include "inflowFunctions.h
  
#include <omp.h>

extern void _main();

vessel* tree;
int N, Nb; //number of vessels in the tree, number of branching points

params par; //simulation parameters

bool updateTree(double t, int** branchingPoints, bool last) { 
    bool terminate1 = false, terminate2 = false;
    bool cancel = false; 
    
    double Q_1RA, Q_1LA, Q_1RM, Q_1LM, Q_1RP, Q_1LP;

	omp_set_num_threads(8);

	#pragma omp parallel for
    for(int i=0; i<N; i++) {
        terminate1 = updateInterior(tree+i, par, t, last);
		//printf("Niter %d\n", i);
        if (terminate1)
           cancel = true;
    }
  
    if (cancel) return cancel; 

	#pragma omp parallel for
    for(int i=0; i<Nb; i++) { //Nb

		if (branchingPoints[i][2] > -.5) {
            updateBranchingPoint(tree + branchingPoints[i][0],
                tree + branchingPoints[i][1],
                tree + branchingPoints[i][2], par, t, branchingPoints[i][3]);
		}
		else {
			updateBranchingPoint(tree + branchingPoints[i][0],
				tree + branchingPoints[i][1],
				tree + branchingPoints[i][1], par, t, branchingPoints[i][3]);
		}

		//if (terminate2)
        //    break;
    }
    
	return false;//terminate1 || terminate2;  
}

void saveOutput() {  
    for(int i=0; i<N; i++){
        (tree+i)->saveCurrentVals();
        //(tree+i)->saveSpaceState();
    }
}

// ################################################################
// #################### PYTHON FUNCTIONS ##########################
// ################################################################

static PyObject* PythonPWASpeed(PyObject* self, PyObject* args){
    // getting simulation settings 
    PyObject *simulationParameters = NULL, *modelParameters = NULL; // parameters related with simulation and model 
    PyObject *treeStructure = NULL, *treeBranchingPoints = NULL;    // tree structure 
    PyObject *whState = NULL, *pC = NULL;                           

    PyObject *simulationParameters_C = NULL, *modelParameters_C = NULL; 
    PyObject *treeStructure_C = NULL, *treeBranchingPoints_C = NULL;    
    PyObject *whState_C = NULL, *pC_C = NULL;   

    PyObject *P_pressure = NULL, *Q_flow = NULL;  

    if (!PyArg_ParseTuple(args, "OOOOOO", &simulationParameters, &modelParameters, 
                                        &treeStructure, &treeBranchingPoints, 
                                        &whState, &pC))
    return NULL;

    simulationParameters_C  = PyArray_FROM_OTF(simulationParameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (simulationParameters_C == NULL) return NULL;
    double *simParams = (double *)PyArray_DATA(simulationParameters_C); 
    
    modelParameters_C       = PyArray_FROM_OTF(modelParameters, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (modelParameters_C == NULL) return NULL;
    double *modParams = (double *)PyArray_DATA(modelParameters_C); 
    
    treeStructure_C         = PyArray_FROM_OTF(treeStructure, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (treeStructure_C == NULL) return NULL;
    double *trStr = (double *)PyArray_DATA(treeStructure_C);
    
    treeBranchingPoints_C   = PyArray_FROM_OTF(treeBranchingPoints, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (treeBranchingPoints_C == NULL) return NULL;
    double *trBrPts = (double *)PyArray_DATA(treeBranchingPoints_C);
    
    whState_C               = PyArray_FROM_OTF(whState, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (whState_C == NULL) return NULL;
    double *wS = (double *)PyArray_DATA(whState_C);
    
    pC_C                    = PyArray_FROM_OTF(pC, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (pC_C == NULL) return NULL;
    double *pc = (double *)PyArray_DATA(pC_C);

    // getting simulation settings ------------------------ 
    int Niter = (int)simParams[0];
    // par.dx    = simParams[1];
    par.dt    = simParams[1];
    int oS    = (int)simParams[2];
    int Nx    = (int)simParams[3];

    // getting model parameters ---------------------------
    par.pT  = modParams[0]; //terminal pressure
    par.p0  = modParams[1]; //reference pressure
    
    // //parameters associated with vessel elasticity
    par.k1  = modParams[2];
    par.k2  = modParams[3];
    par.k3  = modParams[4];
    
    // //blood properites
    par.rho = modParams[5];
    par.mu  = modParams[6];
    
    // //inflow condition
    par.q0  = modParams[7];
    par.T   = modParams[8];
    par.tau = modParams[9];
    par.qB  = modParams[10];

    // parameters associated with 1D heart model
    par.Emin = modParams[11]; 
    par.Emax = modParams[12]; 
    par.a = modParams[13]; 
    par.b = modParams[14]; 
    par.tm = modParams[15];
    par.V0 = modParams[16]; 
    par.VlvInit = modParams[17]; 
    par.R = modParams[18]; 
    par.Vb = modParams[19];
    par.pa = modParams[20];
    par.Llv = modParams[21];
    par.Time = modParams[22];
    par.pla = modParams[23];
    par.constPhase3 = modParams[24];
    par.Vbup = modParams[25];
    par.Lla = modParams[26]; 
    par.Rla = modParams[27];
    par.div = modParams[28];
    par.model = modParams[29];
    par.l = modParams[30];
    par.message = modParams[31];

    // initial conditions for elastance function 
    par.Qprev = 0.0; 
    par.Qlaprev = 0.0;
    par.Vprev = par.VlvInit;
    par.xF0 = 0;
    par.Qparam1 = 0.0;
    par.Qparam2 = 0.0; 

    par.isQZero = 0; 
    par.isQgret = 0; 
    par.Vfin = NULL; 
    par.VfinUp = NULL; 
    par.value = 0; 
    par.valuePrev = 1;
    par.aorticValve = 0; 
    par.phase = 0; 
    
    par.iterator = 0;

    if (par.message){
        printf("\n\nnew computations ===============================\n");
        // printf("Emin = %f, Emax = %f, a = %f, b = %f, tm = %f, R =%f, Vb = %f, Vlv = %f\n", par.Emin, par.Emax, par.a, par.b, par.tm, par.R, par.Vb, par.VlvInit, par.message);
    };

    // getting tree structure and initializing tree
    const npy_intp *dims_array; 
    dims_array = PyArray_DIMS(treeStructure_C); // dimensions of tree structure
    N = (int)dims_array[1];
    int rows = (int)dims_array[0];
    tree = new vessel [N];
    for(int i=0; i<N; i++)
        tree[i].initializeVessel(trStr[4*N+i]==0,
                                 trStr[5*N+i]>0,
                                 trStr[2*N+i],
                                 trStr[3*N+i], trStr[1*N+i], par, 0.2*trStr[5*N+i]*pow(10.,4.),
								 0.8*trStr[5*N+i] * pow(10., 4.), trStr[6*N+i] * pow(10., -6.), 
                                 i, trStr[7*N+i], Nx, trStr[8*N+i], 
                                 trStr[10*N+i],  // add info about vessel-dependent dx
                                 trStr[11*N+i]); // add info about inlets to the cerebral autoregulation tree

    
    // reading branch points 
    dims_array = PyArray_DIMS(treeBranchingPoints_C);
    Nb = dims_array[1];

    // read initial condition -- optional 
    int aux = 0; 
    const npy_intp *dims_array2; 
    dims_array2 = PyArray_DIMS(whState_C);
    if ((int)dims_array2[0] > 1) {
        //I'm assuming that the input variable have proper dimension
        double *initCond = wS;
        double *initpC = pc;
        for (int i = 0; i < N; i++) {//for each vessel in the tree
            tree[i].pC = initpC[i];
            for (int ii = 0; ii <= tree[i].N; ii++) {
                tree[i].A[ii] = initCond[aux];
                tree[i].Q[ii] = initCond[aux + 1];
                aux += 2;
            }
        }
	}

    //declaring 2D array 
    int** branchingPoints;
    branchingPoints = new int*[Nb];
    for (int i = 0;i<Nb;i++)
        branchingPoints[i] = new int[dims_array[0]];
    
    for(int i = 0; i<dims_array[0]; i++)
        for(int j = 0; j<dims_array[1]; j++)
            branchingPoints[j][i]= trBrPts[i*dims_array[1]+j];//trBrPts[j*dims_array[0]+i];

    // saving initial condition 
    saveOutput();

    // simulating
    double t = 0;
    bool terminate;
    
    std::vector<double> Pic; 
    for(int i=0; i<Niter; i++) {
        if (((i+1) % oS) == 0) {    //save output
            saveOutput();
        }
		// printf("Iter %i\n", i);
        terminate = updateTree(t, branchingPoints, i==Niter-1); 
        t = t+par.dt;

        if (terminate) {
			if (par.message){
                printf("Termination at iteration: %i \n", i);
            };
            break;
        }
    }

    // WRITING OUTPUT TO PYTHON
    // dumping the whole current state into output variable 
    std::vector<double> whStateQ;
    std::vector<double> whStateA;
    std::vector<double> pC_;

    for (int i = 0; i < N; i++) {//for each vessel in the tree
        for (int ii = 0; ii <= tree[i].N; ii++) {
            whStateQ.push_back(tree[i].Q[ii]);
            whStateA.push_back(tree[i].A[ii]);
        }
        pC_.push_back(tree[i].pC);
    }

    int Npoints = tree[0].Pt.size();
    npy_intp dimLst[1] = {N*Npoints};
    PyObject *array_pressure = PyArray_SimpleNew(1, dimLst, NPY_DOUBLE); 
    if(!array_pressure) return NULL;
    double *arrayC_pressure = (double *)PyArray_DATA(array_pressure);

    PyObject *array_flow = PyArray_SimpleNew(1, dimLst, NPY_DOUBLE); 
    if(!array_pressure) return NULL;
    double *arrayC_flow = (double *)PyArray_DATA(array_flow);

    npy_intp dimWh[2] = {whStateA.size(), 2};
    PyObject *whState_toPython = PyArray_SimpleNew(2, dimWh, NPY_DOUBLE);
    if(!whState_toPython) return NULL;
    double *whState_toPythonC = (double *)PyArray_DATA(whState_toPython); 

    npy_intp dimPc[1] = {pC_.size()};
    PyObject *pC_toPython = PyArray_SimpleNew(1, dimPc, NPY_DOUBLE);
    if(!pC_toPython) return NULL; 
    double *pC_toPythonC = (double *)PyArray_DATA(pC_toPython);

    npy_intp dimPic[1] = {Pic.size()}; 
    PyObject *pic_toPython = PyArray_SimpleNew(1, dimPic, NPY_DOUBLE); 
    if(!pic_toPython) return NULL; 
    double *pic_toPythonC = (double *)PyArray_DATA(pic_toPython);

    for (int i=0; i<N; i++){
        for (int j=0; j<Npoints; j++){
            arrayC_pressure[i*Npoints+j] = tree[i].Pt.at(j);
            arrayC_flow[i*Npoints+j] = tree[i].Qt.at(j);
        }
    }

    for (int i=0; i<whStateQ.size(); i++){
        whState_toPythonC[2*i] = whStateA.at(i);
        whState_toPythonC[2*i+1] = whStateQ.at(i);
    }

    for (int i=0; i<pC_.size(); i++)
        pC_toPythonC[i] = pC_.at(i);

    for (int i=0; i<Pic.size(); i++)
        pic_toPythonC[i] = Pic.at(i); 
    
    Py_DECREF(simulationParameters_C);
    Py_DECREF(modelParameters_C);
    Py_DECREF(treeStructure_C);
    Py_DECREF(treeBranchingPoints_C);
    Py_DECREF(whState_C); 
    Py_DECREF(pC_C); 
    Py_DECREF(arrayC_flow);
    Py_DECREF(arrayC_pressure);
    Py_DECREF(whState_toPythonC);
    Py_DECREF(pC_toPythonC);
    Py_DECREF(pic_toPythonC);
    
    for (int i = 0;i<Nb;i++)
        delete[] branchingPoints[i];
    delete[] branchingPoints;
    delete[] tree;
    return Py_BuildValue("(O,O,O,O,O)", array_pressure, array_flow, whState_toPython, pC_toPython, pic_toPython);

}

static PyMethodDef Methods[] = {
    {"PythonPWASpeed", PythonPWASpeed, METH_VARARGS, "python PWA code"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef PythonPWAExtension = {
    PyModuleDef_HEAD_INIT,
    "PythonPWAExtension",
    "Test",
    -1,
    Methods
};   

PyMODINIT_FUNC PyInit_PythonPWAExtension(void){
    import_array();
    return PyModule_Create(&PythonPWAExtension);
}
