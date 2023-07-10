# %%
import numpy as np
from itertools import product
import warnings
import time
import copy
import sys, os
from scipy.optimize import shgo, dual_annealing, least_squares
sys.path.append('../packages')
import PythonAdditionalFunctions as PAF

import pkg_resources
pkg_resources.require("PythonPWAExtension")[0].version

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# reading and building tree
tree = {}
tree['treeStructure'], tree['branchingPoints'] = PAF.buildTreeCircle('../PythonPWAExtension_library/Data/treeFull2skipOneleg.xlsx')
# tree['treeStructure'], tree['branchingPoints'] = PAF.buildTree('../PythonPWAExtension_library/Data/treeFull2skipOneleg.xlsx')

# scaling arterial tree to patient height 
# calculating approximate distance from the heart to the foot 
pathLeg = np.array([9, 11, 12, 24, 26, 28, 30, 32, 33, 34, 37, 40])

# gathering data
L = 0 
for i in range(len(pathLeg)):
    L = L + tree['treeStructure'][1, pathLeg[i]]

Patients = np.load('./patientsHD.npy', allow_pickle=True).item()

PatientsID = Patients.keys()
TypeOfTheBreak = ['ShortBreak']
MomentOfTheMeasurement = ['BeforeStart', 'AfterStart', 'BeforeEnd', 'AfterEnd']

Loop = list(product(*[PatientsID, TypeOfTheBreak, MomentOfTheMeasurement]))

method = "PSO"
add_method = "LSQ"
n = 6
# print(f"sys: {sys.argv[0]}")
# %%
for PID, BreakType, MomentType in [(sys.argv[1], sys.argv[2], sys.argv[3])]:
# for PID, BreakType, MomentType in [('Pt15', 'ShortBreak', 'AfterEnd')]:
# for Measurement in [sys.argv[1]]:
    # take this PWA_ measurement key, which has better quality of the data
    start = time.time()

    if np.isnan(Patients[PID][BreakType][MomentType]['PF']['SV']):
        print(f"PF IS NAN: {PID}, {BreakType}, {MomentType}")
        continue

    try: 
        PatientPulse = Patients[PID][BreakType][MomentType]['PWA']['pulse']['PeriphPulse']
        PatientCharacteristics = Patients[PID][BreakType][MomentType]['PWA']['data']
        data = PAF.loadGroupSphygmoCor(PatientCharacteristics, PatientPulse, L)
    except:
        print(f"ERROR: {sys.argv[1]} {sys.argv[2]} {sys.argv[3]}")
        continue

    data['model'] = 2 
    data['message'] = False

    # Scaling tree according to the scale factor
    treeP = copy.deepcopy(tree)
    treeP['treeStructure'][1:4,:] = treeP['treeStructure'][1:4,:]*data['ScaleFactor']
    # we need to round L to given precision
    aux = np.round(treeP['treeStructure'][1,:],2)

    treeP['treeStructure'][1,:] = aux

    # making sure that where to take measurement variable is ok
    indx = treeP['treeStructure'][8,:]>treeP['treeStructure'][1,:]
    treeP['treeStructure'][8,indx] = np.floor(treeP['treeStructure'][1,indx]/2)

    # scale resistances and compliance
    treeP['treeStructure'][5,:] = treeP['treeStructure'][5,:]/(data['ScaleFactor']**3)
    treeP['treeStructure'][6,:] = treeP['treeStructure'][6,:]*(data['ScaleFactor']**3)

    params = PAF.initializeParamsVaso(data, treeP, sph=1)
    # paramsToFit = ['k1','k3','scaleRes','scaleComp', 
    #                'a', 'b', 'tm', 'Emin', 'Emax', 'R', 'Llv']
    paramsToFit = ['tm', 'Emax', 'scaleRes', 'scaleComp']

    simSettings = {}
    simSettings['secondsToSimulate'] = 8.0 # number of seconds to simulate 
    simSettings['pps'] = 1000   # Hz, number of points per second in the output 
    simSettings['dtdx'] = 0.00025

    x0, lb, ub = PAF.startingPointVaso(params, paramsToFit)
    # x0 = np.array([2000000, 900000.0, 1.0, 15.0, 0.4588, 2.49]).T

    # err, pulse = PAF.F_SphygmoCor(np.array(x0), treeP, simSettings, data, paramsToFit)
    # %%
    # ========== PSO ===========
    if method == "PSO":
        print("========== PSO fitting ============\n")
        print(f"== Fitting: {paramsToFit} ===")
        print("====================================\n")
        PSOoptions = {'n_particles': 24,
                      'iters': 10}
        joint_vars, res = PAF.Optimize_PSO(paramsToFit, PSOoptions, lb, ub,
                                           treeP, simSettings, data,
                                           n=n)
    # %% 
    # ============= SHGO ==============
    if method == "SHGO": 
        start = time.time()
        bounds_real = list(zip(lb, ub)) 
        bounds_norm = [(0,1) for _ in  range(len(bounds_real))]

        res = shgo(PAF.opt_funcFFT_scipy, 
                   bounds=bounds_norm, 
                   n=20,
                   iters=2, 
                   sampling_method='sobol',
                   args=(
                        treeP,
                        simSettings,
                        data,
                        paramsToFit,
                        bounds_real,
                        n
                   ),
                   options={"disp": True,},
                   minimizer_kwargs={"qhull_options" : "Qz"})
        
        joint_vars = [res.x[i] * (bounds_real[i][1] - bounds_real[i][0]) + bounds_real[i][0] for i in range(len(res.x))]
    
    if method == "DUAL_ANNEALING": 
        start = time.time()
        bounds_real = list(zip(lb, ub)) 
        bounds_norm = [(0,1) for _ in  range(len(bounds_real))]

        res = dual_annealing(PAF.opt_funcFFT_scipy, 
                             bounds=bounds_norm,
                             maxiter=10,
                             maxfun=24,
                             args=(
                                  treeP,
                                  simSettings,
                                  data,
                                  paramsToFit,
                                  bounds_real
                                  ),
                            )
        
        joint_vars = [res.x[i] * (bounds_real[i][1] - bounds_real[i][0]) + bounds_real[i][0] for i in range(len(res.x))]
     
    if add_method == "LSQ":
        print("========== Additional method: LSQ ==========\n")

        bounds_real = list(zip(lb, ub))
        bounds_norm = (np.zeros_like(lb), np.ones_like(ub))
        
        try: 
            x0 = res.x
        except AttributeError:
            x0 = res

        res_lsq = least_squares(PAF.opt_funcFFT_scipy, 
                                x0 = x0,
                                bounds=bounds_norm,
                                diff_step=0.05,
                                verbose=2,
                                ftol=1e-6,
                                xtol=1e-6,
                                gtol=1e-6, 
                                args=(
                                    treeP,
                                    simSettings,
                                    data,
                                    paramsToFit,
                                    bounds_real, 
                                    n
                                ),
                                )
        joint_vars = [res_lsq.x[i] * (bounds_real[i][1] - bounds_real[i][0]) + bounds_real[i][0] for i in range(len(res_lsq.x))]

    # treeP = copy.deepcopy(tree)
    err, pulse, P, Q = PAF.F_SphygmoCorFFT(joint_vars, treeP, simSettings, data, paramsToFit, n)
    print(f"\n\n Computed error once again: {np.sum([e**2 for e in err])}")

    toSave = {}
    toSave['P']           = P 
    toSave['treeP']       = treeP
    toSave['dataP']       = data
    toSave['simSettings'] = simSettings
    toSave['paramsToFit'] = paramsToFit
    toSave['pulse']       = pulse
    toSave['params']      = joint_vars
    toSave['Q_inlet']     = Q[0, :]
    toSave['error']       = err
    toSave['SV_PF']       = Patients[PID][BreakType][MomentType]['PF']['SV']
    toSave['SV_PFmn']     = Patients[PID][BreakType][MomentType]['PF']['SVmn']

    np.save(f"results/fitResult_{PID}_{BreakType}_{MomentType}.npy", toSave)
    
    print(f"======================================\n"
          f"Patient: {PID}, BT: {BreakType}, MT: {MomentType}\n" 
          f"EVALUATION TIME (IN SECONDS): {(time.time() - start):.2f}\n"
          f"======================================\n")
