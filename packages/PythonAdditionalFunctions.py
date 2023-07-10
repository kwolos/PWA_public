import copy
import pandas as pd 
import numpy as np 
from scipy import interpolate
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.optimize import shgo
from scipy.interpolate import interp1d
import pyswarms as ps
from PythonPWAExtension import PythonPWASpeed
# from PythonPWAExtensionAutoregulation import PythonPWASpeed

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error")

def buildTree(fileName):
    """Function used to read tree structure from xls file and build
    appropriate Python data structure.

    Parmeters
    ---------
        fileName (str): xls file with tree data

    Returns
    -------
        tree, branchingPoints 
    """
    treeDefNum = pd.read_excel(fileName)

    Nart = treeDefNum.shape[0]
    tree = np.zeros((9, Nart))

    for i in range(Nart):      # for each vessel segment
        tree[0,i] = treeDefNum.iloc[i,0]        # ID
        tree[1,i] = treeDefNum.iloc[i,2]        # length
        tree[2,i] = treeDefNum.iloc[i,3]        # rIN
        tree[3,i] = treeDefNum.iloc[i,4]        # rOut
        tree[4,i] = treeDefNum.iloc[i,5]        # parentID
        tree[5,i] = treeDefNum.iloc[i,8]        # RT
        tree[6,i] = treeDefNum.iloc[i,9]        # CT
        tree[7,i] = 10                          # characterisitc flow
        tree[8,i] = treeDefNum.iloc[i,10]       # where to take measurement     
        
    # creating matrix containing information about branching points 
    branchingPoints = np.empty((3,0), dtype='int64')
    for i in range(Nart):
        indx = np.where(tree[4,:] == tree[0, i])[0]
        if len(indx) == 2: 
            branchingPoints = np.append(branchingPoints, np.array([[i, indx[0], indx[1]]]).transpose(), axis=1)
        elif len(indx) == 1: 
            branchingPoints = np.append(branchingPoints, np.array([[i, indx[0], -1]]).transpose(), axis=1)
        
    # calculation of the characteristic flow 
    for i in range(branchingPoints.shape[1]):
        if branchingPoints[2,i] >= 0:
            pFlow = tree[7, branchingPoints[0,i]]
            rD1   = tree[2, branchingPoints[1,i]]
            rD2   = tree[2, branchingPoints[2,i]]
            rS    = rD1+rD2
            tree[7, branchingPoints[1,i]] = pFlow*rD1/rS
            tree[7, branchingPoints[2,i]] = pFlow*rD2/rS
        else:
            pFlow = tree[7, branchingPoints[0,i]]
            rD1   = tree[2, branchingPoints[1,i]]
            rD2   = tree[2, branchingPoints[1,i]] # same vessel
            rS    = rD1+rD2
            tree[7, branchingPoints[1,i]] = pFlow*rD1/rS      
    
    return tree, branchingPoints

def buildTreeCircle(fileName):
    """Function used to read tree structure from xls file and build
    appropriate Python data structure.

    Parmeters
    ---------
        fileName (str): xls file with tree data

    Returns
    -------
        tree, branchingPoints 
    """
    treeDefNum = pd.read_excel(fileName)

    Nart = treeDefNum.shape[0]
    tree = np.zeros((12, Nart))

    for i in range(Nart):                       # for each vessel segment
        tree[0, i]  = treeDefNum.iloc[i,0]        # ID
        tree[1, i]  = treeDefNum.iloc[i,2]        # length
        tree[2, i]  = treeDefNum.iloc[i,3]        # rIN
        tree[3, i]  = treeDefNum.iloc[i,4]        # rOut
        tree[4, i]  = treeDefNum.iloc[i,5]        # parentID
        tree[5, i]  = treeDefNum.iloc[i,8]        # RT
        tree[6, i]  = treeDefNum.iloc[i,9]        # CT
        tree[7, i]  = 10                          # characterisitc flow
        tree[8, i]  = treeDefNum.iloc[i,10]       # where to take measurement       
        tree[9, i]  = treeDefNum.iloc[i,11]       # additional ID for cicle
        tree[10,i]  = treeDefNum.iloc[i,12]       # step for a given branch
        tree[11,i]  = treeDefNum.iloc[i,13]       # branches with autoreg connection

    # creating matrix containing information about branching points 
    branchingPoints = np.empty((4,0), dtype='int64')
    for i in range(Nart):
        indx      = np.where(tree[4,:] == tree[0, i])[0]
        indxCycle = np.where(tree[9,:] == tree[0, i])[0]
        if len(indx) == 2: 
            branchingPoints = np.append(branchingPoints, np.array([[i, indx[0], indx[1], 0]]).transpose(), axis=1)
        elif len(indx) == 1:
            # cycles are still biffurcation, branches causing cycles 
            # should not be Parent ID for any branch 
            if len(indxCycle):
                branchingPoints = np.append(branchingPoints, np.array([[i, indx[0], indxCycle[0], 1]]).transpose(), axis=1)
            else: 
                branchingPoints = np.append(branchingPoints, np.array([[i, indx[0], -1, 0]]).transpose(), axis=1)
        
    # calculation of the characteristic flow 
    for i in range(branchingPoints.shape[1]):
        if branchingPoints[2,i] >= 0:
            pFlow = tree[7, branchingPoints[0,i]]
            rD1   = tree[2, branchingPoints[1,i]]
            rD2   = tree[2, branchingPoints[2,i]]
            rS    = rD1+rD2
            tree[7, branchingPoints[1,i]] = pFlow*rD1/rS
            tree[7, branchingPoints[2,i]] = pFlow*rD2/rS
        else:
            pFlow = tree[7, branchingPoints[0,i]]
            rD1   = tree[2, branchingPoints[1,i]]
            rD2   = tree[2, branchingPoints[1,i]] # same vessel
            rS    = rD1+rD2
            tree[7, branchingPoints[1,i]] = pFlow*rD1/rS      
    
    return tree, branchingPoints


def initializeParamsVaso(patient, tree, sph=None): 
    """Intialize parameters from SphygmoCor or others data. 

    Parameters
    ----------
    patient : dict
        data from SphygmoCor returned from ``loadSphygmoCor`` function. 
    tree : dict
        generated tree from ``build tree`` function
    sph : int, optional
        data may come from other sources, which can differ in some nuances. 
        if you take data from SphygmoCor, take sph = 1. 

    Returns
    -------
    params : dict
        dictionary with parameters for solver
    """
    params = {} 

    params['pT']        = 10 * 1333.322365        # pressure at the venous end of capillary bed, 1333.322 is for unit conversion from mmHG
    params['bounds_pT'] = np.array([5, 15])*1333.322365
    params['p0']        = 100 * 1333.322365        # reference pressure, 1333.322 is for unit conversion from mmHG     
    params['bounds_p0'] = np.array([80, 110])*1333.322365
    params['k1']        = 3*1e6                 # parameter associated with elasticity, see supp materials 
    params['bounds_k1'] = np.array([0.1, 5])*params['k1']   # [0.1 3]*params.k1;
    params['k2']        = -13.5                  # parameter associated with elasticity, see supp materials
    params['bounds_k2'] = np.array([0.25, 10])*params['k2']

    params['k3'] =  0.4*8.65 * 1e5
    params['bounds_k3'] = np.array([0.1, 8.0]) * params['k3']

    if params['k3'] < params['bounds_k3'][0] or params['k3'] > params['bounds_k3'][1]:
        params['k3'] = 8.65 * 1e5 

    params['minr0'] = np.min(tree['treeStructure'][3, :]) # min rout
    params['maxr0'] = np.max(tree['treeStructure'][2, :]) # max rin

    params['w1'] = 4/3 * (params['k1'] * np.exp(params['k2']*params['minr0'])+params['k3'])
    params['bounds_w1'] = np.array([0.05, 5]) * params['w1']
    params['w2'] = 4/3 * (params['k1'] * np.exp(params['k2']*params['maxr0']) + params['k3'])
    params['bounds_w2'] = np.array([0.5, 1.5])*params['w2']

    params['rho']        = 1.04                 # blood density (g/cm^3)
    params['bounds_rho'] = np.array([1, 1.07])
    params['mu']         = 0.04                 # kinematic viscosity
    params['bounds_mu']  = np.array([0.6, 1.4])*params['mu'] 

    params['scaleRes'] =  1#0.26501 + 0.01388 * patient['Weight'] # %1;
    params['bounds_scaleRes'] = np.array([0.1, 4])#np.array([0.1, 3.5])
    if params['scaleRes'] < params['bounds_scaleRes'][0] or params['scaleRes'] > params['bounds_scaleRes'][1]:
        params['scaleRes'] = 1

    params['scaleComp'] = 1#30.6231-0.1245*patient['Height'] #1;
    params['bounds_scaleComp'] = np.array([0.1, 5])
    if params['scaleComp'] < params['bounds_scaleComp'][0] or params['scaleComp'] > params['bounds_scaleComp'][1]:
        params['scaleComp'] = 1

    if sph is None: 
        params['T']     = patient['waveLength']/1000.0 
        params['Time']  = patient['waveLength']/1000.0 
        params['HR']    = patient['HR']
    else: 
        params['HR']    = patient['HR']
        params['T']     = patient['C_PERIOD']/1000.0
        params['Time']  = patient['C_PERIOD']/1000.0

    # params connected with the elastance model
    params['Emin'] = 0.049
    params['bounds_Emin'] = np.array([0.02, 0.1])
    if params['Emin'] < params['bounds_Emin'][0] or params['Emin'] > params['bounds_Emin'][1]:
        params['Emin'] = 0.049 
    
    params['Emax'] = 2.49       # 2.49;%          % maximal systolic value of the left ventricular elastance function [mmHg/mm]
    params['bounds_Emax'] = np.array([1.0, 7]) #np.array([1.7, 3.0])   #[2.0, 3.5];
    if params['Emax'] < params['bounds_Emax'][0] or params['Emax'] > params['bounds_Emax'][1]:
        params['Emax'] = 2.0
    
    params['a'] = 0.9 #0.9;            
    params['bounds_a'] = np.array([0.5, 1.5])
    if params['a'] < params['bounds_a'][0] or params['a']> params['bounds_a'][1]:
        params['a'] = 0.9 

    params['b'] = 0.25 
    params['bounds_b'] = np.array([0.1, 0.7])      # [0.05, 0.7]; 
    if params['b'] < params['bounds_b'][0] or params['b'] > params['bounds_b'][1]:
        params['b'] = 0.25
    
    params['tm'] = 0.29+0.2 * params['T']   # 0.55;
    params['bounds_tm'] = np.array([0.4*params['T'], 0.9*params['T']])
    if params['tm'] < params['bounds_tm'][0] or params['tm'] > params['bounds_tm'][1]:
        params['tm'] = 0.55
    
    params['VlvInit'] = 120*patient['ScaleFactor']**3              # value probably scalable
    # params['VlvInit'] = 0.3*2.20*patient['Weight']+62
    params['bounds_VlvInit'] = np.array([100, 140])
    if params['VlvInit'] < params['bounds_VlvInit'][0] or params['VlvInit'] > params['bounds_VlvInit'][1]:
        params['VlvInit'] = 120
        
    params['Vb'] = 0.2#*patient['ScaleFactor']**3                    # value probably scalable 
    params['bounds_Vb'] = np.array([0.01, 3.0])
    if params['Vb'] < params['bounds_Vb'][0] or params['Vb'] > params['bounds_Vb'][1]:
        params['Vb'] = 2.0

    params['pla'] = 7.5
    params['bounds_pla'] = np.array([6.0, 12.0])
    if params['pla'] < params['bounds_pla'][0] or params['pla'] > params['bounds_pla'][1]:
        params['pla'] = 2.0 
        
    params['R'] = 0.08 #0.08; %0.0334; 
    params['bounds_R'] = np.array([0.01, 0.1])
    if params['R'] < params['bounds_R'][0] or params['R'] > params['bounds_R'][1]:
        params['R'] = 0.08

    params['Llv'] = 0.000416  #0.000416; 
    params['bounds_Llv'] = np.array([0.0001, 0.0007])
    if params['Llv'] < params['bounds_Llv'][0] or params['Llv'] > params['bounds_Llv'][1]:
        params['Llv'] = 0.000416
        
    params['V0']            = 15*patient['ScaleFactor']**3    # value probably scalable
                                                              # left ventr. volume at 0 pressure, Olufsen 
    params['pa']            = 80            # root aortic systemic pressure (should be taken directly from the model)
    params['constPhase3']   = 70
    params['Vbup']          = params['Vb']  #2.0; # backflow in the mitral valve
    params['Lla']           = 0.00005 
    params['Rla']           = 0.000089 
    params['div']           = 10.0          # characteristic flow trough the vessel
    params['model']         = patient['model']
    params['l']             = 0.02          # lenght of backflow (in seconds)
    params['message']       = patient['message']
    
    # # Autoregulation parameters
    # params['r_d0'], params['h_d0'], params['sigma_e0'], params['k_sigma']   = 0.015, 0.003, 0.1425, 10
    # params['sigma_coll'],  params['T_0'],  params['r_m'],  params['r_t']    = 62.79, 2.16, 0.027, 0.018
    # params['n_m'],  params['eta'],  params['K_v'],  params['K_gM'],  params['K_gA']   = 1.83, 232, 7.4*1e3, 1.05*1e6, 7.5*1e5 
    # params['K_gP'],  params['K_e'],  params['C_m'],  params['t_CA'],  params['G_CA']  = 5.9*1e5, 0.077, 1.37, 10, 10
    # params['G_pv'],  params['G_f'],  params['G_cAA'],  params['G_cPP']      = 0.189, 7.0*1e-5, 4.0*1e-2, 1.0*1e-2, 
    # params['G_cAM'],  params['G_cMP'],  params['P_an'],  params['G_o']      = 9.0*1e-3, 8.0*1e-3, 100, 3.17*1e-4
    # params['P_s'],  params['Q_nM'],  params['Q_nA'],  params['Q_nP'], params['P_icn'] = 6, 2.2, 1.48, 1.14, 9.5
    # params['P_ic'] = 14

    return params

def loadGroupSphygmoCor(dataPatient, dataPulse, L): 
    """load group of parameters from SphygmoCor connected with given patient

    Parameters
    ----------
    dataPatient : DataFrame 
        Data frame from SpygmoCor with measured by device parameters as Age, 
        Weight, CAI, SP, DP 
    dataPulse : array 
        Measured pulse waveform 
    L : float
        Calculated length from the heart to the foot - for scaling purposes

    Returns
    -------
    group : dict
        dictionary with characteristics connected with pulse waveform for simulation
    
    """
    group = dict() 
    
    group['OI']          = dataPatient['Operator_Index']
    group['Height']      = dataPatient['HEIGHT']
    group['Weight']      = dataPatient['WEIGHT']
    group['sample_rate'] = dataPatient['SAMPLE_RATE']
    group['HR']          = dataPatient['HR']
    group['ED']          = dataPatient['ED']
    
    group['PT1']         = dataPatient['P_T1']
    
    group['PT2']         = dataPatient['P_T2']
    group['MAP']         = dataPatient['P_MEANP']  # peripheral systolic pressure
    
    group['PPT1']        = dataPatient['P_P1'] 
    group['PPT2']        = dataPatient['P_P2'] 
#   (according to manual, P_P2 is the peripheral P2, where P2 - pressure at T1) 

    # peripheral pressure at ED
    group['PESP']        = dataPatient['P_ESP'] 
    
    # systolic pressure (user input in SpC machine)
    group['SP']          = dataPatient['SP'] 
    # diastolic pressure (use input in SpC machine)
    group['DP']          = dataPatient['DP'] 
    group['Age']         = dataPatient['AGE'] 

    group['PW']          = dataPulse
    group['SUBTYPE']     = dataPatient['SUB_TYPE'] 

    group['C_PERIOD']    = dataPatient['C_PERIOD']

#     %reading central wave
    
#     %calculating
    dt = 1000/(group['sample_rate']) # %in miliseconds
#     % group.(pat).tmesh = dt*(0:(length(group.(pat).CH1)-1));
    group['tmesh']  = dt * np.arange(0, len(group['PW']))
    group['PD']     = group['tmesh'][-1]+dt
    
    group['ScaleFactor'] = group['Height']/L

    return group

def normalization(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def renormalization(data, SP, DP):
    """
    From data after dividing to measured SP, DP 
    """
    return (data - np.min(data))*(SP - DP)/(np.max(data) - np.min(data))+DP 

def loadDataANGE(dataPatient, CH, L): 
    """
    load group of parameters from ANGE device connected with 
    a given patient 

    Parameters
    ----------
        dataPatient : struct
            struct with basic patient characteristics loaded from 
            ANGE device, such as weight, height, age, sex, name
        CH1 : np.ndarray, int
            measured pulse waveform from left wrist
        CH2 : np.ndarray, int
            measured pulse waveform from right wrist
        CH3 : np.ndarray, int
            measured pulse waveform from left ankle
        CH4 : np.ndarray, int
            measured pulse waveform from right wrist
        HR : struct
            struct of the form {n: hr} with computed hr for n-th cuff 
        L : float 
            Calculated length from the heart to the foot - for scaling purposes

    Returns
    ------- 
        group : dict
            dictionary with characteristics connected with pulse waveform for simulation
    """
    group = dict() 
    group['Height']      = dataPatient['height']
    group['Weight']      = dataPatient['weight']
    group['Age']         = dataPatient['age'] 

    HRconsideredCuffs = []
    
    # perform normalization of the data (in the scale 0-1)
    if (CH['CUFF_WAVEFORMS'][1] is not 1) and (CH['HR_CUFFS'][1] is not 1): 
        # group['PW_CH1'] = renormalization(CH['CUFF_WAVEFORMS'][1],
        #                                   CH['SP_LEVELS'][1], 
        #                                   CH['DP_LEVELS'][1])
        group['PW_CH1'] = normalization(CH['CUFF_WAVEFORMS'][1])
    if (CH['CUFF_WAVEFORMS'][2] is not 1) and (CH['HR_CUFFS'][2] is not 1):
        group['PW_CH2'] = normalization(CH['CUFF_WAVEFORMS'][2])
    if (CH['CUFF_WAVEFORMS'][3] is not 1) and (CH['HR_CUFFS'][3] is not 1):
        group['PW_CH3'] = normalization(CH['CUFF_WAVEFORMS'][3])
    if (CH['CUFF_WAVEFORMS'][4] is not 1) and (CH['HR_CUFFS'][4] is not 1):
        group['PW_CH4'] = normalization(CH['CUFF_WAVEFORMS'][4])

    # HR is the mean of HR of the considered cuffs
    group['HR'] = CH['HR']

    # # we need to intra/extrapolate results to a given period
    group['waveLength'] = CH['WAVE_LENGTH']  # in miliseconds
    
    dt = CH['DT'] # in miliseconds
    group['tmesh']  = dt * np.arange(0, group['waveLength'])

    # # interpolate measurements to a given period
    # consideredCuffs = [k for k in group.keys() if k.startswith('PW')]

    # for k in consideredCuffs: 
    #     x = np.linspace(0, group['waveLength'], num=len(group[k]))
    #     f_intp = interp1d(x, group[k])
    #     group[k] = f_intp(group['tmesh'])

    group['PD']     = group['tmesh'][-1]+dt
    group['ScaleFactor'] = group['Height']/L

    return group  

def loadGroupVaso(dataPatient, Patient, Number, L):
    group = {}
    
    PatientNumber = Number
    dataPatient = dataPatient[dataPatient['ID']==PatientNumber]

    group['Height'] = dataPatient['Groesse'].values[0]
    group['Weight'] = dataPatient['Gewicht'].values[0]
    group['sample_rate'] = 1000

    # scale pusle waves to averaged HR for all cuffs 
    lensOfWaves = []
    for _, item in Patient.items(): 
        if isinstance(item, np.ndarray):
            lensOfWaves.append(len(item))
    
    PulseLenAveraged = np.mean(lensOfWaves)
    group['pulseWaveAveraged'] = PulseLenAveraged 

    if isinstance(Patient[1], np.ndarray):
        dt = PulseLenAveraged/len(Patient[1])
        group['tmesh'] = dt * np.arange(len(Patient[1]))
    else:
        dt = PulseLenAveraged/len(Patient[2])
        group['tmesh'] = dt * np.arange(len(Patient[2]))
        
    for key, item in Patient.items():
        if isinstance(item, np.ndarray):
            f = interpolate.interp1d(np.arange(len(item)), item, kind='linear', fill_value='extrapolate')
            group['CH'+str(key)] = f(group['tmesh'])
        else:
            group['CH'+str(key)] = -1
            
        
    group['ScaleFactor'] = group['Height']/L
    
    return group


def pointToParamsVaso(x, params, paramsToFit):
    if isinstance(paramsToFit, (list, tuple, np.ndarray)): 
        for i, key in enumerate(paramsToFit):
            if key in params: 
                    params[key] = x[i]
    else: 
        params[paramsToFit] = x

    return params

def solveModelSpeedVaso(treeP, params, simSettings, state=None):

    if state is None: 
        state = {} 
        state["whState"] = np.array([1])
        state["pC"] = np.array([1])

    qB = 0 # baseline value of the flow 
    CO= 4080
    q0 = (CO - qB*60)/(60/params['T'])
    params['tau'] = 0.1172

    modelParameters = np.array([params['pT'], params['p0'],
                                params['k1'], params['k2'], params['k3'], 
                                params['rho'], params['mu'], 
                                q0, params['T'], params['tau'], qB, 
                                params['Emin'], params['Emax'], params['a'], params['b'],
                                params['tm'], params['V0'], params['VlvInit'], 
                                params['R'], params['Vb'], params['pa'], params['Llv'], 
                                params['Time'], params['pla'], params['constPhase3'], params['Vbup'],
                                params['Lla'], params['Rla'], params['div'], params['model'], 
                                params['l'], params['message']])

    if "dtdx" in simSettings: 
        dtdx = simSettings["dtdx"]
    else:
        dtdx = 0.5                  # dt/dx 
        
    if "Nx" in simSettings:         # number of points on x mesh 
        Nx = simSettings["Nx"]
    else:
        Nx = 100
        
    dx = 0.5
    dt = dtdx * dx

    aux = 1/dt                                              # number of iterations per second
    oS  = np.floor(aux/simSettings['pps'])                  # write state every oS iterations 
    Nit = simSettings['secondsToSimulate']*aux              # number of iterations 
    simulationParameters = np.array([Nit, dt, oS, Nx])

    (P, Q, whState, pC, _) = PythonPWASpeed(simulationParameters, modelParameters, treeP['treeStructure'],
                                        treeP['branchingPoints'], state['whState'], state['pC'])
    
    _, siz = treeP['treeStructure'].shape
    P = P.reshape((siz, -1))
    Q = Q.reshape((siz, -1))
    whState = whState.T
    
    return P, Q, whState, pC


def startingPointVaso(params, paramsToFit):  
    x0 = [params[i] for i in paramsToFit] 
    lb = [params['bounds_'+i][0] for i in paramsToFit]
    ub = [params['bounds_'+i][1] for i in paramsToFit]

    return x0, lb, ub


def selectOnePeriodVaso(P, sample_rate, minPlsDr, diagnostics = False):
    """
    Select one period from given pulse wave signal

    Parameters
    ----------
    P : array
        array with values of the pulse waveform 
    sample_rate : int 
        sample rate, how many samples per second perfomed 
    minPlsDr : float 
        minimal pulse duration in seconds, assuming max HR of 100 bps 
    diagnostics : bool 
        perform diagnostic 

    Returns
    -------
    P : array 
        selected period of pressure 
    t : array 
        two-element array - first element: beginning of selected pulse waveform
        second element: end of selected pulse waveform 
    succ : bool 
        true if beginning of pulse waveform correctly founded 
        false if end of pulse waveform not correctly founded
    """

    # rejecting half of the signal 
    PEndIdx = len(P)
    P = P[round(PEndIdx/2):-1]

    dPdt = np.diff(P)

    # defining minimal peak prominence 
    minPeakProminence = (np.max(dPdt) - np.min(dPdt))/4

    # defining minimal peak separation 
    minimalPulseDuration = minPlsDr    # seconds, assuming max HR of 100 bps 
    minPeakDistance = minimalPulseDuration*0.95*sample_rate

    # finding derivative peaks
    pks, _ = find_peaks(dPdt, distance=minPeakDistance, prominence=minPeakProminence)
    # finding first derivative crossing points from negative to positive 
    W = 2
    crossPoints = [] 

    for i in np.arange(W, len(dPdt)-W):
        if np.all(dPdt[i-(np.arange(W)+1)] <= 0) and np.all(dPdt[i+(np.arange(W)+1)] > 0): 
            crossPoints.append(i)
            
    # plotting auxiliary figure 
    if diagnostics:
        fig, ax = plt.subplots(1,2)
        ax[0].plot(dPdt)
        ax[0].plot(pks, dPdt[pks], 'ro', label='peaks')
        ax[0].plot(crossPoints, np.zeros_like(crossPoints), 'bx', label='crossPoints')
        ax[0].legend()
        ax[0].grid()

    aux = len(pks)

    while aux > 1:
        # taking two peaks within acceptable limits 
        sep = (pks[aux-1] - pks[aux-2])/sample_rate
        if sep < minimalPulseDuration * 1.01:
            pks = pks[(np.arange(aux-2, aux))]
            break 
        else: 
            aux = aux - 1
            
    if aux == 1:
        pks = []

    if len(pks) > 1 and len(crossPoints)>0:
        # taking closest crossing points before derivative peaks
        z1 = pks[0] - crossPoints
        z1 = z1[z1>=0]
        
        z2 = pks[1] - crossPoints
        z2 = z2[z2>=0]

        if len(z1) > 0 and len(z2) > 0:
            z1 = pks[0] - np.min(z1)
            z2 = pks[1] - np.min(z2)
            sep = (z2 - z1)/sample_rate 
        else: 
            sep = np.array([])
        
        if not (sep.size == 0): # it is possible if crossPoints behave badly 
            if sep > minimalPulseDuration*1.03 or sep < minimalPulseDuration*0.97:
                P = []
                t = []
            else:
                P = P[z1:z2]
                t = [z1 + np.round(PEndIdx/2), z2 + np.round(PEndIdx/2)]
                indx = np.argmin(P)
                if indx > len(P)/8 and indx < (len(P) - len(P)/8):
                    P = [] 
                    t = []
                else:
                    if diagnostics:
                        ax[1].plot(P, 'ro')
        else:
            P = []
            t = []
    else: 
        P = []
        t = []

    if diagnostics:
        plt.plot()
    
    succ = False if len(P) == 0 else True 
        
    return P, t, succ


def exportPulse(P, Q, tree, params, simSettings, whState=None, diagnostics=None, which=None): 
    
    if which is None: 
        diagnostics = False
        which = 0

    if whState is None: 
        whState = np.array([1])

    pulse                = {} 
    pulse['params']      = params
    pulse['tree']        = tree 
    pulse['simSettings'] = simSettings
    pulse['whState']     = whState
    
    pulse['sample_rate'] = simSettings['pps']
    pulse['HR']          = params['HR']
    pulse['T']           = params['T']
    pulse['P']           = P
    
    # extracting single profile, we assume that the solution mesh is 
    # multiplcation of period 
    
    # from radial artery
    pulse['PW'], pulse['t_period'], _ = selectOnePeriodVaso(P[21,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
    c = len(pulse['PW']) == 0

    # from central artery 
    pulse['CW'], pulse['t_periodCW'], _ = selectOnePeriodVaso(P[which,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
    c = c or (len(pulse['CW'])==0)
    
    # from brachial artery 
    pulse['BW'], pulse['t_periodBW'], _ = selectOnePeriodVaso(P[20,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
    c = c or (len(pulse['BW'])==0)
    
    if not c: 
        # making sure that CW has the same length as PW 
        if (len(pulse['PW'])>len(pulse['CW'])):
            pulse['CW'] = np.append(pulse['CW'], pulse['CW'][-1]*np.ones(len(pulse['PW']) - len(pulse['CW'])))
        elif (len(pulse['PW'])<len(pulse['CW'])):
            pulse['CW'] = pulse['CW'][:-(len(pulse['CW']) - len(pulse['PW']))] 

        pulse['SP'] = np.max(pulse['BW']) # systolic pressure (we assume this is user input in SpC machine)
        pulse['DP'] = np.min(pulse['BW']) # diastolic pressure (we assume this is user input in SpC machine)
        
        # normalizing radial pressure to SP and DP 
        pulse['PW'] = pulse['PW'] - np.min(pulse['PW'])
        pulse['PW'] = pulse['PW']/np.max(pulse['PW'])
        pulse['PW'] = pulse['PW'] *(pulse['SP'] - pulse['DP']) + pulse['DP']
        
        dt = 1000/pulse['sample_rate'] # in miliseconds 
        pulse['tmesh'] = dt*np.arange(len(pulse['PW']))
        pulse['PD'] = pulse['tmesh'][-1]+dt

        # computing stroke volume 
        Q_inlet = Q[0,:][int(pulse['t_period'][0]) : int(pulse['t_period'][1])]
        pulse['SV'] = np.sum(Q_inlet*dt)/1e3
        pulse['SVup'] = np.sum(Q_inlet[Q_inlet>0]*dt)/1e3

        pulse['succ'] = True
    else:
        pulse['succ'] = False

    return pulse

def exportPulseB(P, Q, tree, params, simSettings, whState, diagnostics=None, which=None): 
    
    if which is None: 
        diagnostics = False
        which = 0
    
    pulse                = {} 
    pulse['params']      = params
    pulse['tree']        = tree 
    pulse['simSettings'] = simSettings
    pulse['whState']     = whState
    
    pulse['sample_rate'] = simSettings['pps']
    pulse['HR']          = params['HR']
    pulse['P']           = P
    
    # extracting single profile, we assume that the solution mesh is 
    # multiplcation of period 
    
    # from radial artery 
    pulse['PW'], pulse['t_period'], _ = selectOnePeriodVaso(P[20,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
    c = len(pulse['PW']) == 0
    
    # from central artery 
    pulse['CW'], pulse['t_period'], _ = selectOnePeriodVaso(P[which,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
    c = c or (len(pulse['CW'])==0)
    
    # from brachial artery 
    pulse['BW'], pulse['t_period'], _ = selectOnePeriodVaso(P[20,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
    c = c or (len(pulse['BW'])==0)
    
    if not c:
        # making sure that CW has the same length as PW 
        if (len(pulse['PW'])>len(pulse['CW'])):
            pulse['CW'] = np.append(pulse['CW'], pulse['CW'][-1]*np.ones(len(pulse['PW']) - len(pulse['CW'])))
        elif (len(pulse['PW'])<len(pulse['CW'])):
            pulse['CW'] = pulse['CW'][:-(len(pulse['CW']) - len(pulse['PW']))]

        pulse['SP'] = np.max(pulse['BW']) # systolic pressure (we assume this is user input in SpC machine)
        pulse['DP'] = np.min(pulse['BW']) # diastolic pressure (we assume this is user input in SpC machine)
        
        # normalizing radial pressure to SP and DP 
        pulse['PW'] = pulse['PW'] - np.min(pulse['PW'])
        pulse['PW'] = pulse['PW']/np.max(pulse['PW'])
        pulse['PW'] = pulse['PW'] *(pulse['SP'] - pulse['DP']) + pulse['DP']
        
        dt = 1000/pulse['sample_rate'] # in miliseconds 
        pulse['tmesh'] = dt*np.arange(len(pulse['PW']))
        pulse['PD'] = pulse['tmesh'][-1]+dt

        # computing stroke volume 
        Q_inlet = Q[0,:][int(pulse['t_period'][0]) : int(pulse['t_period'][1])]
        pulse['SV'] = np.sum(Q_inlet*dt)/1e3
        pulse['SVup'] = np.sum(Q_inlet[Q_inlet>0]*dt)/1e3

        pulse['succ'] = True
    else:
        pulse['succ'] = False
    
    return pulse

def exportPulseANGE(P, Q, tree, params, simSettings, CH, whState=None, diagnostics=None, which=None): 
    """
    Export pulse waves from simulations

    Inputs
    ------
        P : array 
            simulated pulse waves from the model 
        Q : array 
            simulated flow waves from the model 
        params : dict 
            parameters of the simulation, see function `initializeParamsVaso`
        simSettings : dict
            simulation settings
        CH : dict 
            dict which keys as a numbers of the cuffs and values as the bool 
            telling if a given cuff is taken into consideration (val=1) or 
            not (val=0)
        whState : 
        diagnostics : bool, optional, default False
            if True, plot dividings from dPdt
    """

    if which is None: 
        diagnostics = False
    
    if whState is None: 
        whState = np.array([1])

    pulse                = {} 
    pulse['params']      = params
    pulse['tree']        = tree 
    pulse['simSettings'] = simSettings
    pulse['whState']     = whState
    
    pulse['sample_rate'] = simSettings['pps']
    pulse['HR']          = params['HR']
    pulse['T']           = params['T']
    pulse['P']           = P
    
    # extracting single profile, we assume that the solution mesh is 
    # multiplcation of period 

    c_CH1, c_CH2 = True, True 
    # radial artery (left)
    if CH[1]: 
        # here brachial has taken (radial has no: 21)
        pulse['PW_CH1'], pulse['t_periodCH1'], _ = selectOnePeriodVaso(P[21,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
        c_CH1 = len(pulse['PW_CH1']) == 0
        if not c_CH1:
            pulse['PW_CH1'] = normalization(pulse['PW_CH1'])
        else: 
        # if c_CH1:
            pulse['succ'] = False 
            return pulse
    # raial artery (right)
    if CH[2]:
        # here brachial has taken (radial has no: 7)
        pulse['PW_CH2'], pulse['t_periodCH2'], _ = selectOnePeriodVaso(P[7,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
        c_CH2 = len(pulse['PW_CH2']) == 0
        if not c_CH2:
            pulse['PW_CH2'] = normalization(pulse['PW_CH2'])
        else:
        # if c_CH2:
            pulse['succ'] = False 
            return pulse
            
    # # if we have two measurements from wrists, we want to consider two of them 
    # if CH[1] and CH[2]:
    #     if c_CH1 or c_CH2:
    #         pulse['succ'] = False 
    #         return pulse
    # # if we have only one measurement, we want to have non-empty divifing for it
    # else: 
    #     if c_CH1 and c_CH2:
    #         pulse['succ'] = False
    #         return pulse 

    c_CH3, c_CH4 = True, True
    if CH[3]: 
        pulse['PW_CH3'], pulse['t_periodCH3'], _ = selectOnePeriodVaso(P[47,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
        c_CH3 = len(pulse['PW_CH3']) == 0
        if not c_CH3:
            pulse['PW_CH3'] = normalization(pulse['PW_CH3'])
        else:
        # if c_CH3:
            pulse['succ'] = False 
            return pulse
    if CH[4]:
        pulse['PW_CH4'], pulse['t_periodCH4'], _ = selectOnePeriodVaso(P[47,:]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
        c_CH4 = len(pulse['PW_CH4']) == 0
        if not c_CH4:
            pulse['PW_CH4'] = normalization(pulse['PW_CH4'])
        else:
        # if c_CH4:
            pulse['succ'] = False 
            return pulse

    # from central artery -- important to compute SV 
    pulse['CW'], pulse['t_periodCW'], _ = selectOnePeriodVaso(P[0, :]/1333.322365, pulse['sample_rate'], pulse['T'], diagnostics)
    c = len(pulse['CW'])==0
    
    if not c: 
        dt = 1000/pulse['sample_rate'] # in miliseconds 
        
        for k, v in CH.items():
            if v != 0: 
                pulse[f'tmesh_CH{k}'] = dt*np.arange(len(pulse[f'PW_CH{k}']))
                pulse[f'PD_CH{k}'] = pulse[f'tmesh_CH{k}'][-1]+dt

        # computing stroke volume 
        Q_inlet = Q[0,:][int(pulse['t_periodCW'][0]) : int(pulse['t_periodCW'][1])]
        pulse['SV']   = np.sum(Q_inlet*dt)/1e3
        pulse['SVup'] = np.sum(Q_inlet[Q_inlet>0]*dt)/1e3
        pulse['succ'] = True
    else:
        pulse['succ'] = False

    return pulse

def analyzeWaves(data):
    # assumptions: tmesh is such that f(tmesh[0]) = f(tmesh[-1] + dt)
    # Results will be written to the data structure with "est" ending 

    # PARAMETERS OF THE PROCEDURE
    tHz     = 9     # discard frequencies equal to or above given Hz  
    order   = 1     # which order of finite difference scheme (1-6)

    # coefficient for first derivative using forward scheme
    coeff1st = np.array([[-1,       1,  0,      0,      0,      0,      0],
                         [-3/2,	    2,	-1/2,   0,      0,      0,      0],
                         [-11/6,	3,	-3/2,	1/3,	0,      0,      0],
                         [-25/12,	4,	-3,	    4/3,	-1/4,	0,      0],
                         [-137/60,	5,	-5,	    10/3,	-5/4,	1/5,    0],
                         [-49/20,	6,	-15/2,	20/3,	-15/4,	6/5,	-1/6]])

    coeff3rd = np.array([[-1,       3,      -3,         1,          0,          0,          0,          0,      0],
                         [-5/2,     9,      -12,        7,          -3/2,       0,          0,          0,      0],
                         [-17/4,    71/4,   59/2,       49/2,       -41/4,      7/4,        0,          0,      0],
                         [-49/8,	29,     -461/8,     62,         -307/8,	    13,         -15/8,      0,      0],
                         [-967/120,	638/15,	-3929/40,   389/3,	    -2545/24,   268/5,  -   1849/120,   29/15,  0],
                         [-801/80,	349/6,	-18353/120,	2391/10,	-1457/6,	4891/30,	-561/8,	    527/30,	-469/240]])
    
    whichToAnalyze = ['PW', 'CW']   # analyze both peripheral and central waves

    for wh in whichToAnalyze:

        profile = data[wh]
        tmesh   = data['tmesh']

        # FIRST STEP - APPROXIMATING PROFILE WITH FOURIER SERIES
        F = fft(profile)                     # calculating fast Fourier transform of the signal
        F[tHz-1 : -tHz+2] = 0                # neglecting frequencies >= tHz
        data[wh+'fft'] = np.real(ifft(F))    # saving approximated profile, using inverse fast fourier transform

        # SECOND STEP - CALCULATING DERIVATIVES
        Nt = len(tmesh)             # number of points in tmesh
        dt = tmesh[1]-tmesh[0]      # time step
        # defining frequencies
        if np.mod(Nt, 2):
            k = 2*np.pi/(tmesh[-1]+dt)*np.concatenate((np.arange(np.ceil(Nt/2)), np.arange(-np.ceil(Nt/2)+1, 0)), axis=None)
        else:
            k = 2*np.pi/(tmesh[-1]+dt)*np.concatenate((np.arange(Nt/2), 0, np.arange(-Nt/2+1, 0)), axis=None)

        # calculating first derivative in Fourier space 
        dpdtFFT = np.real(ifft(1j*k*F))

        
        data[wh+'dpdtfft'] = dpdtFFT 

        # calculating first derivative using finnite differences
        dpdtFN = coeff1st[order-1, 0] * profile[:-order]
        for ii in np.arange(order):
            if (order - ii - 1) > 0:
                dpdtFN = dpdtFN + coeff1st[order-1, ii+1]*profile[ii+1:-(order-ii-1)]
            else: 
                dpdtFN = dpdtFN + coeff1st[order-1, ii+1]*profile[ii+1:]
    
        dpdtFN = np.concatenate((dpdtFN/dt, np.zeros(order)), axis=None) # padding with zeros at the end
        data[wh+'dpdtfn'] = dpdtFN

        # calculating third derivative in fourier space
        dp3dt3FFTd = np.real(ifft(-1j*k**3*F))
        data[wh+'dp3dt3fftd'] = dp3dt3FFTd

        # calculating third derivative using finite differences (forward scheme)
        dp3dt3FN = coeff3rd[order-1,0]*profile[:-(order+2)]

        for ii in range(order+2):
            try:
                dp3dt3FN = dp3dt3FN + coeff3rd[order-1,ii+1]*profile[(ii+1):-(order+1-ii)]
            except ValueError: 
                dp3dt3FN = dp3dt3FN + coeff3rd[order-1,ii+1]*profile[(ii+1):]
        
        dp3dt3FN = np.concatenate((dp3dt3FN/dt**3, np.zeros(order+2)), axis=None)
        data[wh+'dp3dt3fn'] = dp3dt3FN

        # calculating third derivative using fourier approximation of finite
        # differences
        tHz2 = 9
        aux = fft(dp3dt3FN[:-3])

        aux[tHz2-1 :-tHz2+2] = 0 # neglecting frequencies >= tHz
        dp3dt3FFT = np.concatenate((np.real(ifft(aux)), 0, 0, 0), axis=None)
        data[wh+'dp3dt3fft'] = dp3dt3FFT

        # here select which one to take for analysis
        dpdt    = dpdtFN
        dp3dt3  = dp3dt3FFT

        # THIRD STEP - CALCULATING T1 AND T2
        # finding maximum of the pressure profile
        Pmax  = np.max(profile)
        indxP = np.argmax(profile)
        tPmax = tmesh[indxP]    # moment of the maximum pressure
   
        # finding maximum point of the derivative
        indx = np.argmax(dpdt)
        tdPmax = tmesh[indx]

        # locating first relative peak of third derivative after tdPmax
        l, _ = find_peaks(dp3dt3)
        pk, lk   = dp3dt3[l], tmesh[l]

        aux = tdPmax-lk 
        aux[aux<=0] = np.inf
        indxn = np.argmin(aux)
        tmax3rd = lk[indxn]
        max3rd  = pk[indxn]
        wA = dict()

        if (tPmax >= tmax3rd + 40):
            # print('Type A wave')
            wA['type'] = 1
            wA['P2'] = Pmax     # pressure at T2
            wA['T2'] = tPmax    # moment of T2
            f_P1 = interpolate.interp1d(tmesh, profile)
            wA['P1'] = f_P1(tmax3rd)
            wA['T1'] = tmax3rd
        else: 
            # display('Type B or C wave')
            wA['type'] = 2
            wA['T1'] = tPmax
            wA['P1'] = Pmax
            # finding first negative to positive switch of third derivative to
            # the right of tmax3rd
            Tzc3rdmp = 0
            for i in np.arange(np.argwhere(tmesh>tmax3rd)[0,0], len(tmesh)+1):
                if (dp3dt3[i-1]<0 and dp3dt3[i]>0):
                    f_Tzc3rdmp = interpolate.interp1d(np.array([dp3dt3[i-1], dp3dt3[i]]),
                                                      np.array([tmesh[i-1], tmesh[i]]))
                    Tzc3rdmp = f_Tzc3rdmp(0)
                    break
            
            if ((Tzc3rdmp < tPmax) or (Tzc3rdmp > 300)):
                # display('Second shoulder not found')
                wA['T2'] = 0
                wA['P2'] = 9999
            else:
                wA['T2'] = Tzc3rdmp
                f_P2 = interpolate.interp1d(tmesh,profile)
                wA['P2'] = f_P2(Tzc3rdmp)

        # FOURTH STEP - CALCULATING ED
        if wA['T2'] > 0: # if second shoulder located
            Tins = 0
            # finiding first moment of third derivative crossing from positive
            # to negative after next relatively the biggest peak after (T2)
            aux3 = np.arange(len(lk)-indxn-1) # look at all peaks after
            aux2 = (lk[indxn+aux3+1] - lk[indxn]) < 250 # take only those before 250 ms
            aux3 = aux3[aux2]
            
            if aux3.size == 0: # if no peak after T2 and before 250 ms
                indxn2 = 0
            else:
                indxn2 = np.argmax(pk[indxn+aux3+1])

            for i in np.arange(np.argwhere(tmesh>lk[indxn+indxn2+1])[0,0], len(tmesh)+1):
                if (dp3dt3[i-1]>0 and dp3dt3[i]<0):
                    f_Tins = interpolate.interp1d(np.array([dp3dt3[i-1], dp3dt3[i]]),
                                                  np.array([tmesh[i-1], tmesh[i]]))
                    Tins = f_Tins(0)
                    break
            
            wA['ED']  = Tins
            f_PED = interpolate.interp1d(tmesh, profile)
            wA['PED'] = f_PED(Tins)
        else:
            Tins = 0
            # finiding first moment of third derivative crossing from positive
            # to negative after first shoulder (T1)
            for i in np.arange(np.argwhere(tmesh>wA['T1']+60)[0,0], len(tmesh)+1):
                if (dp3dt3[i-1]>0 and dp3dt3[i]<0):
                    f_Tins = interpolate.interp1d(np.array([dp3dt3[i-1], dp3dt3[i]]),
                                                  np.array([tmesh[i-1], tmesh[i]]))
                    Tins = f_Tins(0)
                break

            wA['ED'] = Tins
            f_PED = interpolate.interp1d(tmesh, profile)
            wA['PED'] = f_PED(Tins)
        
        # calculating augumentation index 
        wA['cAI'] = (wA['P2']-profile[0])/(wA['P1']-profile[0])*100

        wA['SP'] = np.max(profile)
        wA['DP'] = profile[0]

        for k, v in wA.items(): 
            if type(v) is np.ndarray: 
                wA[k] = v.item()

        data[wh+'calc'] = wA
    
    return data

def F_SphygmoCor(x, tree, simSettings, data, paramsToFit):

    # do the deepcopy of the tree (important!)
    treeP = copy.deepcopy(tree)

    params = initializeParamsVaso(data, treeP, sph=1)
    params = pointToParamsVaso(x, params, paramsToFit)

    # scaling tree resistances and compliances
    treeP['treeStructure'][5, :] = treeP['treeStructure'][5, :] * params['scaleRes']
    treeP['treeStructure'][6, :] = treeP['treeStructure'][6, :] * params['scaleComp']

    # solve the model for gicen parameters
    P, Q, whState, _ = solveModelSpeedVaso(treeP, params, simSettings)

    if P.shape[1] < simSettings['secondsToSimulate'] * simSettings['pps']: 
        pulse = None 
        err   = 1000 * np.ones_like(paramsToFit, np.float16)
    else: 
        if data['SUBTYPE'] == 'RADIAL':
            pulse = exportPulse(P, Q, treeP, params, simSettings)
        else: # take brachial
            pulse = exportPulseB(P, Q, treeP, params, simSettings)

        if pulse['succ']:
            # pulse = analyzeWaves(pulse)
            f_ps  = interpolate.interp1d(pulse['tmesh'], pulse['PW'])
            f_PW  = interpolate.interp1d(data['tmesh'], data['PW'])

            # take SP and DP to the error
            Y1 = data['SP'] - pulse['SP']
            Y2 = data['DP'] - pulse['DP']
            
            # take T1 and PT1 to the error
            Y3 = data['PPT1'] - f_ps(data['PT1'])
        
            # take T2 and PT2 to the error
            if data['PPT2'] == 9999 | data['PT2'] == 9999:
                Y4 = 0
            else:
                Y4 = data['PPT2'] - f_ps(data['PT2']) 

            # take ED and PED to the error
            Y5 = data['PESP'] - f_ps(data['ED'])
        
            # take mean pressure to the error
            Y6 = np.mean(data['PW'])-np.mean(pulse['PW'])

            MilisecondsToConsider = data['C_PERIOD']
            NumberOfPoints = 5 
            PointsToConsider = np.linspace(MilisecondsToConsider//2, MilisecondsToConsider-5, NumberOfPoints)
            try:
                YPoints = np.array([f_PW(ms) - f_ps(ms) for ms in PointsToConsider])
            except ValueError as e:
                print(f"ValueError at interpolation: {e}")
                YPoints = 100 * np.ones_like(PointsToConsider)

            # take the moment of the maximum
            # calculate moment of maximum for the data peripheral wave
            indx = np.argmax(data['PW'])
            TmaxD = data['tmesh'][indx]
            # calculate moment of maximum for the model wave
            indx = np.argmax(pulse['PW'])
            TmaxM = pulse['tmesh'][indx]
            Y12 = (TmaxD - TmaxM)/5

            err = np.concatenate((np.array([Y1, Y2, Y3, Y4, Y5, Y6, Y12]), 
                                 YPoints), axis=None)
            err[np.isnan(err)] = 100
            
        else: 
            err = 1000 * np.ones(12)
            pulse = None

    return err, pulse, P, Q


def ToFourierSeries(tmesh, pulse, n=None):
    """based on https://pages.mtu.edu/~tbco/cm416/fft1.pdf
    
    Short Description
    -----------------
    We approximate data by following series:
    
    `f(t) = a0/2 + sum(an * cos(omega t) + bn * sin(omega t))`
    
    where `omega = (2 * pi)/T` and `T` is the period of signal. 
    Approximation will be on tmesh.  

    Parameters 
    ----------
    tmesh : array
        array with mesh with time samples for measured pulse waveform
    pulse : array 
        array with measured pulse waveform
    n : int, optional
        number of harmonics to compute, if n > len(tmesh)//2 or n is None, 
        then n == len(tmesh)//2

    Returns
    -------
    an : array 
        cosinus coefficients from Fourier series
    bn : array
        sinus coefficients from Fourier series
    a0 : a0 coefficient
    fappox : array 
        approximated function 

    Usage
    -----
    We want to expand pulse waveform in Fourier Series. Algorithms 
    with computational integrating may not be efficient in the case 
    when we want to perform many computations. To impove efficency 
    it is worth to use fourier fast transform. Thanks to some 
    transformations we gain coefficients with `n` harmonics, where 
    `n = N//2`, where `N = len(tmesh)` - number of samples. 
    """

    T = tmesh[-1]            # period
    ffreq = 2 * np.pi/T      # fundamental frequency
    N = len(pulse)           # number of discrete data
    N = N - 1                # for trapezoid rule of integration

    f = copy.deepcopy(pulse)
    fhat = f 
    fhat[0] = (f[0] + f[-1])/2

    F = fft(fhat, N)

    F = F[:N//2]
    k = np.arange(0, N//2-1)
    omega = k * ffreq

    an =  2 * np.real(F)/N
    a0 =  an[0]/2
    bn = -2 * np.imag(F)/N

    fapprox = np.ones(len(tmesh)) * a0

    if (n is None) or (n > len(an)//2):
        n = len(an)//2

    for k in range(n+1):
        fapprox = fapprox + an[k+1]*np.cos(omega[k+1]*tmesh)\
                            + bn[k+1]*np.sin(omega[k+1]*tmesh)

    return a0, an[:n], bn[:n], fapprox

def F_SphygmoCorFFT(x, tree, simSettings, data, paramsToFit, n=None, changeParams=None):
    """
    Objective function. Function computes fft from proposed by optimization 
    algorithm pulse, then compares fourier coefficients a0, an and bn up to 
    n-th element of fourier series. Fourier series are computed on mesh 
    from data. 

    Parameters
    ----------
        x : array 
            values of the parameters from paramsToFit
        tree : array
            biffurcation tree array 
        simSettings : array 
            simulation settings
        data : dict
            dictionary with data connected with a patient with pulse recording
        paramsToFit : array 
            search parameters 
        n : int, optional
            number of elements in Forier series to expand. If n is not specified,
            then n = 20
        changeParams : dict, optional
            change values of some aditional parameters (ex. previousely generated)
    
    Returns
    -------
        err : array
            error between n parameters of the Fourier expansion
        pulse : dict 
            dict with elements as in exportPulse function return
        P : matrix 
            matirx with simulated pulse, rows correspond to the following branches
        Q : matrix 
            matrix with simulated flow, rows correspond to the following branches
    """
    if n is None: 
        n = 20 

    # do the deepcopy of the tree (important!)
    treeP = copy.deepcopy(tree)
    params = initializeParamsVaso(data, treeP, sph=1)
    params = pointToParamsVaso(x, params, paramsToFit)
    
    # print chosen parameters
    # print([f"{paramsToFit[i]}: {x[i]} " for i in range(len(x))])

    if changeParams is not None: 
        params = pointToParamsVaso(changeParams['x'], params, changeParams['params'])
    # scaling tree resistances and compliances
    treeP['treeStructure'][5, :] = treeP['treeStructure'][5, :] * params['scaleRes']
    treeP['treeStructure'][6, :] = treeP['treeStructure'][6, :] * params['scaleComp']

    # solve the model for gicen parameters
    P, Q, _, _ = solveModelSpeedVaso(treeP, params, simSettings)

    # plt.figure()
    # plt.plot(P[0, :]/1333)
    # plt.show()
    if P.shape[1] < simSettings['secondsToSimulate'] * simSettings['pps']: 
        pulse = None 
        err   = 100 * np.ones(2*n, np.float16)
    else: 
        if data['SUBTYPE'] == 'RADIAL':
            pulse = exportPulse(P, Q, treeP, params, simSettings)
        else: # take brachial
            pulse = exportPulseB(P, Q, treeP, params, simSettings)

        if pulse['succ']:
            # interpolating pulse from computed data
            # f_ps = interpolate.interp1d(pulse['tmesh'], pulse['PW'])

            # # compare max val of the functions 
            # PmaxDiff = data['SP'] - pulse['SP']
            # PminDiff = data['DP'] - pulse['DP']
            # take the moment of the maximum
            # calculate moment of maximum for the data peripheral wave
            # indx = np.argmax(data['PW'])
            # TmaxD = data['tmesh'][indx]
            # calculate moment of maximum for the model wave
            # indx = np.argmax(pulse['PW'])
            # TmaxM = pulse['tmesh'][indx]

            a0_p, an_p, bn_p, _ = ToFourierSeries(data['tmesh'], pulse['PW'], n)
            a0_d, an_d, bn_d, _ = ToFourierSeries(data['tmesh'], data['PW'], n)
            a0_diff = a0_p - a0_d
            an_diff = an_p - an_d
            bn_diff = bn_p[1:] - bn_d[1:]
            err = np.concatenate((a0_diff, an_diff, bn_diff), axis=None)
            err[np.isnan(err)] = 100
            
            # if pulse['SV'] < 30.0: # penalty for being < 30 ml
            #     err = 3 * err
                # print(f"err_vect: {err}")
        else: 
            # x_list = P[21, :]/1333
            err = 100 * np.ones(2*n, np.float16)
            pulse = None

    return err, pulse, P, Q

def F_ANGE_FFT(x, tree, simSettings, data, paramsToFit, CH, n=None, changeParams=None):
    """
    Objective function for ANGE data where more than one cuff may be considered. 
    Function computes fft from proposed by optimization algorithm pulse, then 
    compares fourier coefficients a0, an and bn up to n-th element of fourier 
    series. Fourier series are computed on mesh from data. 

    Parameters
    ----------
        x : array 
            values of the parameters from paramsToFit
        tree : array
            biffurcation tree array 
        simSettings : array 
            simulation settings
        data : dict
            dictionary with data connected with a patient with pulse recording
        paramsToFit : array 
            search parameters 
        CH : dict
            dict with info about which cuffs we are consider
        n : int, optional
            number of elements in Forier series to expand. If n is not specified,
            then n = 20
        changeParams : dict, optional
            change values of some aditional parameters (ex. previousely generated)
    
    Returns
    -------
        err : array
            error between n parameters of the Fourier expansion
        pulse : dict 
            dict with elements as in exportPulse function return
        P : matrix 
            matirx with simulated pulse, rows correspond to the following branches
        Q : matrix 
            matrix with simulated flow, rows correspond to the following branches
    """
    if n is None: 
        n = 20 

    # do the deepcopy of the tree (important!)
    treeP = copy.deepcopy(tree)
    params = initializeParamsVaso(data, treeP)
    params = pointToParamsVaso(x, params, paramsToFit)
    if changeParams is not None: 
        params = pointToParamsVaso(changeParams['x'], params, changeParams['params'])
    # scaling tree resistances and compliances
    treeP['treeStructure'][5, :] = treeP['treeStructure'][5, :] * params['scaleRes']
    treeP['treeStructure'][6, :] = treeP['treeStructure'][6, :] * params['scaleComp']
    # print(f"Params: Emax = {params['Emax']}, tm = {params['tm']}, k1 = {params['k1']/(3*1e6)}, k3 = {params['k3']/(0.4*8.65*1e5)}, sR = {params['scaleRes']}, sC = {params['scaleComp']}")
    # solve the model for gicen parameters
    P, Q, _, _ = solveModelSpeedVaso(treeP, params, simSettings)
    # P, Q = Pall, Qall 
    # figure(figsize=(12, 5))
    # def make_fig_succ(show_once=True):
    #     plt.plot(tx_list, x_list)
    #     plt.plot(ty_list, y_list)
    #     plt.title("fitting")
    
    # def make_fig_nosucc1(show_once=True):
    #     plt.plot(x_list) 
    #     plt.title("1") 
    # def make_fig_nosucc2(show_once=True):
    #     plt.plot(x_list) 
    #     plt.title("2")     

    if P.shape[1] < simSettings['secondsToSimulate'] * simSettings['pps']: 
        pulse = None 
        err   = 100 * np.ones_like(paramsToFit, np.float16)
        # x_list = P[21, :][::10]/1333
    else: 
        pulse = exportPulseANGE(P, Q, treeP, params, simSettings, CH)
        err = dict()

        if pulse['succ']:
            for k, v in CH.items(): 
                if v == 1: 
                    a0_m, an_m, bn_m, _ = ToFourierSeries(data['tmesh'], pulse[f'PW_CH{k}'], n)
                    a0_s, an_s, bn_s, _ = ToFourierSeries(data['tmesh'], data[f'PW_CH{k}'], n)
                    a0_diff = a0_m - a0_s
                    an_diff = an_m - an_s
                    bn_diff = bn_m - bn_s

                    # if k == 1:
                    #     x_list = pulse['PW_CH1'][::10]
                    #     tx_list = pulse['tmesh_CH1'][::10]
                    #     y_list = data['PW_CH1'][::10]
                    #     ty_list = data['tmesh'][::10]
                    # if k != 1: 
                    #     a0_diff *= 500
                    #     an_diff *= 100 
                    #     bn_diff *= 100
                                # penalty for pressure being < 60 mmHg and > 120 mmHg 

                    if (np.max(P[21, :]/1333) > 135) | (np.min(P[21, 6500:]/1333) < 55):
                        err[k] = np.concatenate((a0_diff, an_diff, bn_diff), axis=None)*10
                    else:
                        err[k] = np.concatenate((a0_diff, an_diff, bn_diff), axis=None)
                    err[k][np.isnan(err[k])] = 100

            # if pulse['SV'] < 30.0: # penalty for being < 30 ml
            #     err = 3 * err
                # print(f"err_vect: {err}")
        else: 
            # x_list = P[21, :][::10]/1333          
            for k, v in CH.items():
                if v == 1:
                    err[k] = 100 * np.ones(2*n+1)
                    pulse = None

    return err, pulse, P, Q

def opt_funcFFT_PSO(X, tree, simSettings, data, paramsToFit, nrm=True, 
                    bounds_real=None, n=None, changeParams=None, ange=None):
    """
    Cost function wich returns errors to minimize. Function dedciated to 
    pyswarms optimization: https://pyswarms.readthedocs.io/en/latest/. 
    Function computes error using parameters from Fourier series,
    see : `F_SphygmoCorFFT`

    Parameters
    ----------
        X : array_like 
            paricles 
        tree : array_like 
            cardiovascular biffurcation tree
        simSettings : array_like 
            settings for the simulation 
        data : dict 
            dict with information about patient with pulse recordings
        paramsToFit : array_like
            array with parameters to optimize
        nrm : bool, default True 
            if True, X is trated as normalized and Y is return to previous
            values
        bounds_real : array_like, default None
            if nrm True, bounds_real are not normalized bouds of 
            parameters
        n : int, default None 
            number of elements in Forier series to expand. If n is not specified,
            then n = 20
        change_params : dict, default None 
            change values of some aditional parameters (ex. previousely generated)
        ange : dict
            TODO

    Returns
    -------
        err : array_like
            Array with errors computed for particular swarms
    """ 
    n_particles = X.shape[0]
    err = 100*np.ones(n_particles)

    for i in range(n_particles):
        Y = np.array([X[i][j] * (bounds_real[j][1] - bounds_real[j][0]) + bounds_real[j][0] for j in range(len(X[i]))]) if nrm else X 
        if ange:
            err0, _, _, _ = F_ANGE_FFT(Y, tree, simSettings, data, paramsToFit, ange, n)
            try:
                err[i] = np.sum([np.sum([(e*100)**2 for e in v]) for _, v in err0.items()])
            except AttributeError: 
                err[i] = 1e6
        else:
            err0, _, _, _ = F_SphygmoCorFFT(Y, tree, simSettings, data, paramsToFit, n, changeParams)
            #err[i]  = np.linalg.norm(err0, 2)
            err[i] = np.sum([e**2 for e in err0])
        
        if data['message']:
            print(f"\n\n ==== err_{i} = {err[i]:.2f} === \n\n")

    return err


def opt_funcFFT_scipy(X, tree, simSettings, data, paramsToFit, 
                      bounds_real, n=None, changeParams=None, ange=None):
    """
    Cost function wich returns errors to minimize. Function dedciated to 
    scipy optimizations. Function computes error using parameters from Fourier series,
    see : `F_SphygmoCorFFT`

    Parameters
    ----------
        X : array_like 
            particle, to evaluate 
        tree : array_like 
            cardiovascular biffurcation tree
        simSettings : array_like 
            settings for the simulation 
        data : dict 
            dict with information about patient with pulse recordings
        paramsToFit : array_like
            array with parameters to optimize 
        ange : dict, optional
            TODO

    Returns
    -------
        err : array_like
            Array with computed error
    """

    Y = np.array([X[i] * (bounds_real[i][1] - bounds_real[i][0]) + bounds_real[i][0] for i in range(len(X))])
    
    if ange:
        err0, _, _, _ = F_ANGE_FFT(Y, tree, simSettings, data, paramsToFit, ange, n)
        k = np.sum(list(ange.values()))
        
        try:
            err  = np.sum([np.sum([(e*100)**2 for e in v]) for _, v in err0.items()])
            err0 = np.concatenate([v for v in err0.values()])
        except AttributeError:
            err0 = np.full(k*(2*n+1), 100)
            err  = np.sum([e**2 for e in err0])
        # lerr= len(err0)
        # print(f"len of err0: {lerr}, len of k: {k*(2*n+1)}")
        
        if len(err0) != k*(2*n+1):
            # lerr= len(err0)
            # print("something wrong with len in opt scipy")
            # print(f"len of err0: {lerr}, len of k: {k*(2*n+1)}")
            err0 = np.full(k*(2*n+1), 100)
            err  = np.sum([e**2 for e in err0])

    else:
        err0, _, _, _ = F_SphygmoCorFFT(Y, tree, simSettings, data, paramsToFit, n, changeParams)
        #err[i]  = np.linalg.norm(err0, 2)
        err = np.sum([e**2 for e in err0])

    if data['message']:
        print(f"\n\n ==== err = {err:.2f} ==== \n\n")

    return err0

def Optimize_PSO(paramsToFit, pso_options, lb, ub, tree, simSettings, data, 
                 nrm=True, n=None, changeParams=None, ange=None):
    """
    Optimizer using function from package `pyswarms`.
    Pyswarms: (https://pyswarms.readthedocs.io/en/latest/)

    Parameters
    ----------
        paramsToFit : list[str]
            list of strings characterizing which parameters will be 
            optimized 
        pso_options: dict,
            dict with options for PSO algorithm. Dict should contain 
            `n_particles` and `iters`
        lb : array_like, list 
            list of lower bounds in searching space
        ub : array_like, list
            list of right bounds in searching space 
        tree : dict(), 
            tree for simulation purposes, see: `F_Spygmocor` functions, 
        simSettings : dict() 
            simulation settings for simulation purposes, see: `F_Spygmocor` functions, 
        data : dict(), 
            data about settings for simulation purposes, see: `F_Spygmocor` functions,
        nrm : bool, optional
            normalization of the parameters in the space        
        options : dict(), optional
            dictionary with parameters `c1`: (coginitive), `c2`: social 
            and `w`: inertia
        n : int, default None 
            number of elements in Forier series to expand. If n is not specified,
            then n = 20
        changeParams : dict, default None 
            change values of some aditional parameters (ex. previousely generated), 
            dict should contain keys `params` -> params which we should change and 
            `x` -> values of these params. These parameters do not take part in 
            fitting procedure (!)
        ange : dict, default None 
    
    Returns
    -------
        joint_vars : array_like
            array with not - normalized parameter values 
        res : array_like
            array with normalized parameter values (if nrm = True)
    """
    dim = len(paramsToFit)
    
    if nrm: 
        bounds_real = list(zip(lb, ub))
        bounds = (np.zeros_like(lb), np.ones_like(ub))
    else:
        bounds_real = None 
        bounds = list(zip(lb, ub))

    if not 'c1' in pso_options.keys():
        pso_options['c1'] = 2.8 
    if not 'c2' in pso_options.keys():
        pso_options['c2'] = 1.3
    if not 'w' in pso_options.keys():
        k = 1 # k can take values between 0 and 1, but is usually set to 1 (Montes de Oca et al., 2006)
        pso_options['w'] = 2*k/(abs(2-(pso_options['c1']+pso_options['c2'])-np.sqrt((pso_options['c1']+pso_options['c2'])**2-4*(pso_options['c1']+pso_options['c2']))))

    optimizer = ps.single.GlobalBestPSO(n_particles=pso_options['n_particles'], 
                                        dimensions=dim,
                                        bounds=bounds,
                                        options=pso_options)

    _, res = optimizer.optimize(opt_funcFFT_PSO,
                                    iters= pso_options['iters'],
                                    tree=tree,
                                    simSettings=simSettings,
                                    data=data,
                                    paramsToFit=paramsToFit,
                                    nrm = nrm, 
                                    bounds_real=bounds_real,
                                    n=n, 
                                    changeParams=changeParams, 
                                    ange=ange)

    if nrm:
        joint_vars = np.array([res[i] * (bounds_real[i][1] - bounds_real[i][0]) + bounds_real[i][0] for i in range(len(res))])
    else: 
        return joint_vars

    return joint_vars, res

def MAPE_MSE(dataReal, dataSimul): 
    """
    Computes mean absolute percentage error between `x` and `y`.

    Parameters
    ----------
        x : array_like
            data from measurement, see `pulse`
        y : array_like 
            data from simulation, see `dataP`
    
    Returns
    -------
        mape : float 
            MAPE error
        mse : float
            MSE error

    More: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    RealMeshPoints = dataReal['tmesh']
    RealData = dataReal['PW']

    SimulMeshPoints = dataSimul['tmesh']
    SimulData = dataSimul['PW']

    if len(SimulData) < len(RealData): 
        RealMeshPoints = dataSimul['tmesh']
        RealData = dataSimul['PW']

        SimulMeshPoints = dataReal['tmesh']
        SimulData = dataReal['PW']

    SimulDataInterpolated = np.zeros(len(RealData))
    finerpolate = interpolate.interp1d(SimulMeshPoints, SimulData,
                                       fill_value="extrapolate")
    
    for i in range(len(RealData)):
        ms = RealMeshPoints[i]
        SimulDataInterpolated[i] = finerpolate(ms)

    At = RealData
    Ft = SimulDataInterpolated

    mape = np.sum(np.abs(At - Ft)/At)/len(Ft)
    mse  = np.mean(np.square((At-Ft)))

    return mape, mse



