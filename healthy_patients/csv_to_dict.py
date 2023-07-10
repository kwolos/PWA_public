# %% csv to dict
import os 
import pandas as pd
import numpy as np 

def csv_to_dict(filename, orient='records'):
    df = pd.read_csv(filename)
    data = df.to_dict(orient)
    return data

patients  = os.listdir('patients_csv')
Patients = {} 

for pt in patients: 
    Patients[pt] = {} 
    files = os.listdir(os.path.join('patients_csv', pt))
    for p in files:    
        file = p.split('.')[0]
        if file.startswith('PWA'):
            
            if file.split('_')[0] not in Patients[pt]:
                Patients[pt][file.split('_')[0]] = {} 
            
            if file.split('_')[1] == 'data':
                Patients[pt][file.split('_')[0]]['data'] = csv_to_dict(os.path.join('patients_csv', pt, p))[0]
            elif file.split('_')[1] == 'pulse':
                Patients[pt][file.split('_')[0]]['pulse'] = csv_to_dict(os.path.join('patients_csv', pt, p), orient='list')
        else: 
            Patients[pt][file] = csv_to_dict(os.path.join('patients_csv', pt, p))[0]

np.save('patients.npy', Patients)