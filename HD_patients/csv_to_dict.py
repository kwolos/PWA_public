# %%
import os 
import pandas as pd
import numpy as np

def csv_to_dict(filename, orient='records'):
    df = pd.read_csv(filename)
    data = df.to_dict(orient)
    return data

patients  = os.listdir('patients_csv_HD')
Patients = {} 

for pt in patients: 
    Patients[pt] = {} 
    for br in os.listdir(os.path.join('patients_csv_HD', pt)):
        Patients[pt][br] = {} 
        for mt in os.listdir(os.path.join('patients_csv_HD', pt, br)):
            Patients[pt][br][mt] = {}
            
            csv_files = os.listdir(os.path.join('patients_csv_HD', pt, br, mt))

            for p in csv_files: 
                if p.startswith('PWA'): 
                    if 'PWA' not in Patients[pt][br][mt]: 
                        Patients[pt][br][mt]['PWA'] = {} 
                    if p.split('_')[1] == 'data.csv':
                        Patients[pt][br][mt]['PWA']['data'] =  csv_to_dict(os.path.join('patients_csv_HD', pt, br, mt, p))[0]
                    elif p.split('_')[1] == 'pulse.csv':
                        Patients[pt][br][mt]['PWA']['pulse'] =  csv_to_dict(os.path.join('patients_csv_HD', pt, br, mt, p), orient='list')
                    else: 
                        print(f"something went wrong with PWA: patient {pt}, {br}, {mt}, {p}")
                elif p.startswith('PF'): 
                    Patients[pt][br][mt]['PF'] = {}
                    Patients[pt][br][mt]['PF'] = csv_to_dict(os.path.join('patients_csv_HD', pt, br, mt, p))[0]
                else: 
                    print(f"something went wrong with PF: patient {pt}, {br}, {mt}, {p}")

np.save('patientsHD.npy', Patients)