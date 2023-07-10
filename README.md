# PWA_public
by **Kamil Wołos** and **Jan Poleszczuk**


## Installation 
1. Clone repository. 
2. Install conda environment:  
    - change folder to `packages` and find `env.yaml` file,  
    - install environment using command `conda env create --name pwa --file env.yaml`, where `pwa` is the name of creating env
3. After installation change conda env to already created `pwa` using command `conda activate pwa`. 
4. Go to the file `PythonPWAExtension_library` &rarr; `SolverSpeed` &rarr; `setup.py`. It is the installation file with python wrapper to cpp solver. 
5. Open `setup.py` with editor and change `extra_compile_args` and `extra_link_args` to paths with `gsl` library (if needed). 
6. Install solver as the python library using command `python3 setub.py install`. After installation we should receive following message: `Finished processing dependencies for PythonPWAExtension==0.9.6`  
    - to be sure, you can still check that everythog was succesful. To do this run (in the given env!) `python3` and execute following commands: 
    ```
    >>> import pkg_resources
    >>> pkg_resources.require("PythonPWAExtension")[0].version
    ```
    You should receive message: `0.9.6`.
7. Additionaly you can change number of threads in `omp_set_num_threads` in the file `PythonPWAExtension_library` &rarr; `SolverSpeed` &rarr; `C++ source code` &rarr; `PythonPWAspeed.cpp` (if needed). 

## Using the wrapper
1. Get the data to fit from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8131393.svg)](https://doi.org/10.5281/zenodo.8131393).  Paste `patients_csv_HD` into folder `HD_patients` and `patients_csv` into folder `healthy_patients`.  
2. Change `.csv` files into numpy dict which is readable to the script, by running command `python3 csv_to_disct.py`. You should receive `patients.npy` or `patientsHD.npy` in the working directory. 
3. In the `healthy_patients` and `HD_patients` you can find file `script.sh`. Make this file executable, using command `chmod +x script.sh`. To start fitting data to the patients use command `./script.sh`. 
