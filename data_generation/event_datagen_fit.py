# %%

import eos
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os

np.random.seed(34)
rng = np.random.mtrand.RandomState(34)

# Change this each time round
save_dir = '/home/hep/md1021/msci_project/dstore_csr_fit'

os.makedirs(save_dir, exist_ok=True)

output_path = Path(save_dir)

# Loop through N-times to produce different configurations' events
for j in range(251):
    
    dstarlnu_kinematics = eos.Kinematics(**{
        'q2':            2.0,  'q2_min':            0.02,     'q2_max':           10.5,
        'cos(theta_l)':  0.0,  'cos(theta_l)_min': -1.0,      'cos(theta_l)_max': +1.0,
        'cos(theta_d)':  0.0,  'cos(theta_d)_min': -1.0,      'cos(theta_d)_max': +1.0,
        'phi':           0.3,  'phi_min':           0.0,      'phi_max':           2.0 * np.pi
    })

    # - Define Model, Form-Factors, and Lepton Considered -  
    options = eos.Options(**{
    'form-factors' : 'BSZ2015',  
    'model': 'WET',              
    'lepton' : 'mu'})

    # - Complicated method to set WC Correctly - Cannot just enter like w/ options - 
    params = eos.Parameters()
    param_wc = params['cbmunumu::Re{cSR}']
    param_wc.set((-10 + (j // 50) * 5))

    # - Initialise the decay PDF Object - 
    # =- This randomly changes q2, cos_theta_l and cos_theta_d for some reason? -=
    dstarlnu_pdf = eos.SignalPDF.make('B->D^*lnu::d^4Gamma', params, dstarlnu_kinematics, options)

    # - Generating MC Samples -
    dstarlnu_samples, _ = dstarlnu_pdf.sample_mcmc(N=200000, stride=5, pre_N=10000, preruns=2, rng=rng)

    # create dataframe out of the samples you get
    col_names = ["q2", "cos_theta_l", "cos_theta_d", "phi"]
    df_samples = pd.DataFrame(dstarlnu_samples, columns = col_names)

    # store data as a csv
    df_samples.to_csv(output_path / f'model_WET_mu_{j}.csv', index=False)

    # store null values for wilson coefficients under SM so inputs have the same shame
    # merge the wilson coeffs with 
    conf_final = {'wc' : params['cbmunumu::Re{cSR}'].evaluate()}

    # store the configuration file - including the wilson coefficients.
    with open(output_path / f'model_WET_mu_{j}.json', 'w') as f:
        json.dump(conf_final, f, indent=4)

    print(f'Configuration Number: {j} Complete. Wilson Coefficient {param_wc.evaluate():.3f}')



# %%
