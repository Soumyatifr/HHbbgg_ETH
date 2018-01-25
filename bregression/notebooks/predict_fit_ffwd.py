# # Characterize FFWD NN output
#load libs
import keras.models
import os
import bregnn.io as io
import matplotlib.pyplot as plt
import sys
import json
from optparse import OptionParser, make_option
sys.path.insert(0, '/users/nchernya/HHbbgg_ETH/bregression/python/')
import plotting_utils as plotting


parser = OptionParser(option_list=[
    make_option("--training",type='string',dest="training",default='mse'),
    make_option("--inp-dir",type='string',dest="inp_dir",default='/users/nchernya//HHbbgg_ETH/root_files/'),
    make_option("--inp-file",type='string',dest='inp_file',default='ttbar_RegressionPerJet_heppy_energyRings3_forTesting.hd5'),
    make_option("--out-dir",type='string',dest="out_dir",default='/users/nchernya//HHbbgg_ETH/bregression/output_root/'),
])

## parse options
(options, args) = parser.parse_args()
input_trainings = options.training.split(',')

# ## Read test data and model
# load data
base_dir = '/scratch/snx3000/musella/bregression'
#data = io.read_data(base_dir+'/ttbar_unweighted_full80M_selected_test.hd5', columns = None )
data = io.read_data('%s%s'%(options.inp_dir,options.inp_file),columns=None)

for idx,name in enumerate(input_trainings):
    # list all model files in the training folder
    #target = '/users/musella/jupyter/bregression/hybrid_cfg'
    target='/users/nchernya/HHbbgg_ETH/bregression/notebooks/'+input_trainings[idx]
    models = get_ipython().getoutput('ls -t $target/*.hdf5')
    models
  
    # read training configuration
    import json
    with open('%s/config.json' % target) as fin:
        config = json.loads(fin.read())
    config
  
    # ## Compute predictions
    features = config['options']['features'].split(',')
    X = data[features].values
    
    model = keras.models.load_model(models[0],compile=False)
    y_pred = model.predict(X)
  
    # *Note*: the target is typically normalized in the training y = (y-mu)/sigma
    # ## Convert raw prediction into actual scale and resolution estimation

    if y_pred.shape[1] == 1: # with one output we only have a scale correction
        corr = y_pred
        res = None
    elif y_pred.shape[1] == 2: # with two outputs first is mean and second is sigma
        corr = y_pred[:,0]
        res = y_pred[:,1]
    elif y_pred.shape[1] == 3: # assume that 3 outputs are mean + 2 quantile
        corr = y_pred[:,0]
        res = 0.5*(y_pred[:,2] - y_pred[:,1])

    # normalize back to energy scale
    if config['options']['normalize_target']:
        corr *= config['y_std']
        corr += config['y_mean']
    
        if res is not None:
            res *= config['y_std']
  
    # Add new prediction to data frame
    data = data.assign(newNNreg=y_pred[:,0])
    data = data.rename(columns={'newNNreg':'Jet_pt_reg_NN_%s'%input_trainings[idx]})
    if res is not None :
        data = data.assign(newNNreg_res=res)
        data = data.rename(columns={'newNNreg_res':'Jet_resolution_NN_%s'%input_trainings[idx]})

# save dataframe with added corrections
outfilename=options.out_dir+'applied_res_'+options.inp_file
data.to_hdf(outfilename,'w')

