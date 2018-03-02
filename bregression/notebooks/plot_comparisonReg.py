import keras.models
import os
import bregnn.io as io
import bregnn.utils as utils
import sys
import json
from optparse import OptionParser, make_option
sys.path.insert(0, '/users/nchernya/HHbbgg_ETH/bregression/python/')
import plotting_utils as plotting
import datetime
import math

parser = OptionParser(option_list=[
    make_option("--training",type='string',dest="training",default='HybridLoss'),
    make_option("--inp-file",type='string',dest='inp_file',default='applied_res_ttbar_RegressionPerJet_heppy_energyRings3_forTesting.hd5'),
    make_option("--inp-dir",type='string',dest="inp_dir",default='/scratch/snx3000/nchernya/bregression/output_root/'),
    make_option("--sample-name",type='string',dest="samplename",default='ttbar'),
    make_option("--labels",type='string',dest="labels",default=''),
    make_option("--where",type='string',dest="where",default=''),
])

## parse options
(options, args) = parser.parse_args()
input_trainings = options.training.split(',')
input_files = options.inp_file.split(',')

now = str(datetime.datetime.now()).split(' ')[0]
scratch_plots ='/scratch/snx3000/nchernya/bregression/plots/quantiles/%s/'%now
#scratch_plots ='/scratch/snx3000/nchernya/bregression/plots/quantiles/2018-02-26/'
dirs=['',input_trainings[0],options.samplename]
for i in range(len(dirs)):
  scratch_plots=scratch_plots+'/'+dirs[i]+'/'
  if not os.path.exists(scratch_plots):
    os.mkdir(scratch_plots)

print(options.where)
whats = ['p_T','\eta','rho']
ranges = [[30,400],[-2.5,2.5],[0,50]]
binning =[50,10,20] #[50,20]
linestyles = ['-.', '--','-', ':']
colors=['green','red','blue','cyan']
markers=['s','o','^','h']
labels=options.labels.split(',')
bins_same = []

for i in range(0,3):
 sigma_mu_array = []
 sigma_array = []
 for ifile in range(len(input_files)):
    # ## Read test data and model
  # load data
    data = io.read_data('%s%s'%(options.inp_dir,input_files[ifile]),columns=None)
    if options.where!='' : data = data.query(options.where)
    data.describe()

    #Regions of pt and eta 
    file_regions = open('/users/nchernya/HHbbgg_ETH/bregression/scripts/regionsPtEta.json')
    regions_summary = json.loads(file_regions.read())
    region_names = regions_summary['pt_regions']+regions_summary['eta_region_names']

 #   y = (data['Jet_mcPt']/data['Jet_pt']).values.reshape(-1,1)
    y = (data['Jet_mcPt']/(data['Jet_pt_raw']*data['Jet_corr_JEC'])).values.reshape(-1,1)
    X_pt = (data['Jet_pt_raw']).values.reshape(-1,1)
    X_eta = (data['Jet_eta']).values.reshape(-1,1)
    X_rho = (data['rho']).values.reshape(-1,1)
    res = (data['Jet_resolution_NN_%s'%input_trainings[ifile]])
    y_pred = (data['Jet_pt_reg_NN_%s'%input_trainings[ifile]]) #bad name because it is actually a correction
    y_hbb = (data['Jet_mcPt']/data['Jet_pt_reg']).values.reshape(-1,1)
    y_corr = (y[:,0]/y_pred).values.reshape(-1,1)


    if i==0 : X = X_pt
    elif i==1 : X = X_eta
    elif i==2 : X = X_rho
    print(i,X)
 
    if (ifile==0) : bins=binning[i]
    if ('ggHHbbgg' in options.samplename) and ('p_T' in whats[i]) : bins=int(binning[i]/2.)
    if ('ZHbbll' in options.samplename) and ('eta' in whats[i]) : ranges[i]=[-2.45,2.45]
 
    if ifile==0 : 
       bins, y_corr_mean_pt, y_corr_std_pt, y_corr_qt_pt = utils.profile(y_corr,X,range=ranges[i],bins=bins,quantiles=np.array([0.25,0.4,0.5,0.75])) 
       bins_same.append(bins)
    else :  
       bins = bins_same[i]
       _, y_corr_mean_pt, y_corr_std_pt, y_corr_qt_pt = utils.profile(y_corr,X,bins=bins,quantiles=np.array([0.25,0.4,0.5,0.75])) 
    y_corr_median_pt = y_corr_qt_pt[2]
    y_corr_25_pt,y_corr_40_pt,y_corr_75_pt = y_corr_qt_pt[0],y_corr_qt_pt[1],y_corr_qt_pt[3]
    y_corr_iqr2_pt =  y_corr_qt_pt[0],y_corr_qt_pt[3]
    err_corr_iqr2 =  0.5*(y_corr_qt_pt[3]-y_corr_qt_pt[0])
    sigma_mu_corr = np.array(err_corr_iqr2)/np.array(y_corr_40_pt)
    sigma_mu_array.append(sigma_mu_corr)
    sigma_array.append(err_corr_iqr2)

    _, y_mean_pt, y_std_pt, y_qt_pt = utils.profile(y,X,bins=bins,quantiles=np.array([0.25,0.4,0.5,0.75])) 
    y_25_pt,y_40_pt,y_75_pt = y_qt_pt[0],y_qt_pt[1],y_qt_pt[3]
    y_iqr2_pt =  y_qt_pt[0],y_qt_pt[3]
    err_jec_iqr2 =  0.5*(y_qt_pt[3]-y_qt_pt[0])
    sigma_mu_jec = np.array(err_jec_iqr2)/np.array(y_40_pt)
    sigma_jec = np.array(err_jec_iqr2)
 
    binc = 0.5*(bins[1:]+bins[:-1])

  #  print(binc.shape,bins.shape,sigma_mu_jec.shape,err_corr_iqr2.shape,y_corr_median_pt.shape) 
 
    ## Draw profile of sigma (0.72-0.25)/2 vs eta and pt
    if (ifile==0) :  plt.scatter(binc,sigma_mu_jec,color='black',marker='*',label='JEC only')
    plt.scatter(binc,sigma_mu_corr,color=colors[ifile],marker=markers[ifile],label='NN %s'%labels[ifile])
 plt.grid(alpha=0.2,linestyle='--',markevery=2)
 axes = plt.gca()
 if (i==0) : axes.set_ylim(0.02,0.20)
 if (i==1) : axes.set_ylim(0.06,0.15)
 if (i==2) : axes.set_ylim(0.08,0.15)
 axes.set_xlim(ranges[i][0],ranges[i][1])
 if (i==0) : axes.set_xlim(0,ranges[i][1])
 ymin, ymax = (plt.gca()).get_ylim()
 xmin, xmax = (plt.gca()).get_xlim()
 plt.text(xmin+abs(xmin)*0.05,ymax*0.95,'%s'%options.samplename, fontsize=20)
 lgd = plt.legend(loc="lower center",numpoints=1,ncol=2,bbox_to_anchor=(0.0,-0.202,1.,-.202),borderaxespad=0.,mode='expand')
 plt.xlabel('$%s$'%whats[i])
 plt.ylabel('[IQR/2]/quantile 0.4')
 where = (options.where).replace(' ','').replace('<','_').replace('>','_').replace('(','').replace(')','')
 savename='/IQR_compare_%s_%s%s'%(whats[i].replace('\\',''),options.samplename,where)
 plt.savefig(scratch_plots+savename+'.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
 plt.savefig(scratch_plots+savename+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
 plt.clf()

# difference = 2*(np.array(sigma_mu_array[0])-np.array(sigma_mu_array[1]))/(np.array(sigma_mu_array[0])+np.array(sigma_mu_array[1]))
# difference = [round(a,4) for a in difference]
# data_csv = pd.DataFrame(np.array(difference).reshape(1,binc.shape[0]), columns=(binc))
 data_csv = pd.DataFrame({whats[i].replace('\\',''):binc})
 data_csv['onlyJEC'] = sigma_mu_jec
 data_csv['sigma_onlyJEC'] = sigma_jec
 for ifile in range(len(input_files)):
     data_csv['%s'%labels[ifile]] = sigma_mu_array[ifile]
     data_csv['sigma_%s'%labels[ifile]] = sigma_array[ifile]
     data_csv['delta_%s_JEC'%labels[ifile]] = 2*(np.array(sigma_mu_array[ifile])-np.array(sigma_mu_jec))/(np.array(sigma_mu_array[ifile])+np.array(sigma_mu_jec))
     data_csv['delta_%s_JEC_rel'%labels[ifile]] = (np.array(sigma_mu_array[ifile])-np.array(sigma_mu_jec))/(np.array(sigma_mu_jec))
     data_csv['delta_sigma_%s_JEC'%labels[ifile]] = 2*(np.array(sigma_array[ifile])-np.array(sigma_jec))/(np.array(sigma_array[ifile])+np.array(sigma_jec))
     for jfile in range(ifile+1,len(input_files)):
         data_csv['delta_%s_%s'%(labels[ifile],labels[jfile])] = 2*(np.array(sigma_mu_array[ifile])-np.array(sigma_mu_array[jfile]))/(np.array(sigma_mu_array[ifile])+np.array(sigma_mu_array[jfile]))
         data_csv['delta_sigma_%s_%s'%(labels[ifile],labels[jfile])] = 2*(np.array(sigma_array[ifile])-np.array(sigma_array[jfile]))/(np.array(sigma_array[ifile])+np.array(sigma_array[jfile]))
              
 savename='/data_IQR_compare_%s_%s%s.csv'%(whats[i].replace('\\',''),options.samplename,where)
 data_csv.to_csv(scratch_plots+savename)
