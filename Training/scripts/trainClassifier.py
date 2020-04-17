import os
import sys; sys.path.append("/work/nchernya//HHbbgg_ETH_devel/Training/python") # to load packages

import matplotlib
matplotlib.use('Agg')

import training_utils as utils
import numpy as np
reload(utils)
import preprocessing_utils as preprocessing
reload(preprocessing)
import plotting_utils as plotting
reload(plotting)
import optimization_utils as optimization
reload(optimization)
import postprocessing_utils as postprocessing
reload(postprocessing)
import pandas as pd
import root_pandas as rpd
import matplotlib.pyplot as plt
import json
from ROOT import TLorentzVector
from optparse import OptionParser, make_option
from  pprint import pprint
import commands
import time
import datetime
start_time = time.time()



def main(options,args):
    year=options.year
    #please specify which year you want 
    Y = 2018
    outstr = "%s_test_C2V0_training"%Y
    doRhoReweight = False
    dirs = ['']
    ntuples = dirs[year]
    SMname = ['hh%s_13TeV_125_13TeV_VBFDoubleHTag_0'%Y]
    VBFname = ['vbfhh%s_13TeV_125_13TeV_VBFDoubleHTag_0'%Y]
    gghname = ['ggh%s_13TeV_125_13TeV_VBFDoubleHTag_0'%Y]
    #gghname = ['GluGluHToGG_M125_13TeV_amcatnloFXFX_pythia8_13TeV_VBFDoubleHTag_0']
    vhname =  ['vh%s_13TeV_125_13TeV_VBFDoubleHTag_0'%Y]
    qqhname = ['qqh%s_13TeV_125_13TeV_VBFDoubleHTag_0'%Y]
    tthname = ['tth%s_13TeV_125_13TeV_VBFDoubleHTag_0'%Y]
    NodesNormalizationFile = '/afs/cern.ch/user/n/nchernya/public/Soumya/reweighting_normalization_26_11_2019.json'
    useMixOfNodes = False
    whichNodes = ['SM']
    ggHHMixOfNodesNormalizations = json.loads(open(NodesNormalizationFile).read())
    # "%" sign allows to interpret the rest as a system command
    get_ipython().magic(u'env data=$utils.IO.ldata$ntuples')
    status,files = commands.getstatusoutput('! ls $data | sort -t_ -k 3 -n')
    files=files.split('\n')   
    print files    
    #signal = [s for s in files if ("VBFHHTo2B2G_CV_1_C2V_1_C3_1" in s) ]
    ggHH = [s for s in files if ("output_GluGluToHHTo2B2G_node_all_%s.root"%Y in s) ]
    diphotonJets = [s for s in files if "DiPhotonJetsBox_" in s]
    diphotonJets_1B = [s for s in files if "DiPhotonJetsBox1B" in s] # will use for limits
    diphotonJets_2B = [s for s in files if "DiPhotonJetsBox2B" in s] # will use for limits
    gJets_lowPt = [s for s in files if "GJet_Pt-20to40" in s]
    gJets_highPt = [s for s in files if "GJet_Pt-40" in s]
    ggh = [s for s in files if "output_ggh" in s]
    qqh = [s for s in files if "output_qqh" in s]
    vh  = [s for s in files if "output_vh" in s]
    tth  = [s for s in files if "output_tth" in s]

    couplings = 'CV_1_C2V_1_C3_1,CV_1_C2V_2_C3_1,CV_1_C2V_1_C3_2,CV_1_C2V_1_C3_0,CV_0_5_C2V_1_C3_1,CV_1_5_C2V_1_C3_1'.split(',') ### THE ORDER IS EXTREMELY IMPRORTANT, DO NOT CHANGE
    signal = []
    for coup in couplings :
          signal.append('output_VBFHHTo2B2G_%s_TuneCP5_PSWeights_13TeV-madgraph-pythia8.root'%coup)
    signal_name = 'vbfhh%s_13TeV_125_13TeV_VBFDoubleHTag_0'%Y
    
    utils.IO.reweightVBFHH = True
    utils.IO.vbfhh_cv = [1.]  
    utils.IO.vbfhh_c2v = [0.]
    utils.IO.vbfhh_kl = [1.]
    for sig in signal:
        utils.IO.add_signal(ntuples,sig,1,'tagsDumper/trees/%s'%signal_name,year)
   
#    utils.IO.add_signal(ntuples,signal,1,'tagsDumper/trees/%s'%VBFname[year],year)

    utils.IO.use_signal_nodes(useMixOfNodes,whichNodes,ggHHMixOfNodesNormalizations)
    utils.IO.add_background(ntuples,ggHH,-1, 'tagsDumper/trees/%s'%SMname[year],year)
    utils.IO.add_background(ntuples,diphotonJets,-2,'tagsDumper/trees/'+diphotonJets[0][diphotonJets[0].find('output_')+7:diphotonJets[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)
    utils.IO.add_background(ntuples,diphotonJets_1B,-2,'tagsDumper/trees/'+diphotonJets_1B[0][diphotonJets_1B[0].find('output_')+7:diphotonJets_1B[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)
    utils.IO.add_background(ntuples,diphotonJets_2B,-2,'tagsDumper/trees/'+diphotonJets_2B[0][diphotonJets_2B[0].find('output_')+7:diphotonJets_2B[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)
    utils.IO.add_background(ntuples,gJets_lowPt,-2,'tagsDumper/trees/'+gJets_lowPt[0][gJets_lowPt[0].find('output_')+7:gJets_lowPt[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)    
    utils.IO.add_background(ntuples,gJets_highPt,-2,'tagsDumper/trees/'+gJets_highPt[0][gJets_highPt[0].find('output_')+7:gJets_highPt[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year) 
    #utils.IO.add_background(ntuples,ggh,-3, 'tagsDumper/trees/%s'%gghname[year],year)    
    #utils.IO.add_background(ntuples,vh,-3, 'tagsDumper/trees/%s'%vhname[year],year)
    #utils.IO.add_background(ntuples,qqh,-3, 'tagsDumper/trees/%s'%qqhname[year],year)
    #utils.IO.add_background(ntuples,tth,-3, 'tagsDumper/trees/%s'%tthname[year],year)

#    utils.IO.add_signal(ntuples,signal,1,'%s'%VBFname[year],year)

#    utils.IO.use_signal_nodes(useMixOfNodes,whichNodes,ggHHMixOfNodesNormalizations)
#    utils.IO.add_background(ntuples,ggHH,-1, '%s'%SMname[year],year)
#    utils.IO.add_background(ntuples,diphotonJets,-2,diphotonJets[0][diphotonJets[0].find('output_')+7:diphotonJets[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)
#    utils.IO.add_background(ntuples,diphotonJets_1B,-2,diphotonJets_1B[0][diphotonJets_1B[0].find('output_')+7:diphotonJets_1B[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)
#    utils.IO.add_background(ntuples,diphotonJets_2B,-2,diphotonJets_2B[0][diphotonJets_2B[0].find('output_')+7:diphotonJets_2B[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)
#    utils.IO.add_background(ntuples,gJets_lowPt,-2,gJets_lowPt[0][gJets_lowPt[0].find('output_')+7:gJets_lowPt[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)
#    utils.IO.add_background(ntuples,gJets_highPt,-2,gJets_highPt[0][gJets_highPt[0].find('output_')+7:gJets_highPt[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)

    for i in range(len(utils.IO.backgroundName)):        
        print "using background file n."+str(i)+": "+utils.IO.backgroundName[i]
    for i in range(len(utils.IO.signalName)):    
        print "using signal file n."+str(i)+": "+utils.IO.signalName[i]


    utils.IO.plotFolder = '/afs/cern.ch/work/m/mukherje/Training_VBFHH/HHbbgg_ETH/Training/plots/%s/'%outstr
    if not os.path.exists(utils.IO.plotFolder):
        print utils.IO.plotFolder, "doesn't exist, creating it..."
        os.makedirs(utils.IO.plotFolder)

    doReweight = False #reweight signal from 2017 to match 2016 (mix of nodes reweight with HH_mass at gen level)


    #use noexpand for root expressions, it needs this file https://github.com/ibab/root_pandas/blob/master/root_pandas/readwrite.py
    ########################new code branches############################
    #branch_names = 'Mjj,leadingJet_DeepFlavour,subleadingJet_DeepFlavour,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),PhoJetMinDr,PhoJetOtherDr'.split(",")
    #branch_names = 'leadingJet_DeepFlavour,subleadingJet_DeepFlavour,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),PhoJetMinDr,PhoJetOtherDr,noexpand:(VBFleadJet_pt/VBFJet_mjj),VBFleadJet_eta,noexpand:(VBFsubleadJet_pt/VBFJet_mjj),VBFsubleadJet_eta,VBFCentrality_jg,VBFCentrality_jb,VBFDeltaR_jg,VBFDeltaR_jb,VBFProd_eta,VBFJet_mjj,VBFJet_Delta_eta,VBFleadJet_QGL,VBFDelta_phi,VBFsubleadJet_QGL,VBF_angleHH,VBF_dRHH,VBF_etaHH'.split(",")  #  
    #branch_names = 'leadingJet_DeepFlavour,subleadingJet_DeepFlavour,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),PhoJetMinDr,PhoJetOtherDr,VBFleadJet_pt,VBFleadJet_eta,VBFsubleadJet_pt,VBFsubleadJet_eta,VBFCentrality_jg,VBFCentrality_jb,VBFDeltaR_jg,VBFDeltaR_jb,VBFProd_eta,VBFJet_mjj,VBFJet_Delta_eta,VBFleadJet_QGL,VBFsubleadJet_QGL,diHiggs_pt,VBFDelta_phi,noexpand:(diVBFjet_pt/VBFJet_mjj)'.split(",")    
    #branch_names = 'leadingJet_DeepFlavour,subleadingJet_DeepFlavour,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),PhoJetMinDr,PhoJetOtherDr,noexpand:(VBFleadJet_pt/VBFJet_mjj),VBFleadJet_eta,noexpand:(VBFsubleadJet_pt/VBFJet_mjj),VBFsubleadJet_eta,VBFCentrality_jg,VBFCentrality_jb,VBFDeltaR_jg,VBFDeltaR_jb,VBFProd_eta,VBFJet_mjj,VBFJet_Delta_eta,VBFleadJet_QGL,VBFsubleadJet_QGL,diHiggs_pt,MX,VBFDelta_phi'.split(",")
    branch_names = 'leadingJet_DeepFlavour,subleadingJet_DeepFlavour,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),PhoJetMinDr,PhoJetOtherDr,noexpand:(VBFleadJet_pt/VBFJet_mjj),VBFleadJet_eta,noexpand:(VBFsubleadJet_pt/VBFJet_mjj),VBFsubleadJet_eta,VBFCentrality_jg,VBFCentrality_jb,VBFDeltaR_jg,VBFDeltaR_jb,VBFProd_eta,VBFJet_mjj,VBFJet_Delta_eta,VBFleadJet_QGL,VBFsubleadJet_QGL,diHiggs_pt,MX'.split(",")
    #branch_names = 'leadingJet_DeepFlavour,subleadingJet_DeepFlavour,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(sigmaMJets*1.4826),PhoJetMinDr,noexpand:(VBFleadJet_pt/VBFJet_mjj),VBFleadJet_eta,noexpand:(VBFsubleadJet_pt/VBFJet_mjj),VBFsubleadJet_eta,VBFCentrality_jg,VBFCentrality_jb,VBFDeltaR_jg,VBFDeltaR_jb,VBFProd_eta,VBFJet_mjj,VBFJet_Delta_eta,VBFleadJet_QGL,VBFsubleadJet_QGL,diHiggs_pt,MX,ProddiphoptOverprodVBFpt,ProddijetptOverprodVBFpt,SumvecptOversumsclpt'.split(",")
    branch_cuts = 'leadingJet_pt,subleadingJet_pt,leadingJet_bRegNNCorr,subleadingJet_bRegNNCorr,noexpand:(leadingJet_pt/leadingJet_bRegNNCorr),noexpand:(subleadingJet_pt/subleadingJet_bRegNNCorr)'.split(',')
    #cuts = 'VBFleadJet_eta < 4.7 & VBFsubleadJet_eta < 4.7 & VBFleadJet_pt > 40'
    #cuts = 'VBFsubleadJet_pt < 50'
    cuts = 'leadingJet_pt > 0'
    nodesWeightBranches=[]
    if utils.IO.signalMixOfNodes : nodesWeightBranches=[ 'benchmark_reweight_%s'%i for i in whichNodes ] 
    #cuts = 'subleadingJet_pt>25'
    ######################
    print (nodesWeightBranches)
    event_branches = ['event','weight','btagReshapeWeight','MX','leadingJet_hflav','leadingJet_pflav','subleadingJet_hflav','subleadingJet_pflav','CMS_hgg_mass','Mjj'] #,'Mjj'  #for the training without Mjj
    event_branches+=['leadingJet_phi','leadingJet_eta','subleadingJet_phi','subleadingJet_eta']
    event_branches+=['leadingPhoton_eta','leadingPhoton_phi','subleadingPhoton_eta','subleadingPhoton_phi']

    resolution_weighting = 'ggbb' # None, gg or ggbb
    doOverlapRemoval=True   #diphotons overlap removal if using b-enriched samples


    branch_names = [c.strip() for c in branch_names]
    print branch_names

    event_bkg,event_sig = None,None
    if (year>=1 and doReweight == False): #not used anymore
        preprocessing.set_signals(branch_names+event_branches+branch_cuts+['genMhh'],True,cuts)
        preprocessing.set_backgrounds(branch_names+event_branches+branch_cuts,True,cuts)
    else :
        preprocessing.set_signals(branch_names+event_branches+branch_cuts,True,cuts)
        preprocessing.set_backgrounds(branch_names+event_branches+branch_cuts,True,cuts)
        #for i in range(len(utils.IO.backgroundName)):
        #    if i == 0: 
        #       print("soumya")
        #       preprocessing.set_backgrounds(branch_names+event_branches+branch_cuts+nodesWeightBranches,True,cuts)
        #    else:
        #       print("tarun")
        #       preprocessing.set_backgrounds(branch_names+event_branches+branch_cuts,True,cuts)
#        if range(len(utils.IO.backgroundName)) == 0:
#            print("soumya")
#            print (branch_names)
#            preprocessing.set_backgrounds(branch_names+event_branches+branch_cuts+nodesWeightBranches,True,cuts)
#        else:
#            print ("Tarun")
#            print (branch_names) 
#            preprocessing.set_backgrounds(branch_names+event_branches+branch_cuts,True,cuts)

    #### Adding new deltaR (photon,jet) branches ####
  #  for i in range(utils.IO.nBkg):
  #     preprocessing.add_deltaR_branches(utils.IO.background_df[i])
  #  for i in range(utils.IO.nSig):
  #     preprocessing.add_deltaR_branches(utils.IO.signal_df[i])
  #  branch_names = branch_names + ['photJetdRmin','photJetdRmin2'] 
   ##### New photon + jet branches added  above #####

    info_file = open(utils.IO.plotFolder+"info_%s.txt"%outstr,"w") 
    info_file.write("\n".join(branch_names))
    info_file.write("Resolution weighting : %s\n"%resolution_weighting)
    info_file.write("Cuts : %s\n"%cuts)
    info_file.write("Signal weighted Events Sum before inverse resolution weighting : \n")
    info_file.write("%.4f \n"%(np.sum(utils.IO.signal_df[0]['weight']))) 
    info_file.write("Background weighted Events Sum : \n")
    sum_bkg_weights = 0
    for bkg_type in range(utils.IO.nBkg):
        bkg_weight = np.sum(utils.IO.background_df[bkg_type]['weight'])
        sum_bkg_weights+=bkg_weight
        info_file.write("proc %d : %.4f \n"%( utils.IO.bkgProc[bkg_type],bkg_weight)) 
    info_file.write("Background weighted Events Sum Total : %.4f \n"%(sum_bkg_weights)) 
    info_file.close()
    if '2016' in gghname[0] and doRhoReweight == True : 
        diphoton_for_rho = ['output_DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa.root','output_DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa.root']
        diphoton_frame2016=rpd.read_root(utils.IO.ldata+'/'+diphoton_for_rho[0],'tagsDumper/trees/DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_VBFDoubleHTag_0', columns = ['weight','rho'])
        diphoton_frame2017=rpd.read_root('/eos/user/m/mukherje/HH_bbgg/2017_Sample/'+diphoton_for_rho[1],'tagsDumper/trees/DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_VBFDoubleHTag_0', columns = ['weight','rho'])
        preprocessing.reweight_rho('rho',diphoton_frame2016,diphoton_frame2017,utils.IO.signal_df[0])


    if 'gg' in resolution_weighting : 
        preprocessing.weight_signal_with_resolution_all(branch='sigmaMOverM')
    if 'bb' in resolution_weighting : 
        preprocessing.weight_signal_with_resolution_all(branch='(sigmaMJets*1.4826)')

    #preprocessing.reweight_rho('rho',diphoton_frame2016,diphoton_frame2017,utils.IO.signal_df[0]) 
    if doOverlapRemoval == True:    
        for i in range(utils.IO.nBkg):
            if 'DiPhotonJetsBox_MGG' in utils.IO.bkgTreeName[i] : preprocessing.cleanOverlapDiphotons(utils.IO.bkgTreeName[i],utils.IO.background_df[i])        


    X_bkg,y_bkg,weights_bkg,event_bkg,X_sig,y_sig,weights_sig,event_sig=preprocessing.set_variables(branch_names,use_event_num=True)




    X_bkg,y_bkg,weights_bkg,event_bkg = preprocessing.randomize(X_bkg,y_bkg,weights_bkg,event_num = np.asarray(event_bkg))
    X_sig,y_sig,weights_sig,event_sig = preprocessing.randomize(X_sig,y_sig,weights_sig,event_num = np.asarray(event_sig))


    #Get training and test samples based on event number : even/odd or %5, set in the function for now
    y_total_train = preprocessing.get_total_training_sample_event_num(y_sig.reshape(-1,1),y_bkg,event_sig.reshape(-1,1),event_bkg).ravel()
    X_total_train = preprocessing.get_total_training_sample_event_num(X_sig,X_bkg,event_sig.reshape(-1,),event_bkg.reshape(-1,))

    y_total_test = preprocessing.get_total_test_sample_event_num(y_sig.reshape(-1,1),y_bkg,event_sig.reshape(-1,1),event_bkg).ravel()
    X_total_test = preprocessing.get_total_test_sample_event_num(X_sig,X_bkg,event_sig.reshape(-1,),event_bkg.reshape(-1,))

    w_total_train = preprocessing.get_total_training_sample_event_num(weights_sig.reshape(-1,1),weights_bkg.reshape(-1,1),event_sig.reshape(-1,1),event_bkg).ravel()
    w_total_test = preprocessing.get_total_test_sample_event_num(weights_sig.reshape(-1,1),weights_bkg.reshape(-1,1),event_sig.reshape(-1,1),event_bkg).ravel()


    ##########Normalize weights for training and testing. Sum(signal)=Sum(bkg)=1. But keep relative normalization
    # between bkg classes
    w_total_train = preprocessing.normalize_process_weights_split_all(w_total_train,y_total_train)
    w_total_test = preprocessing.normalize_process_weights_split_all(w_total_test,y_total_test)


    print "Starting the training now : "
    now = str(datetime.datetime.now())
    print(now)
    
    ################Training a classifier###############
    ########final optimization with all fixed#######
    from sklearn.externals import joblib
    import xgboost as xgb
    n_threads=10

    #optimized parameters with Mjj for 2016 done by Francesco
#Optimized for the 2017 C2V_2 training 
#    clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
#           gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=4,
#           min_child_weight=1e-06,  n_estimators=400,
#           nthread=n_threads, objective='multi:softprob', reg_alpha=0.0,
#           reg_lambda=0.05, scale_pos_weight=1, seed=None, silent=True,
#           subsample=1)

    clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
           gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=4,
           min_child_weight=1e-06,  n_estimators=400,
           nthread=n_threads, objective='multi:softprob', reg_alpha=0.0,
           reg_lambda=0.05, scale_pos_weight=1, seed=None, silent=True,
           subsample=1)


    clf.fit(X_total_train,y_total_train, sample_weight=w_total_train)
    
    
    print 'Training is done. It took', time.time()-start_time, 'seconds.'
    #print(clf.feature_importances_)    
    #from xgboost import plot_importance
    #from matplotlib import pyplot
    #plot_importance(clf)
    #pyplot.rcParams['figure.figsize'] = [5, 5]
    #pyplot.savefig('graph_2018_setII.png')

    _,_,_ = plt.hist(utils.IO.signal_df[0]['rho'], np.linspace(0,100,100), facecolor='b',weights=utils.IO.signal_df[0]['weight'], alpha=0.5,normed=False,label='2016')
    plt.xlabel('rho [GeV]')
    plt.ylabel('A.U.')
    plt.savefig('%s_2016.png'%Y)





    joblib.dump(clf, os.path.expanduser('/afs/cern.ch/work/m/mukherje/Training_VBFHH/HHbbgg_ETH/Training/output_files/training_with_%s.pkl'%outstr), compress=9)

    plot_classifier = plotting.plot_classifier_output(clf,X_total_train,X_total_test,y_total_train,y_total_test,outString=outstr)
    fpr_dipho,tpr_dipho = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-1,outString=outstr,weights=w_total_test)
    fpr_gJets,tpr_gJets = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-2,outString=outstr,weights=w_total_test)
    fpr_singleH,tpr_singleH = plotting.plot_roc_curve_multiclass_singleBkg(X_total_test,y_total_test,clf,-3,outString=outstr,weights=w_total_test)


    roc_df_dipho = pd.DataFrame({"fpr_dipho": (fpr_dipho).tolist(),"tpr_dipho": (tpr_dipho).tolist()})
    roc_df_gJets = pd.DataFrame({"fpr_gJets": (fpr_gJets).tolist(),"tpr_gJets": (tpr_gJets).tolist()})
    #roc_df_singleH = pd.DataFrame({"fpr_singleH": (fpr_singleH).tolist(),"tpr_singleH": (tpr_singleH).tolist()})
    roc_df_dipho.to_hdf(utils.IO.plotFolder+"roc_curves_dipho_%s.h5"%outstr, key='df', mode='w')
    roc_df_gJets.to_hdf(utils.IO.plotFolder+"roc_curves_gJets_%s.h5"%outstr, key='df', mode='w')
    #roc_df_singleH.to_hdf(utils.IO.plotFolder+"roc_curves_singleH_%s.h5"%outstr, key='df', mode='w')





if __name__ == "__main__":

    parser = OptionParser(option_list=[
            make_option("-y","--year",
                        action="store",type=int,dest="year",default=0,
                        help="which year : 2016-0,2017-1,2018-2",
                        )
            ])

    (options, args) = parser.parse_args()
    sys.argv.append("-b")

    pprint(options.__dict__)
    
    main(options,args)
        
