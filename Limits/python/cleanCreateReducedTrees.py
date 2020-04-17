from optparse import OptionParser, make_option
from  pprint import pprint

import os
import sys; sys.path.append("/afs/cern.ch/work/m/mukherje/Training_VBFHH/HHbbgg_ETH/Training/python") # to load packages
import training_utils as utils
import numpy as np
import preprocessing_utils as preprocessing
import plotting_utils as plotting
import optimization_utils as optimization
import postprocessing_utils as postprocessing

import pandas as pd
import root_pandas as rpd
import json

treeDir = 'tagsDumper/trees/'
#samples = ["VBFHHTo2B2G_CV_1_C2V_2_C3_1","VBFHHTo2B2G_CV_1_C2V_1_C3_1","GluGluToHHTo2B2G_node_all","ggh","vh","qqh","tth","DiPhotonJetsBox_","DiPhotonJetsBox2BJets_","DiPhotonJetsBox1BJet_","GJet_Pt-20to40","GJet_Pt-40toInf"]#
samples = ["VBFHHTo2B2G_CV_1_C2V_1_C3_1","DiPhotonJetsBox_","GluGluToHHTo2B2G_node_all"]
#samples = ["VBFHHTo2B2G_CV_1_C2V_2_C3_1","VBFHHTo2B2G_CV_1_C2V_1_C3_1","GluGluToHHTo2B2G_node_all","DiPhotonJetsBox_","DiPhotonJetsBox2BJets","DiPhotonJetsBox1BJet","GJet_Pt-20to40","GJet_Pt-40toInf","ggh","vh","qqh","tth"] #,"DiPhotonJetsBox2BJets","DiPhotonJetsBox1BJet","ttH","TTGJets","TTTo2L2Nu","TTGG_0Jets","GJet_Pt-20to40","GJet_Pt-40toInf"]#
#samples = ["GluGluToHHTo2B2G_node_all","GJet_Pt-20to40","GJet_Pt-40toInf"]#
#samples = ["GluGluToHHTo2B2G_node_all","DiPhotonJetsBox_","DiPhotonJetsBox2BJets","DiPhotonJetsBox1BJet","ttH","TTGJets","TTGG_0Jets"]#
#samples = ["GluGluToHHTo2B2G_node_all","TTTo2L2Nu"]#
background_names = []
#samples = ["VBFHHTo2B2G_CV_1_C2V_1_C3_1","DiPhotonJetsBox_", "DiPhotonJetsBox2BJets"]#
cleanOverlap = True   # Do not forget to change it 
#treeTag="_2017"
treeTag=""

NodesNormalizationFile = '/afs/cern.ch/user/n/nchernya/public/Soumya/reweighting_normalization_26_11_2019.json'
useMixOfNodes = True
whichNodes = ['SM']  #used to create cumulative on SM only

#just a list of all nodes to add weight branches in the trees
nodes_branches = list(np.arange(0,12,1))   #all nodes are used to train. 
nodes_branches.append('SM')
nodes_branches.append('box')
background_names = []

def addSamples():#define here the samples you want to process
    ntuples = options.ntup
    year = options.year
    year_str = ''
    if year==0:
      year_str='2016'
    elif year==1:
      year_str='2017'
    elif year==2:
      year_str='2018'
    gghhname="hh2018_13TeV_125"
    vbfhh_name="vbfhh2018_13TeV_125"
    gghname = "ggh2018_13TeV_125"
    #gghname = "GluGluHToGG_M125_13TeV_amcatnloFXFX_pythia8"
    vhname =  "vh2018_13TeV_125"
    qqhname = "qqh2018_13TeV_125"
    tthname = "tth2018_13TeV_125"
    if options.ldata is not "":
        print("loading files from: "+options.ldata)
        utils.IO.ldata=options.ldata
    ggHHMixOfNodesNormalizations = json.loads(open(NodesNormalizationFile).read())
    utils.IO.use_signal_nodes(useMixOfNodes,whichNodes,ggHHMixOfNodesNormalizations) 
    files= os.listdir(utils.IO.ldata+ntuples)

    couplings = 'CV_1_C2V_1_C3_1,CV_1_C2V_2_C3_1,CV_1_C2V_1_C3_2,CV_1_C2V_1_C3_0,CV_0_5_C2V_1_C3_1,CV_1_5_C2V_1_C3_1'.split(',') ### THE ORDER IS EXTREMELY IMPRORTANT, DO NOT CHANGE
    signal = []
    for coup in couplings :
          signal.append('output_VBFHHTo2B2G_%s_TuneCP5_PSWeights_13TeV-madgraph-pythia8.root'%coup)
    print ("The vbfhh samples are:")
    print signal
    signal_name = 'vbfhh2018_13TeV_125_13TeV_VBFDoubleHTag_0'
    utils.IO.reweightVBFHH = True
    utils.IO.vbfhh_cv = [1.]
    utils.IO.vbfhh_c2v = [2.]
    utils.IO.vbfhh_kl = [1.]
    for sig in signal:
        utils.IO.add_signal(ntuples,sig,1,'tagsDumper/trees/%s'%signal_name,year)
   # for iSample in samples:
    for num,iSample in enumerate(samples):
            process  = [s for s in files if iSample in s]
#        if "VBFHHTo2B2G_CV_1_C2V_2_C3_1" in iSample:
#            utils.IO.add_signal(ntuples,process,1,treeDir+vbfhh_name+'_13TeV_VBFDoubleHTag_0',year)
#        else :
            if "VBFHHTo2B2G" in iSample:
               if utils.IO.reweightVBFHH==False:
                          utils.IO.add_signal(ntuples,process,1,treeDir+vbfhh_name+'_13TeV_VBFDoubleHTag_0',year)
               else :
                          print 'signal samples are already added'
            #print 'adding bkg with process num : ',process[0],"  ",-num
            elif ("GluGluToHHTo2B2G" in iSample) and (useMixOfNodes==False):
              utils.IO.add_background(ntuples,process,-num,treeDir+gghhname+'_13TeV_VBFDoubleHTag_0',year)
              background_names.append(samples[num].replace('-','_'))
            elif ("GluGluToHHTo2B2G"in iSample) and (useMixOfNodes==True) :
              utils.IO.use_signal_nodes(useMixOfNodes,whichNodes,ggHHMixOfNodesNormalizations)
              utils.IO.add_background(ntuples,process,-num,treeDir+gghhname+'_13TeV_VBFDoubleHTag_0',year)
              background_names.append(samples[num].replace('-','_'))
            elif ("ggh" in iSample):
              utils.IO.add_background(ntuples,process,-num,treeDir+gghname+'_13TeV_VBFDoubleHTag_0',year)
              background_names.append(samples[num].replace('-','_'))
            elif ("vh" in iSample):
              utils.IO.add_background(ntuples,process,-num,treeDir+vhname+'_13TeV_VBFDoubleHTag_0',year)
              background_names.append(samples[num].replace('-','_'))
            elif ("qqh" in iSample):
              utils.IO.add_background(ntuples,process,-num,treeDir+qqhname+'_13TeV_VBFDoubleHTag_0',year)
              background_names.append(samples[num].replace('-','_'))
            elif ("tth" in iSample):
              utils.IO.add_background(ntuples,process,-num,treeDir+tthname+'_13TeV_VBFDoubleHTag_0',year)
              background_names.append(samples[num].replace('-','_'))
            else :
              utils.IO.add_background(ntuples,process,-num,treeDir+process[0][process[0].find('output_')+7:process[0].find('.root')].replace('-','_')+'_13TeV_VBFDoubleHTag_0',year)  
              background_names.append(samples[num].replace('-','_'))
            print samples[num]

   

    nBkg = len(utils.IO.backgroundName)
    print 'bkgs : ',utils.IO.backgroundName
 
    Data= [s for s in files if "DoubleEG" in s]
    #utils.IO.add_data(ntuples,Data,-10,'tree')
    dataTreeName = 'Data_13TeV_VBFDoubleHTag_0'
    utils.IO.add_data(ntuples,Data,-10,treeDir+dataTreeName)
 
####################Not used anymore###########################   
#    #add all nodes : old, now we do reweiting inside
#    nodes = []
#    nodesTreeNames = []
#    if options.addnodes:
#        for i in range(2,14): #+ ['box']:
#            nodes.append([s for s in files if "GluGluToHHTo2B2G_reweighted_node_"+str(i) in s])
#            nodesTreeNames.append("GluGluToHHTo2B2G_node_"+str(i)+'_13TeV_madgraph_13TeV_VBFDoubleHTag_0')
#    if options.addrew:
#        for i in range(2,14): #+ ['box']:                   
#            nodes.append([s for s in files if "GluGluToHHTo2B2G_reweighted_nodes" in s])
#            nodesTreeNames.append("GluGluToHHTo2B2G_reweighted_node_"+str(i))
#    for i in range(nBkg,nBkg+len(nodes)):
#        if "reweighted_nodes" not in  str(nodes[i-nBkg]):
#            utils.IO.add_background(ntuples,nodes[i-nBkg],-i,treeDir+nodesTreeNames[i-nBkg])
#        else:
#            utils.IO.add_background(ntuples,nodes[i-nBkg],-i,nodesTreeNames[i-nBkg])
####################################################################

    for i in range(utils.IO.nBkg):        
        print "using background file n."+str(i)+": "+utils.IO.backgroundName[i]
    for i in range(utils.IO.nSig):    
        print "using signal file n."+str(i)+": "+utils.IO.signalName[i]
    print "using data file: "+ utils.IO.dataName[0]
    


def main(options,args):
    

    print options.addnodes
    addSamples()
    
   # branch_names = 'leadingJet_DeepFlavour,subleadingJet_DeepFlavour,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),PhoJetMinDr,PhoJetOtherDr,noexpand:(VBFleadJet_pt/VBFJet_mjj),VBFleadJet_eta,noexpand:(VBFsubleadJet_pt/VBFJet_mjj),VBFsubleadJet_eta,VBFCentrality_jg,VBFCentrality_jb,VBFDeltaR_jg,VBFDeltaR_jb,VBFProd_eta,VBFJet_mjj,VBFJet_Delta_eta,VBFleadJet_QGL,VBFsubleadJet_QGL,diHiggs_pt,MX'.split(",")
    branch_names = 'leadingJet_DeepFlavour,subleadingJet_DeepFlavour,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),PhoJetMinDr,PhoJetOtherDr,noexpand:(VBFleadJet_pt/VBFJet_mjj),VBFleadJet_eta,noexpand:(VBFsubleadJet_pt/VBFJet_mjj),VBFsubleadJet_eta,VBFCentrality_jg,VBFCentrality_jb,VBFDeltaR_jg,VBFDeltaR_jb,VBFProd_eta,VBFJet_mjj,VBFJet_Delta_eta,VBFleadJet_QGL,VBFsubleadJet_QGL,diHiggs_pt,MX'.split(",")
#    branch_names = 'leadingJet_DeepFlavour,subleadingJet_DeepFlavour,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),PhoJetMinDr,PhoJetOtherDr,noexpand:(VBFleadJet_pt/VBFJet_mjj),VBFleadJet_eta,noexpand:(VBFsubleadJet_pt/VBFJet_mjj),VBFsubleadJet_eta,VBFCentrality_jg,VBFCentrality_jb,VBFDeltaR_jg,VBFDeltaR_jb,VBFProd_eta,VBFJet_mjj,VBFJet_Delta_eta,VBFleadJet_QGL,VBFDelta_phi,VBFsubleadJet_QGL,VBF_angleHH,VBF_dRHH,VBF_etaHH'.split(",")
    additionalCut_names = 'CMS_hgg_mass,Mjj,MX,ttHScore,btagReshapeWeight,VBFleadJet_PUID,VBFsubleadJet_PUID'.split(',')
 #   additionalCut_names = 'CMS_hgg_mass,Mjj,MX'.split(',')
  #  if options.addHHTagger:
    additionalCut_names += 'HHbbggMVA'.split(",")
    signal_trainedOn = ['noexpand:(event%2!=0)']   #if 1 the event is trained on, if 0 -> should be used only for limit extraction
#    bkg_trainedOn = ['noexpand:(event%1==0)'] #to accept all events
    bkg_trainedOn = []
    overlap = ['overlapSave']
    data_weight = ['weight']
    additionalCut_names+= ['event','weight']
    additionalCut_names+=['leadingJet_phi','leadingJet_eta','subleadingJet_phi','subleadingJet_eta']
    additionalCut_names+=['leadingPhoton_eta','leadingPhoton_phi','subleadingPhoton_eta','subleadingPhoton_phi']
    event_branches = ['leadingJet_hflav','leadingJet_pflav','subleadingJet_hflav','subleadingJet_pflav','btagReshapeWeight']
    branch_cuts = 'leadingJet_pt,subleadingJet_pt,leadingJet_bRegNNCorr,subleadingJet_bRegNNCorr,noexpand:(leadingJet_pt/leadingJet_bRegNNCorr),noexpand:(subleadingJet_pt/subleadingJet_bRegNNCorr)'.split(',')
    if not options.addData:
   #     branch_cuts = 'leadingJet_pt,subleadingJet_pt,leadingJet_bRegNNCorr,subleadingJet_bRegNNCorr,noexpand:(leadingJet_pt/leadingJet_bRegNNCorr),noexpand:(subleadingJet_pt/subleadingJet_bRegNNCorr)'.split(',')
   #     event_branches += ['leadingJet_hflav','leadingJet_pflav','subleadingJet_hflav','subleadingJet_pflav']
        cuts = 'leadingJet_pt>0 '
    else:
        cuts = 'rho>0'
 #   cuts = 'leadingJet_pt>20 & subleadingJet_pt> 20 & (leadingJet_pt/leadingJet_bRegNNCorr>20) & (subleadingJet_pt/subleadingJet_bRegNNCorr>20) '
    #cuts = 'VBFleadJet_eta < 4.7 & VBFsubleadJet_eta < 4.7 & VBFleadJet_pt > 40'
    #cuts = 'VBFsubleadJet_eta < 4.7'
    cuts = 'leadingJet_pt>0 '


    #if not options.addData:
    #   cuts = 'leadingJet_pt>0 '
    #else:
    #    cuts = 'rho>0'#just because with data we don't save the raw pt (we should)  -->>What  ? (Nadya)
######################
################################################################


    branch_names = [c.strip() for c in branch_names]
    print "using following variables for MVA: " 
    print branch_names
    
    
    # no need to shuffle here, we just count events
    nodesWeightBranches=[]
    if utils.IO.signalMixOfNodes : nodesWeightBranches=[ 'benchmark_reweight_%s'%i for i in nodes_branches ] 
    preprocessing.set_signals(branch_names+branch_cuts+event_branches+additionalCut_names+signal_trainedOn,False,cuts) 
    preprocessing.set_backgrounds(branch_names+branch_cuts+event_branches+additionalCut_names+bkg_trainedOn,False,cuts) 

    #### Adding new deltaR (photon,jet) branches ####
  #  for i in range(utils.IO.nBkg):
  #     preprocessing.add_deltaR_branches(utils.IO.background_df[i])
  #  for i in range(utils.IO.nSig):
  #     preprocessing.add_deltaR_branches(utils.IO.signal_df[i])
  #  branch_names = branch_names + ['photJetdRmin','photJetdRmin2'] 
   ##### New photon + jet branches added  above #####
   # event_branches+=['SumWeight','normalization']

############################ Do THIS ONLY FOR THE CURRENT G Jet 40 for 2017 ########
  #  if options.year==1:
  #     for i in range(utils.IO.nBkg):        
  #        if "GJet_Pt_40toInf" in utils.IO.bkgTreeName[i] :
  #            preprocessing.scale_weight(utils.IO.background_df[i],1.3) # because not all jobs finished
  #  if options.year==2:
  #     for i in range(utils.IO.nBkg):        
  #        if "TTTo2L2Nu" in utils.IO.bkgTreeName[i] :
  #            preprocessing.scale_weight(utils.IO.background_df[i],5.13) # because not all jobs finished
##################################################################################
   # X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.set_variables(branch_names+['year'])  
    X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.set_variables(branch_names)  
 
    data_branches = ["weight"]
    #data_branches = 'weight,leadingJet_DeepFlavour,subleadingJet_DeepFlavour, VBFleadJet_QGL, VBFsubleadJet_QGL,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,noexpand:(leadingPhoton_pt/CMS_hgg_mass),noexpand:(subleadingPhoton_pt/CMS_hgg_mass),noexpand:(leadingJet_pt/Mjj),noexpand:(subleadingJet_pt/Mjj),rho,noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826),PhoJetMinDr,PhoJetOtherDr,VBFleadJet_pt,VBFleadJet_eta,VBFsubleadJet_pt,VBFsubleadJet_eta,VBFCentrality_jg,VBFCentrality_jb,VBFDeltaR_jg,VBFDeltaR_jb,VBFProd_eta,VBFDelta_phi,VBFJet_mjj,VBFJet_Delta_eta'.split(",")

    
    if options.addData:
        #preprocessing.set_data(branch_names+branch_cuts+event_branches,cuts)
        #X_data,y_data,weights_data = preprocessing.set_variables_data(branch_names)
        preprocessing.set_data(branch_names+additionalCut_names,cuts)
        X_data,y_data,weights_data = preprocessing.set_variables_data(branch_names)
        X_data,y_data,weights_data = preprocessing.clean_signal_events_single_dataset(X_data,y_data,weights_data)
    
    #bbggTrees have by default signal and CR events, let's be sure that we clean it
    if y_bkg.shape[1]==1 : 
        X_bkg,y_bkg,weights_bkg = preprocessing.clean_signal_events_single_dataset(X_bkg,y_bkg,weights_bkg)
        X_sig,y_sig,weights_sig = preprocessing.clean_signal_events_single_dataset(X_sig,y_sig,weights_sig)
    else : 
        X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.clean_signal_events(X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig)
    
    
    # load the model from disk
    from sklearn.externals import joblib
    
    bkg = []
    for i in range(0,len(utils.IO.backgroundName)): 
        bkg.append(X_bkg[y_bkg ==utils.IO.bkgProc[i]])
        print utils.IO.backgroundName[i],'with proc num : ',utils.IO.bkgProc[i]
    
    print 'add Tagger' ,options.addHHTagger 
    #compute the MVA
    Y_pred_bkg = []
    if not options.addHHTagger:
        print 'Adding tagger output'
        loaded_model = joblib.load(os.path.expanduser(options.trainingDir+options.trainingVersion+'.pkl'))
        loaded_model._Booster.set_param('nthread', 10)
        print "loading"+options.trainingDir+options.trainingVersion+'.pkl'
#        print(loaded_model.get_xgb_params)
        if options.addData:
            Y_pred_data = loaded_model.predict_proba(X_data)[:,loaded_model.n_classes_-1].astype(np.float64)
            #print Y_pred_data 
        for i in range(0,len(utils.IO.backgroundName)):  
       # for i in range(0,0):   #not to apply MVA on bkg
            print 'evaluating MVA for bkg : ',str(i)
            Y_pred_bkg.append(loaded_model.predict_proba(bkg[i])[:,loaded_model.n_classes_-1].astype(np.float64))
        Y_pred_sig = loaded_model.predict_proba(X_sig)[:,loaded_model.n_classes_-1].astype(np.float64)
    
    
    
    outTag = options.outTag
    outDir=os.path.expanduser("/afs/cern.ch/work/m/mukherje/Training_VBFHH/HHbbgg_ETH/"+outTag)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    branch_names+=branch_cuts
    branch_names+=event_branches
   

###########################  data  block starts  ################################################################ 
    if options.addData:   
      #  data_count_df = (rpd.read_root(utils.IO.dataName[0],utils.IO.dataTreeName[0], columns = branch_names+additionalCut_names+bkg_trainedOn)).query(cuts)
      #  nTot,dictVar = postprocessing.stackFeatures(data_count_df,branch_names+additionalCut_names,isData=1)
        data_count_df = (rpd.read_root(utils.IO.dataName[0],utils.IO.dataTreeName[0], columns =branch_names+data_weight)).query(cuts)
        nTot,dictVar = postprocessing.stackFeatures(data_count_df,branch_names+data_weight,isData=1)
    #apply isSignal cleaning
        nCleaned = nTot[np.where(nTot[:,dictVar['weight']]!=0),:][0]
        print "nCleaned"
        print nCleaned.shape
  
  #save preselection data
        processPath=os.path.expanduser(options.outputFileDir)+outTag+'/'+utils.IO.dataName[0].split("/")[len(utils.IO.dataName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"
        if not options.addHHTagger:        
            postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_data)
        else:
            postprocessing.saveTree(processPath,dictVar,nCleaned)
 
        processPath=os.path.expanduser(options.outputFileDir)+outTag+'/'+utils.IO.dataName[0].split("/")[len(utils.IO.dataName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"
        if not options.addHHTagger:        
            postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_data,nameTree="reducedTree_data%s"%treeTag)
        else:
            postprocessing.saveTree(processPath,dictVar,nCleaned,nameTree="reducedTree_data%s"%treeTag)
###########################   data  block  ends  ##############################################################

 
###########################  signal  block starts  ################################################################
    for isig in range(0,len(utils.IO.signalName)):
       sig_count_df = utils.IO.signal_df[isig]
       print utils.IO.signalName[isig]
       preprocessing.define_process_weight(sig_count_df,utils.IO.sigProc[isig],utils.IO.signalName[isig],utils.IO.signalTreeName[isig],cleanSignal=True,cleanOverlap=cleanOverlap)

 
    #nTot is a multidim vector with all additional variables, dictVar is a dictionary associating a name of the variable
    #to a position in the vector
       nTot,dictVar = postprocessing.stackFeatures(sig_count_df,branch_names+additionalCut_names+signal_trainedOn+overlap)
    #apply isSignal cleaning
       nCleaned = nTot[np.where(nTot[:,dictVar['weight']]!=0),:][0]
    
       processPath=os.path.expanduser(options.outputFileDir)+outTag+'/'+utils.IO.signalName[isig].split("/")[len(utils.IO.signalName[isig].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"


       if not options.addHHTagger:
            postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_sig)
       else:
             postprocessing.saveTree(processPath,dictVar,nCleaned)        
    
       processPath=os.path.expanduser(options.outputFileDir)+outTag+'/'+utils.IO.signalName[isig].split("/")[len(utils.IO.signalName[isig].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"

    if not options.addHHTagger:
             postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_sig,nameTree="reducedTree_sig%s"%treeTag)
    else:    
             postprocessing.saveTree(processPath,dictVar,nCleaned,nameTree="reducedTree_sig%s"%treeTag)
  
##########################  signal  block ends  ################################################################ 
    
    for iProcess in range(0,len(utils.IO.backgroundName)):
  #  for iProcess in range(0,0):  #not to run on bkg
        
        print "Processing sample: "+str(iProcess)
        bkg_count_df = utils.IO.background_df[iProcess]
        preprocessing.define_process_weight(bkg_count_df,utils.IO.bkgProc[iProcess],utils.IO.backgroundName[iProcess],utils.IO.bkgTreeName[iProcess],cleanSignal=True,cleanOverlap=True)
    
        crazySF=1.
        nTot,dictVar = postprocessing.stackFeatures(bkg_count_df,branch_names+additionalCut_names+bkg_trainedOn+overlap,SF=crazySF)
        nCleaned = nTot
        print "nCleaned"
        print nCleaned.shape
    

        bkgName = background_names[iProcess]
        bkgName_idx = len(samples)-1  #how many bkg we have
        print (bkgName)
        print 'bkg Index : ',bkgName_idx 
        print 'predictd bkg len and size of each bkg sample: ',len(Y_pred_bkg),Y_pred_bkg

        processPath=os.path.expanduser(options.outputFileDir)+outTag+'/'+utils.IO.backgroundName[iProcess].split("/")[len(utils.IO.backgroundName[iProcess].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"
        if not options.addHHTagger:
            postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_bkg[iProcess])
        else:
            postprocessing.saveTree(processPath,dictVar,nCleaned)
        
        processPath=os.path.expanduser(options.outputFileDir)+outTag+'/'+utils.IO.backgroundName[iProcess].split("/")[len(utils.IO.backgroundName[iProcess].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"
        if options.addrew and "reweighted_nodes"in processPath:
            processPath = processPath.replace("reweighted_nodes_","reweighted_node_"+str(iProcess-(len(samples)-3)))
        if "GluGluToHHTo2B2G_reweighted_node"in processPath and options.addrew:
         #   treeName = "reducedTree_sig_node_"+str(iProcess-7)
            treeName = "reducedTree_sig_node_"+str(iProcess-(bkgName_idx))+treeTag
        else:
            #treeName = "reducedTree_bkg_"+str(iProcess)+treeTag
            treeName = "reducedTree_bkg_"+bkgName+treeTag

        if not options.addHHTagger:        
            postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_bkg[iProcess],nameTree=treeName)
        else:
            postprocessing.saveTree(processPath,dictVar,nCleaned,nameTree=treeName)    
    
    
    os.system('hadd '+ os.path.expanduser(options.outputFileDir)+outTag+'/'+'Total_preselection_diffNaming.root '+ os.path.expanduser(options.outputFileDir)+outTag+'/'+'*diffNaming.root')

    


if __name__ == "__main__":

    parser = OptionParser(option_list=[
            make_option("-n", "--ntuples",
                        action="store", type="string", dest="ntup",
                        default="",
                        help="ntuples location",
                        ),
            make_option("-a","--addMVAOutput",
                        action="store_true",dest="addHHTagger",default=False,
                        help="add MVAOutput to outTree",
                        ),
            make_option("-t","--training",
                        action="store", type="string",dest="trainingVersion",
                        default="training_with_2018_test_C2V0_training",
                        help="MVA version to apply",
                        ),
            make_option("-x","--trainingDir",
                        action="store",type="string",dest="trainingDir",default="/afs/cern.ch/work/m/mukherje/Training_VBFHH/HHbbgg_ETH/Training/output_files/",
                        help="directory from where to load pklfile",
                        ),
            make_option("-o", "--out",
                        action="store", type="string", dest="outTag",
                        default="",
                        help="output folder name",
                        ),
            make_option("-k","--nodes",
                        action="store_false",dest="addnodes",default=False,
                        help="add or not nodes",
                        ),
            make_option("-w","--reweightednodes",
                        action="store_true",dest="addrew",default=False,
                        help="add or not reweighted nodes",
                        ),
            make_option("-y","--year",
                        action="store",type=int,dest="year",default=2,
                        help="which year : 2016-0,2017-1,2018-2",
                        ),
            make_option("-l","--ldata",
                        action="store",type="string",dest="ldata",default="",
                        help="directory from where to load data (if different from default one)",
                        ),
            make_option("-d","--adddata",
                        action="store_true",dest="addData",default=False,
                        help="decide if you want to process or not data",
                        ),
            make_option("-f","--outputFileDir",
                        action="store",type="string",dest="outputFileDir",default="/afs/cern.ch/work/m/mukherje/Training_VBFHH/HHbbgg_ETH/Training/2018_C2V0/",
                        help="directory where to save output trees",
                        ),
            ]
                          )

    (options, args) = parser.parse_args()
    sys.argv.append("-b")

    
    pprint(options.__dict__)

    import ROOT
    
    main(options,args)
        
