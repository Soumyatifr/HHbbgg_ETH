from optparse import OptionParser, make_option
from  pprint import pprint

import os
import sys; sys.path.append("/shome/nchernya/HHbbgg_ETH_devel/Training/python") # to load packages
import training_utils as utils
import numpy as np
import preprocessing_utils as preprocessing
import plotting_utils as plotting
import optimization_utils as optimization
import postprocessing_utils as postprocessing

import pandas as pd
import root_pandas as rpd


#samples = ["GluGluToHHTo2B2G_node_SM","DiPhotonJetsBox_","GJet_Pt-20to40","GJet_Pt-40","DiPhotonJetsBox2BJets_","DiPhotonJetsBox1BJet_","GluGluHToGG_","VBFHToGG_","VHToGG_","ttHToGG_"]#
samples = ["GluGluToHHTo2B2G_node_SM","DiPhotonJetsBox_","GJet_Pt-20to40","GJet_Pt-40"]#
treeDir = 'tagsDumper/trees/'

def addSamples():#define here the samples you want to process
    ntuples = options.ntup
    year = options.year
   # samples = ["GluGluToHHTo2B2G_node_SM","DiPhotonJetsBox_","GJet_Pt-20to40","GJet_Pt-40","GluGluHToGG","VBFHToGG","VHToGG","bbHToGG_M-125_4FS_yb2","bbHToGG_M-125_4FS_ybyt","ttHToGG"]#is bbH correct?

    files= os.listdir(utils.IO.ldata+ntuples)

   # for iSample in samples:
    for num,iSample in enumerate(samples):
        process  = [s for s in files if iSample in s]
        print 'process : ',process
        if iSample == "GluGluToHHTo2B2G_node_SM":
            utils.IO.add_signal(ntuples,process,1,treeDir+process[0][process[0].find('output_')+7:process[0].find('.root')].replace('-','_')+'_13TeV_DoubleHTag_0',year)
        elif not "GJet" in str(iSample):
            utils.IO.add_background(ntuples,process,-(samples.index(iSample)-(samples.index(iSample)>2)),treeDir+process[0][process[0].find('output_')+7:process[0].find('.root')].replace('-','_')+'_13TeV_DoubleHTag_0',year)
        else:
            utils.IO.add_background(ntuples,process,-2,treeDir+process[0][process[0].find('output_')+7:process[0].find('.root')].replace('-','_')+'_13TeV_DoubleHTag_0',year)

    

    nBkg = len(utils.IO.backgroundName)
    
    Data= [s for s in files if "DoubleEG" in s]
    #utils.IO.add_data(ntuples,Data,-10,'tree')
    dataTreeName = 'Data_13TeV_DoubleHTag_0'
    utils.IO.add_data(ntuples,Data,-10,treeDir+dataTreeName)
    
    #add all nodes
    nodes = []
    nodesTreeNames = []
    if not options.addnodes:
        for i in range(2,14) + ['box']:
            nodes.append([s for s in files if "GluGluToHHTo2B2G_node_"+str(i) in s])
            nodesTreeNames.append("GluGluToHHTo2B2G_node_"+str(i)+'_13TeV_madgraph_13TeV_DoubleHTag_0')
    for i in range(nBkg,nBkg+len(nodes)):
        utils.IO.add_background(ntuples,nodes[i-nBkg],-i,treeDir+nodesTreeNames[i-nBkg])
        
    for i in range(len(utils.IO.backgroundName)):        
        print "using background file n."+str(i)+": "+utils.IO.backgroundName[i]
    for i in range(len(utils.IO.signalName)):    
        print "using signal file n."+str(i)+": "+utils.IO.signalName[i]
    print "using data file: "+ utils.IO.dataName[0]
    


def main(options,args):
    

    addSamples()
    
    #mva variables, use noexpand for root expressions, it needs this file https://github.com/ibab/root_pandas/blob/master/root_pandas/readwrite.py
  #  branch_names = 'leadingJet_bDis,subleadingJet_bDis,noexpand:fabs(CosThetaStar_CS),noexpand:fabs(CosTheta_bb),noexpand:fabs(CosTheta_gg)'.split(",")
  #  branch_names += 'noexpand:diphotonCandidate.Pt()/diHiggsCandidate.M(),noexpand:dijetCandidate.Pt()/diHiggsCandidate.M()'.split(",")
 ###   branch_names += 'customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverMDecorr,PhoJetMinDr'.split(",")
  #  branch_names += 'customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,PhoJetMinDr'.split(",")
   ############ branch names with regression ############
#    branch_names = 'leadingJetCorr_bDis,subleadingJetCorr_bDis,noexpand:fabs(CosThetaStar_CS),noexpand:fabs(CosTheta_bb),noexpand:fabs(CosTheta_gg)'.split(",")
#    branch_names += 'noexpand:diphotonCandidate.Pt()/diHiggsCandidateCorr.M(),noexpand:dijetCandidateCorr.Pt()/diHiggsCandidateCorr.M()'.split(",")
#    branch_names += 'customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverM,PhoJetMinDr'.split(",") #set of variables March 2017 but regressed
#    branch_names += 'noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(dijetSigmaMOverM*1.4826)'.split(",")
######################new codea#################################
   # branch_names = 'leadingJet_bDis,subleadingJet_bDis,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverMDecorr,PhoJetMinDr'.split(",") #set of variables March 2017 but regressed
    branch_names = 'leadingJet_DeepCSV,subleadingJet_DeepCSV,absCosThetaStar_CS,absCosTheta_bb,absCosTheta_gg,diphotonCandidatePtOverdiHiggsM,dijetCandidatePtOverdiHiggsM,customLeadingPhotonIDMVA,customSubLeadingPhotonIDMVA,leadingPhotonSigOverE,subleadingPhotonSigOverE,sigmaMOverMDecorr,PhoJetMinDr'.split(",") #set of variables March 2017 but regressed
    branch_names +=['rho']
    branch_names += 'noexpand:(leadingJet_bRegNNResolution*1.4826),noexpand:(subleadingJet_bRegNNResolution*1.4826),noexpand:(sigmaMJets*1.4826)'.split(",")
    branch_cuts = 'leadingJet_pt,subleadingJet_pt,leadingJet_bRegNNCorr,subleadingJet_bRegNNCorr,noexpand:(leadingJet_pt/leadingJet_bRegNNCorr),noexpand:(subleadingJet_pt/subleadingJet_bRegNNCorr)'.split(',')
    event_branches = ['event','weight','leadingJet_hflav','leadingJet_pflav','subleadingJet_hflav','subleadingJet_pflav']
 #   cuts = 'leadingJet_pt>20 & subleadingJet_pt> 20 & (leadingJet_pt/leadingJet_bRegNNCorr>20) & (subleadingJet_pt/subleadingJet_bRegNNCorr>20) '
    cuts = 'leadingJet_pt>0 '
######################
################################################################


    branch_names = [c.strip() for c in branch_names]
    print "using following variables for MVA: " 
    print branch_names
    
    
    # no need to shuffle here, we just count events
   # preprocessing.set_signals_and_backgrounds("bbggSelectionTree",branch_names,shuffle=False)
    #preprocessing.set_signals_and_backgrounds_drop(branch_names+branch_cuts+event_branches,False,cuts)  #############Temporary fix to drop all NAN events
    preprocessing.set_signals_and_backgrounds(branch_names+branch_cuts+event_branches,False,cuts) 
    X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.set_variables(branch_names+['year'])  
 
#when no data is present   
###    X_data,y_data,weights_data = preprocessing.set_data("bbggSelectionTree",branch_names)
###    X_data,y_data,weights_data = preprocessing.clean_signal_events_single_dataset(X_data,y_data,weights_data)
#    preprocessing.set_data(branch_names+branch_cuts+event_branches,cuts)
#    X_data,y_data,weights_data = preprocessing.set_variables_data(branch_names)
#    X_data,y_data,weights_data = preprocessing.clean_signal_events_single_dataset(X_data,y_data,weights_data)
    
    #bbggTrees have by default signal and CR events, let's be sure that we clean it
    if y_bkg.shape[1]==1 : 
        X_bkg,y_bkg,weights_bkg = preprocessing.clean_signal_events_single_dataset(X_bkg,y_bkg,weights_bkg)
        X_sig,y_sig,weights_sig = preprocessing.clean_signal_events_single_dataset(X_sig,y_sig,weights_sig)
    else : 
        X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig=preprocessing.clean_signal_events(X_bkg,y_bkg,weights_bkg,X_sig,y_sig,weights_sig)
    
    
    # load the model from disk
    from sklearn.externals import joblib
    
    bkg = []
    for i in range(0,len(utils.IO.backgroundName)-1): 
        bkg.append(X_bkg[y_bkg ==-i-1])
    
        
    #compute the MVA
    if not options.addHHTagger:
        loaded_model = joblib.load(os.path.expanduser('/shome/nchernya/HHbbgg_ETH_devel/Training/output_files/'+options.trainingVersion+'.pkl'))
        Y_pred_sig = loaded_model.predict_proba(X_sig)[:,loaded_model.n_classes_-1].astype(np.float64)
        Y_pred_bkg = []
        for i in range(0,len(utils.IO.backgroundName)-1):  
            print str(i)
            Y_pred_bkg.append(loaded_model.predict_proba(bkg[i])[:,loaded_model.n_classes_-1].astype(np.float64))
    
#when no data is present   
  #      Y_pred_data = loaded_model.predict_proba(X_data)[:,loaded_model.n_classes_-1].astype(np.float64)
  #      print Y_pred_data 
    
    
    
    #define MVA cut and additional variables needed
   # additionalCut_names = 'noexpand:diphotonCandidate.M(),noexpand:dijetCandidate.M(),MX,isSignal,event,noexpand:dijetCandidateCorr.M(),noexpand:(event%2!=0),subleadingJet_genHadronFlavourb,leadingJet_genHadronFlavourb'.split(",")
  #  additionalCut_names = 'CMS_hgg_mass,Mjj,noexpand:(event%2!=0),noexpand:(event%5!=0)'.split(',')
    additionalCut_names = 'CMS_hgg_mass,Mjj,MX'.split(',')
    signal_trainedOn = ['noexpand:(event%2!=0)']
    bkg_trainedOn = ['noexpand:(event%1==0)'] #to accept all events
    overlap = ['overlapSave']
    #mva output
    if options.addHHTagger:
        additionalCut_names += 'HHTagger2017,HHTagger2017_transform'.split(",")
    outTag = options.outTag
    outDir=os.path.expanduser("/shome/nchernya/HHbbgg_ETH_devel/outfiles/"+outTag)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    branch_names+=branch_cuts
    branch_names+=event_branches
    
    sig_count_df = (rpd.read_root(utils.IO.signalName[0],utils.IO.signalTreeName[0], columns = branch_names+additionalCut_names+signal_trainedOn)).query(cuts)
    preprocessing.define_process_weight(sig_count_df,utils.IO.sigProc[0],utils.IO.signalName[0],utils.IO.signalTreeName[0])
       
 
    #nTot is a multidim vector with all additional variables, dictVar is a dictionary associating a name of the variable
    #to a position in the vector
    nTot,dictVar = postprocessing.stackFeatures(sig_count_df,branch_names+additionalCut_names+signal_trainedOn)
    #apply isSignal cleaning
    nCleaned = nTot[np.where(nTot[:,dictVar['weight']]!=0),:][0]
    
    processPath=os.path.expanduser('/shome/nchernya/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.signalName[0].split("/")[len(utils.IO.signalName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"

    if not options.addHHTagger:
        postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_sig)
    else:
        postprocessing.saveTree(processPath,dictVar,nCleaned)        
    
    processPath=os.path.expanduser('/shome/nchernya/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.signalName[0].split("/")[len(utils.IO.signalName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"

    if not options.addHHTagger:
        postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_sig,nameTree="reducedTree_sig")
    else:    
        postprocessing.saveTree(processPath,dictVar,nCleaned,nameTree="reducedTree_sig")
    
    ## do gJets not in the loop since they have two samples for one process, to be fixed
    bkg_1_count_df = (rpd.read_root(utils.IO.backgroundName[1],utils.IO.bkgTreeName[1], columns = branch_names+additionalCut_names+bkg_trainedOn)).query(cuts)
    preprocessing.define_process_weight(bkg_1_count_df,utils.IO.bkgProc[1],utils.IO.backgroundName[1],utils.IO.bkgTreeName[1])
    
    crazySF_20=25
    nTot,dictVar = postprocessing.stackFeatures(bkg_1_count_df,branch_names+additionalCut_names+bkg_trainedOn,SF=crazySF_20)
    
    print nTot.shape
    
    bkg_2_count_df = (rpd.read_root(utils.IO.backgroundName[2],utils.IO.bkgTreeName[2], columns = branch_names+additionalCut_names+bkg_trainedOn)).query(cuts)
    preprocessing.define_process_weight(bkg_2_count_df,utils.IO.bkgProc[2],utils.IO.backgroundName[2],utils.IO.bkgTreeName[2])
    
    crazySF_40=3
    nTot_2,dictVar = postprocessing.stackFeatures(bkg_2_count_df,branch_names+additionalCut_names+bkg_trainedOn,SF=crazySF_40)
    
    
    nTot_3 = np.concatenate((nTot,nTot_2))
    
    print nTot_3.shape
    nCleaned = nTot_3[np.where(nTot_3[:,dictVar['weight']]!=0),:][0]
    print "nCleaned"
    print nCleaned.shape
    
    processPath=(os.path.expanduser('/shome/nchernya/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.backgroundName[1].split("/")[len(utils.IO.backgroundName[1].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root").replace("_Pt-20to40","")
    if not options.addHHTagger:
        postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_bkg[1])
    else:
        postprocessing.saveTree(processPath,dictVar,nCleaned) 
   
    processPath=(os.path.expanduser('/shome/nchernya/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.backgroundName[1].split("/")[len(utils.IO.backgroundName[1].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root").replace("_Pt-20to40","")

    if not options.addHHTagger:
        postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_bkg[1],nameTree="reducedTree_bkg_2")
    else:
        postprocessing.saveTree(processPath,dictVar,nCleaned,nameTree="reducedTree_bkg_2")    
    
    
    
    for iProcess in range(0,len(utils.IO.backgroundName)):
        ##gJets which are two samples for one process are skipped
        iSample=iProcess
        if iProcess == 1 or iProcess ==2:
            continue
        if iProcess > 2:
            iSample = iProcess-1
        
        print "Processing sample: "+str(iProcess)
        bkg_count_df = (rpd.read_root(utils.IO.backgroundName[iProcess],utils.IO.bkgTreeName[iProcess], columns = branch_names+additionalCut_names+bkg_trainedOn)).query(cuts)
        preprocessing.define_process_weight(bkg_count_df,utils.IO.bkgProc[iProcess],utils.IO.backgroundName[iProcess],utils.IO.bkgTreeName[iProcess])
    
        crazySF=1
        ##scale diphoton + jets
        if iProcess == 0:
            crazySF=1.45
        nTot,dictVar = postprocessing.stackFeatures(bkg_count_df,branch_names+additionalCut_names+bkg_trainedOn,SF=crazySF)
    
    
        nCleaned = nTot[np.where(nTot[:,dictVar['weight']]!=0),:][0]
        print "nCleaned"
        print nCleaned.shape
    
    #    processPath=os.path.expanduser('~/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.backgroundName[iProcess].split("/")[len(utils.IO.backgroundName[7].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"
        processPath=os.path.expanduser('/shome/nchernya/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.backgroundName[iProcess].split("/")[len(utils.IO.backgroundName[(len(samples)-3)].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"
        if not options.addHHTagger:
            postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_bkg[iSample])
        else:
            postprocessing.saveTree(processPath,dictVar,nCleaned)
        
       # processPath=os.path.expanduser('~/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.backgroundName[iProcess].split("/")[len(utils.IO.backgroundName[7].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"
        processPath=os.path.expanduser('/shome/nchernya/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.backgroundName[iProcess].split("/")[len(utils.IO.backgroundName[(len(samples)-3)].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"
        if "GluGluToHHTo2B2G_node_"in processPath:
         #   treeName = "reducedTree_sig_node_"+str(iProcess-7)
            treeName = "reducedTree_sig_node_"+str(iProcess-(len(samples)-3))
        else:
            treeName = "reducedTree_bkg_"+str(iProcess)
        if not options.addHHTagger:        
            postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_bkg[iSample],nameTree=treeName)
        else:
            postprocessing.saveTree(processPath,dictVar,nCleaned,nameTree=treeName)    
    
###########################  data  block starts  ################################################################### 
 #   ####data_count_df = (rpd.read_root(utils.IO.dataName[0],"bbggSelectionTree", columns = branch_names+additionalCut_names)).query(cuts)
 #   data_count_df = (rpd.read_root(utils.IO.dataName[0],utils.IO.dataTreeName[0], columns = branch_names+additionalCut_names+bkg_trainedOn)).query(cuts)
 #   
 #   nTot,dictVar = postprocessing.stackFeatures(data_count_df,branch_names+additionalCut_names,isData=1)
 #   
 #   
 #   #apply isSignal cleaning
 #   nCleaned = nTot[np.where(nTot[:,dictVar['weight']]!=0),:][0]
 #   print "nCleaned"
 #   print nCleaned.shape
 #   
 #   #save preselection data
  #  processPath=os.path.expanduser('~/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.dataName[0].split("/")[len(utils.IO.dataName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection"+".root"
  #  if not options.addHHTagger:        
  #      postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_data)
  #  else:
  #      postprocessing.saveTree(processPath,dictVar,nCleaned)
    
  #  processPath=os.path.expanduser('~/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+utils.IO.dataName[0].split("/")[len(utils.IO.dataName[0].split("/"))-1].replace("output_","").replace(".root","")+"_preselection_diffNaming"+".root"
  #  if not options.addHHTagger:        
  #     postprocessing.saveTree(processPath,dictVar,nCleaned,Y_pred_data,nameTree="reducedTree_data")
  #  else:
  #      postprocessing.saveTree(processPath,dictVar,nCleaned,nameTree="reducedTree_data")
    
    os.system('hadd '+ os.path.expanduser('/shome/nchernya/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+'Total_preselection_diffNaming.root '+ os.path.expanduser('/shome/nchernya/HHbbgg_ETH_devel/outfiles/')+outTag+'/'+'*diffNaming.root')
###########################   data  block  ends  ################################################################### 
    


if __name__ == "__main__":

    parser = OptionParser(option_list=[
            make_option("-n", "--ntuples",
                        action="store", type="string", dest="ntup",
                        default="20170620",
                        help="ntuples location",
                        ),
            make_option("-a","--addMVAOutput",
                        action="store_true",dest="addHHTagger",default=False,
                        help="add MVAOutput to outTree",
                        ),
            make_option("-t","--training",
                        action="store", type="string",dest="trainingVersion",
                        default="allMC_resWeighting_F_noDR_minDRGJet",
                        help="MVA version to apply",
                        ),

            make_option("-o", "--out",
                        action="store", type="string", dest="outTag",
                        default="20180108_test",
                        help="output folder name",
                        ),
            make_option("-k","--nodes",
                        action="store_true",dest="addnodes",default=False,
                        help="add or not nodes",
                        ),
            make_option("-y","--year",
                        action="store",type=int,dest="year",default=0,
                        help="which year : 2016-0,2017-1,2018-2",
                        ),
            ]
                          )

    (options, args) = parser.parse_args()
    sys.argv.append("-b")

    
    pprint(options.__dict__)

    import ROOT
    
    main(options,args)
        
