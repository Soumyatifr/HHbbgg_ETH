from ROOT import *
from array import array
import os 
import argparse

parser =  argparse.ArgumentParser(description='Add Classification BDT weights')
parser.add_argument('-f', '--file', dest='file', required=True, type=str)
parser.add_argument('-v', '--var', dest='var', required=True, type=str)
parser.add_argument('-c', '--cut', dest='cut', required=True, type=str)
parser.add_argument('-o', '--output', dest='out', required=True, type=str)
parser.add_argument('-e', '--backEff', dest='beff', action='store_true', default=False)
parser.add_argument('--BCR', dest='bcr', action='store_true', default=False)
parser.add_argument('--BSR', dest='bsr', action='store_true', default=False)


opt = parser.parse_args()

os.system("hadd "+opt.file+" "+opt.file.replace("Total_preselection_diffNaming.root","*diffNaming.root"))

f = TFile(opt.file)

t_sig = f.Get("reducedTree_sig")
t_bac = f.Get("reducedTree_bkg")

var_n = opt.var.split(',')[0]
var_b = opt.var.split(',')[1]
var_min = opt.var.split(',')[2]
var_max = opt.var.split(',')[3]

var_step = (float(var_max) - float(var_min))/float(var_b)

eff_sig = TH1F("eff_sig", ";"+var_n+";Signal Efficiency", int(var_b), float(var_min), float(var_max))
rej_bac = TH1F("rej_bac", ";"+var_n+";Background Rejection (1-Eff)", int(var_b), float(var_min), float(var_max))
ROC = TGraph()
ROC.SetName("ROC")

h_sig = TH1F("h_sig", ";"+var_n+";",int(var_b), float(var_min), float(var_max))
h_bac = TH1F("h_bac", ";"+var_n+";",int(var_b), float(var_min), float(var_max))

t_sig.Draw(var_n+">>h_sig", opt.cut+ " && isSignal==1", "goff")
if opt.bcr:
  t_bac.Draw(var_n+">>h_bac", opt.cut+ " && isSignal==0 && !(Mgg>115 && Mgg<135)", "goff")
elif opt.bsr:
  t_bac.Draw(var_n+">>h_bac", opt.cut+ " && isSignal==1 && !(Mgg>115 && Mgg<135)", "goff")
else:
  t_bac.Draw(var_n+">>h_bac", opt.cut+ " && isSignal==0", "goff")

h_sig.Sumw2()
tot_sig = h_sig.Integral()
h_sig.Scale(1./tot_sig)

h_bac.Sumw2()
tot_bac = h_bac.Integral()
h_bac.Scale(1./tot_bac)

for x in range(0, int(var_b)):
  print '#bin',x
  tst = float(var_min) + x*var_step
#  sEff = float(t_sig.GetEntries((opt.cut+ " && isSignal==1" + " && " + var_n + " > " + str(tst))))/float(tot_sig)                                                                                                                        
#  bRej = float(t_bac.GetEntries((opt.cut+ " && isSignal==0" + " && " + var_n + " < " + str(tst))))/float(tot_bac)                                                                                                                        
#  sEff = h_sig.Integral(tst, float(var_max))                                                                                                                                                                                             
#  bRej = h_bac.Integral(float(var_min), tst)                                                                                                                                                                                             
  sEff = h_sig.Integral(x+1, int(var_b)+1)
  bRej = h_bac.Integral(1, x+1)
  if opt.beff: bRej = h_bac.Integral(x+1, int(var_b)+1)
  eff_sig.SetBinContent(x+1, sEff)
  rej_bac.SetBinContent(x+1, bRej)
  ROC.SetPoint(x, sEff, bRej)
  
def myROC(x,par=0):
  return ROC.Eval(x[0])

myROCTF1 = TF1("myROCTF1", myROC, 0, 1, 0)

outfile = TFile(opt.out, "RECREATE")
eff_sig.Write()
rej_bac.Write()
ROC.Write()
h_sig.Write()
h_bac.Write()
myROCTF1.Write()
outfile.Close()
