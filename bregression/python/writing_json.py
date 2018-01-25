import json


data = {"pt_regions" : [ '(Jet_mcPt>0)','(Jet_mcPt<60)','(Jet_mcPt>=60 & Jet_mcPt<100)','(Jet_mcPt>=100 & Jet_mcPt<150)','(Jet_mcPt>=150 & Jet_mcPt<200)','(Jet_mcPt>=200 & Jet_mcPt<250)','(Jet_mcPt>=250 & Jet_mcPt<300)','(Jet_mcPt>=300 & Jet_mcPt<400)','(Jet_mcPt>=400 & Jet_mcPt<600)','(Jet_mcPt>=600)'] ,
	"eta_regions" : ['(Jet_eta<0.5 & Jet_eta>-0.5)','((Jet_eta>=0.5 & Jet_eta<1.0) |(Jet_eta<=-0.5 & Jet_eta>-1.0))','(( Jet_eta>=1.0 & Jet_eta<1.5)|(Jet_eta<=-1.0 & Jet_eta>-1.5))','((Jet_eta>=1.5 & Jet_eta<2.0)|(Jet_eta<=-1.5 & Jet_eta>=-2.0 ))','(Jet_eta>=2.0 | Jet_eta<=-2.0) )'],
	"eta_region_names" : ['|Jet_eta|<0.5','|Jet_eta|>=0.5 & |Jet_eta|<1.0','|Jet_eta|>=1.0 & |Jet_eta|<1.5','|Jet_eta|>=1.5 & |Jet_eta|<2.0','|Jet_eta|>=2.0'] }

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)
