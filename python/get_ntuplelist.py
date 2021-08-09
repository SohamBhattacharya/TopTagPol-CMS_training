import os
import subprocess


basepath = "/pnfs/desy.de/cms/tier2/store/user/sobhatta/TopTagPol/ntuples"
outdir = "sourceFiles"

l_sample = [
"QCD_Pt_170to300_TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-PREMIX_RECODEBUG_102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"QCD_Pt_300to470_TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-PREMIX_RECODEBUG_102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"QCD_Pt_470to600_TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-PREMIX_RECODEBUG_102X_upgrade2018_realistic_v15_ext1-v1_MINIAODSIM",
"QCD_Pt_600to800_TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-PREMIX_RECODEBUG_102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-PREMIX_RECODEBUG_102X_upgrade2018_realistic_v15_ext1-v1_MINIAODSIM",
"QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-PREMIX_RECODEBUG_102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-PREMIX_RECODEBUG_102X_upgrade2018_realistic_v15_ext1-v1_MINIAODSIM",
"QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-PREMIX_RECODEBUG_102X_upgrade2018_realistic_v15_ext1-v1_MINIAODSIM",
"QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8_RunIIAutumn18MiniAOD-PREMIX_RECODEBUG_102X_upgrade2018_realistic_v15_ext1-v1_MINIAODSIM",

"ZprimeToTT_M1000_W10_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M1000_W100_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M1000_W300_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M1250_W12p5_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M1250_W125_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M1250_W375_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M1500_W15_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M1500_W150_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M1500_W450_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M2000_W20_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M2000_W200_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M2000_W600_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M3000_W30_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M3000_W300_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M3000_W900_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M3500_W35_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M3500_W350_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M3500_W1050_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M4000_W40_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M4000_W400_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
"ZprimeToTT_M4000_W1200_TuneCP2_PSweights_13TeV-madgraphMLM-pythia8_RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1_MINIAODSIM",
]


for iSample, sample in enumerate(l_sample) :
    
    print("\n")
    print("Sample: %s" %(sample))
    
    samplepath = "%s/%s" %(basepath, sample)
    
    datedir = subprocess.check_output("ls %s | sort -V | tail -n 1" %(samplepath), shell = True).decode("utf-8").strip()
    
    outfilename = "%s/%s_%s.txt" %(outdir, sample, datedir)
    
    cmd = "find `find {samplepath} -mindepth 1 -maxdepth 1 | sort -V | tail -n 1` -mindepth 1 | sort -V > {outfilename}".format(
        samplepath = samplepath,
        outfilename = outfilename,
    )
    
    print("Command: %s" %(cmd))
    
    os.system(cmd)
    os.system("wc -l %s" %(outfilename))
