from __future__ import print_function

import numpy
import time
import uproot

import ROOT
ROOT.gROOT.SetBatch(1)



def main() :
    
    #tree = uproot.open("/home/soham/naf_dust/work/TopTagPol/TreeMaker/CMSSW_10_5_0/src/ntupleTree_ZprimeToTT_M1000_W10.root:treeMaker/tree")
    #outFileName = "jetImage_lepTop_ZprimeToTT_M1000_W10.pdf"
    #cut = "(fatJet_pT_reco > 200) & (fatJet_eta_reco < 2.4) & (fatJet_nConsti_reco >= 3) & (fatJet_nearestGenTopDR_reco < 1) & (fatJet_nearestGenTopIsLeptonic_reco > 0.5)"
    
    tree = uproot.open("/home/soham/naf_dust/work/TopTagPol/TreeMaker/CMSSW_10_5_0/src/ntupleTree_QCD_Pt_470to600.root:treeMaker/tree")
    outFileName = "jetImage_QCD_Pt_470to600.pdf"
    cut = "(fatJet_pT_reco > 200) & (fatJet_eta_reco < 2.4) & (fatJet_nConsti_reco >= 3) & (fatJet_nearestGenTopDR_reco > 1)"
    
    tree.show()
    
    l_branchName = [
        #"genTop_isLeptonic",
        #
        #"fatJet_pT_reco",
        #"fatJet_eta_reco",
        #"fatJet_m_reco",
        #"fatJet_nearestGenTopIdx_reco",
        #"fatJet_nearestGenTopDR_reco",
        #"fatJet_nearestGenTopIsLeptonic_reco",
        #"fatJet_nConsti_reco",
        "fatJet_constiTrans_x_reco",
        "fatJet_constiTrans_y_reco",
        "fatJet_constiTrans_w_reco",
    ]
    
    tree_branches = tree.arrays(
        expressions = l_branchName,
        cut = cut,
    )
    
    #print(tree_branches.tolist())
    
    #print(len(tree_branches["fatJet_pT_reco"]))
    #print(len(tree_branches["fatJet_constiTrans_x_reco"]))
    #print(tree_branches["fatJet_constiTrans_x_reco"][0])
    #print(tree_branches["fatJet_constiTrans_x_reco"][1])
    
    
    #h1_fatJet_m = ROOT.TH1F("h1_fatJet_m", "h1_fatJet_m", 200, 0, 1000)
    
    h2_fatJet_img = ROOT.TH2F("h2_fatJet_img", "h2_fatJet_img", 50, -1.0, 1.0, 50, -1.0, 1.0)
    
    
    nEvent = len(tree_branches[l_branchName[0]])
    #nEvent = 50
    
    nJet_selected = 0
    
    for iEvent in range(0, nEvent) :
        
        nJet = len(tree_branches["fatJet_constiTrans_x_reco"][iEvent])
        
        for iJet in range(0, nJet) :
            
            #pt = tree_branches["fatJet_pT_reco"][iEvent][iJet]
            #eta = tree_branches["fatJet_eta_reco"][iEvent][iJet]
            #m = tree_branches["fatJet_m_reco"][iEvent][iJet]
            #
            #if (pt < 200 or abs(eta) > 2.4) :
            #    
            #    continue
            #
            #if (isSig) :
            #    
            #    genTop_idx = int(tree_branches["fatJet_nearestGenTopIdx_reco"][iEvent][iJet])
            #    
            #    if (genTop_idx < 0) :
            #        
            #        continue
            #    
            #    #print(genTop_idx)
            #    
            #    #isLeptonic = tree_branches["genTop_isLeptonic"][iEvent][genTop_idx] > 0.5
            #    isLeptonic = tree_branches["fatJet_nearestGenTopIsLeptonic_reco"][iEvent][iJet] > 0.5
            #    
            #    if (not isLeptonic) :
            #        
            #        continue
            
            a_x = tree_branches["fatJet_constiTrans_x_reco"][iEvent][iJet].to_numpy()
            a_y = tree_branches["fatJet_constiTrans_y_reco"][iEvent][iJet].to_numpy()
            a_w = tree_branches["fatJet_constiTrans_w_reco"][iEvent][iJet].to_numpy()
            
            nConsti = len(a_x)
            
            #if (nConsti < 3) :
            #    
            #    continue
            
            #print(
            #    "[%d/%d] "
            #    "jet %d/%d: "
            #    "pT %0.2f, "
            #    "m %0.2f, "
            #    "nConsti %d (%d), "
            #    "genDR %0.2f, "
            #    "isLep %d, "
            #%(
            #    iEvent+1, nEvent,
            #    iJet+1, nJet,
            #    tree_branches["fatJet_pT_reco"][iEvent][iJet],
            #    tree_branches["fatJet_m_reco"][iEvent][iJet],
            #    nConsti,
            #    tree_branches["fatJet_nConsti_reco"][iEvent][iJet],
            #    tree_branches["fatJet_nearestGenTopDR_reco"][iEvent][iJet],
            #    isLeptonic,
            #))
            
            
            #h1_fatJet_m.Fill(m)
            
            h2_fatJet_img.FillN(nConsti, a_x, a_y, a_w)
            
            nJet_selected += 1
    
    
    print("Selected jets: %d" %(nJet_selected))
    
    h2_fatJet_img.Scale(1.0 / nJet_selected)
    
    h2_fatJet_img.SetMinimum(1e-6)
    h2_fatJet_img.SetMaximum(1)
    
    
    ROOT.gStyle.SetPalette(ROOT.kVisibleSpectrum)
    ROOT.gStyle.SetNumberContours(50)
    
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)
    
    
    canvas = ROOT.TCanvas("canvas", "canvas")
    canvas.SetCanvasSize(800, 600)
    
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.21)
    canvas.SetBottomMargin(0.15)
    canvas.SetTopMargin(0.1)
    
    #h1_fatJet_m.Draw("hist")
    
    h2_fatJet_img.Draw("colz")
    
    canvas.SetLogz(True)
    
    canvas.SaveAs(outFileName)
    
    #time.sleep(100000)



if (__name__ == "__main__") :
    
    main()
