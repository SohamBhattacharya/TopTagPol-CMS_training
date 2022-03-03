import argparse
import datetime
import numpy
import os
import subprocess

import utils


def main() :
    
    cwd = os.getcwd()
    
    # Argument parser
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "--config",
        help = "Configuration file",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--force",
        help = "Will rename existing output/condor directory with its timestamp",
        default = False,
        action = "store_true",
    )
    
    parser.add_argument(
        "--submit",
        help = "Submit the jobs",
        default = False,
        action = "store_true",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    d_config = utils.load_config(args.config)
    
    config_name = args.config.split("/")[-1]
    condorconfig_name = d_config["condorconfig"].split("/")[-1]
    condorscript_name = d_config["condorscript"].split("/")[-1]
    
    nSample = len(d_config["samples"])
    
    nJob_total = 0
    d_jobSummary = {}
    
    for iSample, sampleSource in enumerate(d_config["samples"]) :
        
        fileAndTreeNames = utils.get_fileAndTreeNames([sampleSource])
        nFile = len(fileAndTreeNames)
        
        sourceFile = sampleSource.strip().split(":")[0]
        sampleName = sourceFile.split("/")[-1].split(".txt")[0]
        
        outputTag = d_config["outputTag"].strip()
        outputTag = "_%s"%(outputTag) if len(outputTag) else ""
        
        outDir = "%s/%s/%s%s" %(d_config["outputDirBase"], d_config["modelName"], sampleName, outputTag)
        condordir = "%s/%s/%s%s" %(d_config["condordir"], d_config["modelName"], sampleName, outputTag)
        
        nFile = len(fileAndTreeNames)
        
        print("")
        print("*"*100)
        print("*"*100)
        print("Sample %d/%d:", sampleName, iSample+1, nSample)
        print("Sample file list:", sampleSource)
        print("Output directory:", outDir)
        print("Condor directory:", condordir)
        print("# units:", nFile)
        print("# jobs:", nFile)
        #print("# units per job:", args.nUnitPerJob)
        print("*"*100)
        print("*"*100)
        print("")
        
        
        d_jobSummary[sampleSource] = {
            "skipped": False,
            "nJob": nFile,
            "outDir": outDir,
            "condordir": condordir,
        }
        
        
        if (os.path.exists(outDir) or os.path.exists(condordir)) :
            
            print("~"*10, "Paths already exist:")
            if (os.path.exists(outDir)) : print(outDir)
            if (os.path.exists(condordir)) : print(condordir)
            
            if (args.force) :
                
                print("~"*10, "Renaming. New paths:")
                
                if (os.path.exists(outDir)) :
                    
                    newname = utils.get_name_withtimestamp(outDir)
                    print(newname)
                    os.system("mv %s %s" %(outDir, newname))
                
                if (os.path.exists(condordir)) :
                    
                    newname = utils.get_name_withtimestamp(condordir)
                    print(newname)
                    os.system("mv %s %s" %(condordir, newname))
            
            else :
                
                print("~"*10)
                print("~"*10, "Skipping sample", "~"*10)
                print("~"*10)
                
                d_jobSummary[sampleSource]["skipped"] = True
                
                continue
        
        os.system("mkdir -p %s" %(outDir))
        os.system("mkdir -p %s" %(condordir))
        
        config = "%s/%s" %(condordir, config_name)
        os.system("cp %s %s" %(args.config, config))
        
        
        for iFile, fileAndTreeName in enumerate(fileAndTreeNames) :
            
            print("")
            
            l_cmd = []
            
            verbosetag = "[sample %d/%d] [unit %d/%d]" %(iSample+1, nSample, iFile+1, nFile)
            
            inFileName = fileAndTreeName.strip().split(":")[0].split("/")[-1]
            
            outFileName = "%s/%s" %(outDir, inFileName)
            
            script_cmd = (
                "python -u {treemaker} "
                "--config {config} "
                "--inFileNames {fileAndTreeName} "
                "--outFileName {outFileName} "
            ).format(
                treemaker = d_config["treemaker"],
                config = args.config,
                fileAndTreeName = fileAndTreeName,
                outFileName = outFileName,
            )
            
            
            condorconfig = "%s/%s" %(condordir, condorconfig_name)
            condorscript = "%s/%s" %(condordir, condorscript_name)
            
            condorconfig = condorconfig[0: condorconfig.rfind(".")] + "_%d" %(iFile+1) + condorconfig[condorconfig.rfind("."):]
            condorscript = condorscript[0: condorscript.rfind(".")] + "_%d" %(iFile+1) + condorscript[condorscript.rfind("."):]
            
            l_cmd.extend([
                "cp %s %s" %(d_config["condorconfig"], condorconfig),
                "cp %s %s" %(d_config["condorscript"], condorscript),
                
                "chmod +x %s" %(condorscript),
            ])
            
            
            condor_log = "%s/job_%d.log" %(condordir, iFile+1)
            condor_out = "%s/job_%d.out" %(condordir, iFile+1)
            condor_err = "%s/job_%d.err" %(condordir, iFile+1)
            
            print("%s condor config: %s" %(verbosetag, condorconfig))
            print("%s condor script: %s" %(verbosetag, condorscript))
            print("%s condor log   : %s" %(verbosetag, condor_log))
            print("%s condor out   : %s" %(verbosetag, condor_out))
            print("%s condor err   : %s" %(verbosetag, condor_err))
            
            d_format = {
                "@dir@": cwd,
                "@exe@": condorscript,
                "@log@": condor_log,
                "@out@": condor_out,
                "@err@": condor_err,
                "@cmd@": script_cmd,
            }
            
            l_cmd.extend(utils.format_file(condorconfig, d_format))
            l_cmd.extend(utils.format_file(condorscript, d_format))
            
            utils.run_cmd_list(l_cmd)
            
            nJob_total += 1
            
            submit_cmd = "condor_submit %s" %(condorconfig)
            commandReturn = 1
            
            if (args.submit) :
                
                # Repeat until job is submission is successful (returns 0)
                while (commandReturn) :
                    
                    commandReturn = os.system(submit_cmd)
                    
    
    
    print("\n")
    print("="*100)
    print("Summary")
    print("="*100)
    print("")
    
    print("Total jobs: %d" %(nJob_total))
    print("")
    
    print("Skipped samples:")
    
    hasSkipped = False
    
    for iSample, sampleSource in enumerate(d_config["samples"]) :
        
        if (not d_jobSummary[sampleSource]["skipped"]) :
            
            continue
        
        hasSkipped = True
        
        print("")
        print("Sample: %s" %(sampleSource))
        print("Paths already exist:")
        if (os.path.exists(outDir)) : print(outDir)
        if (os.path.exists(condordir)) : print(condordir)
    
    if (not hasSkipped) :
        
        print("None")
    
    return 0

if (__name__ == "__main__") :
    
    main()
