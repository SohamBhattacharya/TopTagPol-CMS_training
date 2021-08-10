import argparse
import datetime
import numpy
import os
import subprocess


cwd = os.getcwd()
#cwd = "%s/src" %(subprocess.check_output(["echo", "$CMSSW_BASE"]).strip())
#proxy = subprocess.check_output(["voms-proxy-info", "--path"]).strip()


# Argument parser
parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)

parser.add_argument(
    "--tag",
    help = "Will append \"_<tag>_<datetime>\" to the output directories",
    type = str,
    required = True,
)

parser.add_argument(
    "--trainconfig",
    help = "Training configuration file",
    type = str,
    required = True,
)

parser.add_argument(
    "--condordir",
    help = "Condor base directory (will create the job directory here)",
    type = str,
    required = False,
    default = "training_results/condor_jobs"
)

parser.add_argument(
    "--condorconfig",
    help = "Configuration template",
    type = str,
    required = False,
    default = "scripts/condor_config_template.submit"
)

parser.add_argument(
    "--condorscript",
    help = "Script (executable) template",
    type = str,
    required = False,
    default = "scripts/condor_script_template.sh"
)


if (__name__ == "__main__") :
    
    # Parse arguments
    args = parser.parse_args()
    
    trainconfig_name = args.trainconfig.split("/")[-1]
    
    condorconfig_name = args.condorconfig.split("/")[-1]
    condorscript_name = args.condorscript.split("/")[-1]
    
    args.tag = args.tag.strip()
    
    if (not len(args.tag)) :
        
        print("Error: \"tag\" cannot be empty string")
        exit(1)
    
    out_tag = "%s_%s" %(args.tag, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    dirname = "%s" %(out_tag)
    
    condorDir = "%s/%s" %(args.condordir, dirname)
    os.system("mkdir -p %s" %(condorDir))
    
    trainconfig = "%s/%s" %(condorDir, trainconfig_name)
    condorconfig = "%s/%s" %(condorDir, condorconfig_name)
    condorscript = "%s/%s" %(condorDir, condorscript_name)
    
    os.system("cp %s %s" %(args.trainconfig, trainconfig))
    os.system("cp %s %s" %(args.condorconfig, condorconfig))
    os.system("cp %s %s" %(args.condorscript, condorscript))
    
    condor_log = "%s/job.log" %(condorDir)
    condor_out = "%s/job.out" %(condorDir)
    condor_err = "%s/job.err" %(condorDir)
    
    d_format = {
        "@dir@": cwd,
        "@tag@": out_tag,
        "@cfg@": trainconfig,
        
        "@exe@": condorscript,
        "@log@": condor_log,
        "@out@": condor_out,
        "@err@": condor_err,
    }
    
    def format_file(filename, d) :
        
        for key in d :
            
            val = d[key]
            
            cmd = "sed -i \"s#{find}#{repl}#g\" {filename}".format(
                find = key,
                repl = val,
                filename = filename,
            )
            
            os.system(cmd)
    
    format_file(condorconfig, d_format)
    format_file(condorscript, d_format)
    
    print("")
    print("Created directory: %s" %(condorDir))
    
    print("Log file:          %s" %(condor_log))
    print("Out file:          %s" %(condor_out))
    print("Err file:          %s" %(condor_err))
    
    print("***** VERIFY THE FOLLOWING FILES ARE OKAY BEFORE SUBMITTING *****")
    print("Training config:   %s" %(trainconfig))
    print("Submit file:       %s" %(condorconfig))
    print("Script file:       %s" %(condorscript))
    
    print("")
    
