import argparse
import os
import subprocess


def main() :
    
    # Argument parser
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    # List of directories
    parser.add_argument(
        "--samplenames",
        help = "List of samples",
        nargs = "*",
        type = str,
        required = True
    )
    
    parser.add_argument(
        "--samplebasepath",
        help = "Base path for samples",
        type = str,
        required = False,
        default = "/pnfs/desy.de/cms/tier2/store/user/sobhatta/TopTagPol/ntuples",
    )
    
    parser.add_argument(
        "--outdir",
        help = "Output directory",
        type = str,
        required = False,
        default = "sourceFiles",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    
    for iSample, sample in enumerate(args.samplenames) :
        
        sample = sample.strip()
        
        if (not len(sample) or "#" in sample) :
            
            continue
        
        print("\n")
        print("Sample: %s" %(sample))
        
        samplepath = "%s/%s" %(args.samplebasepath, sample)
        
        datedir = subprocess.check_output("ls %s | sort -V | tail -n 1" %(samplepath), shell = True).decode("utf-8").strip()
        
        outfilename = "%s/%s_%s.txt" %(args.outdir, sample, datedir)
        
        # Will get the latest one when the directory has the ending tag YYYY-MM-DD_hh-mm-ss
        cmd = "find `find {samplepath} -mindepth 1 -maxdepth 1 | sort -V | tail -n 1` -mindepth 1 | sort -V > {outfilename}".format(
            samplepath = samplepath,
            outfilename = outfilename,
        )
        
        print("Command: %s" %(cmd))
        
        os.system(cmd)
        os.system("wc -l %s" %(outfilename))


if (__name__ == "__main__") :
    
    main()
