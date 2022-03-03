# Training configuration file details
* Located under `configs`.
* Rule of thumb for the option `loadEventPerJob`: use 1/10th of the max jets per sample.

# Run/debug the training over a small dataset
1. Get interactive condor session: `condor_submit -i scripts/gpu_interactive.submit`. This can take some time.
2. Inititialize the conda environment and cd to the training directory.
3. `mkdir -p logs`.
4. `nohup python -u python/train.py --config configs/tensorflow_training/test_config.yml > logs/train.log &`.

The model will be saved in `training_results/model_checkpoints` with the date and time.
The tensorboard info will be saved in `training_results/tensorboard` with the date and time.

# Submit a condor job for training over a dataset
1. Initialize the conda environment and cd to the training directory. No need to get an interactive session in this case.
2. `python -u python/run_condor.py --tag <tag> --trainconfig <trainconfig> --condorconfig <condorconfig>`.
    * Use a meaningful name for `<tag>` (no spaces or special characters except `-` and `_`). This is just a label to help identify the training.
    * `<trainconfig>`: the training config file you want. This will be copied. The copied file can be edited before submission.
    * `<condorconfig>`: \
    For a large dataset (about 100K or more per sample): `condor_config_highmem_template.submit`. \
    For a small dataset (about 10K or fewer per sample): `condor_config_template.submit`.
3. This will create and list a few files. **Check them (especially the training config file) before submitting**.
4. If things look okay, submit using: `condor_submit <submit file>`. \
    Here `<submit file>` is the same as that printed by the previous command.
5. Check the status of the condor job(s) with: `condor_q <username>`.
6. If you want to abort your job(s):
    * `condor_rm <jobid>` will abort a particular condor job.
    * `condor_rm <username>` will abort all your condor jobs.
7. Once the job starts to run, it will create `job.out` in the condor directory. To monitor it, do `tail -f <file>`.
8. Warnings and errors will be in `job.err`.
9. Once the training starts, the model checkpoint and tensorboard directories will be created, and the latter can be viewed on a browser, even when the training is ongoing.

# View the tensorboard page
1. On another terminal, log into NAF with: \
    `ssh -L 6006:127.0.0.1:6006 <username>@naf-cms.desy.de`.
2. `cd` to the training directory and run one of the following:
    * View a particular training: \
    `tensorboard --logdir <dir>`. \
    Here `<dir>` is any particular training under `training_results/tensorboard`.
    * View multiple trainings: \
    `tensorboard --port 6006 --reload_multifile True --logdir_spec <name1>:<dir1>,<name2>:<dir2>, ...`. \
    That is, a comma-separated list of `<name>:<dir>`. You can enter any `<name>` for a particular `<dir>`.
    * View latest training (note, this will sort using the directory name): \
    `tensorboard --port 6006 --reload_multifile True --logdir $(find training_results/tensorboard/ -mindepth 1 -maxdepth 1 | sort -V | tail -n 1)`.
    * If you want the latest training of a particular `<tag>`:
    `tensorboard --port 6006 --reload_multifile True --logdir $(find training_results/tensorboard/ -mindepth 1 -maxdepth 1 | grep <tag> | sort -V | tail -n 1)`.
3. From the browser on your machine, open: `http://127.0.0.1:6006` or `http://localhost:6006`.

# Tensorboard tips
1. If port `6006` is already being used on the remote machine, you'll get an error when running tensorboard -- change the port to some other number in that case, or log into a different machine.
2. When viewing images, tensorboard will not always load the images for all epochs. In that case, the flag `--samples_per_plugin images=100` will load images for at most 100 epochs.

# Miscellaneous
* Install the terminator terminal: makes life significantly easier! \
https://github.com/gnome-terminator/terminator/blob/master/INSTALL.md
