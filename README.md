How to run the training:
1. Get interactive condor session: `condor_submit -i scripts/gpu_interactive.submit`. This can take some time.
2. Inititialize conda environment and cd to the training directory.
3. `mkdir logs`.
4. `nohup python -u python/train.py --config configs/test_config.yml > logs/train.log &`.

The model will be saved in `training_results/model_checkpoints` with the date and time.
The tensorboard info will be saved in `training_results/tensorboard` with the date and time.

To view the tensorboard page:
1. On another shell, login to NAF, cd to the training directory and run one of the following:
    * `tensorboard --logdir training_results/tensorboard/<dir>`, where `<dir>` is any particular training under `training_results/tensorboard`.
    * `tensorboard --logdir `find training_results/tensorboard/ -maxdepth 1 | sort -V | tail -n 1`; will load the latest training.
2. On your machine run: `ssh -L 6006:127.0.0.1:6006 <username>@<host>.desy.de`. N.B. Use the same host as in step 1. For example, if you log into `naf-cms20.desy.de`, use the same as host here.
3. From the browser on your machine, open: `http://127.0.0.1:6006`
