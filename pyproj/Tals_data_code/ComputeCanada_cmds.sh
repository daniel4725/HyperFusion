ssh duenias@narval3.computecanada.ca
cd /home/duenias/projects/def-arbeltal/duenias
rm -r pyproj

ssh danieldu@cygnusx.cim.mcgill.ca
cd /usr/local/data/danieldu

# from yoshi
rsync -az --exclude='lightning_logs' --exclude='.ipynb_checkpoints' --exclude='.vscode' --exclude='archive' --exclude='old_DeepVisionCode' --exclude='pvg-pipeline' --exclude='venv' --exclude='wandb' --exclude='wandb_logs' --exclude='wandb_sweeps' danieldu@yoshi.cim.mcgill.ca:/usr/local/faststorage/danieldu/pyproj .
rsync -az  danieldu@yoshi.cim.mcgill.ca:/usr/local/faststorage/danieldu/pyproj/metadata_by_features_sets .

# from cygnusx
rsync -az --exclude='lightning_logs' --exclude='.ipynb_checkpoints' --exclude='.vscode' --exclude='archive' --exclude='old_DeepVisionCode' --exclude='pvg-pipeline' --exclude='venv' --exclude='wandb' --exclude='wandb_logs' --exclude='wandb_sweeps' danieldu@cygnusx.cim.mcgill.ca:/usr/local/data/danieldu/pyproj .
rsync -az danieldu@cygnusx.cim.mcgill.ca:/usr/local/data/danieldu/metadata_by_features_sets .

# from narval 
rsync -az duenias@narval3.computecanada.ca:/home/duenias/projects/def-arbeltal/duenias .



sbatch exp.sh

# You can look at the log files that are generated in the directory you launched the job in.
# Use sq to check the status of the job.

wandb sync --project HyperNets_imgNtabular --sync-all /home/duenias/projects/def-arbeltal/duenias/pyproj/wandb_logs/wandb

# interactive session
salloc --time=1:0:0 --cpus-per-task=8 --nodes=1 --mem=32G --account=rrg-arbeltal
scancel 11677723

pip install numpy --no-index
pip install torch --no-index
pip install tqdm --no-index
pip install torchvision --no-index
pip install matplotlib --no-index
pip install pandas --no-index
pip install lz4 --no-index
pip install pytorch_lightning --no-index
pip install sklearn --no-index
pip install monai --no-index
pip install wandb --no-index

pip install torchvision==0.11.1
pip install torch==1.10.0

# copy torch package to compute canada
rsync -az danieldu@yoshi:/usr/local/faststorage/danieldu/pyproj/venv/lib/python3.6/site-packages/torch .
rsync -az danieldu@yoshi:/usr/local/faststorage/danieldu/pyproj/venv/lib/python3.6/site-packages/torch-1.10.1.dist-info .
rsync -az --progress danieldu@cygnusx.cim.mcgill.ca:/usr/local/data/danieldu/torch .
rsync -az --progress danieldu@cygnusx.cim.mcgill.ca:/usr/local/data/danieldu/torch-1.10.1.dist-info .

