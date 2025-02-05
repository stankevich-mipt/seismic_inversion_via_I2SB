<h1 align="center"> Acoustic Waveform Inversion based on Image-to-Image Schrödinger Bridges </h1>


A PyTorch-base implementation of acousic FWI pipeline based on Image-to-Image Schrödinger Bridges. 

## Setup

The only available option right now is the deployment based on Docker image with support of nvidia-docker

### Prerequisites  

1. Install Docker following the platform-specific instructions [https://docs.docker.com/engine/installation/](https://docs.docker.com/engine/installation/)
2. Install Nvidia drivers on your machine either from [Nvidia](http://www.nvidia.com/Download/index.aspx?lang=en-us) directly or follow the instructions [here](https://github.com/saiprashanths/dl-setup#nvidia-drivers). Note that you _don't_ have to install CUDA or cuDNN. These are included in the Docker container.
3. Install nvidia-docker replacement for the docker CLI: [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker), following the instructions [here](https://github.com/NVIDIA/nvidia-docker/wiki/Installation).

### Obtaining the Docker image
To build the image locally clone the repository and execute the following comand
```bash
cd docker
docker build -t stankevich-mipt/seismic_inversion_via_i2sb:latest -f Dockerfile .
```

## Running Docker containers 
The image built yields all the necessary dependencies to run the code in the repository. To run experiments, one should create a container using the current image. 

```bash
docker run -d --gpus=all -p 'host_port':'container_port' -v 'host_data_volume':'container_data_volume' -v 'host_output_volume':'container_output_volume' --name 'container_name' stankevich-mipt/seismic_inversion_via_i2sb:latest
```

| Parameter      | Explanation |
|----------------|-------------|
|`-d`             | Launches the container in detached mode |
|`-p 'host_port':'container-post`    | Exposes the ports inside the container so they can be accessed from the host. The default iPython Notebook runs on port 8888 and Tensorboard on 6006|
|`-v 'host_data_volume':'container_data_volume'` | Attaches the volume `host_data_volume` on your host machine to `container_data_volume` inside your container. Any data written to the folder by the container is persistent. Multiple volumes can be assigned to a single container by the means of passing another `-v` flag|
|`--name 'container_name'`| Specifies the name of newly created container |
|`stankevich-mipt/seismic_inversion_via_i2sb:latest` | The image that is used to create a container. SHA-256 hash could be specified instead of image name and tag|

To attach to a running container with interactive terminal, use
```bash
docker exec -it 'container-id' bash 
```
where `container-id` should be replaced either with the name or the SHA256 ID of the given container.  

## Data

We train and evaluate our models on [OpenFWI](https://smileunc.github.io/projects/openfwi) dataset collection. The file `dataset/config/openfwi_dataset_config.json`, borrowed from the [OpenFWI repo](https://github.com/lanl/OpenFWI), contains metadata, nesessary for preprocessing. 


LMDB database conversion is kept as a feature inherited from the [original **I<sup>2</sup>SB** project](https://github.com/NVlabs/I2SB) and utilized to cache static data preprocessing steps (e.g, normalization and reshaping). For LMDB conversion script to work correctly, the dataset folder has to have following file structure 

```bash
$DATA_DIR/                  # dataset directory
├── train/
    ├── model/              # folder with training velocity models as .npy files   
        └── model*.npy                
    └── data/               # folder with training seismograms as .npy files
        └── data*.npy       
└── val
    ├── model/              # folder with validation velocity models as .npy files 
        └── model*.npy      
    └── data/               # folder with validation seismograms as .npy files
        └── data*.npy       
```
LMDB database is built on the first call of the script. Tensors in database are normalized, reduced to the same shape, and stored as **pickled python dictionaries**. Concatenations of absolute paths to samples serve as sample retrieval keys. The former are saved separately in pickled dataset class instance. 

The expected `$DATA_DIR` structure with LMDB cache is as follows  
```bash
$DATA_DIR/                           # dataset directory
├──  train/
    ├── model/
    ├── data/
    ├── database.lmdb                # normalized training samples in LMDB format                           
    └── database.lmdb.pt             # pickled training dataset instance 
├── val
    ├── model/ 
    ├── data/
    ├── database.lmdb                # normalized val samples in LMDB format                   
    └── database.lmdb.pt             # pickled val dataset instance
```

## Training

To train a baseline model instance on a single node with minimal customization, execute the following command line in terminal attached to the running Docker container
```bash
python /home/seismic_inversion_via_I2SB/train.py --result-dir 'container_output_volume' --dataset-dir 'container_data_volume' --dataset-name $DATASET_NAME --name $EXP_NAME
```
### All launch options

#### General 
| Parameter      | Explanation |
|----------------|-------------|
|`--result-dir`  | (Required, Path) Directory for output files. Specify the directory attached to container as persistent volume `container_output_volume` to save results at host |
|`--dataset-dir` | (Required, Path) Directory containing OpenFWI dataset in the format listed above. Specify the directory attached to container as persistent volume `container_output_volume` to get the data from host and save LMDB cache at the same volume |
|`--dataset-name` | (Required, Path) OpenFWI dataset name required to fetch metadata for preprocessing. Possible values  <ul> <li> FlatVel_A </li> <li> FlatVel_B </li> <li> CurveVel_A </li> <li> CurveVel_B </li> <li> FlatFault_A </li> <li> FlatFault_B </li> <li> CurveFault_A </li> <li> CurveFault_B </li> <li> Style_A </li> <li> Style_B </li> </ul> |
|`--name` | (Optional, srt, default=str(seed)) Experiment ID|
|`--seed` | (Optional, int, default=0) Random Seed|
|`--master-port`| (Optional, int, default=6020) Master port for process group initialized with torch.DDP module. Specify this if multiple script instances have to be launched within the same container |
|`--ckpt` |(Optional, Path, default=None) Relative path to checkpoint weights within the output directory. If specified, training will resume from the checkpoint|
|`--gpu`  |(Optional, int, default=None) Specify to run on a particular device|

#### Model-related
| Parameter      | Explanation |
|----------------|-------------|
|`--model`|(Optional, str, default="unet_ch32") Model architecture. Possible values <ul>  <li> "unet_ch32" </li> <li> "unet_ch64" </li> </ul>|
|`--image-size`|(Optional, int, default=256) Height and width of model inputs and outputs. Data samples are resized once at the stage of lmdb database creation|
|`--corrupt`|(Optional, str, default="blur-openfwi_custom") Restoration task. Possible values <ul>  <li> "blur-openfwi_custom" - gaussian smoothing with variable kernel size combined with the addition of zero-centered normal noise </li> <li> "blur-openfwi_benchmark" - gaussian smoothing with kernel_size=9 (ref. https://arxiv.org/pdf/2410.21776)  </li> </ul>|
|`--val-batches`|(Optional, int, default=100) Upper bound for the amount of batches to run validation on|
|`--t0`|(Optional, float, default=0.0001) Initial time in network parameterization|
|`--T` |(Optional, float, default=1.)   Final time in network parameterization  |
|`--interval`|(Optional, type=int, default=1000) Maximum amount of discrete timesteps to divide the time interval into|
|`--beta-max`|(Optional, type=float, default=0.3) Square root of the standard noise deviation at the end of time interval during DDPM sampling|
|`--drop_cond`|(Optional, type=float, default=0.25) Probability to replace conditional input with zero-valued tensor. Inspired by https://arxiv.org/pdf/2207.12598|
|`--pred-x0`|(Optional) If set, replaces the original objective with the one described in https://arxiv.org/pdf/2206.00364|
|`--ot-ode`|(Optional) If set, uses the ODE model instead of SDE one, _i.e.,_ the limit when the diffusion vanishes.  Hence, sampling becomes deterministic|
|`--clip-denoise`| (Optional) If set, clip predicted image to [-1, 1] value range at each sampling iteration|

#### Optimization-related
| Parameter      | Explanation |
|----------------|-------------|
|`--batch-size`| (Optional, type=int, default=256)|
|`--microbatch`| (Optional, type=int, default=2) Accumulate gradient over batch shards, processing only `microbatch` samples at a time. Useful for memory conservation when total batch size does not fit the GPU memory|
|`--num-itr`|(Optional, type=int, default=1000000) Total amount of batches to process during training run|
|`--lr`|(Optional, type=float, default=5e-5) Initial learning rate for AdamW optimizer|
|`--lr-gamma`|(Optional, type=float, default=0.99) Learning rate decay ratio|
|`--clip_grad_norm`|(Optional, type=float, default=None) Clip gradient value to a set value. Stabilizes the training procedure, making it more resilient to large-scale gradients caused by outliers in data|
|`--ema`|(Optional, type=float, default=0.99) Exponential moving average rate for network parameters through time. Instance of EMA parameters is saved independenty in parralel with the runtime checkpoint.|

#### Metadata and Logging

| Parameter      | Explanation |
|----------------|-------------|
|`--log-writer`|(Optional, type=str, default=None) Specify the logger backend. At the moment only tensorbard logging is supported|
|`--json-data-config`| (Optional, type=Path) Full path to JSON config for OpenFWI. Used for data normalization. Default path is `/home/seismic_inversion_via_I2SB/dataset/config/`|


## Multi-Node Training

For multi-node training, we recommand the MPI backend.
```bash
mpirun --allow-run-as-root -np $ARRAY_SIZE -npernode 1 bash -c \
    'python train.py $TRAIN \
    --num-proc-node $ARRAY_SIZE \
    --node-rank $NODE_RANK \
    --master-address $IP_ADDR '
```
where `TRAIN` wraps all the the [original (single-node) training options](https://github.com/NVlabs/I2SB#training), `ARRAY_SIZE` is the number of nodes, `NODE_RANK` is the index of each node among all the nodes that are running the job, and `IP_ADDR` is the IP address of the machine that will host the process with rank 0 during training; see [here](https://pytorch.org/tutorials/intermediate/dist_tuto.html#initialization-methods).


## Citation

```
@article{liu2023i2sb,
  title={I{$^2$}SB: Image-to-Image Schr{\"o}dinger Bridge},
  author={Liu, Guan-Horng and Vahdat, Arash and Huang, De-An and Theodorou, Evangelos A and Nie, Weili and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2302.05872},
  year={2023},
}
```

## License
Copyright © 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC.

The model checkpoints are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).