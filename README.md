<h1 align="center"> Acoustic Waveform Inversion based on Image-to-Image Schrödinger Bridges </h1>

Current reposiory contains official implementation of acousic FWI pipeline based on Image-to-Image Schrödinger Bridges written in PyTorch.

<p align="center">
    <img src="assets\i2sb_training.jpg" alt="Conditional I2SB Training" width="500"/>
</p>

<p align="center">
    <img src="assets\i2sb_inference.jpg" alt="Conditional I2SB Inference" width="500"/>
</p>



## Description

### Preface

Consider the continuous medium governed with the 2D acoustic wave equation
```math
\bigtriangleup \mathbf{p} - \dfrac{1}{\mathbf{c}^2} \dfrac{\partial^2\mathbf{p}}{\partial t^2} = \mathbf{s}
```
where $\bigtriangleup$ is a 2D Laplace operator, $\mathbf{p} (x, y, t)$ is the 
pressure field, $\mathbf{s}(x, y, t)$ is the source function, an $\mathbf{c}(x, y)$ is the velocity field. 
Regularized Full Waveform Inversion problem statement for such equation is formulated as an optimization task of the form
```math
    \min\limits_{\mathbf{c}} \, J(\mathbf{d}_{\mathbf{model}},  \mathbf{d}_{\text{obs}}) + \lambda R(\mathbf{c})
    \quad \text{s.t.} \quad \mathbf{d}_{\text{model}} = F(\mathbf{c}) 
```
where $J$ measures the discrepancy between modelled $`\mathbf{d}_{\text{model}}`$ 
and observed $`\mathbf{d}_{\text{obs}}`$ seismic data, $F$ is the forward modelling operator, 
and $\lambda R(\mathbf{c})$ is the regularization term that limits the capacity of model space.
To put it another way, the target of full waveform inversion is to estimate the value of ''pseudoinverse'' of 
$F$ applied to the observed seismic data $\mathbf{d}_{\text{obs}}$
```math
    \mathbf{c}^* = \hat{F}^{-1} (\mathbf{d}_{\text{obs}})
```

Supervised learning-based appoaches to acoustic waveform inversion seek $\hat{F}^{-1}_{\mathbf{\theta}}$ - 
a parametric approximation of $\hat{F}^{-1}$
The tuning of parameters $\mathbf{\theta}$ is carried 
out with gradient optimization using training dataset, which contains coupled instances of velocity models 
$\mathbf{c}_i^{\text{train}}$ and corresponding observed data $`\mathbf{d}_i^{\text{train}}`$.
Once the parameter fitting is done, $`\hat{F}^{-1}_{\mathbf{\theta}^*} \left( \mathbf{d}_i^{\text{test}} \right)`$ 
yields the reconstruced velocity model for seismogram $`\mathbf{d}_i^{\text{test}}`$ (illustration courtesy of [[1]](#1))

<p align="center">
    <img src="assets\data-driven-FWI.png" alt="Data Driven Acoustic FWI" width="500"/>
</p>

### Problem Statement and Proposed Solution

Consider the acoustic FWI problem statement coupled with additional information, given by smooth velocity model $\mathbf{c}_{\text{smooth}}$ of the medium under investigation. 
```math
    \begin{cases}
        \min\limits_{\mathbf{c}} \, J(\mathbf{d}_{\text{model}},  \mathbf{d}_{\text{obs}}) + \lambda R(\mathbf{c}) \\
        \text{s.t.} \quad \mathbf{d}_{\text{model}} = F(\mathbf{c}),\quad \mathbf{c}_1 = \mathbf{c}_{\text{smooth}}
    \end{cases}
```
$`\mathbf{c}_{\text{smooth}}`$ is believed to be reasonably close to ground truth velocity model $\mathbf{c}^*$,  yet lacking high-frequency details. Hence, in realistic inversion scenarios $\mathbf{c}_{\text{smooth}}$ is commmonly employed as a starting point for nonlinear optimization. 

We propose a novel way to utilize such piece of information in context of recently 
proposed diffusion-based deep learning approach to acoustic waveform inversion
[[2]](#2), [[3]](#3), [[4]](#4).
Specifically, we calculate 
$`\mathbf{c}^* = \hat{F}^{-1}_{\mathbf{\theta}^*} (\mathbf{d}^{\text{obs}}, \mathbf{c}_{\mathbf{smooth}})`$ 
by running inference process of conditional I$`^2`$SB [[5]](#5) model 
under hypothesis of $`\mathbf{c}_{\text{smooth}} \sim p_\text{prior} \left(\cdot | \mathbf{c}^* \right)`$ (algorithm 2).
To make our model more flexible, we augment the training procedure of I$`^2`$SB with classifier-free diffusion guidance [[6]](#6) (algorithm 1)
<p align="center">
    <img src="assets\i2sb_training_algo.png" alt="Conditional I2SB Training" width="500"/>
</p>
<p align="center">
    <img src="assets\i2sb_sampling_algo.png" alt="Conditional I2SB Training" width="500"/>
</p>


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

## Evaluation and sampling
To estimate performance metrics proposed in https://arxiv.org/pdf/2111.02926 for checkpoint `ckpt.pt` recorded during `$EXPERIMENT` run given the output folder `$RESULT_DIR`, use
```bash
python /home/seismic_inversion_via_I2SB/evaluation/record_openfwi_metrics.py --name $EXPERIMENT_NAME --result-dir $RESULT_DIR --ckpt ckpt.pt
```
By default dataset file paths and metadata are fetched from `$RESULT_DIR/checkpoints/options.pkl`. Evaluation is run on the val split of the dataset used for training. Such behaviour could be customized with providing different `--dataset-name` and `--dataset-dir` options.  

Compared to the `compute_metrices.py` script provided in the original repo, `record_openfwi_metrics.py` does not utilize external .pt files with pre-sampled batches of data. 

To save several data batches from the same checkpoint, enter the following command
```bash
python /home/seismic_inversion_via_I2SB/sample.py --name $EXPERIMENT_NAME --result-dir $RESULT_DIR --ckpt ckpt.pt
```
After the execution is finished, `$RESULT_DIR/$EXPERIMENT_NAME/samples` will contain .pt files with batched model inputs and outputs. 

### All launch options

`sample.py` and `record_openfwi_metrics.py` share the same set of launch options provided below

### General 
| Parameter      | Explanation |
|----------------|-------------|
|`--result-dir`  | (Required, Path) Directory for output files generated with training script. |
|`--name`        | (Required, str) Experiment ID. Model checkpoints are fetched from `result-dir/name`|
|`--ckpt` |(Required, Path) Checkpoint instance within `result-dir/name` directory to sample from|
|`--corrupt`     | (Optional, str, default="blur_openfwi-baseline") Image restoration task. Values supported at the moment are <ul> <li> "blur_openfwi-baseline" </li> <li> "blur_openfwi-dist_shift" </li> </ul>|
|`--dataset-dir` | (optional, Path, default=None) Directory containing OpenFWI dataset. If not provided directly, this argument will be loaded from saved training options. |
|`--dataset-name` | (optional, Path, default=None) OpenFWI dataset name required to fetch metadata for preprocessing. If not provided directly, this argument will be loaded from saved training options. Possible values  <ul> <li> FlatVel_A </li> <li> FlatVel_B </li> <li> CurveVel_A </li> <li> CurveVel_B </li> <li> FlatFault_A </li> <li> FlatFault_B </li> <li> CurveFault_A </li> <li> CurveFault_B </li> <li> Style_A </li> <li> Style_B </li> </ul> |
|`--seed` | (Optional, int, default=0) Random Seed |
|`--partition` | (Optional, str, default=None) Separate evaluation data into multiple chunks . E.g. '0_4' means the first 25% of the dataset is used|
|`--master-port`| (Optional, int, default=6020) Intercom port for process group initialized with torch.DDP module. Specify this if multiple script instances have to be launched within the same container |

### Sampling-related
| Parameter      | Explanation |
|----------------|-------------|
|`--batch_size`|(Optional, int, default=32) Evaluation batch size |
|`--nfe`|(Optional, int, default=None) Number of neural network calls to get a single sample batch. If not provided directly, the argument '--interval' from training options will be used.|
|`--clip-denoise`|(Optional) If provided, clamp predicted image to [-1, 1] range at each sampling iteration|
|`--use-fp16`|(Optional) If provided, uses network weights with reduced floating point precision for greater sampling speed at the cost of accuracy|
|`--stochastic`|(Optional) Use stochastic sampling during inference. Mutually exclusive with `--deterministic`|
|`--deterministic`|(Optional) Use deterministic sampling during inference. Mutually exclusive with `--stochastic`|
|`--test-var-reduction`|(Optional) If set, register inference results for batches of smooth models obtained through application of degradation operator to the same reference model|
|`--guidance-scale`|(Optional, default=None) Employ linear combination of  unconditional and conditional starting point predictions with weights `guidance-scale` and 1 - `guidance-scale` respectively| 

## Training

To train a baseline model instance on a single node with minimal customization, execute the following command line in terminal attached to the running Docker container
```bash
python /home/seismic_inversion_via_I2SB/train.py --result-dir 'container_output_volume' --dataset-dir 'container_data_volume' --dataset-name $DATASET_NAME --name $EXP_NAME
```
### All launch options

### General 
| Parameter      | Explanation |
|----------------|-------------|
|`--result-dir`  | (Required, Path) Directory for output files. Specify the directory attached to container as persistent volume `container_output_volume` to save results at host |
|`--dataset-dir` | (Required, Path) Directory containing OpenFWI dataset in the format listed above. Specify the directory attached to container as persistent volume `container_output_volume` to get the data from host and save LMDB cache at the same volume |
|`--dataset-name` | (Required, Path) OpenFWI dataset name required to fetch metadata for preprocessing. Possible values  <ul> <li> FlatVel_A </li> <li> FlatVel_B </li> <li> CurveVel_A </li> <li> CurveVel_B </li> <li> FlatFault_A </li> <li> FlatFault_B </li> <li> CurveFault_A </li> <li> CurveFault_B </li> <li> Style_A </li> <li> Style_B </li> </ul> |
|`--name` | (Optional, str, default=str(seed)) Experiment ID|
|`--seed` | (Optional, int, default=0) Random Seed|
|`--master-port`| (Optional, int, default=6020) Master port for process group initialized with torch.DDP module. Specify this if multiple script instances have to be launched within the same container |
|`--ckpt` |(Optional, Path, default=None) Relative path to checkpoint weights within the output directory. If specified, training will resume from the checkpoint|
|`--gpu`  |(Optional, int, default=None) Specify to run on a particular device|

### Model-related
| Parameter      | Explanation |
|----------------|-------------|
|`--model`|(Optional, str, default="i2sb_small_cond") Specify to select one of the model architectures implemented in scope of the paper. Possible values are <ul>  <li> "inversionnet_small" </li> <li> "inversionnet_small_cond" </li> <li> "inversionnet_large" </li> <li> "inversionnet_large_cond" </li> <li> "ddpm_small" </li> <li> "ddpm_small_cond" </li> <li> "ddpm_large" </li> <li> "ddpm_large_cond" </li> <li> "i2sb_small" </li> <li> "i2sb_small_cond" </li> <li> "i2sb_large" </li> <li> "i2sb_large_cond" </li> </ul>|
|`--image-size`|(Optional, int, default=64) Height and width of model inputs and outputs. Data samples are resized once at the stage of lmdb database creation|
|`--corrupt`|(Optional, str, default="blur-ci2sb_baseline") Distortion operator that determines the endpoint distributions. Possible values <ul>  <li> "blur-ci2sb_baseline" - gaussian smoothing with variable kernel size combined with the addition of zero-centered normal noise </li> <li> "uni" - uniform filter from the original I$`^2`$SB paper  </li> <li> "gauss" - gaussian filter from the original I$`^2`$SB paper  </li> </ul>|
|`--val-batches`|(Optional, int, default=100) Upper bound for the amount of batches to run validation on|
|`--t0`|(Optional, float, default=0.0001) Initial time in network parameterization|
|`--T` |(Optional, float, default=1.)   Final time in network parameterization|
|`--interval`|(Optional, type=int, default=1000) Maximum amount of discrete timesteps to divide the time interval into|
|`--beta-max`|(Optional, type=float, default=0.3) Square root of the standard noise deviation at the end of time interval during DDPM sampling|
|`--drop_cond`|(Optional, type=float, default=0.25) Probability to replace conditional input with zero-valued tensor. Inspired by https://arxiv.org/pdf/2207.12598|
|`--pred-c0`|(Optional) If set, predict the initial point of noising process directly|
|`--ot-ode`|(Optional) If set, uses the ODE model instead of SDE one, _i.e.,_ the limit when the diffusion vanishes.  Hence, sampling becomes deterministic|
|`--clip-denoise`| (Optional) If set, clip predicted image to [-1, 1] value range at each sampling iteration|

### Optimization-related
| Parameter      | Explanation |
|----------------|-------------|
|`--batch-size`| (Optional, type=int, default=256)|
|`--microbatch`| (Optional, type=int, default=2) Accumulate gradient over batch shards, processing only `microbatch` samples at a time. Useful for memory conservation when total batch size does not fit the GPU memory|
|`--num-itr`|(Optional, type=int, default=1000000) Total amount of batches to process during training run|
|`--lr`|(Optional, type=float, default=5e-5) Initial learning rate for AdamW optimizer|
|`--lr-gamma`|(Optional, type=float, default=0.99) Learning rate decay ratio|
|`--clip_grad_norm`|(Optional, type=float, default=None) Clip gradient value to a set value. Stabilizes the training procedure, making it more resilient to large-scale gradients caused by outliers in data|
|`--ema`|(Optional, type=float, default=0.99) Exponential moving average rate for network parameters through time. Instance of EMA parameters is saved independenty in parralel with the runtime checkpoint.|

### Metadata and Logging

| Parameter      | Explanation |
|----------------|-------------|
|`--loss-log-freq`|(Optional, type=int, default=10) Frequency in batches with which the training loss value is registered by logger during script execution |   
|`--save-freq`|(Optional, type=int, default=5000) Frequency in batches with which the model state is saved to the storage during script execution |
|`--val-freq`| (Optional, type=int, default=5000) Frequency in batches with which the model performance is evaluated during script execution |            
|`--log-writer`|(Optional, type=str, default=None) Specify the logger backend. At the moment only tensorbard logging is supported|
|`--json-data-config`| (Optional, type=Path) Full path to JSON config for OpenFWI. Used for data normalization. Default path is `/home/seismic_inversion_via_I2SB/dataset/config/openfwi_dataset_config.json`|

## Illustrative sets of options for training and evaluation 
### Training 
```bash
python /home/seismic_inversion_via_I2SB/train_on_everything.py --model i2sb_large_cond --result-dir /home/seismic_inversion_via_I2SB/artifacts/ --dataset-dir /opt/data/openfwi/converted --name i2sb_large_cond --seed 42 --corrupt blur-ci2sb_baseline  --clip-grad-norm 1.0 --image-size 64 --pred_c0 --drop_cond 0.5 --num-itr 300000 --batch-size 256 --microbatch 64 --log-writer tensorboard
```
### Evaluation 

```bash
python /home/seismic_inversion_via_I2SB/record_openfwi_metrics.py --master-port 6090 --result-dir /home/seismic_inversion_via_I2SB/artifacts/ --name i2sb_large_cond --dataset-dir /opt/data/openfwi/converted/CurveFault_B --dataset-name CurveFault_B --ckpt ckpt0.pt --guidance-scale 0.2 --nfe 50 --total_batches 16
```

### Sampling  

```bash
python /home/seismic_inversion_via_I2SB/sample.py --master-port 6090 --result-dir /home/seismic_inversion_via_I2SB/artifacts/ --name i2sb_large_cond --dataset-dir /opt/data/openfwi/converted/CurveFault_B --dataset-name CurveFault_B --ckpt ckpt0.pt --guidance-scale 0.2 --nfe 50 --total_batches 16
```

## References 
<a id="1">[1]</a> Deng, C., Feng, S., Wang, H., Zhang, X., Jin, P., Feng, Y., ... & Lin, Y. (2022). OpenFWI: Large-scale multi-structural benchmark datasets for full waveform inversion. Advances in Neural Information Processing Systems, 35, 6007-6020.

<a id="2">[2]</a> Wang, F., Huang, X., & Alkhalifah, T. A. (2023). A prior regularized full waveform inversion using generative diffusion models. IEEE transactions on geoscience and remote sensing, 61, 1-11

<a id="3">[3]</a> Wang, F., Huang, X., & Alkhalifah, T. (2024). Controllable seismic velocity synthesis using generative diffusion models. Journal of Geophysical Research: Machine Learning and Computation, 1(3), e2024JH000153.

<a id="4">[4]</a> Zhang, H., Li, Y., & Huang, J. (2024). DiffusionVel: Multi-information integrated velocity inversion using generative diffusion models. arXiv preprint arXiv:2410.21776.

<a id="5">[5]</a> Liu, G. H., Vahdat, A., Huang, D. A., Theodorou, E. A., Nie, W., & Anandkumar, A. (2023). I$^2$SB: Image-to-Image Schrodinger Bridge. arXiv preprint arXiv:2302.05872.

<a id="6">[6]</a> Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598.

## License
Copyright © 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC.

The model checkpoints are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).
