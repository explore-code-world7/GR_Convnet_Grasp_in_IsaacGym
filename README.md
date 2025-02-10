The project integrates Antipodal Robotic Grasping into Isaac Gym Environment

# Result Display

* Parallel env, multi-objects

<video src="./video/multi_env_multi_obj.mp4"></video>

* single env, single-object

<video src="./video/sing_env_obj.mp4"></video>

* single env, multi-object

<video src="./video/single_env_multi_obj.mp4"></video>

# Install required packages

1. GR-Convnet
```bash
pip install -r grgrasp_requirement.txt
```

2. IsaacGym
* System Requirement: Ubuntu 18.04/20.04/22.04/24.04
* Env Requirement: python 3.6/3.7/3.8

```bash
cd isaac_packages
pip install -e .
```
* for trouble shooting of IsaacGym installation, see [link](https://junxnone.github.io/isaacgymdocs/install.html#troubleshooting)

## Download Cornell Dataset
* see README.md in dataset/

# Folder Introduction
1. /hardware, /inference, /results, /trained-models, /utils were inherited from [Antipodal Robotic Grasping](https://github.com/skumra/robotic-grasping);

2. /grtest includes a self-defined dataset Class and dataset from IsaacGym env to predict RGB-D images from Isaac Gym, and resuls from a simle test of GR-Convnet.

3. /logs stores trained GR-Convnet Model;

4. /dataset includes Cornell Dataset to train GR-Convnet;

5. /franka_img includes rgb images taken each frame of simulation environment;

6. /box_predicted includes depth images taken when robotic arm start a new grasp and their corresponding greyscale values;
(5&6 can be integrated in fact, but I choose the above set for faciliated debugging)



# Code Implement

## train GR-Convnet Model
* train on Gornell Grasp Dataset
```bash
python train_network.py --dataset  cornell  --dataset-path  ./dataset/cornell-grasp/versions/1/  --description  training_cornell
```
* evaluate on Cornell Grasp Dataset
```bash
python evaluate.py --network logs/250131_2338_training_cornell/epoch_44_iou_0.96   --dataset   cornell  --dataset-path  ./dataset/cornell-grasp/versions/1/   --iou-eval
```

## test GR-Convnet Model
```bash
python test.py
```
* use model trained myself——"logs/250131_2338_training_cornell/epoch_44_iou_0.96"
* the test output object centre and angle, and draw grasp rectangle in plt default page
* dataset is in "grtest/depth", "grtest/rgb", result is in "grtest/predicted"

## utilize GR-Convnet for Robotic Arm conotrol

```bash
python franka_cube_stage.py --num_envs 16  --multiple True
```
* --num_envs denotes number of simulating environment
* `--multiple True` means laying multiple objects on table
* `--test True` means playing totally 100 grasp tests and output average differ, the num_envs is better to be set as 100 in case
some environmen doesn't grasp successfully and will taken same photo continuously, resulting in invarinant differ in x-axis, y-axis and rotation angle

