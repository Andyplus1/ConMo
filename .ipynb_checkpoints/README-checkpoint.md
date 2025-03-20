# ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer
<!-- <a href="https://diffusion-motion-transfer.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/"><img src="https://img.shields.io/badge/arXiv-2311.17009-b31b1b.svg"></a> -->

This is the official implementation of the paper:

By Jiayi Gao, Zijin Yin, Changcheng Hua, Yuxin Peng, Kongming Liang, Zhanyu Ma, Jun Guo and Yang Liu

<!-- [**Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer**](https://diffusion-motion-transfer.github.io/)
<br/>

[Danah Yatim*](https://www.linkedin.com/in/danah-yatim-4b15231b5/),
[Rafail Fridman*](https://www.linkedin.com/in/rafail-fridman/),
[Omer Bar-Tal](https://omerbt.github.io/),
[Yoni Kasten](https://ykasten.github.io/),
[Tali Dekel](https://www.weizmann.ac.il/math/dekel/)
<br/>
(*equal contribution)

https://github.com/diffusion-motion-transfer/diffusion-motion-transfer/assets/22198039/4fe912d4-0975-4580-af7f-19fd73b0cbfe -->



Introducing a zero-shot method for transferring motion across objects and scenes with multiple application. without any training or finetuning.

![method](pipeline.png)

>The development of Text-to-Video (T2V) generation has made motion transfer possible, enabling the control of video motion based on existing footage. However, current methods have two limitations: 1) struggle to handle multi-subjects videos, failing to transfer specific subject motion; 2) struggle to preserve the diversity and accuracy of motion as transferring to subjects with varying shapes.
To overcome these, we introduce ConMo, a training-free framework that disentangle and recompose the motions of subjects and camera movements. ConMo isolates individual subject and background motion cues from complex trajectories in source videos using only subject masks, and reassembles them during target video generation. This approach enables more accurate motion control across diverse subjects and improves performance in multi-subject scenarios. Additionally, we propose soft motion guidance in the recomposition stage which controls the retention of original motion to adjust shape constraints, aiding subject shape adaptation and semantic transformation. Unlike previous methods, ConMo unlocks a wide range of applications, including subject size and position editing, subject removal, semantic modifications, and camera motion simulation. Extensive experiments demonstrate that ConMo significantly outperforms state-of-the-art methods in motion fidelity and temporal consistency.

<!-- For more, visit the [project webpage](https://diffusion-motion-transfer.github.io/). -->

# Installation
Clone the repo and create a new environment:
```
git clone https://github.com/Andyplus1/ConMo.git
cd ConMo
conda create --name conmo python=3.9
conda activate conmo
```
Install our environment requirements:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

# Download Sample Data
The sample data is provided in [this link](https://pan.baidu.com/s/1W5Sap5V_Xw_BXnftTDfd5A) (BaiduYun Extraction Code: uyi4 )

# Motion Transfer
<!-- * Our method is designed for transferring motion across objects and scenes -->
* Our method is based on [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) text-to-video model and [DMT](https://github.com/diffusion-motion-transfer/diffusion-motion-transfer) motion transfer method. Therefore, we can edit videos of **24 frames**.
* In some cases, the combination of the target object and the motion from the input video may be out of the T2V model's distribution, potentially resulting in visual artifacts in the generated video. It might be necessary to try several different seeds for sampling.
* Method was tested to run on a single NVIDIA A40 48GB, and takes ~32GB of video memory.
 It takes approximately 5 minutes on a single NVIDIA A100 40GB for inference.

# Preprocess
Following [DMT](https://github.com/diffusion-motion-transfer/), to preprocess a video, update configuration file `configs/preprocess_config.yaml':

Arguments to update:
* ```video_path``` - the input video frames should be located in this path
* ```save_dir``` - the latents will be saved in this path
* ```prompt``` - empty string or a string describing the video content

Optional arguments to update:
* ```--save_ddim_reconstruction``` if True, the reconstructed video will be saved in ```--save_dir```

After updating config file, run the following command:
```
python preprocess_video_ddim.py --config_path configs/preprocess_config.yaml
```
Once the preprocessing is done, the latents will be saved in the ```save_dir``` path. 

# Editing
To edit the video, update configuration file `configs/guidance_config.yaml`
Arguments to update:
* ```data_path``` - the input video frames should be located in this path
* ```output_path``` - the edited video will be saved in this path
* ```latents_path``` - the latents of the input video should be located in this path
* ```source_prompt``` - prompt used for inversion
* ```target_prompt``` - prompt used for editing
    
Optional arguments to update:
* ```negative_prompt``` - prompt used for unconditional classifier free guidance
*  ```seed``` - By default it is randomly chosen, to specify seed change thise value.
*  ```optimization_step``` - number of optimization steps for each denoising step
* ```optim_lr``` - learning rate
* ```with_lr_decay```  - if True, overrides `optim_lr`, and the learning rate will decay during the optimization process in the range of `scale_range`

<!-- After updating the config file, run the following command:
```
python run.py --config_path configs/guidance_config.yaml
``` -->
We provide editing example for the application we mentioned in the paper(preprocess steps are also included here)

For motion transfer involving no geometric changes (single entity/multiple entities), run the following command:
```
#  For transferring fine-grained motions between small shape differences.
python preprocess_video_ddim.py --config_path  configs/common/preprocess_config_finegrained.yaml
python run.py --config_path configs/common/guidance_config_finegrained.yaml 

#  For transferring multi-subject motions.
python preprocess_video_ddim.py --config_path  configs/common/preprocess_config_multisubject.yaml
python run.py --config_path configs/common/guidance_config_multisubject.yaml

#  For transferring motions with with drastic shape differences.
python preprocess_video_ddim.py --config_path  configs/common/preprocess_config_drasticshape.yaml
python run.py --config_path configs/common/guidance_config_drasticshape.yaml
```

For motion transfer involving geometric changes (reposition/resize), run the following command:
```
#  For reposition
python preprocess_video_ddim.py --config_path configs/reposition_resize/preprocess_config_reposition.yaml
python run_resizeorreposition.py --config_path configs/reposition_resize/guidance_config_reposition.yaml

#  For resize
python preprocess_video_ddim.py --config_path configs/reposition_resize/preprocess_config_resize.yaml
python run_resizeorreposition.py --config_path configs/reposition_resize/guidance_config_resize.yaml
```

For selective transferring motions from the original video / transferring only the camera motion, run the following command:
```
#  For selective transferring 
python preprocess_video_ddim.py --config_path configs/erase_camera/preprocess_config_erase.yaml
python run_erase.py --config_path configs/erase_camera/guidance_config_erase.yaml
#  For transferring camera motion only.

python preprocess_video_ddim.py --config_path configs/erase_camera/preprocess_config_camera.yaml
python run_erase.py --config_path configs/erase_camera/guidance_config_camera.yaml
```

Once the method is done, the video will be saved to the ```output_path``` under `result.mp4`.


# Tips
* To get better samples from the T2V model, we used the prefix text ```"Amazing quality, masterpiece, "``` for inversion and edits.
* If the video contains more complex motion/small objects, try increasing number of optimization steps - ```optimization_step: 30```.
* For large deviation in structure between the source and target objects, try using a lower lr - ```scale_range:[0.005, 0.002]```,
*  or adding the source object to the negative prompt text.

<!-- # Measuring motion fidelity
We also provide the code for calculating the motion fidelity metric introduced in the paper (Section 5.1).
To calculate the motion fidelity metric, first follow the instructions [here](https://github.com/facebookresearch/co-tracker) to install Co-Tracker and download their checkpoint.
Then, run the following command:
```
python motion_fidelity_score.py --config_path configs/motion_fidelity_config.yaml
``` -->



<!-- # Citation
```
@article{yatim2023spacetime,
        title = {Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer},
        author = {Yatim, Danah and Fridman, Rafail and Bar-Tal, Omer and Kasten, Yoni and Dekel, Tali},
        journal={arXiv preprint arxiv:2311.17009},
        year={2023}
        }
``` -->
