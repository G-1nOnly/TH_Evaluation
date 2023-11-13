# TH_Evaluation
Evaluation Script for Multi-modal Learning for Audio-driven Talking Head Generation, Task 3 in AAAI2024 Workshop AI for Digital Human.

### Evaluation
1. Put the Ground-Truth in the "gt" directory, and put your results in the "pred" directory.
2. Generate Ground-Truth 3DMM coefficients.
```
bash gt_3dmm.sh
```
3. Evaluate your results.
```
export target=[Your directory name]
bash eval_single.sh $target
# for Single Image Setting 
bash eval_video.sh $target
# for Single Image Setting 
```

### Acknowledgement
Some evaluation scripts are modified from [vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline). The original repository of Deep3DFaceRecon_pytorch is [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch) and the original repository for syncnet_python is [syncnet](https://github.com/joonson/syncnet_python).
Thanks for these authors' wonderful projects.

