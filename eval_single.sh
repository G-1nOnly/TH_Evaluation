export CUDA_VISIBLE_DEVICES=3
export target=Exciting_AI

python eval_single.py --gt_video_folder ./gt --pd_video_folder ./pred/$target/ --teamname $target