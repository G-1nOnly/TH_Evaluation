export CUDA_VISIBLE_DEVICES=2
export target=VastGen

# cd ./Deep3DFaceRecon_pytorch/

# python extract_kp_videos.py \
#   --input_dir ../pred/$target/ \
#   --output_dir ../keypoints/ \
#   --device_ids $CUDA_VISIBLE_DEVICES \
#   --workers 1

# python face_recon_videos.py \
#   --input_dir ../pred/$target/ \
#   --keypoint_dir ../keypoints/$target \
#   --output_dir ../recons/ \
#   --inference_batch_size 128 \
#   --name=face_recon_feat0.2_augment \
#   --epoch=20 \
#   --model facerecon

# cd ..

python eval_video.py --gt_video_folder ./gt --pd_video_folder ./pred/$target/ --teamname Exciting_AI
