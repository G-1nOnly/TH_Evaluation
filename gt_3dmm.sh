export CUDA_VISIBLE_DEVICES=3

cd ./Deep3DFaceRecon_pytorch/

python extract_kp_videos.py \
  --input_dir ../gt/ \
  --output_dir ../keypoints/ \
  --device_ids $CUDA_VISIBLE_DEVICES \
  --workers 1

python face_recon_videos.py \
  --input_dir ../gt/ \
  --keypoint_dir ../keypoints/gt \
  --output_dir ../recons/ \
  --inference_batch_size 128 \
  --name=face_recon_feat0.2_augment \
  --epoch=20 \
  --model facerecon