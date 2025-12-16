export PROJECT_ROOT=/root/workspace/main/external/FoundationPose-plus-plus
export TEST_CASE=/root/workspace/main/test_case/lego_20fps

cd $PROJECT_ROOT

# wo camK will wa, wo est_refine_iter, track_refine_iter will be bad at big rotation
python src/obj_pose_track.py \
--rgb_seq_path $TEST_CASE/color \
--depth_seq_path $TEST_CASE/depth \
--mesh_path $TEST_CASE/mesh/1x4.stl \
--init_mask_path $TEST_CASE/0_mask.png \
--pose_output_path $TEST_CASE/pose.npy \
--mask_visualization_path $TEST_CASE/mask_visualization \
--bbox_visualization_path $TEST_CASE/bbox_visualization \
--pose_visualization_path $TEST_CASE/pose_visualization \
--cam_K "[[426.8704833984375, 0.0, 423.89471435546875], [0.0, 426.4277648925781, 243.5056915283203], [0.0, 0.0, 1.0]]" \
--activate_2d_tracker \
--apply_scale 0.01 \
--est_refine_iter 10 \
--track_refine_iter 3 \
--force_apply_color \
--apply_color "[0, 159, 237]"
# should disable these two arguments, otherwise, it will got a bad tracker in test case
# --activate_kalman_filter \
# --kf_measurement_noise_scale 0.05 \