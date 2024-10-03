set -e

workdir=$1
echo "Processing all scenes in $workdir..."
export CUDA_VISIBLE_DEVICES=1

for input_folder in "$workdir"/*
do
    scene=$(basename $input_folder)

    if [ -d $input_folder ]; then
        
        echo "Processing scene: $input_folder"

        # Step 1: Image undistortion
        colmap image_undistorter \
            --image_path $input_folder/images \
            --input_path $input_folder/sparse/0 \
            --output_path $input_folder/dense_undistorted \
            --output_type COLMAP

        # Step 2: Compute depth maps
        colmap patch_match_stereo \
            --workspace_path $input_folder/dense_undistorted \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true

        # Step 3: Fuse depth maps into a dense point cloud
        colmap stereo_fusion \
            --workspace_path $input_folder/dense_undistorted \
            --workspace_format COLMAP \
            --input_type geometric \
            --output_path $input_folder/dense_undistorted/fused.ply

    fi
done
echo "Done with all scenes!"