#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nYou need to register at https://camerahmr.is.tue.mpg.de/"
read -p "Username (CameraHMR):" username
read -p "Password (CameraHMR):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p extract_motion/CameraHMR/data/pretrained-models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=cam_model_cleaned.ckpt' -O 'extract_motion/CameraHMR/data/pretrained-models/cam_model_cleaned.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=camerahmr_checkpoint_cleaned.ckpt' -O 'extract_motion/CameraHMR/data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=model_final_f05665.pkl' -O 'extract_motion/CameraHMR/data/pretrained-models/model_final_f05665.pkl' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=smpl_mean_params.npz' -O 'extract_motion/CameraHMR/data/smpl_mean_params.npz' --no-check-certificate --continue

echo -e "\nAll pretrained models downloaded successfully!"
