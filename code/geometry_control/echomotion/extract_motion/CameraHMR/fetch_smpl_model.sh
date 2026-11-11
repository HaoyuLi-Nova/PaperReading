#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nYou need to register at https://camerahmr.is.tue.mpg.de/"
read -p "Username (CameraHMR):" username
read -p "Password (CameraHMR):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p extract_motion/CameraHMR/data/models/SMPL
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=SMPL_NEUTRAL.pkl' -O 'extract_motion/CameraHMR/data/models/SMPL/SMPL_NEUTRAL.pkl' --no-check-certificate --continue

echo -e "\nSMPL model downloaded successfully!"
