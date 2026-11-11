basedir=eval_folder
folder1_path=${basedir}/base_t2v_eval_sets
folder2_path=${basedir}/eval_sets

# calculate FVD
python calculate_fvd_styleganv.py -v1_f ${folder1_path} -v2_f ${folder2_path}

# calculate FID
python -m pytorch_fid ${basedir}/eval_1 ${basedir}/eval_2

# calculate CLIP-SIM
python calculate_clip.py -v_f ${folder2_path}

rm -rf ${basedir}/eval_1
rm -rf ${basedir}/eval_2