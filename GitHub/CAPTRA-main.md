[CAPTRA: CAtegory-level Pose Tracking for Rigid and Articulated Objects from Point Clouds (ICCV 2021)](https://github.com/halfsummer11/CAPTRA)

- 完成环境配置、数据集下载和预处理、预训练模型、cuda编译网络等。
  - 若只测试，下载：[nocs_model_corners.tar](http://download.cs.stanford.edu/orion/captra/nocs_model_corners.tar)，[real_test.zip](http://download.cs.stanford.edu/orion/nocs/real_test.zip)，[nocs_ckpt.tar](http://download.cs.stanford.edu/orion/captra/nocs_ckpt.tar)，[sapien_ckpt.tar](http://download.cs.stanford.edu/orion/captra/sapien_ckpt.tar)即可。
- 代码目前似乎仍有些bug：
  - [Overall Structure](https://github.com/halfsummer11/CAPTRA#overall-structure)：注意`data`和`runs`应该与`CAPTRA`是同级别的文件夹。
  - [Datasets](https://github.com/halfsummer11/CAPTRA#datasets)：建议将`CAPTRA/configs/obj_config/obj_info_*.yml`中的`basepath`都改为绝对路径。
  - [Test](https://github.com/halfsummer11/CAPTRA#test)：将`tar -xzvf nocs_real_corners.tar`改为：`tar -xzvf nocs_model_corners.tar。如果``data的目录位置变更，需要修改``nocs_preproc.sh中的路径。`
  - `Dataset Folder Structure：将文件夹nocs_model_corners重命名为model_corners。`
  - [Testing](https://github.com/halfsummer11/CAPTRA#testing)：`如果``runs的目录位置变更，需要修改scripts/track/nocs/*.`sh脚本中的路径。
  - [Evaluation](https://github.com/halfsummer11/CAPTRA#evaluation)：将`bash CAPTRA/scripts/eval.sh改为：bash scripts/eval.sh。``如果``runs的目录位置变更，需要修改scripts/eval.sh各行中runs`的路径。
  - [Visualization](https://github.com/halfsummer11/CAPTRA#visualization)：将`bash CAPTRA/scripts/visualize.sh改为：bash scripts/visualize.sh。``如果``data`和`runs``的目录位置变更，需要修改scripts/visualize.sh各行中``data`和`runs`的路径。
- `cd CAPTRA/datasets/nocs_data/preproc_nocs`
  `python generate_all.py --data_path ../../../../data/nocs_data --data_type=test_only --parallel --num_proc=10 > nocs_preproc.sh`
  `bash nocs_preproc.sh; cd ../../..`
- `bash scripts/track/nocs/1_bottle.sh; bash scripts/track/nocs/2_bowl.sh; bash scripts/track/nocs/3_camera.sh; bash scripts/track/nocs/4_can.sh; bash scripts/track/nocs/5_laptop.sh; bash scripts/track/nocs/6_mug.sh`
- `bash scripts/eval.sh # eval all nocs & sapien categoriespython misc/eval/eval.py --config config_track.yml --obj_config obj_info_nocs.yml --obj_category=1 --experiment_dir=../runs/1_bottle_rot # just eval bottle category`
- `bash scripts/visualize.sh # visualize nocs & sapienpython misc/visualize/visualize_tracking_nocs.py --img_path ../data/nocs_data/nocs_full/real_test --exp_path ../runs --output_path ../nocs_viz --save_fig # just visualize nocs`
