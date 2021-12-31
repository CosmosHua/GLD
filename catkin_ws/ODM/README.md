### RTK校准GPS 使用说明

---

#### 1、运行环境（链接中含安装方法）

- [Ubuntu](https://releases.ubuntu.com/18.04/) >= 18.04
- [Python3](https://www.digitalocean.com/community/tutorials/ubuntu-18-04-python-3-zh) >= 3.6.9
- [Docker](https://docs.docker.com/engine/install/ubuntu/) >= 20.10.16
- [OpenSfM](https://opensfm.org/docs/building.html)

#### 2、安装脚本程序

```bash
pip3 install -U odm-sfm --user
```

> 更多信息可参见：https://pypi.org/project/odm-sfm/

#### 3、运行脚本程序

- **Bash方式**：

```bash
odm_sfm_lla
--rtk=./RTK # RTK图像的目录，尚不支持子目录搜索
--gps=./GPS # RPS图像的目录，尚不支持子目录搜索
--dst=./out # 工作目录，存放输出的结果
--log=log.txt # 将屏幕输出保存到{dst}/log.txt
--cpu=3 # 最大并行计算/并发数；范围[1,cpu内核数]
--type=sift # 特征提取的类型；默认SIFT
--min=4000 # 每张图像特征点数的下限；默认4000
--quality=0.5 # 每张图像的特征粒度，越大特征越多约细致；默认0.5，范围(0,1]
```

> - 更多参数项目，详见：`odm_sfm_lla --help`
> - 可能`--log`功能失效，可手动重定向，即在上述命令后添加`>> log.txt 2>&1`。例如：
>   - `odm_sfm_lla --gps=GPS --rtk=RTK --dst=out >> out/log.txt 2>&1`
> - 首次运行程序时，docker会自动下载odm的镜像：需要联网，耗时较长。
> - 程序运行过程中需要sudo权限，会提示输入sudo密码。若想永不超时：
>   - 输入`sudo visudo`，在`Defaults env_reset`后添加`, timestamp_timeout=-1`

- **Python方式**：

```python
from odm_sfm import *
ODM_SfM.SfM_CFG['processes'] = 3 # 等价于Bash方式中：--cpu=3
ODM_SfM.SfM_CFG['feature_type'] = 'SIFT' # 等价于Bash方式中：--type=sift
ODM_SfM.SfM_CFG['feature_process_size'] = 0.3 # 等价于Bash方式中：--quality=0.3
ODM_SfM.SfM_CFG['feature_min_frames'] = 5000 # 等价于Bash方式中：--min=5000
ODM_img_lla2(GPS='./GPS', RTK='./RTK', dst='./out') # GPS图像、RTK图像与结果存放的目录
```

#### 4、运行结果：

> 下述符号说明：`{dst}`、`{rtk}`、`{gps}`分别是上述`--dst`、`--rtk`、`--gps`中的目录名。

- GCP列表的路径：`{dst}/sfm-GCP-{rtk}-{gps}/gcp_list.txt`
  - 或者：`{dst}/odm-GCP-{rtk}-{gps}/opensfm/gcp_list.txt`

> 文件首行是编码格式，之后每行的格式是：<geo_x> <geo_y> <geo_z> <im_x> <im_y> <image_name>
>
> 其中，<geo_x> = longitude, <geo_y> = latitude, <geo_z> = altitude，<im_x> <im_y>是GCP在图像中的像素坐标，<image_name>是图像的文件名。

- POS文件的路径：`{dst}/odm-GCP-{rtk}-{gps}/opensfm/image_geocoords.tsv`

> 文件每行的格式是：<image_name> <geo_x> <geo_y> <geo_z>。
>
> 其中，<image_name>是图像的文件名，<geo_x> = longitude, <geo_y> = latitude, <geo_z> = altitude。

- POS的差异比对：`{dst}/odm-GCP-{rtk}-{gps}/opensfm/image_geocoords_dif.txt`

> 文件每行的格式是：<image_name> <lla> <xyz_dif>。
>
> 其中，<image_name>是图像的文件名，<lla> = [longitude, latitude, altitude]，xyz_dif = 在局部空间坐标系下，原始经纬坐标与<lla>的差值（单位是米）。

