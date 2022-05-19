### RTK校准GPS（高低精度匹配）

---

#### 1、安装与配置

- **For Ubuntu：**（[Ubuntu](https://releases.ubuntu.com/18.04/)18.04+、[Python](https://www.digitalocean.com/community/tutorials/ubuntu-18-04-python-3-zh)3.6+）

  - 安装[OpenSfM](https://opensfm.org/docs/building.html)、[Docker](https://docs.docker.com/engine/install/ubuntu/) (20.10.16+)。下载docker镜像、安装脚本及环境：

  ```shell
  sudo docker pull opendronemap/odm:latest # docker image ls
  #sudo docker pull opendronemap/odm:gpu # for GPU Acceleration
  pip3 install -U odm-sfm --user # 详见https://pypi.org/project/odm-sfm/
  ```
  
- **For Win10 Installer：**

  - 安装[ODM](https://github.com/OpenDroneMap/ODM/releases/download/v2.8.6/ODM_Setup_2.8.6.exe) (2.8.6+)。安装pip依赖：（先运行`{ODM的安装路径}/console.bat`）
  
  ```shell
  python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  pip3 install -U py-cpuinfo plyfile ipython
  ```
  
- **For Win10 Docker：**

  - 安装[ODM](https://github.com/OpenDroneMap/ODM/releases/download/v2.8.6/ODM_Setup_2.8.6.exe) (2.8.6+)、[Docker](https://docs.docker.com/desktop/windows/install/) (4.10.1.0+)、[WSL2](https://docs.microsoft.com/en-us/windows/wsl/install-manual)、[Python](https://docs.conda.io/en/latest/miniconda.html#windows-installers) (3.8+)。
  - 下载docker镜像、更换pip源、安装ODM的pip依赖[requirements.txt](https://github.com/OpenDroneMap/ODM/blob/master/requirements.txt)：

  ```shell
  docker pull opendronemap/odm:latest # 请确保ODM的installer与docker的版本一致
  python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  pip3 install -U -r requirements.txt py-cpuinfo plyfile opencv-python --user
  ```

  - 手动安装[rasterio](https://rasterio.readthedocs.io/en/latest/installation.html)：下载[GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)和[raterio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio)，然后用pip安装下载的whl文件。

  ```python
  import os, sys, cv2, numpy as np
  sys.path.append('{ODM的安装路径}/SuperBuild/install/bin/opensfm')
  from opensfm.dataset import DataSet
  ```

- **For Win10 (Docker & Installer)：**

  - 配置系统变量：此电脑->属性->高级系统设置->环境变量->系统变量的Path，添加：
    `{ODM的安装路径}\SuperBuild\install\bin`。
  

---

#### 2、运行脚本

- **For Ubuntu：**

  ```shell
  python3 odm_sfm.py # OR: odm_sfm_bash
  --rtk=./RTK # RTK图像的目录
  --gps=./GPS # GPS图像的目录
  --dst=./out # 工作目录，存放输出结果
  --type=sift # 特征提取的类型；默认SIFT
  --cpu=2 # 最大并行计算/并发数；范围[1,cpu内核数]
  --mod=odm # GPS特征的提取方式；默认odm，可选[odm,sfm]
  --svd # 只计算刚体变换矩阵，跳过导出POS或用GCP重建；默认不开启
  --gpu='' # 提取SIFT特征时，启用GPU加速；默认不开启，可选['',all]
  --size=0.5 # 每张图像的特征粒度，越大特征越多约细致；默认0.5，范围(0,1]
  ```

  > 更多参数详见：`python odm_sfm.py --help`

  > 程序的运行需要sudo权限，会提示输入sudo密码。若要sudo权限不超时：
  > 输入`sudo visudo`，在`Defaults env_reset`后添加`, timestamp_timeout=-1`

- **For Win10 Installer：**先运行`{ODM的安装路径}/console.bat`

  ```shell
  python odm_sfm.py
  --rtk=./RTK # RTK图像的目录
  --gps=./GPS # GPS图像的目录
  --dst=./out # 工作目录，存放输出结果
  --root=D:/ODM # 设定{ODM的安装路径}
  --type=sift # 特征提取的类型；默认SIFT
  --cpu=2 # 最大并行计算/并发数；范围[1,cpu内核数]
  --mod=odm # GPS特征的提取方式；默认odm，可选[odm,sfm]
  --svd # 只计算刚体变换矩阵，跳过导出POS或用GCP重建；默认不开启
  --gpu='' # 提取SIFT特征时，启用GPU加速；默认不开启，可选['',all]
  --size=0.5 # 每张图像的特征粒度，越大特征越多约细致；默认0.5，范围(0,1]
  ```

- **For Win10 Docker：**

  ```shell
  python odm_sfm.py
  --rtk=./RTK # RTK图像的目录
  --gps=./GPS # GPS图像的目录
  --dst=./out # 工作目录，存放输出结果
  --root=D:/ODM # 设定{ODM的安装路径}
  --type=sift # 特征提取的类型；默认SIFT
  --cpu=2 # 最大并行计算/并发数；范围[1,cpu内核数]
  --mod=odm # GPS特征的提取方式；默认odm，可选[odm,sfm]
  --docker # 强制win10采用docker版为后端；默认以exe版为后端
  --svd # 只计算刚体变换矩阵，跳过导出POS或用GCP重建；默认不开启
  --gpu='' # 提取SIFT特征时，启用GPU加速；默认不开启，可选['',all]
  --size=0.5 # 每张图像的特征粒度，越大特征越多约细致；默认0.5，范围(0,1]
  ```

---

#### 3、运行结果：

- GCP列表的路径：

  - 当`mod=sfm`时：`{dst}/sfm-GCP-{rtk}-{gps}/gcp_list.txt`
  - 当`mod=odm`时：`{dst}/odm-GCP-{rtk}-{gps}/opensfm/gcp_list.txt`

  > 文件首行是编码格式，之后每行的格式是：<geo_x> <geo_y> <geo_z> <im_x> <im_y> <image_name>
  >
  > 其中，<geo_x> = longitude, <geo_y> = latitude, <geo_z> = altitude，<im_x> <im_y>是GCP在图像中的像素坐标，<image_name>是图像的文件名。

- POS文件的路径：`{dst}/odm-GCP-{rtk}-{gps}/opensfm/image_geocoords.tsv`

  > 文件每行的格式是：<image_name> <geo_x> <geo_y> <geo_z>
  >
  > 其中，<image_name>是图像的文件名，<geo_x> = longitude, <geo_y> = latitude, <geo_z> = altitude。

- POS的差异比对：`{dst}/odm-GCP-{rtk}-{gps}/opensfm/image_geocoords_dif.txt`

  > 文件每行的格式是：<image_name> <lla> <xyz_dif>
  >
  > 其中，<image_name>是图像的文件名，<lla> = [longitude, latitude, altitude]，<xyz_dif> = 在ENU坐标系下，<lla>与图像初始经纬坐标的差值，即修正量（单位是米）。

- 符号说明：`{dst}`、`{rtk}`、`{gps}`分别为`--dst`、`--rtk`、`--gps`中的目录名。

---

|         |    Ubuntu    |    Win10     |       Win10       |
| :-----: | :----------: | :----------: | :---------------: |
|         |  **Docker**  |  **Docker**  | **Installer/EXE** |
| **CPU** | ODM=√, SFM=√ | ODM=√, SFM=√ |   ODM=√, SFM=√    |
| **GPU** | ODM=√, SFM=X | ODM=X, SFM=X |   ODM=X, SFM=X    |

- 所有GPU的SFM模式失效：因SFM用CPU提特征，ODM用GPU提特征，[两种特征无法交叉匹配](https://github.com/OpenDroneMap/ODM/issues/1503)。
- Win10的EXE的GPU的ODM模式失效：是在GPU提取GPS特征时失败，不是匹配时，属Bug。
- Win10的Docker的GPU的ODM模式失效：匹配失败，原因未知。
