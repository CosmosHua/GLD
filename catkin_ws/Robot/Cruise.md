> 类别命名规则：类别\_子类/组件_状态/材质/属性
>
> 数据采集：远近+水平角（蛇形），俯仰角+高低，光照（上午+下午+晚上）

#### 场景：会议室前

**Category**：

- 整门：`door_open`（门开：木门+毛玻璃门），`door_close`（门关：木门+毛玻璃门）
- 门板组件：`door_leaf_wood`（木门板），`door_leaf_glass`（带毛玻璃的门板），`door_leaf_lateral`（侧门板），`door_leaf_window`（门板上玻璃窗）
- 遮挡组件：`glass_frosted`（毛玻璃），`shutter`（遮光板/帘）
- 巡检目标：`person`（人），`laptop`（便携电脑）

---

**Key_Info**：

- 会议室的`door_leaf_glass`上，都没有海报遮挡
- 会议室的`door_leaf_wood`，旁边肯定有`door_leaf_window`
- 会议室的`door_leaf_window`，除2层以外，几乎都不是`glass_frosted`

---

**Preset**：

0. 预设会议室的观察点：door=门，wall_glass=透明玻璃墙

1. 各个观察点的左右边界坐标；各个观察点的探察主方向；
2. 各个观察点的类型：door_wood=木门，door_glass=玻璃门，wall_glass=透明玻璃墙

**Procedure**：

1. 机器人走到观察点（door/wall_glass）前
2. 检测观察点的可透视区域：

   - if 观察点类型为wall_glass：
     - 若无`shutter`遮挡：ok
     - 若有`shutter`遮挡：skip

   - elif 观察点类型为door_glass：
     - 检测并排除`glass_frosted`的区域：ok
   - elif `door_open`：# 此时观察点必为door_wood
     - 检测并排除`door_leaf_wood`与`door_leaf_lateral`的区域：ok
   - elif `door_close`：# 此时观察点必为door_wood
     - 检测`door_leaf_window`区域：
       - 若非`glass_frosted`：ok
       - 若为`glass_frosted`：skip
3. 在门外扫视检测：`person`或`laptop`即可
4. 类别小结：
   - `door`：`open`，`close`
   - `door_leaf`：`wood`，`glass`，`lateral`，`window`
   - `glass_frosted`，`shutter`
   - `person`，`laptop`


---

> PS：下列表中，glass=全透明玻璃，frosted=全毛玻璃，glass_pf=部分透明部分毛玻璃，glass_shutter=透明玻璃被shutter遮挡；left-right表示左-中-右都扫描。

|  Floor_1  | door_window | door_towards | wall_glass | wall_towards |
| :-------: | :---------: | :----------: | :--------: | :----------: |
| 101/101-2 | wood+glass  |  right/left  |            |              |
|   129T    | wood+glass  |  left-right  |            |              |
|    151    |  glass_pf   |    right     |            |              |
|    152    |  glass_pf   |     left     |            |              |
|    153    |  glass_pf   |    right     |            |              |
|    155    |  glass_pf   |    right     |            |              |
|    156    | 2*glass_pf  |  left-right  |            |              |
|    157    |  glass_pf   |    right     |            |              |
|    158    |  glass_pf   |     left     |            |              |
|    159    |  glass_pf   |    right     |            |              |

---

| Floor_2 | door_window  | door_towards | wall_glass      | wall_towards |
| :-----: | :----------: | :----------: | :-------------: | :----------: |
| 202T    | wood+frosted | left         |                 |              |
| 224     | wood+frosted | right        | frosted         | left         |
| 222     | wood+frosted | right        | glass_shutter   | left         |
| 220     | wood+frosted | right        | glass_shutter   | left         |
| 218     | wood+frosted | left         | 2*glass_shutter | mid+right    |
| 208T    | wood+frosted | left         |                 |              |
| 206T    | wood+frosted | left         |                 |              |

---

| Floor_3 | door_window  | door_towards | wall_glass | wall_towards |
| :-----: | :----------: | :----------: | :--------: | :----------: |
|  320T   | wood+frosted |    right     |            |              |
|   319   |   glass_pf   |     room     |            |              |
|  306T   |  wood+glass  |     left     |            |              |
|   301   |   glass_pf   |     room     |            |              |
|  302T   |  wood+glass  |     left     |            |              |

---

| Floor_4 | door_window | door_towards | wall_glass | wall_towards |
| :-----: | :---------: | :----------: | :--------: | :----------: |
|  402T   | wood+glass  |    right     |            |              |
|  426T   | wood+glass  |    right     |            |              |
|  424T   | wood+glass  |    right     |   glass    |     left     |
|  422T   | wood+glass  |     left     |   glass    |    right     |
|   420   | wood+glass  |    right     |   glass    |     left     |
|   418   | wood+glass  |     left     |   glass    |    right     |
|   419   |  glass_pf   |     room     |            |              |
|   401   |  glass_pf   |     room     |            |              |

---

| Floor_5 | door_window | door_towards | wall_glass | wall_towards |
| :-----: | :---------: | :----------: | :--------: | :----------: |
|   522   | wood+glass  |    right     |            |              |
|  520T   |  glass_pf   |     left     |            |              |
|   523   |  glass_pf   |     room     |            |              |
|   511   | wood+glass  |     left     |            |              |
|   501   |  glass_pf   |     room     |            |              |
|   502   | wood+glass  |    right     |            |              |

---

| Floor_6 | door_window | door_towards | wall_glass | wall_towards |
| :-----: | :---------: | :----------: | :--------: | :----------: |
|  626T   | wood+glass  |    right     |            |              |
|   618   | wood+glass  |     left     |            |              |
|   633   |  glass_pf   |     room     |            |              |
|   621   | wood+glass  |     left     |            |              |
|   601   |  glass_pf   |     room     |            |              |

---

#### 场景：电梯门前

> 已知：目前所在楼层为A，与目的楼层为B。

1. 机器人走到预定的电梯口前：
   - 计算sign(B-A)，表明要上楼/下楼：
     - 设bt = `button_up`或`button_down`
2. 检测电梯旁的`panel`面板与bt按钮：
   - 通过`panel`周围的墙面，测算距离d
   - 若bt是off灭的：计算bt中心的空间坐标，向前触碰bt，直至bt变on
   - 若bt是on亮的：wait
3. 若bt由on变off，检测电梯上方的LED：
   - 若为`other`：更换电梯，或抛出异常+退出
   - 若A!=LED数字：goto 2
   - 否则，持续检测`door_elevator`的状态：
     - 若为`closed`：goto 2
     - 若为`closing`：wait，直至`closed`，然后goto 2
     - 若为`opening`：wait，直至`opened`
     - 若为`opened`：检测电梯内是否有`person`
       - 若无`person`：进入电梯
       - 若有`person`：放弃进入，goto 2
4. 类别小结：
   - `panel`，`person`；~~`floor`~~
   - `button_on`：`up`，`down`；~~`open`，`close`，`ring`；`1`，`2`，`3`，`4`，`5`，`6`，`-1`，`-2`~~
   - `button_off`：`up`，`down`；~~`open`，`close`，`ring`；`1`，`2`，`3`，`4`，`5`，`6`，`-1`，`-2`~~
   - `LED`：`1`，`2`，`3`，`4`，`5`，`6`，`-1`，`-2`，`other`
   - `door_leaf_metal`/`door_elevator`：`opening`，`closing`，`opened`，`closed`

---

#### 场景：电梯内

> PS：因电梯内金属反射，须分清哪个是真电梯门，哪个是投影；或预设好方向。

1. 已知目的楼层为B，检测按钮bt=B：
   - 若bt是off灭的：计算bt中心的空间坐标，向前触碰bt，直至bt变on
   - 若bt是on亮的：wait
2. 若bt由on变off，持续检测`door_elevator`的状态：
   - 若为`closed`：wait
   - 若为`opening`：wait，直至`opened`
   - 若为`opened`：走出电梯
   - 若为`closing`：wait，直至`closed`，然后goto 1
3. 类别小结：
   - `button_on`：~~`up`，`down`；`open`，`close`，`ring`~~；`1`，`2`，`3`，`4`，`5`，`6`，`-1`，`-2`
   - `button_off`：~~`up`，`down`；`open`，`close`，`ring`~~；`1`，`2`，`3`，`4`，`5`，`6`，`-1`，`-2`
   - ~~`LED`：`1`，`2`，`3`，`4`，`5`，`6`，`-1`，`-2`，`other`~~
   - `door_leaf_metal`/`door_elevator`：`opening`，`closing`，`opened`，`closed`

---

追加类别：`doorplate`（门牌），`floor`（地面），`wall`（墙面）