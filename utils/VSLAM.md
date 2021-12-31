#### 视觉SLAM14讲 - 高翔 Note

---

##### Chaper 3.4 四元数

- **求证**：$\mathbf p' = \mathbf q \mathbf p \mathbf q^{-1} = [0,\mathbb R\vec p]$。其中：

  > 四元数：$\mathbf p = [0,\vec p] = [0,x,y,z]$，$\mathbf q = [cos\frac{θ}{2}, \vec n sin\frac{θ}{2}]$，$\mathbf q^{-1} = \mathbf q^* / |\mathbf q|^2$。
  >
  > 旋转矩阵：$\mathbb R = cosθ\mathbb I + (1−cosθ) \vec n \vec n^T + sinθ\vec n ^∧$，其中$\vec a×\vec b \triangleq \vec a^∧ \vec b$。
  >
  > 旋转向量：$θ\vec n = θ[n_x,n_y,n_z]$，其中$|\vec n|^2 = 1$，因此$|\mathbf q|^2 = 1$。

- **相关公式**：

  > 四元数乘法：$\mathbf p \mathbf q = [p_s, \vec p_v][q_s,\vec q_v] = [p_s q_s − \vec p_v \cdot \vec q_v,  p_s \vec q_v + q_s \vec p_v + \vec p_v×\vec q_v]$。
  >
  > 双叉乘公式：$\vec a×(\vec b×\vec c) = (\vec a \cdot \vec c)\vec b - (\vec a \cdot \vec b)\vec c$。

- **证明**：

  For short, set $\alpha = θ/2$, then $\mathbf q = [cos\alpha, \vec ns in\alpha]$, $\mathbf q^{-1} = \mathbf q^* = [cos\alpha, -\vec n sin\alpha]$.

  $\mathbf q \mathbf p \mathbf q^{-1} = [cos\alpha, sin\alpha \vec n][0,\vec p][cos\alpha, -sin\alpha \vec n]$

  $= [0, \vec p cos^2\alpha + 2\vec n×\vec p sin\alpha cos\alpha + \vec n(\vec n \cdot \vec p)sin^2\alpha - \vec n×(\vec p×\vec n)sin^2\alpha]$

  $= [0, \vec p (cos^2\alpha-sin^2\alpha) + (\vec n×\vec p) sin2\alpha + 2\vec n(\vec n \cdot \vec p)sin^2\alpha]$

  $= [0, \vec p cos2\alpha + (\vec n×\vec p) sin2\alpha + \vec n(\vec n \cdot \vec p)(1-cos2\alpha)]$

  $= [0, \vec p cosθ + (\vec n×\vec p) sinθ + \vec n(\vec n \cdot \vec p)(1-cosθ)]$

  $\mathbb R\vec p = [cosθ\mathbb I + (1−cosθ) \vec n \vec n^T + sinθ\vec n ^∧]\vec p = \vec p cosθ + (1−cosθ) \vec n (\vec n \cdot \vec p) + sinθ(\vec n×\vec p)$

  

