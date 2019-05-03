.. _sec:notation:

符号一览
========

大写字母用于随机变量，而小写字母用于随机变量的具体值或标量函数。
小写、粗体的字母用于实数向量(即使是随机变量)。大写的粗体字母用于矩阵。

============================================== ================================================================================================
:math:`\doteq`                                 由定义得到的等于关系
:math:`\approx`                                约等于
:math:`\propto`                                正比于
:math:`\Pr \{X=x\}`                            随机变量 :math:`X` 取值为 :math:`x` 的概率
:math:`X \sim p`                               随机变量 :math:`X` 满足分布 :math:`p(x) \doteq \Pr\{X = x\}`
:math:`\mathbb{E}[X]`                          随机变量 :math:`X` 的期望值, 也就是说 :math:`\mathbb{E}[X] = \sum_x p(x)x`
:math:`\arg \max_a f(a)`                       当 :math:`f(a)` 取最大值时 :math:`a` 的取值
:math:`\ln (x)`                                :math:`x` 的自然对数
:math:`e^x, exp(x)`                            自然对数 :math:`e \approx 2.71828` 的 :math:`x` 次方；:math:`e^{\ln x}=x`
:math:`\mathbb{R}`                             实数集
:math:`f: \mathcal{X} \rightarrow \mathcal{y}` 函数 :math:`f` 表示从集合 :math:`\mathcal X` 中元素到集合 :math:`\mathcal{y}` 中元素的映射
:math:`\leftarrow`                             赋值
:math:`(a, b]`                                 左开右闭的实数区间
\
:math:`\varepsilon`                            在 :math:`\varepsilon` -贪婪策略中采取随机动作的概率
:math:`\alpha, \beta`                          步长参数
:math:`\gamma`                                 折扣率参数
:math:`\lambda`                                资格迹中的衰减率
:math:`\mathbb{1}_{predicate}`                 指示函数(当 *谓词* :math:`predicate` 为真时 :math:`\mathbb{1}_{predicate} \doteq 1`, 反之为0)
\
============================================== ================================================================================================

在多摇臂赌博机问题中:

======================= =========================================================================
:math:`k`               动作(摇臂)的数量
:math:`t`               离散的时间步或玩的次数
:math:`q_*(a)`          动作 :math:`a` 的真实值(预期奖励)
:math:`Q_t(a)`          :math:`q_*(a)` 在时步 :math:`t` 的估计值
:math:`N_t(a)`          在时步 :math:`t` 前动作 :math:`a` 被选中的概率
:math:`H_t(a)`          由学习得到的、在时步 :math:`t` 时选择动作 :math:`a` 的偏好值
:math:`\pi_t(a)`        在时步 :math:`t` 选择动作 :math:`a` 的概率
:math:`\overline{R}_t`  在给定策略 :math:`\pi_t` 的情况下, 预期奖励在时步 :math:`t` 时的估计值
\
======================= =========================================================================

在马尔科夫决策过程中:

===================================================== ===============================================================================================
:math:`s, s^{\prime}`                                 状态
:math:`a`                                             动作
:math:`r`                                             奖励
:math:`\mathcal{S}`                                   所有非末状态的集合
:math:`\mathcal{S}^+`                                 所有状态的集合, 包括末状态
:math:`\mathcal{A}(s)`                                在状态 :math:`s` 下所有可行的动作的集合
:math:`\mathcal{R}`                                   所有可能奖励的集合, 为 :math:`\mathbb{R}` 的有限子集
:math:`\subset`                                       含于, 例如 :math:`\mathcal{R} \subset \mathbb{R}`
:math:`\in`                                           属于, 例如 :math:`s \in \mathcal{S}`, :math:`r \in \mathcal{R}`
:math:`\lvert \mathcal{S} \rvert`                     集合 :math:`\mathcal{S}` 中元素的个数
\
:math:`t`                                             离散的时步
:math:`T, T(t)`                                       回合的最后一个时步, 或包含了时步 :math:`t` 的回合的最后一步
:math:`A_t`                                           在时步 :math:`t` 中所选择的动作
:math:`S_t`                                           时步 :math:`t` 时的状态, 通常由 :math:`S_{t-1}` 和 :math:`A_{t-1}` 概率性地决定
:math:`R_t`                                           在时步 :math:`t` 中的奖励, 通常由 :math:`S_{t-1}` 和 :math:`A_{t-1}` 概率性地决定
:math:`\pi`                                           策略(决策准则)
:math:`\pi(s)`                                        在 *确定性* 策略 :math:`\pi` 下, 在状态 :math:`s` 中所采取的动作
:math:`\pi(a | s)`                                    在 *概率性* 策略 :math:`\pi` 下, 在状态 :math:`s` 中采取动作 :math:`a` 的概率
\
:math:`G_t`                                           在时步 :math:`t` 后的回报
:math:`h`                                             水平，在前瞻多看的时步（horizon, the time step one looks up to in a forward view）
:math:`G_{t:t+n}, G_{t:h}`                            从 :math:`t+1` 到 :math:`t+n` 或到 :math:`h` （折扣的且校正的） 的n步回报
:math:`overline{G}_{t:h}`                             从 :math:`t+1` 到 :math:`h` 的平坦回报（未折扣且未校正的）（5.8节）
:math:`G_{t}^{\lambda}`                               :math:`\lambda` -回报（12.1节）
:math:`G_{t:h}^{\lambda}`                             截断的，校正的:math:`\lambda` -回报（12.3节）
:math:`G_t^{\lambda s}, G_t^{\lambda a}`              估计状态或动作，价值校正的:math:`\lambda` -回报（12.8节）
\
:math:`p(s^{\prime}, r | s, a)`                       从状态 :math:`s` 与动作 :math:`a` 起, 以 :math:`r` 的奖励转移到状态 :math:`s^{\prime}` 的概率
:math:`p(s^{\prime} | s, a)`                          从状态 :math:`s` 起采取动作 :math:`a`, 转移到状态 :math:`s^{\prime}` 的概率
:math:`r(s, a)`                                       动作 :math:`a` 后状态 :math:`s` 的预期即时奖励
:math:`r(s, a, s^{\prime})`                           动作 :math:`a` 下从状态 :math:`s` 到状态 :math:`s^{\prime}` 的转移的预期即时奖励
\
:math:`v_\pi(s)`                                      在策略 :math:`\pi` 下状态 :math:`s` 的价值(预期回报)
:math:`v_*(s)`                                        在最优策略下状态 :math:`s` 的价值
:math:`q_\pi(s, a)`                                   在策略 :math:`\pi` 下, 在状态 :math:`s` 中采取动作 :math:`a` 的价值
:math:`q_*(s, a)`                                     在最优策略下, 在状态 :math:`s` 中采取动作 :math:`a` 的价值
\
:math:`V, V_t`                                        状态价值函数 :math:`v_\pi` 或 :math:`v_*` 的表格估计值
:math:`Q, Q_t`                                        动作价值函数 :math:`q_\pi` 或 :math:`q_*` 的表格估计值
:math:`\overline{V}_t(s)`                             预期的近似动作价值, 如 :math:`\overline{V}_{t}(s) \doteq \sum_{a} \pi(a | s) Q_{t}(s, a)`
:math:`U_t`                                           在时步 :math:`t` 估计的目标
:math:`\delta_t`                                      在时步 :math:`t` （随机变量）的时序差分（TD）误差（6.1节）
:math:`\delta_t^s, \delta_t^a`                        TD误差的状态和行动特定形式（第12.9节）
:math:`n`                                             在n步方法中，:math:`n` 是自举的步骤数
\
:math:`d`                                             维度── :math:`\mathbf{w}` 的分量数量
:math:`d^{\prime}`                                    备用维度── :math:`\mathrm{\theta}` 的分量数量
:math:`\mathbf{w}, \mathbf{w}_{t}`                    近似价值函数的权重 :math:`d` 维向量
:math:`w_{i}, w_{t, i}`                               第 :math:`i` 个可学习的权重向量的组成部分
:math:`\hat{v}(s, \mathbf{w})`                        给定权重向量 mathbf{w} 的状态 :math:`s` 的近似价值
:math:`v_{\mathbf{w}}(s)`                             :math:`\hat{v}(s, \mathbf{w})` 的备用表示
:math:`\hat{q}(s, a, \mathbf{w})`                     状态-动作对 :math:`s,a` 的近似价值，给定权重向量 :math:`\mathbf{w}`
:math:`\hat{\nabla} \hat{v}(s, \mathbf{w})`           关于 :math:`\mathbf{w}` 的 :math:`\hat{v}(s, \mathbf{w})` 的偏导数的列向量
:math:`\nabla \hat{q}(s, a, \mathbf{w})`              关于 :math:`\mathbf{w}` 的 :math:`\hat{q}(s, a, \mathbf{w})` 的偏导数的列向量
\
:math:`\mathbf{x}(s)`                                 在状态 :math:`s` 可见的特征向量
:math:`\mathbf{x}(s, a)`                              在状态 :math:`s` 采取动作 :math:`a` 时可见的特征向量
:math:`x_{i}(s), x_{i}(s, a)`                         向量 :math:`\mathbf{x}(s)`  或 :math:`\mathbf{x}(s, a)` 的分量
:math:`\mathbf{x}_{t}`                                :math:`\mathbf{x}(S_t)` 或 :math:`\mathbf{x}(S_t, A_t)` 的简写
:math:`\mathbf{W}^{\top} \mathbf{x}`                  向量的内积，:math:`\mathbf{w}^{\top} \mathbf{x} \doteq \sum_{i} w_{i} x_{i}`；比如 :math:`\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^{\top} \mathbf{x}(s)`
:math:`\mathbf{V}, \mathbf{V}_{t}`                    用于学习 :math:`\mathbf{w}` 的权重的次要 :math:`d` 维向量（第11章）
:math:`\mathbf{Z}_{t}`                                时步 :math:`t` 的资格迹 :math:`d` 维向量（第12章）
\
:math:`\mathbf{\theta}, \mathbf{\theta}_{t}`          目标策略的参数向量（第13章）
:math:`\pi(a | s, \mathbf{\theta})`                   在给定参数向量 :math:`\mathbf{\theta}` 的状态 :math:`s` 下采取动作 :math:`a` 的概率
:math:`\pi_{\mathbf{\theta}}`                         与参数 :math:`\mathbf{\theta}` 对应的策略
:math:`\nabla \pi(a | s, \mathbf{\theta})`            关于 :math:`\mathbf{\theta}` 的 :math:`\pi(a|s,\mathbf{\theta})` 的偏导数的列向量
:math:`\mathbf{J}(\mathbf{\theta})`                   策略的性能衡量指标
:math:`\nabla \mathbf{J}(\mathbf{\theta})`            关于 :math:`\mathbf{\theta}` 的 :math:`\mathbf{J}(\mathbf{\theta})` 的偏导数的列向量
:math:`h(s, a, \mathbf{\theta})`                      选择基于 :math:`\mathbf{\theta}` 的状态 :math:`s` 中的动作 :math:`a` 的优先指标
\
:math:`b(a|s)`                                        用于在了解目标策略 ;math:`\pi` 时选择动作的行为策略
:math:`b(s)`                                          基线函数 :math:`b : \mathcal{S} \mapsto \mathbb{R}` 用于策略梯度方法
:math:`b`                                             MDP或搜索树的分支因子
:math:`\rho_{t : h}`                                  时步 :math:`t` 到时步 :math:`h` 的重要采样比率（第5.5节）
:math:`\rho_{t}`                                      时间 :math:`t` 的重要采样比率，:math:`\rho_{t} \doteq \rho_{t:t}`
:math:`r(\pi)`                                        策略 :math:`\pi` 的平均回报（奖励率）（第10.3节）
:math:`\overline{R}_{t}`                              在时间 :math:`t` 估计 :math:`r(\pi)`
\
:math:`\mu(s)`                                        各状态的在策略分布（第9.2节）
:math:`\mathbf{\mu}`                                  所有 :math:`s\in\mathcal{S}` 的 :math:`\mu(s)` 的 :math:`|\mathcal{S}|` 维向量
:math:`\|v\|_{\mu}^{2}`                               价值函数 :math:`v` 的 :math:`\mu` 加权平方范数，即 :math:`\|v\|_{\mu}^{2} \doteq \sum_{s \in \mathcal{S}} \mu(s) v(s)^{2}`
:math:`\eta(s)`                                       每回合到状态 :math:`s` 的预期访问次数（第199页）
:math:`\Pi`                                           价值函数的投影算子（第268页）
:math:`B_{\pi}`                                       价值函数的Bellman算子（第11.4节）
\
:math:`\mathbf{A}`                                    :math:`d \times d` 矩阵 :math:`\mathbf{A} \doteq \mathbb{E}\left[\mathbf{x}_{t}\left(\mathbf{x}_{t}-\gamma \mathbf{x}_{t+1}\right)^{\top}\right]`
:math:`\mathbf{b}`                                    :math:`d` 维向量 :math:`\mathbf{b} \doteq \mathbb{E}\left[R_{t+1} \mathbf{x}_{t}\right]`
:math:`\mathbf{w}_{TD}`                               TD不动点 :math:`\mathbf{w}_{\mathrm{TD}} \doteq \mathbf{A}^{-1} \mathbf{b}`（:math:`d` 维向量，第9.4节）
:math:`\mathbf{I}`                                    单位矩阵
:math:`\mathbf{P}`                                    :math:`\pi` 下的 :math:`|\mathcal{S}|\times||mathcal{S}|` 状态转移概率矩阵
:math:`\mathbf{D}`                                    在对角线上具有 :math:`\mathbf{\mu}` 的 :math:`|\mathcal{S}|\times||mathcal{S}|` 对角矩阵
:math:`\mathbf{X}`                                    以 :math:`\mathbf{x}(s)` 为行的 :math:`|\mathcal{S}| \times d` 矩阵
\
:math:`\overline{\delta}_{\mathbf{w}}(s)`             状态 :math:`s` 下 :math:`v_{\mathbf{w}}` 的Bellman误差（预期TD误差）（第11.4节）
:math:`\overline{\delta}_{\mathbf{w}},\mathrm{BE}`    Bellman误差向量，包含分量 :math:`\overline{\delta}_{\mathbf{w}}(s)`
:math:`\overline{\mathrm{VE}}(\mathbf{w})`            均方值误差 :math:`\overline{\mathrm{VE}}(\mathbf{w}) \doteq\left\|v_{\mathbf{w}}-v_{\pi}\right\|_{\mu}^{2}` （第9.2节）
:math:`\overline{\mathrm{BE}}(\mathbf{w})`            均方Bellman误差 :math:`\overline{\mathrm{BE}}(\mathbf{w}) \doteq\|\overline{\delta}_{\mathbf{w}}\|_{\mu}^{2}`
:math:`\overline{\mathrm{PBE}}(\mathbf{w})`           均方投影Bellman误差 :math:`\overline{\mathrm{PBE}}(\mathbf{w}) \doteq\left\|\Pi \overline{\delta}_{\mathbf{w}}\right\|_{\mu}^{2}`
:math:`\overline{\mathrm{TDE}}(\mathbf{w})`           均方时序差分误差 :math:`\overline{\operatorname{TDE}}(\mathbf{w}) \doteq \mathbb{E}_{b}\left[\rho_{t} \delta_{t}^{2}\right]` （第11.5节）
:math:`\overline{\mathrm{RE}}(\mathbf{w})`            均方回报误差（第11.6节）
===================================================== ===============================================================================================
