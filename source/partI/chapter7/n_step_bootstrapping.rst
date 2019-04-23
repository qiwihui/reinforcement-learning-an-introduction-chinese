第7章 :math:`n` 步引导（Bootstrapping）方法
==============================================

在本章中，我们统一了蒙特卡罗（MC）方法和前两章中介绍的一步时序差分（TD）方法。MC方法和一步法TD方法都不是最好的。
在本章中，我们将介绍 *n步TD方法*，这些方法概括了两种方法，以便可以根据需要平滑地从一种方法转换到另一种方法，以满足特定任务的需求。
n步方法在一端采用MC方法，在另一端采用一步TD方法。最好的方法通常介于两个极端之间。

另一种看待n步方法的好处的方法是让它们摆脱时间步骤的独裁。
使用一步TD方法，同一时间步骤确定可以更改操作的频率以及完成引导的时间间隔。
在许多应用程序中，人们希望能够非常快速地更新动作以考虑已经发生变化的任何事情，
但是如果引导在一段时间内发生了重大且可识别的状态变化，则引导效果最佳。
使用一步TD方法，这些时间间隔是相同的，因此必须做出妥协。
n步方法使引导能够在多个步骤中发生，使我们摆脱单一时间步骤的暴政。

n步方法的概念通常用作 *资格跟踪* （第12章）算法思想的介绍，它可以同时实现多个时间间隔的引导。
在这里，我们反过来考虑n步引导的想法，将资格跟踪机制的处理推迟到以后。
这使我们能够更好地分离问题，在更简单的n步设置中处理尽可能多的问题。

像往常一样，我们首先考虑预测问题，然后考虑控制问题。
也就是说，我们首先考虑n步方法如何帮助预测作为固定策略的状态函数的回报（即，估计 :math:`v_\pi`）。
然后我们将想法扩展到行动价值和控制方法。


7.1 :math:`n` 步TD预测
---------------------------

蒙特卡罗和TD方法之间的方法空间是什么？考虑使用 :math:`\pi` 生成的样本回合估计v⇡。
蒙特卡罗方法基于从该状态到回合结束的观察到的奖励的整个序列来执行每个状态的更新。
另一方面，一步法TD方法的更新仅基于下一个奖励，一步之后从状态价值引导作为剩余奖励的代理。
然后，一种中间方法将基于中间数量的奖励执行更新：多于一个，但是在终止之前少于所有奖励。
例如，两步更新将基于前两个奖励以及两个步骤后的估计的状态值。
同样，我们可以进行三步更新，四步更新等。图7.1显示了 :math:`v_\pi` 的 *n步更新* 频谱的备份图，
左侧是一步TD更新，右侧是直到最后终止的蒙特卡罗更新。

.. figure:: images/figure-7.1.png

    **图7.1：** n步方法的备份图。这些方法组成从一步TD方法到蒙特卡罗方法的频谱范围。

使用n步更新的方法仍然是TD方法，因为它们仍然根据它从后来的估计中如何变化来改变先前的估计。
现在后来的估计不是一步之后，而是n步之后。时序差分在n个步骤上延伸的方法称为 *n步TD方法*。
前一章介绍的TD方法都使用了一步更新，这就是我们称之为一步TD方法的原因。

更正式地，考虑状态 :math:`S_t` 的估计值的更新，作为状态奖励序列的结果，
:math:`S_{t}, R_{t+1}, S_{t+1}, R_{t+2}, \ldots, R_{T}, S_{T}` （省略动作）。
我们知道在蒙特卡洛更新中，:math:`v_\pi(S_t)` 的估计值会在完全回报的方向上更新：

.. math::

    G_{t} \doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\cdots+\gamma^{T-t-1} R_{T}

其中T是回合的最后一个步骤。我们将此数量称为更新的 *目标*。
而在蒙特卡洛更新中目标是回报，在一步更新中，目标是第一个奖励加上下一个状态的折扣估计值，我们称之为 *一步回报*：

.. math::

    G_{t : t+1} \doteq R_{t+1}+\gamma V_{t}\left(S_{t+1}\right)

其中这里的 :math:`V_{t} : \mathcal{S} \rightarrow \mathbb{R}` 是 :math:`v_\pi` 在时刻t的估计值。
:math:`G_{t:t+1}` 的下标表示它是时间t的截断回报，使用奖励直到时间 :math:`t+1`，
折扣估计 :math:`\gamma V_{t}\left(S_{t+1}\right)` 代替其他项
:math:`\gamma R_{t+2}+\gamma^{2} R_{t+3}+\cdots+\gamma^{T-t-1} R_{T}` 的完全回报，
如前一章所述。我们现在的观点是，这个想法在经过两个步骤之后就像在一个步骤之后一样有意义。
两步更新的目标是两步回报：

.. math::

    G_{t : t+2} \doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2} V_{t+1}\left(S_{t+2}\right)

其中现在 :math:`\gamma^{2} V_{t+1}\left(S_{t+2}\right)` 纠正了项
:math:`\gamma^{2} R_{t+3}+\gamma^{3} R_{t+4}+\cdots+\gamma^{T-t-1} R_{T}` 的缺失。
同样，任意n步更新的目标是n步回报：

.. math::

    G_{t : t+n} \doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1} R_{t+n}+\gamma^{n} V_{t+n-1}\left(S_{t+n}\right)
    \tag{7.1}

而所有其他状态的值保持不变：对于所有 :math:`s \neq S_{t}`，:math:`V_{t+n}(s)=V_{t+n-1}(s)`。
我们将此算法称为 *n步TD*。 请注意，在每回合的前n-1个步骤中，根本不会进行任何更改。
为了弥补这一点，在回合结束后，终止后和开始下一集之前，会进行相同数量的额外更新。
完整的伪代码在下面的框中给出。

.. admonition:: :math:`n` 步TD(0)估计 :math:`V \approx v_\pi`
    :class: important

    输入：策略 :math:`\pi`

    算法参数：步长 :math:`\alpha \in (0,1]`，正整数 :math:`n`

    对 :math:`s \in \mathcal{S}`，任意初始化 :math:`V(s)`

    所有存储和访问操作（对于 :math:`S_t` 和 :math:`R_t`）都可以使其索引 :math:`mod n + 1`

    对每个回合循环：

        初始化并存储 :math:`S_0 \ne` 终点

        :math:`T \leftarrow \infty`

        对 :math:`t=0,1,2, \ldots` 循环：

            如果 :math:`t < T`，则：

                根据 :math:`\pi(\cdot|S_t)` 采取行动

                观察并将下一个奖励存储为 :math:`R_{t+1}`，将下一个状态存储为 :math:`S_{t+1}`

                如果 :math:`S_{t+1}` 是终点，则 :math:`T \leftarrow t+1`

            :math:`\tau \leftarrow t - n + 1` （:math:`\tau` 是状态估计正在更新的时间）

            如果 :math:`\tau geq 0`：

                :math:`G \leftarrow \sum_{i=\tau+1}^{\min (\tau+n, T)} \gamma^{i-\tau-1} R_{i}`

                如果 :math:`\tau + n < T`， 则 :math:`G \leftarrow G+\gamma^{n} V\left(S_{\tau+n}\right)`

                :math:`V\left(S_{\tau}\right) \leftarrow V\left(S_{\tau}\right)+\alpha\left[G-V\left(S_{\tau}\right)\right]`  :math:`\quad\quad\quad`   :math:`\left(G_{\tau : \tau+n}\right)`

        直到 :math:`\tau = T - 1`

*练习7.1* 在第6章中，我们注意到如果值估计值不是逐步变化，则蒙特卡罗误差可写为TD误差之和（6.6）。
证明（7.2）中使用的n步误差也可以写为推导早期结果的总和TD误差（如果值估计没有改变则再次）。

*练习7.2（编程）* 使用n步法，值估计 *会* 逐步变化，因此使用TD误差之和（参见上一个练习）代替（7.2）中的误差的算法实际上的算法略有不同。
它会是一个更好的算法还是更糟糕的算法？设计和编写一个小实验来根据经验回答这个问题。


n步回报使用价值函数 :math:`V_{t+n-1}` 来校正超出 :math:`R_{t+n}` 的缺失奖励。
n步回报的一个重要特性是，在最恶劣的意义上，它们的期望被保证是比 :math:`V_{t+n-1}` 更好的 :math:`v_\pi` 估计。
也就是说，对于所有 :math:`n \ge 1`，预期的n步回报的最差误差保证小于或等于 :math:`V_{t+n-1}` 下最差误差的n倍：

.. math::

    \max _{s}\left|\mathbb{E}_{\pi}\left[G_{t : t+n} | S_{t}=s\right]-v_{\pi}(s)\right| \leq \gamma^{n} \max _{s}\left|V_{t+n-1}(s)-v_{\pi}(s)\right|
    \tag{7.3}

这称为 *n步回报的误差减少属性*。由于误差减少属性，可以正式显示所有n步TD方法在适当的技术条件下收敛到正确的预测。
因此，n步TD方法形成一系列声音（sound）方法，一步法TD方法和蒙特卡罗方法作为极端其情况。

**例7.1：随机行走的n步TD方法** 考虑在例6.2中描述的5个状态随机游走任务中使用n步TD方法。
假设第一回合直接从中心状态 **C** 向右进行，通过 **D** 和 **E**，然后在右边终止，回报为1。
回想所有状态的估计值从中间值开始，:math:`V(s)=0.5`。
作为这种经验的结果，一步法只改变最后一个状态 :math:`V(E)` 的估计值，该值将增加到1，观察到的返回值。
另一方面，两步法将增加终止前两个状态的值： :math:`V(D)` 和 :math:`V(E)` 都将增加到1。
三步法或任何n步法（对 :math:`n \ge 2`），将所有三个访问状态的值增加到1，全部增加相同的量。

哪个n的值更好？图7.2显示了对较大随机行走过程进行简单经验测试的结果，其中包含19个状态而不是5个
（从左侧离开回报为 :math:`-1`，所有值都初始化为0），我们在本章中将其用作运行示例。
结果显示了涉及大范围n和 :math:`\alpha` 的值的n步TD方法。
垂直轴上显示的每个参数设置的性能度量是19个状态的回合结束时，预测与其真实值之间的均方误差的平方根，
然后在整个实验的前10回合和100次重复中取平均值（所有参数设置都使用相同的行走集合）。
请注意，中间值为n的方法效果最好。
这说明了TD和蒙特卡罗方法对n步方法的推广能够比两种极端方法中的任何一种方法表现更好。

.. figure:: images/figure-7.2.png

    **图7.2：** 对于19个状态的随机行走任务的不同n值，n步TD方法的性能是 :math:`\alpha` 的函数（例7.1）。

*练习7.3* 为什么你认为在本章的例子中使用了更大的随机游走任务（19个州而不是5个）？
较小的步行会将优势转移到不同的n值吗？在较大的步行中左侧结果从0变为 :math:`-1` 是怎么发生的？
你认为这对n的最佳价值有任何不同吗？


7.2 :math:`n` 步Sarsa
----------------------------

如何使用n步方法不仅用于预测，还用于控制？在本节中，我们将展示如何以简单的方式将n步方法与Sarsa结合以产生一种策略上的TD控制方法。
Sarsa的n步版本我们称之为n步Sarsa，而前一章中提到的原始版本我们称之为 *一步Sarsa* 或 *Sarsa(0)*。

主要思想是简单地切换动作状态（状态-动作对），然后使用 :math:`\varepsilon` -贪婪策略。
n步Sarsa的备份图（如图7.3所示），就像n步TD一样（图7.1） ），是交替状态和动作的字符串，
除了Sarsa所有都以动作而不是状态开始和结束。我们根据估计的动作值重新定义n步回报（更新目标）：

.. math::

    G_{t : t+n} \doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1} R_{t+n}+\gamma^{n} Q_{t+n-1}\left(S_{t+n}, A_{t+n}\right), \quad n \geq 1,0 \leq t<T-n
    \tag{7.4}

如果 :math:`t+n \geq T`，则 :math:`G_{t : t+n} \doteq G_{t}`。那么自然算法就是

.. math::

    Q_{t+n}\left(S_{t}, A_{t}\right) \doteq Q_{t+n-1}\left(S_{t}, A_{t}\right)+\alpha\left[G_{t : t+n}-Q_{t+n-1}\left(S_{t}, A_{t}\right)\right], \quad 0 \leq t<T
    \tag{7.5}

而所有其他状态的值保持不变：:math:`Q_{t+n}(s, a)=Q_{t+n-1}(s, a)`，
对于所有 :math:`s, a` 使得 :math:`s \ne S_t` 或 :math:`a \ne A_t`。
这是我们称之为 *n步Sarsa* 的算法。
伪代码显示在下面的框中，图7.4给出了与一步法相比可以加速学习的原因示例。

.. figure:: images/figure-7.3.png

    **图7.3：** 状态-动作值的n步方法频谱的备份图。
    它们的范围从Sarsa(0)的一步更新到蒙特卡罗方法的直到终止更新。
    在两者之间是n步更新，基于实际奖励的n个步骤和第n个下一个状态-动作对的估计值，都被适当地折扣。
    最右边是n步预期Sarsa的备份图。

.. admonition:: :math:`n` 步Sarsa估计 :math:`Q \approx q_*` 或者 :math:`q_\pi`
    :class: important

    对所有 :math:`s\in\mathcal(S)`，:math:`a\in\mathcal(A)`，任意初始化 :math:`Q(s,a)`

    初始化 :math:`\pi` 关于 :math:`Q` 或固定的给定策略为 :math:`\varepsilon` -贪婪

    算法参数：步长 :math:`\alpha \in (0,1]`，小 :math:`\varepsilon > 0`，正整数 :math:`n`

    所有存储和访问操作（对于 :math:`S_t`，:math:`A_t` 和 :math:`R_t`）都可以使其索引 :math:`mod n + 1`

    对每个回合循环：

        初始化并存储 :math:`S_0 \ne` 终点

        选择并存储动作 :math:`A_{0} \sim \pi\left(\cdot | S_{0}\right)`

        :math:`T \leftarrow \infty`

        对 :math:`t=0,1,2, \ldots` 循环：

            如果 :math:`t < T`，则：

                采取行动 :math:`A_t`

                观察并将下一个奖励存储为 :math:`R_{t+1}`，将下一个状态存储为 :math:`S_{t+1}`

                如果 :math:`S_{t+1}` 是终点，则

                    :math:`T \leftarrow t+1`

                否则：

                    选择并存储动作 :math:`A_{t+1} \sim \pi\left(\cdot | S_{t=1}\right)`

            :math:`\tau \leftarrow t - n + 1` （:math:`\tau` 是状态估计正在更新的时间）

            如果 :math:`\tau \geq 0`：

                :math:`G \leftarrow \sum_{i=\tau+1}^{\min (\tau+n, T)} \gamma^{i-\tau-1} R_{i}`

                如果 :math:`\tau + n < T`， 则 :math:`G \leftarrow G+\gamma^{n} Q\left(S_{\tau+n}, A_{\tau+n}\right)` :math:`\quad\quad\quad` :math:`\left(G_{\tau : \tau+n}\right)`

                :math:`Q\left(S_{\tau}, A_{\tau}\right) \leftarrow Q\left(S_{\tau}, A_{\tau}\right)+\alpha\left[G-Q\left(S_{\tau}, A_{\tau}\right)\right]`

                如果 :math:`\pi` 正在被学习，那么确保 :math:`\pi\left(\cdot | S_{\tau}\right)` 是关于 :math:`Q` :math:`\varepsilon` -贪婪

        直到 :math:`\tau = T - 1`

.. figure:: images/figure-7.4.png

    **图7.4：** 由于使用n步方法而导致的策略学习加速的网格世界示例。
    第一个面板显示了一个个体在单个回合中所采用的路径，在一个高回报的位置结束，用 **G** 标记。
    在这个例子中，这些值最初都是0，除了 **G** 的正奖励，所有奖励都是零。
    其他两个面板中的箭头显示了通过一步Sarsa方法和n步Sarsa方法通过该路径加强了的动作价值。
    一步Sarsa法只强化导致高回报的动作序列的最后一个动作，而n步法强化序列的最后n个动作，
    因此从一个回合中学习了更多。

*练习7.4* 证明Sarsa（7.4）的n步回报可以完全按照新的TD误差写成如下：

.. math::

    G_{t : t+n}=Q_{t-1}\left(S_{t}, A_{t}\right)+\sum_{k=t}^{\min (t+n, T)-1} \gamma^{k-t}\left[R_{k+1}+\gamma Q_{k}\left(S_{k+1}, A_{k+1}\right)-Q_{k-1}\left(S_{k}, A_{k}\right)\right]
    \tag{7.6}

那么预期的Sarsa呢？预期Sarsa的n步版本的备份图显示在图7.3的最右侧。
它由一系列样本动作和状态的线性组成，就像在n步Sarsa中一样，
除了它的最后一个元素一如既往是在 :math:`\pi` 下的概率加权的所有动作可能性的分支。
该算法可以用与n步Sarsa（上面）相同的等式来描述，除了将n步回报重新定义为

.. math::

    G_{t : t+n} \doteq R_{t+1}+\cdots+\gamma^{n-1} R_{t+n}+\gamma^{n} \overline{V}_{t+n-1}\left(S_{t+n}\right), \quad t+n<T
    \tag{7.7}

对 :math:`t+n \geq T`，:math:`G_{t : t+n} \doteq G_{t}`。
其中 :math:`\overline{V}_{t}(s)` 是状态s的 *预期近似值*，使用目标策略下时间 :math:`t` 的估计行动值：

.. math::

    \overline{V}_{t}(s) \doteq \sum_{a} \pi(a | s) Q_{t}(s, a), \quad \text { 对所有 } s \in \mathcal{S}
    \tag{7.8}

在本书的其余部分中，使用预期近似值来开发许多动作价值方法。如果s是终点，则其预期近似值被定义为0。


7.3 :math:`n` 步离策略学习
------------------------------

回想一下，离策略学习是学习一项策略的价值函数 :math:`\pi`，同时遵循另一项策略 :math:`b`。
通常，:math:`\pi` 是当前行动-价值-函数估计的贪婪策略，
而 :math:`b` 是一种更具探索性的策略，也许是 :math:`\varepsilon` -贪婪。
为了使用 :math:`b` 中的数据，我们必须考虑到两种策略之间的差异，使用它们采取所采取行动的相对概率（参见第5.5节）。
在n步法中，回报是在n个步骤上构建的，因此我们对这些n个动作的相对概率感兴趣。
例如，在构建n步TD的简单离策略版本，时间 :math:`t` 的更新（实际上在时间 :math:`t+n` 进行）
可以简单地通过 :math:`\rho_{t : t}+n-1` 加权：

.. math::

    V_{t+n}\left(S_{t}\right) \doteq V_{t+n-1}\left(S_{t}\right)+\alpha \rho_{t : t+n-1}\left[G_{t : t+n}-V_{t+n-1}\left(S_{t}\right)\right], \quad 0 \leq t<T
    \tag{7.9}

其中 :math:`\rho_{t : t}+n-1` 称为 *重要性采样比率*，
是在两个策略下从 :math:`A_t` 到 :math:`A_{t+n-1}` 采取 :math:`n` 个动作的相对概率（参见方程5.3）：

.. math::

    \rho_{t : h} \doteq \prod_{k=t}^{\min (h, T-1)} \frac{\pi\left(A_{k} | S_{k}\right)}{b\left(A_{k} | S_{k}\right)}
    \tag{7.10}

例如，如果任何一个动作永远不会被 :math:`\pi` 占用（即 :math:`\pi\left(A_{k} | S_{k}\right)=0`），
那么n步回报应该被赋予零权重并被完全忽略。
另一方面，如果偶然采取行动，:math:`\pi` 将采取比 :math:`\b` 更大的概率，那么这将增加否则将给予回报的权重。
这是有道理的，因为该动作是 :math:`\pi` 的特征（因此我们想要了解它），但很少被 :math:`\b` 选择，因此很少出现在数据中。
为了弥补这一点，我们必须在它发生时增加权重（over-weight）。
请注意，如果这两个策略实际上是相同的（在策略情况下），则重要性采样比率始终为1。
因此，我们的新更新（7.9）概括并可以完全取代我们之前的n步TD更新。
同样，我们之前的n步Sarsa更新可以完全被简单的离策略形式所取代：

.. math::

    Q_{t+n}\left(S_{t}, A_{t}\right) \doteq Q_{t+n-1}\left(S_{t}, A_{t}\right)+\alpha \rho_{t+1 : t+n}\left[G_{t : t+n}-Q_{t+n-1}\left(S_{t}, A_{t}\right)\right]
    \tag{7.11}

对 :math:`0 \leq t<T`。注意，这里的重要性采样比率比n步TD（7.9）晚一步开始和结束。
这是因为我们正在更新状态-动作对。我们不必关心我们选择行动的可能性；
现在我们选择了它，我们希望从发生的事情中充分学习，只对后续行动进行重要性采样。
完整算法的伪代码如下面的框所示。

.. admonition:: 离策略 :math:`n` 步Sarsa估计 :math:`Q \approx q_*` 或者 :math:`q_\pi`
    :class: important

    输入：对所有 :math:`s\in\mathcal(S)`，一个任意的行为策略 :math:`b` 使得 :math:`b(a | s)>0`

    对所有 :math:`s\in\mathcal(S)`，:math:`a\in\mathcal(A)`，任意初始化 :math:`Q(s,a)`

    初始化 :math:`\pi` 关于 :math:`Q` 或固定的给定策略为贪婪

    算法参数：步长 :math:`\alpha \in (0,1]`，正整数 :math:`n`

    所有存储和访问操作（对于 :math:`S_t`，:math:`A_t` 和 :math:`R_t`）都可以使其索引 :math:`mod n + 1`

    对每个回合循环：

        初始化并存储 :math:`S_0 \ne` 终点

        选择并存储动作 :math:`A_{0} \sim \pi\left(\cdot | S_{0}\right)`

        :math:`T \leftarrow \infty`

        对 :math:`t=0,1,2, \ldots` 循环：

            如果 :math:`t < T`，则：

                采取行动 :math:`A_t`

                观察并将下一个奖励存储为 :math:`R_{t+1}`，将下一个状态存储为 :math:`S_{t+1}`

                如果 :math:`S_{t+1}` 是终点，则

                    :math:`T \leftarrow t+1`

                否则：

                    选择并存储动作 :math:`A_{t+1} \sim \pi\left(\cdot | S_{t=1}\right)`

            :math:`\tau \leftarrow t - n + 1` （:math:`\tau` 是状态估计正在更新的时间）

            如果 :math:`\tau \geq 0`：

                :math:`\rho \leftarrow \prod_{i=\tau+1}^{\min (\tau+n-1, T-1)} \frac{\pi\left(A_{i} | S_{i}\right)}{b\left(A_{i} | S_{i}\right)}` :math:`\quad\quad\quad` :math:`\left(\rho_{\tau}+1 : t+n-1\right)`

                :math:`G \leftarrow \sum_{i=\tau+1}^{\min (\tau+n, T)} \gamma^{i-\tau-1} R_{i}`

                如果 :math:`\tau + n < T`， 则 :math:`G \leftarrow G+\gamma^{n} Q\left(S_{\tau+n}, A_{\tau+n}\right)` :math:`\quad\quad\quad` :math:`\left(G_{\tau : \tau+n}\right)`

                :math:`Q\left(S_{\tau}, A_{\tau}\right) \leftarrow Q\left(S_{\tau}, A_{\tau}\right)+\alpha \rho\left[G-Q\left(S_{\tau}, A_{\tau}\right)\right]`

                如果 :math:`\pi` 正在被学习，那么确保 :math:`\pi\left(\cdot | S_{\tau}\right)` 是关于 :math:`Q` 贪婪

        直到 :math:`\tau = T - 1`

n步预期Sarsa的离策略版本将对n步Sarsa使用与上述相同的更新，除了重要性采样比率将减少一个因素。
也就是说，上面的等式将使用 :math:`\rho_{t}+1 : t+n-1` 而不是 :math:`\rho_{t}+1 : t+n`，
当然它将使用n步回报（7.7）的预期Sarsa版本。这是因为在预期的Sarsa中，在最后一个状态中考虑了所有可能的行为；
实际采取的那个没有效果，也没有必要纠正。


7.4 \*具有控制变量的每个决策（per-decision）方法
-------------------------------------------------

上一节中介绍的多步离策略方法简单且概念清晰，但可能不是最有效的方法。
更复杂的方法将使用每个决策重要性采样的想法，如第5.9节中介绍的那样。
要理解这种方法，首先要注意普通的n步回报（7.1），就像所有回报一样，可以递归写入。
对于以水平线 :math:`h` 结束的n个步骤，n步回报可以写成

.. math::

    G_{t : h}=R_{t+1}+\gamma G_{t+1 : h}, \quad t<h<T
    \tag{7.12}

其中 :math:`G_{h : h} \doteq V_{h-1}\left(S_{h}\right)`。
（回想一下，此回报在时间 :math:`h` 使用，先前表示为 :math:`t+n`。）
现在考虑遵循与目标策略不同的行为策略 :math:`b` 的效果。
所有得到的经验，包括第一个奖励 :math:`R_{t+1}` 和下一个状态 :math:`S_{t+1}`，
必须通过时间 :math:`t` 的重要性采样比率加权，
:math:`\rho_{t}=\frac{\pi\left(A_{t} | S_{t}\right)}{b\left(A_{t} | S_{t}\right)}`。
你可能只想对上面的方程的右侧进行加权，但你可以做得更好。
假设时间 :math:`t` 的动作永远不会被 :math:`\pi` 选择，因此 :math:`\rho_{t}` 为零。
然后，简单的加权将导致n步回报为零，这可能导致当它用作目标时的高方差。
相反，在这种更复杂的方法中，人们使用在水平线 :math:`h` 结束的n步回报的替代的离策略定义，即

.. math::

    G_{t : h} \doteq \rho_{t}\left(R_{t+1}+\gamma G_{t+1 : h}\right)+\left(1-\rho_{t}\right) V_{h-1}\left(S_{t}\right), \quad t<h<T
    \tag{7.13}

再次 :math:`G_{h : h} \doteq V_{h-1}\left(S_{h}\right)`。
在这种方法中，如果 :math:`\rho_{t}` 为零，则目标与估计相同并且不会导致变化，而不是目标为零并导致估计收缩。
重要性采样比率为零意味着我们应该忽略样本，因此保持估计不变似乎是合适的。
（7.13）中的第二个附加项称为 *控制变量* （由于不明原因）。
请注意，控制变量不会更改预期的更新；重要性采样比率具有期望值1（第5.9节）并且与估计值不相关，因此控制变量的期望值为零。
另请注意，离策略定义（7.13）是对n步回报（7.1）的早期在策略定义的严格概括，
因为这两者在在策略情况下是相同的，其中 :math:`\rho_{t}` 永远是1。

对于传统的n步方法，与（7.13）结合使用的学习规则是n步TD更新（7.2），
除了嵌入在回报中的那些之外，它没有明确的重要性采样比率。

*练习7.5* 编写上述离策略状态价值预测算法的伪代码。

对于动作价值，n步回报的离策略定义略有不同，因为第一个动作在重要性采样中不起作用。
第一个行动是正在学习的行动；在目标政策下是否不太可能甚至不可能无关紧要──它已被采纳，现在必须对其后的奖励和状态给予全部单位权重。
重要性采样仅适用于其后的操作。

首先请注意，对于动作价值，n步在策略回报结束于水平线 :math:`h`，期望形式（7.7）可以像（7.12）一样递归写入，
除了对于动作价值，
递归以 :math:`G_{h : h} \doteq \overline{V}_{h-1}\left(S_{h}\right)` 结尾，如（7.8）所示。
具有控制变量的离策略形式是

.. math::

    \begin{aligned}
    G_{t : h} & \doteq R_{t+1}+\gamma\left(\rho_{t+1} G_{t+1 : h}+\overline{V}_{h-1}\left(S_{t+1}\right)-\rho_{t+1} Q_{h-1}\left(S_{t+1}, A_{t+1}\right)\right) \\
    &=R_{t+1}+\gamma \rho_{t+1}\left(G_{t+1 : h}-Q_{h-1}\left(S_{t+1}, A_{t+1}\right)\right)+\gamma \overline{V}_{h-1}\left(S_{t+1}\right), t<h \leq T & \text{(7.14)}
    \end{aligned}

如果 :math:`h<T`，则递归以 :math:`G_{h : h} \doteq Q_{h-1}(S_{h}, A_{h})` 结束，
而如果是 :math:`t \geq T`，递归以 :math:`G_{T-1 : h} \doteq R_{T}` 结束。
得到的预测算法（在与（7.5）组合之后）类似于预期的Sarsa。

*练习7.6* 证明上述方程中的控制变量不会改变回报的预期值。

\**练习7.7* 为上面描述的离策略动作价值预测算法写下伪代码。在达到水平线或回合结束时，要特别注意递归的终止条件。

*练习7.8* 如果近似状态值函数没有改变，则表明n步回报（7.13）的通用（离策略）版本仍然可以作为基于状态的TD误差（6.5）的精确和紧凑地书写。

*练习7.9* 对离政策n步回报（7.14）和预期Sarsa TD误差（公式6.9中括号内的数量）的动作版本重复上述练习。

*练习7.10（编程）* 设计一个小的离策略预测问题并用它来表明使用（7.13）和（7.2）的离策略学习算法比使用（7.1）和（7.9）的更简单算法更有效。

我们在本节，上一节和第5章中使用的重要性采样实现了声音（sound）离策略学习，但也导致了高方差更新，
迫使使用小步长参数，从而不可避免地导致学习变慢。离策略训练比在策略训练慢 - 毕竟，数据与所学内容的相关性较低。
然而，可能也可以改进这些方法。对照变量是减少方差的一种方法。另一个是快速调整步长到观察到的方差，
如Autostep方法（Mahmood，Sutton，Degris和Pilarski，2012）。
另一种有希望的方法是由Tian（在准备中）扩展到TD的Karampatziakis和Langford（2010）的不变更新。
Mahmood（2017；Mahmood和Sutton，2015）的使用技巧也可能是解决方案的一部分。
在下一节中，我们将考虑一种不使用重要性采样的离策略学习方法。


7.5 无重要性采样的离策略学习：n步树备份算法
---------------------------------------------

没有重要性采样，是否可以进行离策略学习？
第6章中的Q-learning和预期的Sarsa针对一步式案例进行了此操作，但是是否有相应的多步算法？
在本节中，我们将介绍一种称为 *树备份算法* 的n步方法。

.. figure:: images/the-3-step-tree-backup-update.png
    :align: right
    :width: 120px

右侧显示的3步树备份备份图表明了该算法的思想。
沿中央脊柱向下并在图中标记的是三个样本状态和奖励，以及两个样本动作。
这些是表示在初始状态-动作对 :math:`S_t`，:math:`A_t` 之后发生的事件的随机变量。
悬挂在每个状态的两侧是 *未* 选择的行动。（对于最后一个状态，所有操作都被认为尚未被选中。）
因为我们没有未选择操作的样本数据，所以我们引导并使用它们的值估计来形成更新目标。
这略微扩展了备份图的概念。到目前为止，我们总是将图表顶部节点的估计值更新为一个目标，
其中包括沿途的奖励（适当折扣）和底部节点的估计值。
在树形备份更新中，目标包括所有这些内容以及悬挂在各个层面的悬空动作节点的估计值。
这就是它被称为 *树形备份* 更新的原因；它是整个估计动作值树的更新。

更确切地说，更新来自树的 *叶节点* 的估计动作值。内部的动作节点对应于所采取的实际动作，不参与更新。
每个叶节点对目标的贡献与其在目标策略 :math:`\pi` 下发生的概率成比例。
因此，每个第一级动作 :math:`a` 的权重为 :math:`\pi\left(a | S_{t+1}\right)`，
除了实际采取的动作 :math:`A_{t+1}`，根本没有贡献。
它的概率 :math:`\pi\left(A_{t+1} | S_{t+1}\right)` 用于加权所有第二级动作值。
因此，每个未选择的第二级动作 :math:`a^{\prime}` 贡献权重
:math:`\pi\left(A_{t+1} | S_{t+1}\right) \pi\left(a^{\prime} | S_{t+2}\right)`。
每个第三级动作都有权重 :math:`\pi\left(A_{t+1} | S_{t+1}\right) \pi\left(A_{t+2} | S_{t+2}\right) \pi\left(a^{\prime \prime} | S_{t+3}\right)`，依此类推。
就好像图中动作节点的每个箭头都被动作在目标策略下被选中的概率加权，如果动作下面有一棵树，则该权重适用于树中的所有叶节点。

我们可以将3步树备份更新视为由6个半步骤组成，在从动作到后续状态的样本半步骤之间交替，
以及根据策略从该状态考虑所有可能的动作及其概率的预期半步骤。

现在让我们开发n步树备份算法的详细方程。一步回报（目标）与预期的Sarsa相同，

.. math::

    G_{t : t+1} \doteq R_{t+1}+\gamma \sum_{a} \pi\left(a | S_{t+1}\right) Q_{t}\left(S_{t+1}, a\right)
    \tag{7.15}

对 :math:`t<T-1`。两步树备份回报为

.. math::

    \begin{align}
    G_{t : t+2} &\doteq R_{t+1}+\gamma \sum_{a \neq A_{t+1}} \pi\left(a | S_{t+1}\right) Q_{t+1}\left(S_{t+1}, a\right) \\
    & \quad +\gamma \pi\left(A_{t+1} | S_{t+1}\right)\left(R_{t+2}+\gamma \sum_{a} \pi\left(a | S_{t+2}\right) Q_{t+1}\left(S_{t+2}, a\right)\right) \\
    &=R_{t+1}+\gamma \sum_{a \neq A_{t+1}} \pi\left(a | S_{t+1}\right) Q_{t+1}\left(S_{t+1}, a\right)+\gamma \pi\left(A_{t+1} | S_{t+1}\right) G_{t+1 : t+2}
    \end{align}

对 :math:`t<T-2`。后一种形式表明了树备份n步回报的一般递归定义：

.. math::

    G_{t : t+n} \doteq R_{t+1}+\gamma \sum_{a \neq A_{t+1}} \pi\left(a | S_{t+1}\right) Q_{t+n-1}\left(S_{t+1}, a\right)+\gamma \pi\left(A_{t+1} | S_{t+1}\right) G_{t+1 : t+n}
    \tag{7.16}

对 :math:`0 \leq t<T`，而所有其他状态-动作对的值保持不变：
:math:`Q_{t+n}(s, a)=Q_{t+n-1}(s, a)`，
对于所有 :math:`s,a` 使得 :math:`s\ne S_t` 或 :math:`a\ne A_t`。
该算法的伪代码显示在下面的框中。

.. admonition:: n步树备份估计 :math:`Q \approx q_*` 或者 :math:`q_\pi`
    :class: important

    对所有 :math:`s\in\mathcal(S)`，:math:`a\in\mathcal(A)`，任意初始化 :math:`Q(s,a)`

    初始化 :math:`\pi` 关于 :math:`Q` 或固定的给定策略为贪婪

    算法参数：步长 :math:`\alpha \in (0,1]`，正整数 :math:`n`

    所有存储和访问操作都可以使其索引 :math:`mod n + 1`

    对每个回合循环：

        初始化并存储 :math:`S_0 \ne` 终点

        选择并存储动作 :math:`A_{0} \sim \pi\left(\cdot | S_{0}\right)`

        任意选择动作 :math:`A_{0}` 作为 :math:`S_{0}` 的函数；存储 :math:`A_{0}`

        :math:`T \leftarrow \infty`

        对 :math:`t=0,1,2, \ldots` 循环：

            如果 :math:`t < T`，则：

                采取行动 :math:`A_t`，观察并将下一个奖励存储为 :math:`R_{t+1}`，将下一个状态存储为 :math:`S_{t+1}`

                如果 :math:`S_{t+1}` 是终点，则

                    :math:`T \leftarrow t+1`

                否则：

                    任意选择动作 :math:`A_{t+1}` 作为 :math:`S_{t+1}` 的函数；存储 :math:`A_{t+1}`

            :math:`\tau \leftarrow t + 1 -n` （:math:`\tau` 是状态估计正在更新的时间）

            如果 :math:`\tau \geq 0`：

                如果 :math:`t+1 \geq T`:

                    :math:`G \leftarrow R_{T}`

                否则：

                    :math:`G \leftarrow R_{t+1}+\gamma \sum_{a} \pi\left(a | S_{t+1}\right) Q\left(S_{t+1}, a\right)`

                对 :math:`k=\min (t, T-1)` 下降到 :math:`\tau + 1` 循环：

                    :math:`G \leftarrow R_{k}+\gamma \sum_{a \neq A_{k}} \pi\left(a | S_{k}\right) Q\left(S_{k}, a\right)+\gamma \pi\left(A_{k} | S_{k}\right) G`

                如果 :math:`\pi` 正在被学习，那么确保 :math:`\pi\left(\cdot | S_{\tau}\right)` 是关于 :math:`Q` 贪婪

        直到 :math:`\tau = T - 1`

*练习7.11* 显示如果近似动作值不变，则树备份返回（7.16）可以写为基于预期的TD误差的总和：

.. math::

    G_{t : t+n}=Q\left(S_{t}, A_{t}\right)+\sum_{k=t}^{\min (t+n-1, T-1)} \delta_{k} \prod_{i=t+1}^{k} \gamma \pi\left(A_{i} | S_{i}\right)

其中 :math:`\delta_{t} \doteq R_{t+1}+\gamma \overline{V}_{t}\left(S_{t+1}\right)-Q\left(S_{t}, A_{t}\right)`
并且 :math:`\overline{V}_{t}` 由（7.8）给出。


7.6 \*统一算法：n步 :math:`Q(\sigma)`
--------------------------------------

到目前为止，在本章中我们已经考虑了三种不同类型的动作价值算法，对应于图7.5中所示的前三个备份图。
n步Sarsa具有所有样本转换，树备份算法将所有状态-动作转换完全分支而不进行采样，
n步预期Sarsa具有除最后一个状态到动作之外的所有样本转换，这是完全分支，具有预期值。
这些算法在多大程度上可以统一？

图7.5中的第四个备份图表明了统一的一个想法。这是一个想法，
人们可以逐步决定是否想要将动作作为样本，如在Sarsa中，或者考虑对所有动作的期望，如树备份更新。
然后，如果一个人总是选择采样，那么就会获得Sarsa，而如果一个人选择永远不会采样，那么就会得到树备份算法。
预期的Sarsa就是这样一种情况，即除了最后一步之外，所有步骤都选择采样。

.. figure:: images/figure-7.5.png

    **图7.5：** 本章到目前为止考虑的三种n步动作值更新的备份图（4步案例）以及将它们统一起来的第四种更新的备份图。
    :math:`\rho` 表示在离策略情况下需要重要采样的一半过渡。
    第四种更新通过逐个状态地选择是否采样（:math:`\sigma_{t}=1`）或不采样（:math:`\sigma_{t}=0`）来统一所有其他更新。

当然，如图中的最后一个图所示，还有许多其他可能性。为了进一步提高可能性，我们可以考虑采样和期望之间的连续变化。
令 :math:`\sigma_{t} \in[0,1]` 表示步骤 :math:`t` 的采样程度，
其中 :math:`\sigma=1` 表示完全采样，:math:`\sigma=0` 表示没有采样的纯期望。
随机变量 :math:`\sigma_t` 可以在时间 :math:`t` 被设置为状态，动作或状态 - 动作对的函数。
我们将这个提议的新算法称为n步 :math:`Q(\sigma)`。

现在让我们开发n步 :math:`Q(\sigma)` 的方程。
首先，我们根据水平线 :math:`h+n` 写出树备份n步回报（7.16），然后根据预期的近似值 :math:`\overline{V}` （7.8）：

.. math::

    \begin{aligned}
    G_{t : h} &=R_{t+1}+\gamma \sum_{a \neq A_{t+1}} \pi\left(a | S_{t+1}\right) Q_{h-1}\left(S_{t+1}, a\right)+\gamma \pi\left(A_{t+1} | S_{t+1}\right) G_{t+1 : h} \\
    &=R_{t+1}+\gamma \overline{V}_{h-1}\left(S_{t+1}\right)-\gamma \pi\left(A_{t+1} | S_{t+1}\right) Q_{h-1}\left(S_{t+1}, A_{t+1}\right)+\gamma \pi\left(A_{t+1} | S_{t+1}\right) G_{t+1 : h} \\
    &=R_{t+1}+\gamma \pi\left(A_{t+1} | S_{t+1}\right)\left(G_{t+1 : h}-Q_{h-1}\left(S_{t+1}, A_{t+1}\right)\right)+\gamma \overline{V}_{h-1}\left(S_{t+1}\right)
    \end{aligned}

之后它与控制变量（7.14）的Sarsa的n步回报完全相同，
除了动作概率 :math:`\pi\left(A_{t+1} | S_{t+1}\right)` 代替重要性采样率 :math:`\rho_{t+1}`。
对于 :math:`Q(\sigma)`，我们在这两种情况之间线性滑动：

.. math::

    \begin{aligned}
    G_{t : h} \doteq R_{t+1} &+\gamma\left(\sigma_{t+1} \rho_{t+1}+\left(1-\sigma_{t+1}\right) \pi\left(A_{t+1} | S_{t+1}\right)\right)\left(G_{t+1 : h}-Q_{h-1}\left(S_{t+1}, A_{t+1}\right)\right) \\
    &+\gamma \overline{V}_{h-1}\left(S_{t+1}\right) & \text{(7.17)}
    \end{aligned}

对 :math:`t<h \leq T`。
如果 :math:`h < T`，递归以 :math:`G_{h : h} \doteq Q_{h-1}\left(S_{h}, A_{h}\right)` 结束，
或者如果 :math:`h = T`，则以 :math:`G_{T-1 : T} \doteq R_{T}`。
那么我们使用n步Sarsa（7.11）的一般（离策略）更新。框中给出了完整的算法。

.. admonition:: 离策略 :math:`n` 步Sarsa估计 :math:`Q \approx q_*` 或者 :math:`q_\pi`
    :class: important

    输入：对所有 :math:`s\in\mathcal(S)`，一个任意的行为策略 :math:`b` 使得 :math:`b(a | s)>0`

    对所有 :math:`s\in\mathcal(S)`，:math:`a\in\mathcal(A)`，任意初始化 :math:`Q(s,a)`

    初始化 :math:`\pi` 关于 :math:`Q` 或固定的给定策略为 :math:`\varepsilon` -贪婪

    算法参数：步长 :math:`\alpha \in (0,1]`，小 :math:`\varepsilon>0`，正整数 :math:`n`

    所有存储和访问操作都可以使其索引 :math:`mod n + 1`

    对每个回合循环：

        初始化并存储 :math:`S_0 \ne` 终点

        选择并存储动作 :math:`A_{0} \sim b\left(\cdot | S_{0}\right)`

        :math:`T \leftarrow \infty`

        对 :math:`t=0,1,2, \ldots` 循环：

            如果 :math:`t < T`，则：

                采取行动 :math:`A_t`，观察并将下一个奖励存储为 :math:`R_{t+1}`，将下一个状态存储为 :math:`S_{t+1}`

                如果 :math:`S_{t+1}` 是终点，则：

                    :math:`T \leftarrow t+1`

                否则：

                    选择并存储动作 :math:`A_{t+1} \sim b\left(\cdot | S_{t+1}\right)`

                    选择并存储 :math:`\sigma_{t+1}`

                    存储 :math:`\frac{\pi\left(A_{t+1} | S_{t+1}\right)}{b\left(A_{t+1} | S_{t+1}\right)}` 为 :math:`\rho_{t+1}`

            :math:`\tau \leftarrow t - n + 1` （:math:`\tau` 是状态估计正在更新的时间）

            如果 :math:`\tau \geq 0`：

                :math:`G \leftarrow 0`：

                对 :math:`k=\min (t+1, T)` 下降到 :math:`\tau + 1` 循环：

                    如果 :math:`k=T`：

                        :math:`G \leftarrow R_{T}`

                    否则：

                        :math:`\overline{V} \leftarrow \sum_{a} \pi\left(a | S_{k}\right) Q\left(S_{k}, a\right)`

                        :math:`G \leftarrow R_{k}+\gamma\left(\sigma_{k} \rho_{k}+\left(1-\sigma_{k}\right) \pi\left(A_{k} | S_{k}\right)\right)\left(G-Q\left(S_{k}, A_{k}\right)\right)+\gamma \overline{V}`

                :math:`Q\left(S_{\tau}, A_{\tau}\right) \leftarrow Q\left(S_{\tau}, A_{\tau}\right)+\alpha\left[G-Q\left(S_{\tau}, A_{\tau}\right)\right]`

                如果 :math:`\pi` 正在被学习，那么确保 :math:`\pi\left(\cdot | S_{\tau}\right)` 是关于 :math:`Q` 贪婪

        直到 :math:`\tau = T - 1`


7.7 总结
----------

在本章中，我们开发了一系列时序差分学习方法，介于前一章的一步法TD方法和之前章的蒙特卡罗方法之间。
涉及中间量引导的方法很重要，因为它们通常比任何一个极端都要好。

.. figure:: images/4-step-TD-and-4-step-Q-sigma.png
    :align: right
    :width: 120px

我们在本章中的重点是n步方法，它们展望n步奖励，状态和行动。右边的两个4步备份图总结了大多数介绍的方法。
所示的状态值更新用于具有重要性采样的n步TD，并且动作值更新用于n步 :math:`Q(\sigma)`，其概括了预期的Sarsa和Q-learning。
所有n步方法都涉及在更新之前延迟n个时间步骤，因为只有这样才能知道所有必需的未来事件。
另一个缺点是它们涉及每个时间步骤比先前方法更多的计算。
与一步法相比，n步法还需要更多内存来记录最后n个时间步骤中的状态，动作，奖励以及有时其他变量。
最后，在第12章中，我们将看到如何使用资格迹以最小的内存和计算复杂性实现多步TD方法，但除了一步方法之外总会有一些额外的计算。
这样的成本非常值得为逃避单一时间步骤的暴政而付出代价。

尽管n步法比使用资格迹的方法更复杂，但它们具有概念上清晰的巨大好处。
我们试图通过在n步案例中开发两种离策略学习方法来利用这一点。
一，基于重要性采样在概念上很简单，但可能具有很大的差异。如果目标和行为策略非常不同，它可能需要一些新的算法思想才能有效和实用。
另一种是基于树形备份更新，是Q-learning到具有随机目标策略的多步情况的自然延伸。
它不涉及重要性采样，但是，如果目标和行为策略实质上不同，则即使n很大，引导也可能仅跨越几个步骤。


书目和历史评论
---------------

n步回报的概念归功于Watkins（1989），他也首先讨论了它们的误差减少性质。
在本书的第一版中探讨了n步算法，其中它们被视为概念上的兴趣，但在实践中是不可行的。
Cichosz（1995）和特别是van Seijen（2016）的工作表明它们实际上是完全实用的算法。
鉴于此，以及它们的概念清晰度和简洁性，我们选择在第二版中突出显示它们。
特别是，我们现在推迟所有关于后向观点和资格迹的讨论，直到第12章。

**7.1-2** 根据Sutton（1988）和Singh和Sutton（1996）的工作，本文的随机行走实例的结果。
本章中使用备份图来描述这些算法和其他算法是新的。

**7.3-5** 这些部分的发展基于Precup，Sutton和Singh（2000），Precup，Sutton和Dasgupta（2001）
以及Sutton，Mahmood，Precup和van Hasselt（2014）的工作。

树备份算法归功于Precup，Sutton和Singh（2000），但这里的表示是新的。

**7.6** :math:`Q(\sigma)` 算法是本文的新内容，
但De Asis，Hernandez-Garcia，Holland和Sutton（2017）进一步探讨了密切相关的算法。
