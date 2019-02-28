第2章 k臂老虎机问题
====================

区分强化学习与其他类型学习的最重要特征是，它使用训练信息来 *评估* 所采取的行动，而不是通过给予正确的行动来 *指导*。
这就是为了明确寻找良好行为而产生积极探索的需要。
纯粹的评价反馈表明所采取的行动有多好，但不表明它是最好还是最坏的行动。
另一方面，纯粹的指导性反馈表明采取的正确行动，与实际采取的行动无关。
这种反馈是监督学习的基础，包括模式分类，人工神经网络和系统识别的大部分。
在它们的纯粹形式中，这两种反馈是截然不同的：评价反馈完全取决于所采取的行动，而指导性反馈则与所采取的行动无关。

在本章中，我们在简化的环境中研究强化学习的评价方面，该方法不涉及学习如何在多种情况下行动。
这种 *非关联性* 设置是大多数先前涉及评估反馈的工作已经完成的，并且它避免了完整强化学习问题的大部分复杂性。
研究这个案例使我们能够最清楚地看到评价性反馈如何与指导性反馈的不同，但可以与之相结合。

我们探索的特定非关联评价性反馈问题是k臂老虎机问题的简单版本。
我们使用这个问题来介绍一些基本的学习方法，我们将在后面的章节中对其进行扩展，以应用于完全强化学习问题。
在本章的最后，我们通过讨论当老虎机问题变为关联时会发生什么，即在不止一种情况下采取行动，这种情况更接近完整的强化学习问题。

2.1 一个 :math:`k` 臂老虎机问题
-------------------------------

考虑以下学习问题。你可以反复面对 :math:`k` 种不同的选择或行动。在每次选择之后，你会收到一个数值奖励，该奖励取决于你选择的行动的固定概率分布。
你的目标是在一段时间内最大化预期的总奖励，例如，超过1000个操作选择或 *时间步骤*。

这是 :math:`k` 臂老虎机问题的原始形式，通过类比于老虎机或“单臂强盗”命名，除了它有k个拉杆而不是一个。
每个动作选择就像一个老虎机的拉杆游戏，奖励是击中累积奖金的奖金。
通过反复的行动选择，你可以通过将你的行动集中在最佳杠杆上来最大化你的奖金。
另一个类比是医生在一系列重病患者的实验治疗之间进行选择。每个动作都是治疗的选择，每个奖励都是患者的生存或幸福。
今天，“老虎机问题”一词有时用于上述问题的概括，但在本书中我们用它来指代这个简单的情况。

在我们的 :math:`k` 臂老虎机中，只要选择了该动作，:math:`k` 个动作的每一个都有预期的或平均的奖励，让我们称之为该行动的 *价值*。
我们将在时间步 :math:`t` 选择的动作表示为 :math:`A_t`，并将相应的奖励表示为 :math:`R_t`。
然后，对于任意动作 :math:`a` 的价值，定义 :math:`q_{*}(a)` 是给定 :math:`a` 选择的预期奖励：

.. math::

    q_{*}(a) \doteq \mathbb{E}[R_t|A_t=a]

如果你知道每个动作的价值，那么解决 :math:`k` 臂老虎机问题将是轻而易举的：你总是选择具有最高价值的动作。
我们假设你不确定地知道动作价值，尽管你可能有估计值。
我们将在时间步骤 :math:`t` 的动作 :math:`a` 的估计值表示为 :math:`Q_t(a)`。
我们希望 :math:`Q_t(a)` 接近 :math:`q_{*}(a)`。

如果你保持对动作价值的估计，那么在任何时间步骤中至少有一个其估计值最大的动作。我们把这些称为 *贪婪* 行为。
当你选择其中一个动作时，我们会说你正在 *利用* 你当前对动作价值的了解。
相反，如果你选择了一个非常规动作，那么我们就说你正在 *探索*，因为这可以让你提高你对非行动动作价值的估计。
利用是在一步中最大化预期的奖励的最好的方法，但从长远来看，探索可能会产生更大的总回报。
例如，假设贪婪行为的价值是确定的，而其他一些动作估计几乎同样好，但具有很大的不确定性。
不确定性使得这些其他行动中的至少一个实际上可能比贪婪行动更好，但你不知道哪一个。
如果你有很多时间步骤可以选择行动，那么探索非贪婪行动并发现哪些行动比贪婪行动可能会更好。
在短期内，奖励在探索期间较低，但从长远来看更高，因为在你发现更好的行动之后，你可以多次利用 *它们*。
因为无法探索和利用任何单一行动选择，人们通常会提到探索和利用之间的“冲突”。

在任何特定情况下，探索或利用是否更好在某种复杂方式上取决于估计的精确值，不确定性和剩余步骤的数量。
有许多复杂的方法可以平衡探索和利用 :math:`k` 臂老虎机的特定数学公式和相关问题。
然而，这些方法中的大多数都对关于平稳性和先验知识做出了强有力的假设，这些假设在应用程序中被违反或无法验证，
在随后的章节中我们会考虑完整的强化学习问题。当这些方法的假设不适用时，对这些方法的最优性或有限损失的保证并不太好。

在本书中，我们不担心以复杂的方式平衡探索和开发；我们只担心平衡它们。
在本章中，我们为 :math:`k` 臂老虎机提出了几种简单的平衡方法，并表明它们比总是利用的方法更好。
平衡探索和开发的需要是强化学习中出现的一个独特挑战；我们的 :math:`k` 臂老虎机问题的简单性使我们能够以一种特别清晰的形式展示这一点。

2.2 行动-价值方法
------------------

.. math::
    :label: 2.1

    Q_t(a) \doteq \frac{在t之前采取a动作的奖励总和}{在t之前采取a动作的次数}
    = \frac{\sum_{i=1}^{t-1}R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1}\mathbb{1}_{A_i=a}}

.. math::
    :label: 2.2

    A_t = \mathop{argmax} \limits_{a} Q_t(a)


2.4
------

.. math::

    Q_n \doteq \frac{R_1 + R_2 + \dots + R_{n-1}}{n-1}

.. math::
    :label: 2.3

    \begin{align*}
    Q_{n+1} &= \frac{1}{n}\sum_{i=1}^{n}R_i \\
            &= \frac{1}{n}(R_n + \sum_{i=1}^{n-1}R_i) \\
            &= \frac{1}{n}(R_n + (n-1)\frac{1}{n-1} \sum_{i=1}^{n-1}R_i) \\
            &= \frac{1}{n}(R_n + (n-1)Q_n) \\
            &= \frac{1}{n}(R_n + nQ_n-Q_n) \\
            &= Q_n + \frac{1}{n}(R_n - Q_n)
    \end{align*}

.. math::
    :label: 2.4

    NewEstimate \leftarrow OldEstimate + StepSize [Target - OldEstimate]

2.5
----

.. math::
    :label: 2.5

    Q_{n+1} \doteq Q_n + \alpha(R_n - Q_n)

.. math::
    :label: 2.6

    \begin{align*}
    Q_{n+1} &= Q_n + \alpha(R_n - Q_n) \\
    &= \alpha R_n + (1-\alpha)Q_n \\
    &= \alpha R_n + (1-\alpha)[\alpha R_{n-1} + (1-\alpha)Q_{n-1}] \\
    &= \alpha R_n + (1-\alpha)\alpha R_{n-1} + (1-\alpha)^2 \alpha R_{n-2} + \\
    & \qquad \qquad \dots + (1-\alpha)^{n-1}\alpha R_1 + (1-\alpha)^nQ_1 \\
    &= (1-\alpha)^nQ_1 + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i
    \end{align*}

.. math::
    :label: 2.7

    \sum_{n=1}^{\infty}\alpha_n(a) = \infty 和 \sum_{n=1}^{\infty}\alpha_n^2(a) < \infty

.. math::
    :label: 2.8

    \beta_n \doteq \alpha / \overline{o}_n

.. math::
    :label: 2.9

    \overline{o}_n \doteq \overline{o}_{n-1} + \alpha(1-\overline{o}_{n-1}) for n \ge 0, with \overline{o}_0 \doteq 0

.. math::
    :label: 2.10

    A_t \doteq \mathop{argmax} \limits_{a} \left[Q_t(a) + c \sqrt{\frac{\ln{t}}{N_t(a)}}\right]


.. math::
    :label: 2.11

    Pr\{A_t=a\} \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^{k}e^{H_t(b)}} \doteq \pi_t(a)

.. math::
    :label: 2.12

    \begin{align*}
    H_{t+1}(A_t) &\doteq H_t(A_t) + \alpha(R_t-\overline{R}_t)(1-\pi_t(A_t))， &和 \\
    H_{t+1}(a) &\doteq H_t(a) - \alpha(R_t-\overline{R}_t)\pi_t(a)，&对所有 a \ne A_t
    \end{align*}

The Bandit Gradient Algorithm as Stochastic Gradient Ascent

.. math::
    :label: 2.13

    H_{t+1}(a) \doteq H_t(a) + \alpha\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)}

.. math::

    \mathbb{E}[R_t] = \sum_{x}\pi_t(x)q_*(x)

.. math::

    \begin{align*}
    \frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)}\left[\sum_{x}\pi_t(x)q_*(x)\right] \\
    &= \sum_{x}q_*(x)\frac{\partial \pi_t(x)}{\partial H_t(a)} \\
    &= \sum_{x}(q_*(x)-B_t)\frac{\partial \pi_t(x)}{\partial H_t(a)}
    \end{align*}

.. math::

    \frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} =
        \sum_{x}\pi_t(x)(q_*(x)-B_t)\frac{\partial \pi_t(x)}{\partial H_t(a)}/\pi_t(x)

.. math::

    \begin{align*}
    &= \mathbb{E}\left[ (q_*(A_t)-B_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t) \right] \\
    &= \mathbb{E}\left[ (R_t-\overline{R}_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t) \right]
    \end{align*}


.. math::

    \begin{align*}
    &= \mathbb{E}\left[ (R_t-\overline{R}_t) \pi_t(A_t) (\mathbb{1}_{a=A_t}-\pi_t(a))/\pi_t(A_t) \right] \\
    &= \mathbb{E}\left[ (R_t-\overline{R}_t)(\mathbb{1}_{a=A_t}-\pi_t(a)) \right]
    \end{align*}

.. math::

    H_{t+1}(a) = H_t(a) + \alpha(R_t-\overline{R}_t)(\mathbb{1}_{a=A_t}-\pi_t(a))，对于所有a

.. math::

    \frac{\partial}{\partial x} \left[ \frac{f{x}}{g{x}} \right] =
        \frac{ \frac{\partial f(x)}{\partial x}g(x) - f(x)\frac{\partial g(x)}{\partial x}}{g(x)^2}

.. math::

    \begin{align*}
    \frac{\partial \pi_t(x)}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)}\pi_t(x) \\
    &= \frac{\partial}{\partial H_t(a)}\left[ \frac{e^{H_t(x)}}{\sum_{y=1}^{k}e^{H_t(y)}} \right] \\
    &= \frac{ \frac{\partial e^{H_t(x)}}{\partial H_t(a)} \sum_{y=1}^{k}e^{H_t(y)} - e^{H_t(x)}\frac{\partial \sum_{y=1}^{k}e^{H_t(y)}}{\partial H_t(a)} }{(\sum_{y=1}^{k}e^{H_t(y)})^2} \\
    &= \frac{ \mathbb{1}_{a=x}e_{H_t(x)}\sum_{y=1}^{k}e^{H_t(y)} - e^{H_t(x)}e^{H_t(a)} }{(\sum_{y=1}^{k}e^{H_t(y)})^2} (因为 \frac{\partial e^x}{\partial x}=e^x) \\
    &= \frac{\mathbb{1}_{a=x}e_{H_t(x)}}{\sum_{y=1}^{k}e^{H_t(y)}} - \frac{e^{H_t(x)}e^{H_t(a)}}{(\sum_{y=1}^{k}e^{H_t(y)})^2} \\
    &= \mathbb{1}_{a=x}\pi_t(x) - \pi_t(x)\pi_t(a) \\
    &= \pi_t(x)(\mathbb{1}_{a=x} - \pi_t(a)) &Q.E.D.
    \end{align*}
