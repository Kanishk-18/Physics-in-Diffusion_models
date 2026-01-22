---
title: "Diffusion"
date: 2026-01-20
draft: false
---

# From Physical Diffusion to Diffusion Models: Building Intuition

Coming from a physics background and now working in data science for remote sensing, I was naturally intrigued by diffusion models. While reading about their impressive generative capabilities, I initially struggled to understand how a physical process like diffusion could inspire a modern generative model. How does a phenomenon traditionally associated with randomness, entropy, and information loss become a mechanism for learning complex data distributions?

In this blog, I aim to build intuition for diffusion models by tracing their roots back to physical diffusion processes and unpacking how these ideas translate into the forward process of diffusion models. The focus here is on understanding how data is progressively corrupted in a principled way, and how this mirrors classical diffusion in physics.

I will deliberately postpone a detailed discussion of the reverse (generation) process to the next part. However, I assume a basic familiarity with the overall workflow of diffusion models, so that the connection between physical diffusion and the forward noising process can be properly appreciated. Rather than treating diffusion models as black-box neural networks, the goal here is to understand the principles that make their construction both physically intuitive and mathematically sound.



## Diffusion in Physics: The Intuition

Before diffusion became a generative model, it was a physical process used to describe how order gradually gives way to randomness. Understanding this physical intuition is key to appreciating why diffusion models work so well in machine learning.



## Brownian Motion and Stochasticity

Diffusion originates from the study of Brownian motion—the seemingly random movement of microscopic particles suspended in a fluid. Although each particle follows a chaotic trajectory due to countless molecular collisions, the process itself is not arbitrary. Instead, it is governed by well-defined statistical laws.

Mathematically, Brownian motion is modeled as a stochastic process, where the future state depends on the current state and random fluctuations.

To understand this mathematically, let a particle follow a random walk. At each time step $ \Delta t $:

- the particle moves left or right with equal probability  
- the step size is $ \Delta x $

The probability evolution at time $ \Delta t $ can be written as

$$
\begin{equation}
p(x, t + \Delta t ) = \frac{1}{2}\left[p(x-\Delta x, t) + p(x + \Delta x, t)\right]
\end{equation}
$$

Since this equation is discrete (random walk), we approximate it using Taylor expansions around $x$ and $t$:

$$
p(x, t + \Delta t) = p(x,t) + \Delta t \frac{\partial p}{\partial t}
$$

$$
p(x+\Delta x, t) = p(x,t) + \Delta x \frac{\partial p}{\partial x} + \frac{\Delta x^2}{2} \frac{\partial^2 p}{\partial x^2}
$$

$$
p(x-\Delta x, t) = p(x,t) - \Delta x \frac{\partial p}{\partial x} + \frac{\Delta x^2}{2} \frac{\partial^2 p}{\partial x^2}
$$

Substituting these approximations into Equation (1) and simplifying, we obtain

$$
\frac{\partial p}{\partial t} = \frac{\Delta x^2}{2 \Delta t} \frac{\partial^2 p}{\partial x^2}
$$

Defining the diffusion coefficient $D = \frac{\Delta x^2}{2 \Delta t}$, we arrive at the diffusion equation:

$$
\begin{equation}
\frac{\partial p}{\partial t} = D \frac{\partial^2 p}{\partial x^2}
\end{equation}
$$

The diffusion coefficient $D$ controls how quickly uncertainty spreads. In fact, the variance of the particle position grows linearly with time:

$$
\text{Var}(x_t) = 2Dt
$$

From Equation (2), we see that as time increases, uncertainty in the particle’s position increases, and hence entropy increases. Diffusion is therefore a process of **progressive information loss**.



## Why Gaussian Noise Appears Naturally

Now, with this foundation, we can understand one of the key reasons why **Gaussian noise** is used in diffusion models.

Assume a particle starts at position $x = 0$ at time $t = 0$. The initial distribution is

$$
p(x,0) = \delta(x)
$$

where $\delta(\cdot)$ is the Dirac delta function, representing complete certainty.

In diffusion, particle motion arises from the accumulation of an extremely large number of tiny, independent random steps. After time $t$, the position can be written as

$$
x_t = \sum_{i=1}^{N} \zeta_i
$$

where $\zeta_i$ are independent random variables with finite variance.

By the Central Limit Theorem, the sum of many independent random variables converges to a Gaussian distribution, regardless of the exact form of each step:

$$
x_t \approx \mathcal{N}(0, \sigma^2 t)
$$

Solving the diffusion equation with the delta-function initial condition yields the fundamental solution:

$$
\begin{equation}
p(x,t) = \frac{1}{\sqrt{4\pi Dt}} \exp\left( \frac{-x^2}{4Dt} \right)
\end{equation}
$$

Thus, diffusion evolves a system from **absolute certainty** to a **Gaussian distribution**, which is the maximum-entropy distribution under a fixed variance constraint.

This observation is crucial: **Gaussian noise represents the most unstructured distribution possible when energy (variance) is controlled**. In other words, diffusion systematically removes structure while preserving mathematical tractability.



## Motivation for Gaussian Noise in Diffusion Models

Diffusion models adopt this physical intuition as a **modeling choice**. The goal of the forward process is to gradually destroy the structure of real data until it becomes pure noise. Gaussian noise is particularly well-suited for this purpose because:

- it naturally arises from the accumulation of small independent perturbations  
- it maximizes entropy for a fixed variance  
- it allows closed-form expressions for marginal distributions  
- it leads to stable and tractable learning objectives 

**Key Note:**\
*While Gaussian noise is a natural choice from a physical diffusion perspective, its true importance lies in the reverse process. Gaussian corruption ensures that the reverse-time dynamics remain locally Gaussian, enabling tractable variational inference, stable learning, and an efficient denoising-based parameterization.*



## Forward Process in Diffusion Models

In diffusion models, the forward process is defined as a Markov process, where each state depends only on the previous one. At each step, a small amount of Gaussian noise is added:

$$
\begin{equation}
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)
\end{equation}
$$

Here, $\beta_t$ is the **noise schedule**, controlling how much noise is injected at each step.



The scaling of the mean by $\sqrt{1-\beta_t}$ ensures that the total variance remains bounded across time. This is important both physically (to control the diffusion rate) and practically (to prevent exploding activations when training neural networks).

Repeated application of Equation (4) allows us to derive a closed-form expression
for the marginal distribution of $x_t$ conditioned on the original data $x_0$:

$$
\begin{equation}
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,
\quad \epsilon \sim \mathcal{N}(0, I)
\end{equation}
$$

where $\alpha = 1-\beta$ and $\bar{\alpha}_t = \prod\limits_{s=1}^{t}\alpha_s$

Since the forward process is Markovian, the joint distribution over the entire trajectory factorizes as 
$$
\begin{equation}
    q(x^{(0...T)}) = q(x^{(0)})\prod_{t=1}^{T} q(x^{(t)}|x^{t-1})
\end{equation}
$$


After many steps, this forward process transforms structured data into nearly isotropic Gaussian noise, completing the connection between physical diffusion and generative modeling.

---

*With this forward process defined, the central challenge of diffusion models becomes clear: how do we reverse this irreversible-looking diffusion process to recover meaningful data?*
