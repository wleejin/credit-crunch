# Credit crunch and the economy

This python code solves the main model in [*Guerrieri, Veronica and Guido Lorenzoni. 2017. "Credit Crises, Precautionary Savings, and the Liquidity Trap." Quarterly Journal of Economics*](https://doi.org/10.1093/qje/qjx005).

## Baseline Model
$$\begin{aligned}
\max_{c_{it}, n_{it}, b_{it+1}} & \mathbb{E}\left[\sum_{t=0}^{\infty} \beta^{t} U\left(c_{i t}, n_{i t}\right)\right]\\
\text{s.t.} \ \ &q_{t} b_{i t+1}+c_{i t} \leq b_{i t}+\theta_{i t}n_{i t}-\widetilde{\tau}_{i t}, \\
& n_{i t}\geq 0,\\
&b_{i t+1} \geq -\phi, \\
&\widetilde{\tau}_{i t}={
  \begin{cases}
    \tau_t & \text{if } \theta_{i t}>0\\
    \tau_t - \nu_t & \text{if } \theta_{i t}=0
  \end{cases}
}
\end{aligned}$$

Notice that there is no capital in the model. Also, note that the households (HHs) want to save only for precautionary motive. An unexpected, one-time shock will be on $\phi$. $\theta_{i t}$ is a random variable following a Markov chain.  The relationship between the bond price and interest rate is $q_t = \frac{1}{1+r_t}$.

Let $B_t$ be the aggregate supply of bonds. In the model, the only supply of bonds outside the HH sector comes from the government. The government 's choice variables are $B_t$, $\nu_t$, and $\tau_t$. Its budget constraint is as follows:

$$\begin{gather*}
B_{t}+u v_{t}=q_{t} B_{t+1}+\tau_{t},
\end{gather*}$$

where $u = \Pr(\theta_{i t}=0)$ is the fraction of unemployed agents in the population. Note that the tax revenue is

$$\begin{gather*}
\sum_i \widetilde{\tau}_{i t} = \left( 1 - u \right) \tau_{t} + u \left( \tau_{t}- v_{t} \right) = \tau_{t} - u v_{t}.
\end{gather*}$$

Here, the government fixes $B$  and $\nu$, and chooses $\tau_t$.

$$\begin{gather*}
  \tau_{t} = (1-q_{t}) B + u v.
\end{gather*}$$

Note that $B$ is the sum of all liquid assets held by the HH sector. The model is closed by assuming $B$ is a constant.

### Definition for Equilibrium
An equilibrium is  sequences of policy functions $\{c_t(b,\theta), n_t(b,\theta),b_{t+1}(b,\theta)\}$, price $\{ r_t\}$, taxes $\{\tau_t \}$, and joint distribution $\{\Psi_t(b,\theta)\}$ s.t.
1. Policy functions are optimal.
1. $\Psi_t$ is consistent with policy functions.
1. $\tau_t$ satisfies the government's budget constraint: $\tau_{t} = \frac{r_t}{1+r_t} B + u v$.
1. Bond market clears: ${\int b d\Psi_t(b,\theta)} = {B}$.

### Euler Equations 
Note that $U_1(0) = \infty$ implying $c>0$.

$$\begin{aligned}
\mathcal{L} =& \ \mathbb{E} \left[ \sum_{t=0}^\infty \beta^t \left[ U\left(c_{i t}, n_{i t}\right) + \lambda_{it} \left(   b_{i t}+\theta_{i t}n_{i t}-\widetilde{\tau}_{i t}   - q_{t} b_{i t+1} - c_{i t}  \right) + \mu_{it+1} \left( b_{it+1}+\phi \right)  +  \omega_{it}  n_{it}  \right] \right].
\end{aligned}$$

FOCs are as follows: 

$$\begin{aligned}
c_{it}&: \ \ U_c\left(c_{i t}, n_{i t}\right) - \lambda_{it} = 0,\\
n_{it}&: \ \ U_n\left(c_{i t}, n_{i t}\right) + \lambda_{it} \theta_{i t} +  \omega_{it}= 0,\\ 
b_{it+1}&:  \ \  - \lambda_{it}q_{t} + \mu_{it+1} +\beta  \mathbb{E}_{t}\left[\lambda_{it+1}\right] =0.
\end{aligned}$$

By combining the above equations, we have the following equilibrium conditions:

$$\begin{gather*}
U_{c}\left(c_{i t}, n_{i t}\right) \geq \beta\left(1+r_{t}\right) \mathbb{E}_{t}\left[U_{c}\left(c_{i t+1}, n_{i t+1}\right)\right] \ \ \text{with equality if $b_{it+1}>-\phi$},\\
\theta_{i t} U_{c}\left(c_{i t}, n_{i t}\right)+U_{n}\left(c_{i t}, n_{i t}\right) \leq 0  \ \ \text{with equality if $n_{it}>0$}.
\end{gather*}$$

Note that for constrained consumers, 

$$c_{i t} = b_{i t}+\theta_{i t}n_{i t}-\widetilde{\tau}_{i t} + q_{t} \phi .$$



### Calibration
The utility function is 

$$\begin{gather*}
U(c, n)=\frac{c^{1-\gamma}}{1-\gamma}+\psi \frac{(1-n)^{1-\eta}}{1-\eta}.
\end{gather*}$$

By using the equilibrium conditions, 

$$\begin{gather*}
c_{it}^{-\gamma} \geq \beta\left(1+r_{t}\right) \mathbb{E}_{t}\left[c_{it+1}^{-\gamma} \right],\\
n_{it}=\max \left[0, 1- \left(\frac{\psi}{\theta_{it}} \right)^\frac{1}{\eta} c_{it}^\frac{\gamma}{\eta} \right].
\end{gather*}$$

Also, 

$$\begin{aligned}
&{
  \begin{cases}
    \ln \theta_t = \rho \ln \theta_{t-1} + \epsilon_t & \text{if }\ \ \theta>0, \\
    \text{when first employed, $\theta$  drawn from  uncond'l dist. } & \text{if }\ \ \theta=0.
  \end{cases}
}
\end{aligned}$$

The wage process is approximated by a 12-state Markov chain, $\theta \in [0.35, \dotsb, 2.89]$.


### Algorithm for Steady State and Calibration
Note that $\gamma, \eta, r$ are fixed parameters. Start with calibration targets and an initial guess for calibration parameters.
1. Endogenous Grid Method for policy functions.
1. Stationary distribution $\Psi(\theta,b)$.
1. Check market clearing condition and calibration statistics.
1. Update $\beta, \nu, B, \phi, \psi$.
1. Repeat until the model moments match the target moments.

### Figures
[Policy functions and bond distribution](https://github.com/wleejin/credit-crunch/tree/main/fig)

### Credit Crunch
Gradually, $\phi = 0.959\longrightarrow \phi^\prime = 0.525$. The adjustment lasts six quarters.

