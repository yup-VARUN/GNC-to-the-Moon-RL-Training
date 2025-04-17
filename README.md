# GNC-to-the-Moon-RL-Training
Project for Cloud Computing Class Spring 2025, GSU.

The Soft Actor–Critic (SAC) training loop in Stable‑Baselines3 begins by instantiating a stochastic actor πₜheta and two Q‑value critics Q_{φ₁}, Q_{φ₂}—each paired with a Polyak‑averaged target network φ_{targ,1}, φ_{targ,2} via a soft update coefficient τ—before any interactions with the environment :contentReference[oaicite:0]{index=0}.  
The agent then collects transitions (sₜ, aₜ, rₜ, sₜ₊₁, dₜ) by sampling actions aₜ ∼ πₜheta(·|sₜ) and storing them in a replay buffer for off‑policy learning :contentReference[oaicite:1]{index=1}.  
During each gradient step, the critics are updated by minimizing the soft Bellman residual  
$$
L(φᵢ)=\mathbb{E}_{(s,a,r,s',d)\sim D}\bigl[\,Q_{φᵢ}(s,a)-y(r,s',d)\bigr]^2,
\quad
y=r+\gamma(1-d)\Bigl(\min_{j=1,2}Q_{φ_{targ,j}}(s',\tilde a')-\alpha\logπ_{θ}(\tilde a'|s')\Bigr),
\;\tilde a'∼π_{θ}(\cdot|s'),
$$  
which injects an entropy regularizer α to stabilize learning :contentReference[oaicite:2]{index=2}.  
Next, the policy is updated using the reparameterization trick to minimize  
$$
L(θ)=\mathbb{E}_{s,ξ}\bigl[\,α\logπ_{θ}(\tilde a_{θ}(s,ξ)|s)-\min_{j}Q_{φ_{j}}(s,\tilde a_{θ}(s,ξ))\bigr],
$$  
thereby maximizing expected return plus entropy :contentReference[oaicite:3]{index=3}.  
Optionally, α itself is auto‑tuned by minimizing  
$$
L(α)=\mathbb{E}_{a∼π_{θ}}\bigl[-α\bigl(\logπ_{θ}(a|s)+\bar H\bigr)\bigr],
$$  
to match a target entropy :contentReference[oaicite:4]{index=4}. Finally, after each update the target networks undergo a Polyak update  
$$
φ_{targ,i}\leftarrow τ\,φ_{i} + (1-τ)\,φ_{targ,i},
$$  
before the loop repeats until convergence :contentReference[oaicite:5]{index=5}.
