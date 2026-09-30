# Distillation for Continuous DLMs

# MediaTek Research

## 1 Motivation

Continuous difusion language models (DLMs) are now competitive with discrete and autoregressive models — LangFlow (Chen et al., 2026) shows a continuous formulation can match discrete difusion in language-modeling quality. But to be worth deploying, they also need to be small and cheap to serve, without losing generation quality.

Existing distillation work, however, targets a diferent axis. DLM-One (Chen et al., 2025), Flow Map Language Models (Lee et al., 2026), and progressive ELF distillation (Qiu et al., 2026) all reduce the number of sampling steps. None of them reduce capacity — training a structurally smaller model while keeping the sampling process itself unchanged. That’s the regime that matters for deployment, and the one we focus on.

Capacity distillation is exactly where classical knowledge distillation breaks down for difusion models. Forward KL against the teacher’s soft targets is simple and easy to optimize, but for DLMs what actually matters is the state distribution the loss is evaluated on. Train a student of-policy — on teacher states with noise added — and it can fit those states well while still generating poorly: at inference it follows its own reverse process, and drifts into regions the teacher’s data never covered (Rethinking Knowledge Distillation for Difusion Language Models, 2026). A wider capacity gap between teacher and student makes both problems worse at once: forward KL becomes a weaker local approximation, and the student’s occupancy drifts further from the teacher’s.

Continuous DLMs, however, expose more structure than a bare token distribution. LangFlow’s hierarchy links a token posterior to a denoised embedding, and that embedding defines the ODE vector field that actually drives generation. So distillation should match the objects that produce motion — the local vector field and the integrated trajectory — not only the output distribution. This raises the question motivating this proposal: which teacher signal does a smaller continuous DLM actually need, and does supervising the student on its own ODE occupancy fix the state-distribution mismatch that breaks naive distillation? Our hypothesis: capacity-distilled continuous DLMs should be supervised on the student’s own ODE occupancy.

## 2 Impact

Continuous DLMs are a fast-moving, increasingly popular area, and capacity distillation for them hasn’t been addressed yet. Work that gets there early is well placed to become a reference point for the field.

A second, broader contribution is conceptual. Our objective generalizes classical knowledge distillation: forward-KL distillation on the teacher’s soft probabilities is the special case where only the posterior-matching term is active (forward KL is the Bregman divergence from negative entropy — the same geometry already used to train continuous DLMs). Separating the two failure modes — divergence choice vs. occupancy — also explains a confusing empirical fact: a better of-policy fit can coexist with worse generation. The on-policy, DAgger-style (Ross et al., 2011) fix for this transfers directly to distillation and self-training in other continuous generative models, like image and audio difusion, not just language.

The project also lowers the barrier to entry: if capacity distillation reliably produces smaller continuous DLMs without changing the sampling process, more groups can work with continuous DLMs without training at the teacher’s scale — often the main driver of a model family’s adoption.

Finally, there’s no standard evaluation protocol or baseline set for capacity distillation of continuous DLMs yet. Establishing the first one could make it the default others measure against.

## 3 Novelty

On-policy distillation has matured quickly for continuous generative models, but almost entirely along axes orthogonal to ours: recent flow- and difusion-matching methods use it for reward alignment, multi-teacher consolidation, or cutting the number of steps — always for image generation, and always with the student’s parameter budget held fixed. We instead distill a strong continuous language model into a structurally smaller student while keeping the sampling process fixed. To our knowledge, this is the first work on on-policy distillation for capacity reduction in continuous difusion language models and the first to ask which teacher signal that regime actually needs, rather than assuming the full velocity field is necessary.

The second contribution is analytical. On LangFlow’s afine probability path, we show that the embedding-matching and velocity-matching losses are equivalent: both reduce to the posterior discrepancy under the embedding metric, difering only in a known, noise-dependent weighting. So at a fixed state, the local flow terms add no supervisory signal beyond the posterior — the only quantity genuinely independent of the posterior is the state distribution the loss is evaluated on.

This collapses a large design space into one well-posed question: once occupancy is corrected, and divergence choice is held fixed by an $\alpha - \beta$ control, does flow supervision help beyond posterior matching? Either answer is informative. A positive result would isolate the specific teacher signal that drives generation quality under capacity reduction; a negative result would show that a much simpler, posterior-only recipe is enough. Either way, this gives the first characterization of what teacher signal capacity distillation of continuous DLMs actually needs — a conclusion that should extend to on-policy flow distillation methods more broadly.

## 4 Method

We start from the standard distillation loss, explain why it is not enough for difusion language models, and then build up the objects a continuous distillation should match: the posterior, the local vector field, and the trajectory.

Classical knowledge distillation uses the forward Kullback–Leibler divergence

$$
D _ { \mathrm { K L } } ( p _ { T } \parallel p _ { S } ) = \sum _ { x } p _ { T } ( x \mid s ) \log \frac { p _ { T } ( x \mid s ) } { p _ { S } ( x \mid s ) } .
$$

Because the teacher distribution is fixed, this is just cross-entropy with the teacher’s soft probabilities as targets — simple, dense, and easy to optimize.

For difusion language models, the main failure isn’t the divergence choice — it’s the state distribution the loss is evaluated on. In of-policy masked-DLM distillation, teacher samples are forward-noised, and teacher and student are matched on those noised states. But at inference, the student follows its own reverse process, so it can enter regions the forward-noised teacher data never touches. There, the loss gives no corrective signal — which is why a better of-policy fit can coexist with worse generation (Rethinking Knowledge Distillation for Difusion Language Models, 2026).

So there are two separate issues: forward KL can be a poor local approximation under a large capacity gap, and any of-policy divergence can be evaluated on entirely the wrong states. The $\alpha \mathrm { - } \beta$ divergence helps with the first issue in discrete DLMs, but doesn’t fix the state-distribution mismatch. We keep it for continuous DLMs too, but only as an explicit divergence control (see the experimental design below): crossing divergence choice with evaluation-state distribution is what lets us attribute any gain to occupancy, rather than to a better local approximation.

## 4.1 Distill the flow, not only the distribution

LangFlow (Chen et al., 2026) gives a useful hierarchy; Section Appendix A collects the full parameterization we rely on. At noise level 𝛾, the model predicts a token posterior $p ( x \mid z _ { \gamma } , \gamma )$ , which maps to a denoised embedding,

$$
\begin{array} { r } { \hat { z } = E ^ { \top } p , } \end{array}
$$

and the denoised embedding defines the ODE vector field used for generation (Chen et al., 2026). With $\Phi _ { \theta } ( z _ { 0 } , \gamma )$ as the deterministic flow map from initial noise $z _ { \mathrm { 0 } } .$

$$
\frac { \mathrm { d } z _ { \gamma } } { \mathrm { d } \gamma } = v _ { \theta } \big ( z _ { \gamma } , \gamma \big ) , v _ { \theta } \big ( z _ { \gamma } , \gamma \big ) = \frac { \partial \Phi _ { \theta } ( z _ { 0 } , \gamma ) } { \partial \gamma } .
$$

So continuous distillation has three linked objects to match: the posterior, the local vector field, and the integrated trajectory.

## 4.1.1 Posterior matching

Use the same Bregman geometry that underlies continuous-model training:

$$
\mathcal { L } _ { \mathrm { p o s t } } = \mathbb { E } _ { z _ { \gamma } , \gamma } \big [ D _ { \mathrm { K L } } \big ( p _ { T } \big ( \cdot \mid z _ { \gamma } , \gamma \big ) , p _ { S } \big ( \cdot \mid z _ { \gamma } , \gamma \big ) \big ) \big ] .
$$

With negative entropy as the generator, $D _ { \mathrm { K L } }$ is forward KL — making it a principled baseline for LangFlow, not just a divergence borrowed from classical KD.

Posterior matching preserves lexical information, but it doesn’t directly control the ODE’s motion. So the student also needs to match the quantities that generate the continuous path.

## 4.1.2 Denoised embedding and vector-field matching

For a shared embedding matrix $E ,$

$$
\begin{array} { r } { \hat { z } _ { T } = E ^ { \top } p _ { T } , \quad \hat { z } _ { S } = E ^ { \top } p _ { S } . } \end{array}
$$

A direct denoiser loss is

$$
\mathcal { L } _ { \mathrm { e m b } } = \mathbb { E } _ { z _ { \gamma } , \gamma } \left[ w ( \gamma ) \lVert \hat { z } _ { T } - \hat { z } _ { S } \rVert ^ { 2 } \right] .
$$

This matches the continuous representation that turns token probabilities into motion, but shouldn’t replace posterior matching: the map from a vocabulary distribution to its expected embedding is many-to-one.

The more direct dynamical objective is

$$
\mathcal { L } _ { \mathrm { v f } } = \mathbb { E } _ { z _ { \gamma } , \gamma } \Big [ w ( \gamma ) \big \| v _ { T } \big ( z _ { \gamma } , \gamma \big ) - v _ { S } \big ( z _ { \gamma } , \gamma \big ) \big \| ^ { 2 } \Big ] .
$$

This asks the student to match the teacher’s local ODE direction directly — the mechanism actually used at sampling time, not just the output logits.

LangFlow uses an afine probability path $z _ { \gamma } = \alpha _ { \gamma } \hat { z } + \sigma _ { \gamma } \varepsilon$ with schedule $\left( \alpha _ { \gamma } , \sigma _ { \gamma } \right)$ and $\varepsilon \mathrm { ~ a ~ }$ standard Gaussian; the VP path additionally fixes $\alpha _ { \gamma } ^ { 2 } + \sigma _ { \gamma } ^ { 2 } = 1$ . On such a path, denoised-embedding matching and vector-field matching are not independent at a fixed state. Their diference satisfies

$$
v _ { T } - v _ { S } = \left( \frac { \partial \alpha _ { \gamma } } { \partial \gamma } - \frac { \alpha _ { \gamma } } { \sigma _ { \gamma } } \frac { \partial \sigma _ { \gamma } } { \partial \gamma } \right) ( \hat { z } _ { T } - \hat { z } _ { S } ) .
$$

For the VP 𝛾-path this coeficient is $- \frac { \alpha _ { \gamma } } { 2 }$ , so

$$
\left\| v _ { T } - v _ { S } \right\| ^ { 2 } = \frac { \alpha _ { \gamma } ^ { 2 } } { 4 } \| \hat { z } _ { T } - \hat { z } _ { S } \| ^ { 2 } .
$$

Thus, embedding and vector-field losses are two parameterizations of the same local flow-matching signal, with diferent noise-level weighting. Posterior matching remains distinct because $E ^ { \intercal } p$ discards information about the full token distribution.

Putting the three losses in one picture: both flow losses are quadratic forms in the posterior discrepancy $E ^ { \mathsf { T } } ( p _ { \mathsf { T } } - p _ { S } )$ under the metric $E E ^ { \mathsf { T } }$ , difering only in 𝛾-weighting; posterior matching penalizes the same discrepancy $p _ { T } - p _ { S }$ under Bregman (KL) geometry instead. So at a fixed state, the local flow terms don’t add a target independent of the posterior — they just re-weight and re-geometrize it, and miss any discrepancy in the null space of $E ^ { \top }$ . The one lever genuinely independent of the posterior is the state distribution the loss is evaluated on. That’s why the design below treats occupancy as the primary axis, and loss geometry as secondary.

## 4.1.3 Trajectory matching

A deterministic probability-flow ODE gives a natural coupling: start teacher and student from the same Gaussian noise seed $z _ { \mathrm { 0 } } ,$ and compare their states at the same noise levels:

$$
\mathcal { L } _ { \mathrm { t r a j } } = \mathbb { E } \left[ \sum _ { k } \omega _ { k } \Big \Vert z _ { T , \gamma _ { k } } - z _ { S , \gamma _ { k } } \Big \Vert ^ { 2 } \right] .
$$

Posterior and vector-field losses are local; trajectory matching instead checks whether those local approximations integrate to a similar transport map. It’s a stronger constraint than querying the teacher on student-visited states — a smaller student may reach a similar endpoint by a diferent path — so paired trajectory matching should stay optional. It also assumes teacher and student share the space $z _ { 0 }$ lives in, which holds under a shared embedding $E ;$ that’s a further reason $\mathcal { L } _ { \mathrm { t r a j } }$ is fragile under aggressive capacity reduction.

A compact objective is

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { p } \mathcal { L } _ { \mathrm { p o s t } } + \lambda _ { e } \mathcal { L } _ { \mathrm { e m b } } + \lambda _ { v } \mathcal { L } _ { \mathrm { v f } } + \lambda _ { T } \mathcal { L } _ { \mathrm { t r a j } } . } \end{array}
$$

Because ${ \mathcal { L } } _ { \mathrm { e m b } }$ and ${ \mathcal { L } } _ { \mathrm { v f } }$ are locally equivalent up to known weighting, a practical first model should use posterior matching plus one local flow-matching term. Add paired trajectory matching as a controlled ablation.

## 4.2 Of-policy and on-policy distillation

The of-policy stage is cheap and stable. Generate a static corpus from the teacher, perturb clean embeddings on the same 𝛾-path, and train the student with posterior and local flow-matching losses. This stage transfers broad teacher knowledge without repeated teacher rollouts.

The on-policy stage corrects the state mismatch: roll out the current student ODE, stop gradients through the visited states, and query the teacher at those same states and the same $\gamma .$ Match posterior and local flow there. This is the continuous analogue of DAgger-style distillation – the teacher supervises the states the student actually visits. Two details follow the DAgger analogy: we aggregate visited-state data across iterations rather than training only on the newest rollouts (a purely fresh dataset can oscillate; a purely static one goes stale), and querying the teacher at a student-visited state needs a selfconditioning input, so we feed it the student’s own self-conditioning (the quantity actually present at inference) and treat the teacher’s reliability of its own distribution as something to measure, not assume (see Risks). In parallel, paired trajectories can start from the same noise seed and use $\mathcal { L } _ { \mathrm { t r a j } }$ to test whether constraining accumulated path drift helps.

We show a pseudocode in the Appendix discuss the experimental design in Appendix Section Appendix B.

## 5 Risks

Competition. Continuous difusion is an active area, so other groups may pursue similar ideas. We mitigate this by scoping the project tightly and front-loading the decisive test at gate G2, so the central question is answered early rather than late (see the timeline).

The central hypothesis may not hold. Local flow supervision may fail to improve on posterior matching once occupancy is corrected. This is framed as an empirical question rather than an assumption, and the ablation design isolates it directly, so a null result is informative and publishable.

The local losses may be redundant. For LangFlow’s afine path the denoised-embedding and vector-field losses are equivalent up to a known, noise-level-dependent factor, so combining ${ \mathcal { L } } _ { \mathrm { e m b } }$ and $\mathcal { L } _ { \mathrm { v f } }$ risks adding a term without adding signal. The mitigation is to carry a single local flow-matching term and treat the noise-level weighting 𝑤(𝛾) — not the choice between the two parameterizations — as the real design variable.

The teacher may be an unreliable oracle of its own distribution. On-policy querying evaluates the teacher at student-visited states, which can lie far from the teacher’s own occupancy; there its posterior and vector field are themselves extrapolations, so the fix could import teacher error. We monitor this with the occupancy diagnostic and by checking teacher agreement at queried states, and fall back to aggregated of-policy data in regions where the teacher signal looks unreliable.

Cost and optimization stability. On-policy rollouts are more expensive than static of-policy training, and combining several loss terms can destabilize optimization. The staged design controls both: a cheap, stable of-policy stage transfers broad knowledge first, and the on-policy stage stops gradients through visited states and introduces terms incrementally with a weighting schedule.

Student initialization and layer selection. Training a randomly initialized student through distillation may require substantial compute and still fail to recover the teacher’s quality. We therefore initialize from the teacher’s weights and remove a subset of blocks, using random layer removal as a simple baseline. The risk is that a poor choice of layers wastes the recovery budget. To reduce this risk, we will collect teacher activations across noise levels and fit least-squares linear replacements for candidate blocks. Candidates with low held-out reconstruction error and limited degradation in model outputs will be prioritized. In this process, Progressive Structural Distillation and Shapley value analysis may be employed.

## ## Appendix A The LangFlow parameterization

This appendix collects the LangFlow (Chen et al., 2026) details used in the main text, so the derivations there are selfcontained. We follow LangFlow’s conventions and refer to the original paper for architecture and training specifics.

## A.1 Generative process

LangFlow generates text by transporting Gaussian noise to a clean token embedding along a deterministic ODE, then reading tokens of that embedding. At noise level $\gamma ,$ the model sees a noised latent $z _ { \gamma }$ and predicts a token posterior $p ( x \mid z _ { \gamma } , \gamma )$ . Averaging the embedding matrix 𝐸 under this posterior gives the denoised embedding

$$
\hat { z } = E ^ { \mathsf { T } } p = \sum _ { x } p \big ( x \mid z _ { \gamma } , \gamma \big ) E _ { x } ,
$$

the model’s running estimate of the clean embedding. Self-conditioning feeds the previous 𝑧̂ back in as an extra input, so we treat it as part of the state throughout.

## A.2 Afine path and probability-flow ODE

Training and sampling use an afine path that mixes the clean embedding 𝑧̂ with noise 𝜀 (a standard Gaussian),

$$
z _ { \gamma } = \alpha _ { \gamma } \hat { z } + \sigma _ { \gamma } \varepsilon ,
$$

under a schedule $\left( \alpha _ { \gamma } , \sigma _ { \gamma } \right)$ ; the variance-preserving (VP) choice fixes $\alpha _ { \gamma } ^ { 2 } + \sigma _ { \gamma } ^ { 2 } = 1$ . Generation integrates the probabilityflow ODE

$$
\frac { \mathrm { d } z _ { \gamma } } { \mathrm { d } \gamma } = v _ { \theta } \big ( z _ { \gamma } , \gamma \big ) , ~ v _ { \theta } \big ( z _ { \gamma } , \gamma \big ) = \frac { \partial \Phi _ { \theta } ( z _ { 0 } , \gamma ) } { \partial \gamma } ,
$$

where $\Phi _ { \theta } ( z _ { 0 } , \gamma )$ is the flow map from the initial noise $z _ { 0 }$

## A.3 Vector field from the denoised embedding

The velocity is fixed once the denoised embedding is known. Solving the path for the implied noise, $\begin{array} { r } { \varepsilon = \frac { z _ { \gamma } - \alpha _ { \gamma } \hat { z } } { \sigma _ { \gamma } } } \end{array}$ , and substituting,

$$
v _ { \theta } ( z _ { \gamma } , \gamma ) = \frac { \partial \alpha _ { \gamma } } { \partial \gamma } \hat { z } + \frac { \partial \sigma _ { \gamma } } { \partial \gamma } \varepsilon = \left( \frac { \frac { \partial \sigma _ { \gamma } } { \partial \gamma } } { \sigma _ { \gamma } } \right) z _ { \gamma } + \left( \frac { \partial \alpha _ { \gamma } } { \partial \gamma } - \frac { \alpha _ { \gamma } } { \sigma _ { \gamma } } \frac { \partial \sigma _ { \gamma } } { \partial \gamma } \right) \hat { z } .
$$

Two models compared at the same state $z _ { \gamma }$ share the first term, so it cancels in the diference, leaving the relation used in the main text:

$$
v _ { T } - v _ { S } = \left( \frac { \partial \alpha _ { \gamma } } { \partial \gamma } - \frac { \alpha _ { \gamma } } { \sigma _ { \gamma } } \frac { \partial \sigma _ { \gamma } } { \partial \gamma } \right) ( \hat { z } _ { T } - \hat { z } _ { S } ) .
$$

At a fixed state, then, velocity matching and denoised-embedding matching difer only by the scalar coeficient $\begin{array} { r } { c ( \gamma ) = \frac { \partial \alpha _ { \gamma } } { \partial \gamma } - \left( \frac { \alpha _ { \gamma } } { \sigma _ { \gamma } } \right) \frac { \partial \sigma _ { \gamma } } { \partial \gamma } } \end{array}$ . Under LangFlow’s VP 𝛾-schedule this reduces to $c ( \gamma ) = - \frac { \alpha _ { \gamma } } { 2 }$ (Chen et al., 2026), giving

$$
\left\| v _ { T } - v _ { S } \right\| ^ { 2 } = \frac { \alpha _ { \gamma } ^ { 2 } } { 4 } \| \hat { z } _ { T } - \hat { z } _ { S } \| ^ { 2 } .
$$

## A.4 Why posterior matching is not redundant

The readout $\hat { z } = E ^ { \top } p$ is linear and many-to-one: two posteriors that share the same embedding-space mean $E ^ { \intercal } p$ give the same 𝑧̂, and hence the same local velocity. Matching 𝑧̂ or 𝑣 therefore constrains only the part of the posterior in the row space of $E ^ { \top }$ , and is blind to diferences in its null space. Posterior matching acts on the full distribution 𝑝 and closes that gap — which is why the combined objective keeps both terms.

## Appendix B Experiments

## B.1 Experimental design and evaluation

Capacity axis. We shrink the student backbone in depth and width, keeping tokenizer, embedding 𝐸, and sampler fixed, across a range of capacity gaps (roughly $2 \times \tan 8 \times$ fewer non-embedding parameters). Since both failure modes are predicted to worsen with the gap, we treat the gap as an independent variable rather than fixing one operating point.

Baselines and minimum publishable result. The method has to beat the trivial route. Two null baselines: (i) a same-size student trained directly on teacher-generated text with the ordinary LangFlow objective (the fallback noted below), and (ii) standard sequence-level KD on teacher samples. The minimum publishable result: posterior matching with on-policy occupancy correction beats both nulls on generation at a fixed capacity gap. Local flow, trajectory, $\alpha \mathrm { - } \beta ,$ and the teacheraccess study are all upside beyond that.

Metrics. Since the motivating fact is that a better fit can come with worse generation, every gate is decided on generationquality metrics distinct from the training loss, reported at a fixed number of function evaluations (NFE): generative perplexity under a larger held-out scorer, MAUVE against held-out text, an 𝑛-gram diversity/entropy measure to catch mode collapse, and at least one downstream task. Training-loss or ELBO improvements are reported, but never suficient on their own to pass a gate.

Occupancy diagnostic. We measure the mechanism directly rather than only inferring it from generation. At each noise level 𝛾 we estimate a divergence between the forward-noised teacher state distribution and the student-rollout state distribution, and track how much occupancy correction shrinks it. This validates the causal story behind gate G2 even when end-to-end generation is noisy, and is itself the diagnostic vocabulary claimed in the impact.

Attributing gains: occupancy × divergence. To avoid confounding the sampling distribution with the choice of divergence, the two are crossed rather than confounded. The $\alpha \mathrm { - } \beta$ lane is run as a divergence control: forward KL and $\alpha \cdot \beta$ are each evaluated under both of-policy and on-policy occupancy, so a gain from on-policy training cannot be re-explained as merely a better local projection. The central claim — that occupancy is the dominant lever — is made only if it survives this cross.

## B.2 What continuous distillation requires

Teacher and student need a common token space, or an explicit vocabulary map. Posterior matching needs compatible output semantics, while local flow matching needs a shared embedding coordinate system or a learned alignment. Compare teacher and student at the same noise level 𝛾, not merely at the same solver-step index, and treat self-conditioning as part of the state.

Full access to teacher internals is not necessary. If the teacher returns its posterior or denoised prediction at an arbitrary $z _ { \gamma } \mathrm { . }$ , the known LangFlow parameterization gives $\hat { z } _ { T }$ and therefore the vector field without teacher weights or hidden states. If only intermediate continuous sampler states are exposed, their displacements can provide trajectory or finite-diference flow targets. If the teacher exposes only final text samples, its vector field is not identifiable from those endpoints; true onpolicy querying is unavailable, and the natural baseline is ordinary LangFlow training on teacher-generated text.

## B.3 Datasets

We start with unconditional generation on OpenWebText (Gokaslan et al., 2019) and the One Billion Word Benchmark (Chelba et al., 2014), using the same preprocessing and evaluation protocol for teacher and student. OpenWebText2 (Gao et al., 2020) and selected text subsets from the Nemotron pre-training collection (NVIDIA, 2025) provide extensions for testing whether the findings hold across corpora and data scales. For conditional generation, we consider German-to-English translation on WMT14 (Bojar et al., 2014) and abstractive summarization on XSum (Narayan et al., 2018).

## B.4 Psuedocode

python

```python
1 state = off_policy_state(dot..) or stopgrad(student_rollout_state(dot..))
2
3 p_T = teacher.posterior(state, gamma, self_cond)
4 p_S = student.posterior(state, gamma, self_cond)
5
6 z_T = E.T @ p_T
7 z_S = E.T @ p_S
8 v_T = velocity(state, z_T, gamma)
9 v_S = velocity(state, z_S, gamma)
10
11 loss = lambda_p * KL(p_T, p_S) + lambda_flow * MSE(v_T, v_S)
12 loss += lambda_T * trajectory_MSE(path_T, path_S) # optional
```

## Appendix C Related Work

Continuous language-model distillation mainly targets fewer sampling steps. DLM-One (Chen et al., 2025) uses score distillation for one-step generation, Flow Map Language Models (Lee et al., 2026) learn mappings across noise levels, and progressive ELF distillation (Qiu et al., 2026) trains one student update to replace multiple teacher steps. PlaidQ (Peng et al., 2026) extends this direction to code, using distribution matching for few-step generation and paired noise-to-output supervision for one-step generation. Image difusion ofers two complementary lines of work: progressive distillation (Salimans & Ho, 2022), consistency models (Song et al., 2023), and distribution matching (Yin, Gharbi, Zhang, et al., 2024) reduce sampling steps, while BK-SDM (Kim et al., 2024) and TinyFusion (Fang et al., 2025) shrink the network and recover quality through distillation. The choice of training states has also been studied: Imagine Flash (Kohler et al., 2024) trains along the student’s own sampling path, DMD2 (Yin, Gharbi, Park, et al., 2024) simulates student sampling to reduce training– inference mismatch, and DifusionOPD (Li et al., 2026) matches teacher and student transitions on student rollouts for capability consolidation.

## Appendix D Extended Timeline

The critical path runs Infrastructure → of-policy baseline → on-policy occupancy correction → occupancy-plus-localflow ablation → scaling sweep → write-up. The gates are staged so that the project can stop or simplify as early as the evidence allows:

Each gate is decided on the generation metrics of the experimental design at a fixed capacity gap and fixed NFE. The pass threshold 𝛿 and the two null baselines are frozen at G1 so that later gates are pre-registered rather than chosen after seeing results.

G1 (end of week 4) The of-policy student trains stably and reproduces a sane baseline; the two null baselines (student trained directly on teacher text, and sequence-level KD) and the per-metric thresholds 𝛿 are calibrated and frozen. Early exit: if training is unstable or the baseline is far of, repair the pipeline or teacher access before any on-policy work.

G2 (end of week 8) On-policy occupancy correction beats the of-policy student and both null baselines by at least 𝛿 on the pre-registered metrics at equal NFE, and the occupancy diagnostic confirms the state-distribution gap has shrunk. This tests the central hypothesis. Early exit: if occupancy correction gives no gain above the nulls, stop and pivot — the premise is unsupported, and this is the cheapest point to learn it.

G3 (end of week 10) Local flow supervision adds at least 𝛿 beyond posterior-only, occupancy-corrected training, and the gain survives the 𝛼-𝛽 divergence control (it is not explained by the divergence choice). Early exit: if not, drop $\mathcal { L } _ { \mathrm { v f } }$ and ${ \mathcal { L } } _ { \mathrm { e m b } }$ , ship the simpler posterior-only recipe, and skip the trajectory work.

G4 (end of week 12) Paired trajectory matching helps rather than over-constrains the smaller student, at the same capacity gap. Early exit: if it over-constrains, exclude $\mathcal { L } _ { \mathrm { t r a j } }$ and move straight to finalization.