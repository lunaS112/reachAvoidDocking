# CDC Paper Writing Guide — Patterns from Accepted Papers

> **Purpose**: This document distills structural, tonal, and presentational patterns from five recent papers accepted to CDC and adjacent controls venues. Use it as a reference while drafting the CDC 2026 spacecraft docking BRAT paper. This guide is self-contained — no access to the source papers is needed.

## Reference Papers Analyzed

1. **"Provably-Safe Neural Network Training Using Hybrid Zonotope Reachability Analysis"** — Chung & Kousik (Georgia Tech). Proposes a training method for ReLU networks using scaled hybrid zonotopes and MILPs to enforce collision-free constraints. Demonstrates forward-invariant controllers and reach-avoid planning for a drifting vehicle.

2. **"Approximate Hamilton-Jacobi Reachability Analysis for a Class of Two-Timescale Systems"** — Hirsch & Herbert (UC San Diego). Uses singular perturbation theory to reduce HJ reachability computation for systems with fast and slow dynamics. Derives inner/outer approximations of backward reachable sets. Applies to biological system models.

3. **"Secure Safety Filter Design for Sampled-data Nonlinear Systems under Sensor Spoofing Attacks"** — Tan, Ong, Tabuada & Ames (Caltech / UCLA). Extends secure safety filters from linear to nonlinear systems using observability maps and control barrier functions. Validates on a unicycle with 5 partly-compromised sensors.

4. **"Control Synthesis for Multiple Reach-Avoid Tasks via Hamilton-Jacobi Reachability Analysis"** — Chen, Li & Yin (Shanghai Jiao Tong). Solves sequential multi-target reach-avoid problems by cascading HJ variational inequalities, treating future feasible sets as dynamic targets. Includes LTL connection and four case studies (single integrators through spacecraft rendezvous).

5. **"Uncertainty Removal in Verification of Nonlinear Systems against Signal Temporal Logic via Incremental Reachability Analysis"** — Besset, Tillet & Sandretto (ENSTA Paris). Extends STL verification over reachable sets using Boolean interval arithmetic. Decomposes satisfaction signals to track and selectively refine sources of uncertainty.

---

## 1. Overall Structure

### 1.1 Section Layout

All five papers follow a nearly identical skeleton. The CDC template expects this and reviewers parse it on autopilot — deviating from it is risky.

```
I.   Introduction
     A. Related Work (sometimes I-A, sometimes I-B after a motivation paragraph)
     B. Contributions (always a numbered or bulleted list)
II.  Preliminaries / Problem Setup
III. Main Method / Technical Approach
IV.  (Optional) Additional Technical Results
V.   Experiments / Simulations / Applications
VI.  Conclusion
     References
```

**Key observations:**

- **Introduction is 1–1.5 columns.** Nobody writes a 2-page intro. The intro sets up the problem, says why existing approaches fall short, and immediately states contributions. Chung & Kousik, Tan et al., and Chen et al. all accomplish this in under 1.5 columns.
- **Preliminaries are short and functional.** They define only what the reader needs for the next section. Hirsch & Herbert is a good model: it defines the singularly perturbed system, the differential game, the BRS, and the value function — nothing extra.
- **The method section is the bulk.** Chung & Kousik, Tan et al., and Chen et al. each dedicate 2–3 pages to the core technical machinery, with theorems, propositions, and proofs.
- **Experiments are compact but convincing.** Usually 1–1.5 pages. Tables and figures do the heavy lifting — prose is minimal.
- **Conclusion is 0.25–0.5 columns max.** Restates the contribution in one sentence, then lists 2–4 concrete limitations or future directions.

### 1.2 Page Budget (8-page CDC format)

| Section | Typical Length |
|---------|--------------|
| Abstract | 10–15 lines |
| I. Introduction (incl. related work, contributions) | 1.0–1.5 pages |
| II. Preliminaries | 0.5–1.0 pages |
| III–IV. Method / Theory | 2.0–3.0 pages |
| V. Experiments / Applications | 1.0–2.0 pages |
| VI. Conclusion | 0.25–0.5 pages |
| References | 0.5–1.0 pages |

---

## 2. Abstract

### 2.1 Structure

Every abstract follows the same 4-beat pattern:

1. **Context/Problem** (1–2 sentences): What class of problems does this address?
2. **Gap** (1–2 sentences): What can't existing methods do?
3. **Contribution** (2–4 sentences): What does this paper propose, and what properties does it have?
4. **Validation** (1–2 sentences): How was it demonstrated?

### 2.2 Tone

- Assertive but precisely scoped. Claims are bounded: Chung & Kousik say "effective and fast for networks with up to 240 neurons" — not "for arbitrary networks." Hirsch & Herbert say "we identify a class of systems which can be readily reduced" — not "all multi-timescale systems."
- No hedging words like "we hope" or "we attempt." Instead: "we propose," "we prove," "we demonstrate."
- The abstract names the specific technical tools used: "scaled hybrid zonotopes," "mixed-integer linear programs," "Hamilton-Jacobi variational inequalities." This signals to reviewers that the work is concrete, not hand-wavy.

### 2.3 Examples of Good Scoping

- Chung & Kousik: "...for networks with up to 240 neurons, with the computational complexity dominated by inverse operations on matrices that scale linearly in size with the number of neurons."
- Chen et al.: "We prove that the super-level set of the final value function computed is exactly the feasible set of the MRA task."
- Tan et al.: "The proposed approach provides theoretical safety guarantees for nonlinear systems in the presence of sensor attacks."

**Takeaway for the BRAT paper**: State the dimensionality (13D), the application (spacecraft docking), the method (neural BRAT + MPC), and the validation (Monte Carlo). Scope the claim — don't say "guarantees safety" unqualified; say "provides empirical evidence of safety under the trained value function" or frame the infinite-horizon claim with its precise conditions.

---

## 3. Introduction

### 3.1 Opening Paragraph

The first paragraph always does two things:

1. **Motivates the problem class** with a broad but grounded statement. Not "AI is everywhere" but "neural networks are increasingly deployed in safety-critical control applications" (Chung & Kousik) or "Hamilton-Jacobi reachability is an increasingly useful tool for analyzing and controlling safety-critical systems" (Hirsch & Herbert).
2. **Identifies the core difficulty** within 2–3 sentences. Tan et al. does this well: "a fundamental assumption in many of these works is that the system state is accessible to the controller. In reality... they are susceptible to cyber-physical attacks."

### 3.2 Related Work

All five papers organize related work into **named subcategories** with clear paragraph breaks. This is essential — it shows reviewers you understand the landscape and have surveyed it systematically.

Chung & Kousik use four labeled subsections:
- Training with Constraints
- Neural Network Verification
- Training with Verification
- Hybrid Zonotopes

Chen et al. use a prose-based "Related Works" subsection that groups prior art thematically and explains the gap relative to each group.

**Pattern for each subcategory:**
1. What these methods do (1–2 sentences).
2. Why they are insufficient for your problem (1 sentence, specific).
3. Transition to the next category or to your contribution.

### 3.3 Contributions

**Always a numbered list.** Every single paper does this. The list is typically 2–4 items. Each item is one self-contained sentence.

Chung & Kousik list three contributions:
> 1) Given a non-convex input set and a non-convex unsafe region, we propose a differentiable loss function that...
> 2) We show that our method is fast and scales well...
> 3) We showcase the practicality of our method by synthesizing...

Chen et al. list four:
> - First, we prove that the feasible set of the MRA task can be exactly characterized...
> - We then propose an efficient online algorithm...
> - Additionally, we discuss how the proposed MRA framework is related to linear temporal logic...
> - Finally, we provide a comprehensive set of case studies...

**Key**: Each contribution should be independently verifiable by a reviewer. They will check these against the body of the paper.

**Takeaway for the BRAT paper**: Frame contributions as: (1) extending BRATs to 13D for spacecraft docking, (2) the iterative refinement pipeline combining DeepReach training with MPC-based reach-avoid control, (3) Monte Carlo validation demonstrating safety properties under the trained value function. The framing should read as "application drove method development" — not "method then applied."

---

## 4. Preliminaries

### 4.1 What to Include

Only definitions and notation that are directly referenced in the method section. Every paper is disciplined about this:

- Chung & Kousik define hybrid zonotopes and ReLU networks — the exact two building blocks of their method.
- Hirsch & Herbert define the singularly perturbed system, the differential game, and the value function.
- Tan et al. define the nonlinear system, the attack model, and the zero-order CBF.

### 4.2 Format

- Formal definitions use `Definition X` environments.
- Equations are numbered and referenced later.
- Notation is introduced inline or in a compact "Notation" paragraph at the start of Section II. Tan et al. do this well with a single paragraph defining index sets, cardinality, vector restriction, function spaces, and ball notation.

### 4.3 What NOT to Include

- Background the reader can look up (e.g., "a neural network is a function approximator...").
- Long derivations that are standard in the field.
- Material that is only used in supplementary sections.

---

## 5. Technical Sections (Method / Theory)

### 5.1 Theorem-Proof Structure

Four of the five papers use a formal theorem-proof structure. The pattern is:

1. **Motivating paragraph** explaining the intuition (1–3 sentences).
2. **Formal statement** as a Theorem, Proposition, Lemma, or Corollary.
3. **Proof** — either inline or sketched with a reference to an appendix.

Chung & Kousik chain `Definition 2 (Scaled Hybrid Zonotope)` → `Corollary 3` → `Lemma 4` → `Proposition 5` → `Theorem 6` in a logical sequence where each result builds on the previous.

Chen et al. use `Lemma 1` (existing result, cited) → `Proposition 1` (new recursive construction) → `Theorem 1` (full MRA feasible set characterization) → `Theorem 2` (control synthesis correctness). This cascading structure is very common at CDC and reviewers find it easy to follow and verify.

### 5.2 Remarks

All papers use `Remark` environments to:
- **Discuss limitations or edge cases**: Chung & Kousik use a Remark to say "we opted to demonstrate our method only on affine dynamical systems" to avoid overstating the contribution.
- **Connect to related formulations**: Hirsch & Herbert use Remarks to address technical nuances about domain extensions and viscosity solution definitions.
- **Provide practical implementation notes**: Chen et al. use a Remark to show the QP formulation for online control input selection.

**Takeaway**: Remarks are where you show self-awareness and preempt reviewer objections. Use one to address the infinite-horizon claim: "We note that the infinite-horizon safety guarantee holds under the assumption that the neural value function has converged to a stationary solution of the HJ variational inequality. Verifying this convergence for neural network approximations of high-dimensional value functions remains an open problem."

### 5.3 Equation Density

These papers are equation-heavy. A typical technical section has an equation every 3–5 lines of prose. The surrounding prose exists to:
- State what the next equation defines or achieves.
- Explain the intuition behind a transformation.
- Connect one result to the next.

The prose does NOT re-derive what the equations already show. If the math is self-explanatory, a single sentence of context suffices.

---

## 6. Figures

### 6.1 Types of Figures Used

| Type | Where Used | Purpose |
|------|-----------|---------|
| **System/method flowchart** | Chung & Kousik (Fig. 1), Tan et al. (Fig. 1) | Overview of the full pipeline. Placed prominently — top of page 1 or top of the method section. |
| **Before/after comparison** | Chung & Kousik (Figs. 1–3) | Shows the input set, reachable set, and unsafe set before and after training. Extremely effective for visually communicating the core contribution. |
| **Value function contour plot** | Hirsch & Herbert (Figs. 2–3) | Contour plots of value functions with BRS boundary overlaid. Standard for HJ reachability papers. |
| **Trajectory plot in workspace** | Tan et al. (Fig. 2), Chen et al. (Fig. 2) | Shows closed-loop system trajectory with start/end markers, obstacles, and goals annotated. |
| **Time-series of safety metric** | Tan et al. (Figs. 3–4) | CBF values or control inputs over time. Shows when the safety margin is tight and when the filter activates. |
| **Schematic / conceptual diagram** | Hirsch & Herbert (Fig. 1 — biological circuit), Besset et al. (Fig. 1 — satisfaction tree) | Makes abstract concepts concrete. Useful when the problem domain is specialized. |

### 6.2 Figure Design Principles

1. **Every figure has a detailed caption.** Captions are 3–6 lines and explain what the reader is looking at, what the colors/markers mean, and what the takeaway is. Chung & Kousik's Fig. 1 caption is essentially a one-paragraph method summary.

2. **Color is functional, not decorative.** Sets are color-coded consistently — Chung & Kousik use green for input sets, blue for reachable sets, yellow for unsafe sets, and red for collisions, and maintain this scheme across all figures.

3. **Figures appear at the top of columns**, never floating mid-text. This is standard IEEE formatting.

4. **Axes are labeled with units.** Grid lines are subtle or absent.

5. **Before/after figures use identical axis limits.** This is critical for honest visual comparison.

### 6.3 Figure Count

8-page papers typically contain 3–4 figures. Each figure earns its space by conveying something that would take a full paragraph to describe in words.

**Takeaway for the BRAT paper**: Include (1) a system diagram showing the chaser-target geometry and coordinate frames, (2) 2D slices of the BRAT value function (XY/XZ planes) with goal/failure sets color-coded, (3) Monte Carlo trajectory overlay or success rate visualization, and (4) optionally a refinement progression showing how the BRAT evolves across training iterations. Maintain consistent color coding for the goal set, failure set, and BRAT zero-level-set boundary across all figures.

---

## 7. Tables

### 7.1 When Tables Are Used

Tables appear primarily in experiment sections for quantitative comparisons. Chung & Kousik have one table comparing computation time across network sizes and against a baseline method, with columns for hidden layer widths, training time per iteration, MILP verification time, and total iterations to convergence.

### 7.2 Table Design

- **Compact.** No unnecessary columns.
- **Comparison-oriented.** Always includes a baseline or prior method.
- **Reports variability**: Standard deviations or ranges are included (e.g., "0.140 ± 0.005 s/iteration").
- **Failure cases are stated explicitly**: Chung & Kousik mark "Timeout" for the baseline on larger networks rather than omitting those rows.

---

## 8. Experiments / Applications

### 8.1 Structure

The experiments section typically follows one of two patterns:

**Pattern A — Benchmark + Application**: Chung & Kousik do this. One section is a scalability benchmark on a controlled problem (known input set, known unsafe set, varying network sizes). A second section applies the method to two motivating problems (forward invariance for a non-convex safe region, reach-avoid for a drifting vehicle).

**Pattern B — Multiple Case Studies**: Chen et al. do this. Four case studies of increasing complexity: single integrators → double integrators → spacecraft rendezvous → unicycle robots. Each demonstrates a different aspect of the method.

### 8.2 Elements of a Good Experiment Section

1. **Setup paragraph**: Defines the system dynamics, parameters, dimensions, and the specific input/target/unsafe sets. Everything needed to reproduce the experiment.
2. **Hypotheses** (optional but strong): Chung & Kousik explicitly state "Since the complexity scales linearly for our method and exponentially for [prior work], we expect our method to significantly outperform..." before showing results. This signals scientific rigor.
3. **Results**: Primarily communicated through figures and tables. Prose highlights the key numbers and refers to specific figures/table rows.
4. **Discussion**: Short paragraph interpreting the results, noting surprising findings, and contextualizing relative to baselines.

### 8.3 Comparison to Prior Work

- Chung & Kousik compare directly to their own prior work on computation time, and explicitly note fairness measures: "we do not include the objective loss and only add the constraint loss when it is positive."
- Tan et al. compare trajectory behavior with and without the safety filter active, showing the ground-truth vs. fake trajectory side by side.

**Important**: If comparing to prior work, explain what you did to ensure fairness.

### 8.4 Computing Environment

Always stated explicitly. Chung & Kousik: "All experiments were performed using Python on a desktop computer with a 24-core i9 CPU, 32 GB RAM, and an NVIDIA RTX 4090 GPU." Besset et al. report per-method timing breakdowns on a specific laptop model. This is expected at CDC.

---

## 9. Conclusion

### 9.1 Structure

1. **One-sentence restatement** of the contribution.
2. **Limitations** — explicitly named. This is NOT a weakness in the paper; it shows maturity and gives reviewers confidence that you understand the scope of your claims.
3. **Future work** — 2–3 concrete directions, each paired with the limitation it addresses.

### 9.2 Example

Chung & Kousik's conclusion lists four numbered limitations:
1. Sensitivity to hyperparameters (future: rigorous selection procedures).
2. Matrix inverse bottleneck (future: alternate gradient computation via HSD formulation).
3. Cannot verify whether the problem is solvable (future: joint training of input set and network).
4. Limited to fully-connected ReLU networks (future: extension to CNNs and RNNs).

Each limitation is specific and each future direction is actionable. This is the gold standard for a CDC conclusion.

---

## 10. Tone and Language

### 10.1 Voice

- **Active voice, first person plural**: "We propose," "We show," "We demonstrate." All five papers use this consistently.
- **Present tense for contributions and method descriptions**: "Our method handles non-convex sets."
- **Past tense for experiments**: "The training was successful after 625 iterations."

### 10.2 Hedging vs. Assertion

These papers are assertive about what they prove and hedge only where appropriate:

- ✅ "We prove that the super-level set... is exactly the feasible set." (backed by a theorem)
- ✅ "The proposed method was shown to be effective and fast for networks with up to 240 neurons." (scoped empirical claim)
- ✅ "We are optimistic that future work can extend to CNN and RNN." (hedged on unproven extensions)
- ❌ Avoid: "We believe our method might be useful..." (too weak for a contribution claim)

### 10.3 Connecting Paragraphs

Transition sentences are formulaic but effective:
- "We now review key approaches to..." (starting related work)
- "We now apply scaled hybrid zonotopes to Problem 1." (transitioning from definition to application)
- "To proceed, we first cast the input set Z as..." (step-by-step derivation)
- "The following lemma and proposition construct..." (previewing the logical chain)

### 10.4 Self-Referencing

Papers frequently refer to their own structure to help the reviewer navigate:
- "As per Sec. I-A.2..."
- "We will showcase our method against non-convex sets in Sec. VI."
- "The results of our demo are shown in Fig. 1."

This signals that the paper is well-organized and the authors are guiding the reader deliberately.

---

## 11. References

### 11.1 Count and Recency

Reference counts across the five papers: 37, 30, 30, 39, 20. **Typical range: 25–40.** Most are from the last 5 years. A small number of seminal/older references ground the theoretical foundations (e.g., viscosity solutions from the 1980s, foundational differential games texts).

### 11.2 What to Cite

- **The methods you build on** — cite DeepReach, HJ reachability survey papers, relevant MPC references.
- **The methods you compare to** — cite NeuralPARC or other baselines.
- **The application domain** — cite spacecraft proximity operations and docking literature.
- **The tools you use** — cite PyTorch, numerical solvers, experiment tracking tools if relevant.
- **The invited session's community** — if submitting to a session on behavioral systems / data-driven MPC, cite 2–3 papers from that community even if the connection is tangential. This signals you are aware of the audience.

---

## 12. Specific Advice for the BRAT Spacecraft Docking Paper

### 12.1 Framing

Based on the patterns in these reference papers, the strongest framing is application-first:

> "Spacecraft proximity operations motivate the need for reach-avoid guarantees in high-dimensional systems. We present an extension of backward reach-avoid tubes (BRATs) to 13 dimensions using neural network value functions, with application to autonomous satellite docking."

This mirrors Chung & Kousik's approach: the application (forward invariance, reach-avoid for a drifting car) opens the abstract as the motivating problem, and the method (scaled hybrid zonotopes, differentiable MILP loss) is introduced as the tool developed to address it.

### 12.2 Suggested Contribution List

Three items:
1. Formulation of the 13D chaser-target satellite docking problem as a neural BRAT, including the quaternion-dependent collision envelope and attitude-aware goal/failure sets.
2. An iterative refinement pipeline combining DeepReach value function training with MPC-based reach-avoid control, with specific attention to balancing goal set and failure set sharpness during refinement.
3. Monte Carlo validation on the full 13D system demonstrating safety properties under the trained value function.

### 12.3 Figures to Include

1. **System diagram**: Chaser-target geometry, coordinate frames, docking cone, and collision envelope. Similar in purpose to Tan et al.'s secure safety filter block diagram or Hirsch & Herbert's biological circuit schematic. This orients the reader before the math begins.
2. **Value function slices**: 2D XY, XZ, or isometric slices of the BRAT with the goal set and failure set color-coded. Similar to Hirsch & Herbert's contour plots of the value function with BRS boundaries overlaid.
3. **Monte Carlo results**: Trajectory overlay in position space or a success rate visualization. Similar in spirit to Chung & Kousik's before/after figures showing the reachable set driven out of the unsafe region.
4. **Optional**: Training convergence plot or refinement progression showing how the BRAT boundary evolves across iterations.

Maintain consistent color coding across all figures — e.g., green for goal set, red for failure set, blue for BRAT zero-level-set.

### 12.4 Handling the Infinite-Horizon Claim

The reference papers handle theoretical claims carefully:
- Chung & Kousik include a concrete verification step (MILP feasibility check) that provides a hard stop criterion for training.
- Hirsch & Herbert state three explicit Assumptions and conditions under which the approximation results hold, and the proofs reference these assumptions throughout.

For the BRAT paper, include a Remark that explicitly states the conditions: "The infinite-horizon safety guarantee holds under the assumption that the neural value function has converged to a stationary solution of the HJ variational inequality. Verifying this convergence for neural network approximations of high-dimensional value functions remains an open problem. In practice, we assess convergence empirically via [training loss plateau / Monte Carlo validation]."

### 12.5 Positioning Relative to the Invited Session

Chung & Kousik include a Remark acknowledging that not all choices of safe set can be made forward-invariant. Chen et al. remark that their approach is sound but not complete for LTL synthesis. Both demonstrate awareness of scope boundaries.

If the BRAT paper sits outside the typical intellectual community of the target invited session (behavioral systems / data-driven MPC), include a brief connecting statement in the introduction — something like: "Our iterative refinement procedure can be viewed as a data-driven approach to reach-avoid computation, where the neural value function is progressively refined using trajectory data, connecting to recent work on learning-based predictive control."
