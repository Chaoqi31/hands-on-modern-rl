<div align="center">
  <img src="docs/public/readme/logo-symbol.svg" alt="Hands-On Modern RL Logo" width="220" />
  <h1>Hands-On Modern RL</h1>
  <p><strong>A hands-on modern reinforcement learning course</strong></p>
  <p><em>A practice-first guide to modern RL, from classic control to LLM post-training, RLVR, and multimodal agents.</em></p>

  <p>
    <a href="https://walkinglabs.github.io/hands-on-modern-rl/"><img src="https://img.shields.io/badge/Course-Online-2563eb?style=flat-square" alt="Online Course" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/releases/latest"><img src="https://img.shields.io/badge/PDF-Download-e11d48?style=flat-square" alt="PDF Download" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-111827?style=flat-square" alt="CC BY-NC-SA 4.0 License" /></a>
    <img src="https://img.shields.io/badge/Node-%3E%3D18-16a34a?style=flat-square" alt="Node >= 18" />
    <img src="https://img.shields.io/badge/Docs-VitePress-646cff?style=flat-square" alt="VitePress" />
  </p>

  <p>
    <a href="README.md">English</a> ·
    <a href="README.zh.md">中文</a>
  </p>

  <p>
    <a href="#wechat-group-微信">WeChat Group (微信)</a>
  </p>

  <p>
    <a href="#course-preview">Course Preview</a> ·
    <a href="#overview">Overview</a> ·
    <a href="#news">News</a> ·
    <a href="#contents">Contents</a> ·
    <a href="#course-outline">Course Outline</a> ·
    <a href="#experiment-code">Experiment Code</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#contributing">Contributing</a>
  </p>
</div>

## Course Preview

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-learning-path.png" alt="Course learning map screenshot" width="100%" />
      <br />
      <strong>A clear learning map</strong>
      <br />
      <sub>From the preface and foundations to frontier topics, the chapter tree and page outline help you navigate quickly.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-code-focus.png" alt="PPO code focus screenshot" width="100%" />
      <br />
      <strong>Line-by-line code focus</strong>
      <br />
      <sub>Key PPO, DPO, and GRPO implementations include code maps that connect formulas to readable code.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-training-metrics.png" alt="CartPole training metrics screenshot" width="100%" />
      <br />
      <strong>Training metric visualization</strong>
      <br />
      <sub>Real curves, metric explanations, and failure signals sit together so you can debug while running experiments.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-rlhf-pipeline.png" alt="RLHF pipeline screenshot" width="100%" />
      <br />
      <strong>LLM post-training pipelines</strong>
      <br />
      <sub>RLHF, DPO, GRPO, RLVR, and related topics are tied together through flows, artifacts, and cases.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-agentic-rl.png" alt="Agentic RL experiment page screenshot" width="100%" />
      <br />
      <strong>Agentic RL experiment</strong>
      <br />
      <sub>DeepCoder-style GRPO training curves connect tool-use agents, response length, and reward dynamics in a reproducible lab.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/public/readme/feature-atari-game.png" alt="Atari Pong DQN experiment page screenshot" width="100%" />
      <br />
      <strong>Atari game experiment</strong>
      <br />
      <sub>Atari Pong gameplay screenshots and DQN training notes show how pixel-based agents turn screens into decisions.</sub>
    </td>
  </tr>
</table>

---

> [!NOTE]
> We hope this open course gives more learners the courage to climb toward the frontier of intelligence and solve more of the hard problems on the path to AGI.
>
> The course is evolving quickly. We recommend focusing on chapters that are not marked as under construction; chapters still in progress may contain mistakes, and corrections or suggestions are welcome.

> **Help Wanted**
>
> Because compute resources are limited, we are seeking GPU support. If you can help with GPU access, please contact physicoada@gmail.com.

## Contents

- [Course Preview](#course-preview)
- [Overview](#overview)
- [Course Outline](#course-outline)
- [Experiment Code](#experiment-code)
- [Quick Start](#quick-start)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

**Hands-On Modern RL** is an open course for learning modern reinforcement learning through practice. Instead of the usual "formula first, black-box API later" route, this course takes a **practice-first** path: learners begin with runnable code and observable training behavior, then use those concrete traces to understand states, value functions, policy gradients, reward modeling, credit assignment, and the rest of the mathematical structure behind RL.

The course spans classic control and connects directly to current AI frontiers, including large language model (LLM) post-training, preference alignment with DPO and GRPO, reinforcement learning with verifiable rewards (RLVR), multi-turn tool-use agents, Agentic RL, and vision-language model (VLM) reinforcement learning.

The goal is to provide a solid ladder: from solving CartPole for the first time to building modern post-training and agent systems.

### Design Principles

The course is organized around these engineering and teaching principles:

1. **Practice before formalism.** Each major topic starts from experiments, metrics, failure cases, or implementation details, then introduces the mathematical abstraction.
2. **Theory explains behavior.** MDPs, Bellman equations, policy gradients, GAE, PPO clipping, DPO objectives, and GRPO-style group advantages are introduced as tools for explaining what the code does.
3. **Modern RL goes beyond classic RL.** The course covers classic control and deep RL, then moves into RLHF, preference optimization, RLVR, VLM reinforcement learning, and multi-turn agent training.
4. **Debugging is first-class.** Training collapse, reward hacking, KL drift, entropy decay, OOM failures, and evaluation blind spots are treated as core material.
5. **Readable systems beat black boxes.** Examples favor explicit implementations, inspectable metrics, and clear experiment boundaries so learners can modify and extend them.

### Audience

This course is for learners who want to understand reinforcement learning by building and inspecting working systems.

It is especially useful for:

- Machine learning engineers moving from supervised learning into RL.
- Researchers and students preparing to read modern RL and alignment papers.
- LLM practitioners interested in RLHF, DPO, GRPO, RLVR, and post-training systems.
- Builders of tool-use agents, web agents, code agents, and evaluation pipelines.
- Self-learners who prefer code, experiments, and visual intuition before dense derivations.

Recommended background:

- Python programming experience.
- Basic PyTorch familiarity.
- Introductory linear algebra, probability, and calculus for machine learning.
- Ability to read papers and trace open-source training scripts.

The course includes math review appendices, so full mathematical fluency is not required on day one.

### Learning Goals

After completing the course, learners should be able to:

- Implement and explain the core RL loop: environment interaction, trajectory collection, reward feedback, policy updates, and evaluation.
- Connect MDPs, value functions, Bellman equations, TD learning, policy gradients, and advantage estimation to concrete training behavior.
- Read and modify implementations of DQN, REINFORCE, Actor-Critic, PPO, DPO, GRPO, and related methods.
- Reason about LLM post-training pipelines, including SFT, reward modeling, PPO-style RLHF, DPO-family methods, and RLVR training.
- Understand multi-turn interaction and credit assignment, and build tool-use, trajectory-synthesis, and Agentic RL systems.
- Extend reinforcement learning ideas to VLMs, embodied intelligence, multi-agent self-play, and other frontier areas.
- Diagnose common RL failure modes and design reasonable algorithms, engineering evaluations, and debugging workflows for new RL problems.

### Current Status

This repository is an active courseware project. Content is being expanded chapter by chapter, with emphasis on correctness, runnable examples, and a stable learning path.

- Course site: [walkinglabs.github.io/hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/)
- Source content: [`docs/`](docs/)
- Runnable examples: [`code/`](code/)
- Local verification: `npm run verify`
- License: [CC BY-NC-SA 4.0](LICENSE)

Issues and pull requests are welcome for typo fixes, conceptual corrections, reproducibility improvements, references, and focused course extensions.

## News

> **Note:** This course was created with AI assistance and has not yet been fully reviewed. It may contain factual mistakes or code that does not run as expected. Issues and pull requests are very welcome.

- **[2026-05-13]** 🚀 **Major Upgrade: LLM and Traditional RL Hands-on Labs**: Added reproducible training examples for **Agentic RL** (Deep Research / rLLM) and **Traditional RL** (Actor-Critic continuous control). Includes complete code and fine-tuning analysis for building an Agentic training system from scratch, along with new VLM RL (GeoQA geometry reasoning) hands-on experiments!
- **[2026-05-02]** Initial browsable open-source release for testing and feedback.

## Roadmap

The course is under active development. Planned milestones:

- [x] **2026-05-02:** Initial open-source browsable release for community testing and feedback.
- [ ] **2026-05-10:** Publish a first stable minor version, fix early typos, and stabilize Part 1 and Part 2 content and code.
- [ ] **Late May 2026:** Improve reproducible LLM RL experiments and add a full RLVR hands-on module with evaluation.
- [ ] **Early June 2026:** Deliver Agentic RL projects step by step, from single-tool use to complex Deep Research trajectory synthesis.
- [ ] **Late June 2026:** Add Unity-based embodied RL environments and trainable project examples.
- [ ] **July 2026 and later:** Expand multimodal frontier content with full VLM RL or Diffusion RL hands-on cases.

## Course Outline

The course is divided into four parts plus appendices. The README keeps only the main modules; the online site contains the full chapter tree, diagrams, code references, and detailed navigation.

### Preface

| Module                                                                           | Description                                                        |
| :------------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| [Course Guide](docs/preface/intro.md)                                            | Course positioning, learning path, and how to use the materials.   |
| [A Brief History of Reinforcement Learning](docs/preface/brief-history/index.md) | From trial-and-error learning to AlphaGo, RLHF, and LLM alignment. |
| [Environment Setup](docs/preface/env-setup.md)                                   | Installation and dependency setup for the course.                  |

### Part 1: Foundations by Practice

| Chapter | Main Topic                                                | What It Covers                                                                                                          |
| :------ | :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- |
| 01      | [CartPole](docs/chapter01_cartpole/intro.md)              | States, actions, rewards, policies, values, entropy, and training curves through a first runnable control task.         |
| 02      | [DPO Preference Fine-tuning](docs/chapter02_dpo/intro.md) | Preference data, DPO objectives, reward margins, accuracy, and the first bridge from RL intuition to LLM post-training. |
| Summary | [Part 1 Summary](docs/summaries/part1-summary.md)         | The practical intuition learners should have before entering formal RL theory.                                          |

### Part 2: Core Theory and Methods

| Chapter | Main Topic                                                               | What It Covers                                                                                                                      |
| :------ | :----------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| 03      | [MDPs and Value Functions](docs/chapter03_mdp/intro.md)                  | Bandits, MDPs, value functions, Bellman equations, TD learning, Q-learning, policy objectives, data sources, and reward design.     |
| 04      | [Deep Q-Networks](docs/chapter04_dqn/intro.md)                           | From tabular Q-learning to DQN, replay buffers, target networks, CNN encoders, LunarLander, Atari, and visual game projects.        |
| 05      | [Policy Gradient and REINFORCE](docs/chapter05_policy_gradient/intro.md) | Direct policy optimization, sampling-based gradients, baselines, and variance reduction.                                            |
| 06      | [Actor-Critic](docs/chapter06_actor_critic/intro.md)                     | Actor-critic architecture, advantage functions, TD-error critic training, and game-playing agents.                                  |
| 07      | [PPO](docs/chapter07_ppo/intro.md)                                       | PPO experiments, clipped objectives, trust-region intuition, GAE, reward models, long-horizon planning, and BipedalWalker practice. |
| Summary | [Part 2 Summary](docs/summaries/part2-summary.md)                        | The algorithmic patterns that repeat across classic and modern RL.                                                                  |

### Part 3: LLM-era RL

| Chapter | Main Topic                                                   | What It Covers                                                                                                                                                        |
| :------ | :----------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 08      | [The Full RLHF Pipeline](docs/chapter08_rlhf/intro.md)       | SFT, reward modeling, PPO-style RLHF, evaluation, scaling, and reward hacking.                                                                                        |
| 09      | [Post-Training Alignment](docs/chapter09_alignment/intro.md) | DPO-family methods, GRPO, DeepSeek-R1 and DAPO, RLVR, financial tool-calling GRPO, policy distillation, sandboxed training, and industrial post-training practice.    |
| 10      | [Agentic RL](docs/chapter10_agentic_rl/intro.md)             | Multi-turn credit assignment, tool-use trajectories, agent evaluation, SWE/DeepCoder/FinQA-style labs, Deep Research agents, and end-to-end agentic training systems. |
| Summary | [Part 3 Summary](docs/summaries/part3-summary.md)            | What changes when RL is applied to language models, tools, and multi-step agent behavior.                                                                             |

### Part 4: Frontier and Advanced Systems

| Chapter | Main Topic                                                   | What It Covers                                                                                              |
| :------ | :----------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| 11      | [VLM Reinforcement Learning](docs/chapter11_vlm_rl/intro.md) | VLM GRPO, visual rewards, multimodal reasoning frameworks, visual generation RL, and EasyR1 GeoQA practice. |
| 12      | [Future Trends](docs/chapter12_future_trends/intro.md)       | Embodied intelligence, model-based RL, self-play, multi-agent systems, offline RL, and scaling trends.      |
| Summary | [Part 4 Summary](docs/summaries/part4-summary.md)            | Frontier directions to explore after finishing the core course.                                             |

### Appendices

| Appendix | Main Topic                                                                           | What It Covers                                                                                                               |
| :------- | :----------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| A        | [Training Debugging Guide](docs/appendix_common_pitfalls/intro.md)                   | Common RL training failures, symptoms, root causes, and fixes.                                                               |
| B        | [RL Engineering Practice](docs/appendix_industrial_training/intro.md)                | Training infrastructure, agent sandboxes, parallelism, monitoring, evaluation benchmarks, metrics, and industrial exercises. |
| C        | [Handwritten Code Cheatsheet](docs/appendix_code_cheatsheet/intro.md)                | Compact code notes for SFT, PPO, DPO, GRPO, sampling, attention, and DAPO.                                                   |
| D        | [Learning Resources and Reproduction Projects](docs/appendix_game_projects/intro.md) | Curated resources and reproduction projects for expanding course examples.                                                   |
| E        | [Math Foundations for Reinforcement Learning](docs/appendix_math/intro.md)           | Linear algebra, probability, calculus, optimization, and information theory for RL.                                          |

## Experiment Code

The [`code/`](code/) directory contains runnable examples aligned with course chapters. Each chapter's code is intentionally compact so it can be inspected, run, and modified independently.

| Area                   | Code Path                                                                                                          | Representative Experiments                                                                         |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| Classic control        | [`code/chapter01_cartpole/`](code/chapter01_cartpole/)                                                             | Train CartPole, inspect rewards and episode length, and compare PPO implementations.               |
| Preference fine-tuning | [`code/chapter02_dpo/`](code/chapter02_dpo/)                                                                       | Generate preference data, train with DPO, and compare model behavior before and after fine-tuning. |
| MDP and value learning | [`code/chapter03_mdp/`](code/chapter03_mdp/)                                                                       | Run bandit strategies, solve GridWorld, and verify Bellman updates numerically.                    |
| Deep Q-learning        | [`code/chapter04_dqn/`](code/chapter04_dqn/)                                                                       | Implement replay buffers, target networks, and Double DQN variants.                                |
| Policy gradient        | [`code/chapter05_policy_gradient/`](code/chapter05_policy_gradient/)                                               | Compare REINFORCE, baseline variants, and Actor-Critic updates.                                    |
| PPO                    | [`code/chapter07_ppo/`](code/chapter07_ppo/)                                                                       | Train LunarLander, inspect clipping, visualize GAE, and compare training stability.                |
| RLHF                   | [`code/chapter08_rlhf/`](code/chapter08_rlhf/)                                                                     | Walk through SFT, reward model training, and PPO-style alignment.                                  |
| Alignment and RLVR     | [`code/chapter09_alignment/`](code/chapter09_alignment/), [`code/chapter09_grpo_rlvr/`](code/chapter09_grpo_rlvr/) | Explore DPO rewards, GRPO group advantages, and rule-based verifiable rewards.                     |
| VLM and agents         | [`code/chapter10_agentic_rl/`](code/chapter10_agentic_rl/), [`code/chapter11_vlm_rl/`](code/chapter11_vlm_rl/)     | Build tool-use agent trajectory synthesis and implement multimodal model RL examples.              |
| Advanced topics        | [`code/chapter12_future_trends/`](code/chapter12_future_trends/)                                                   | Study frontier directions including multi-agent RL and model-based RL.                             |

See [`code/README.md`](code/README.md) for a code index and chapter-specific dependency notes.

## Recommended Learning Path

A practical path through the repository:

1. Read the [course guide](docs/preface/intro.md) and run the CartPole example.
2. Skim the DPO chapter early, even before finishing all theory, to anchor the motivation for LLM post-training.
3. Study Chapters 03-07 in order; this is the conceptual core.
4. After understanding policy gradients and PPO, return to RLHF, DPO, GRPO, and RLVR.
5. Use the debugging and engineering appendices whenever a training run behaves strangely.
6. Treat frontier chapters as extensions: VLM reinforcement learning, Agentic RL, continuous control, multi-agent systems, and test-time reasoning.

## Quick Start

### Read Online

Published course site:

```text
https://walkinglabs.github.io/hands-on-modern-rl/
```

### Run the Documentation Site Locally

Requirements:

- Node.js >= 18.0.0
- npm

```bash
git clone https://github.com/walkinglabs/hands-on-modern-rl.git
cd hands-on-modern-rl
npm install
npm run dev
```

Then open the local VitePress URL shown in the terminal, usually:

```text
http://localhost:5173
```

### Verify the Site

Before submitting a pull request that changes documentation structure, theme code, navigation, build scripts, or generated assets, run:

```bash
npm run verify
```

This checks formatting, lints the VitePress theme, builds the site, and verifies expected build artifacts.

### Run Course Code

Most code examples use Python and are organized by chapter.

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For smaller installs, use chapter-specific requirements files:

```bash
pip install -r chapter01_cartpole/requirements.txt
python chapter01_cartpole/1-ppo_cartpole.py
```

Some chapters may require additional system libraries, GPU support, model downloads, or environment-specific setup. Start with Chapter 01 before running examples that involve LLMs, VLMs, or heavy simulators.

## Repository Structure

```text
hands-on-modern-rl/
|-- docs/                      # VitePress course content
|   |-- .vitepress/            # Site config, navigation, and theme overrides
|   |-- public/                # Static assets copied into the built site
|   |-- preface/               # Course introduction and history
|   |-- chapter*/              # Main course chapters
|   |-- appendix*/             # Supplementary material and references
|   `-- summaries/             # Part-level review and summary notes
|-- code/                      # Runnable examples aligned with chapters
|-- scripts/                   # Maintenance and verification scripts
|-- package.json               # Site scripts and dependencies
|-- AGENTS.md                  # Repository maintenance guide
`-- README.md                  # Main project overview
```

## Development Commands

```bash
npm run dev           # Start the local documentation server
npm run build         # Build the static site
npm run preview       # Preview the built site locally
npm run format        # Format repository files with Prettier
npm run format:check  # Check formatting
npm run lint          # Lint VitePress theme code
npm run verify        # Run format check, lint, build, and artifact verification
```

## Contributing

Contributions should make the course clearer, more accurate, easier to reproduce, or easier to navigate.

Good contributions include:

- Fixing conceptual errors, formulas, diagrams, broken links, or typos.
- Improving explanations without changing the intended learning path.
- Adding small, reproducible experiments that clarify existing chapters.
- Improving scripts, build reliability, navigation, or accessibility.
- Adding high-quality references to papers, official documentation, or widely used open-source implementations.

Please keep pull requests focused. A good PR usually changes one chapter, one experiment, one group of diagrams, or one infrastructure issue at a time.

When adding content:

1. Put course material under [`docs/`](docs/).
2. Use kebab-case for new directories and files.
3. Prefer directory-based routes with `index.md`.
4. Update [`docs/.vitepress/config.mjs`](docs/.vitepress/config.mjs) when adding navigable pages.
5. Run `npm run verify` before requesting review if your change touches config, theme, scripts, or generated site output.
6. Use Conventional Commits, such as `docs: clarify ppo clipping` or `fix: repair chapter link`.

For repository-specific maintenance rules, see [`AGENTS.md`](AGENTS.md).

## Other Courses

Our team has also created other courses. Take a look:

[![LEARN HARNESS ENGINEERING](https://img.shields.io/badge/LEARN_HARNESS_ENGINEERING-0052cc?style=for-the-badge)](https://github.com/walkinglabs/learn-harness-engineering)

## WeChat Group (微信)

For suggestions or feedback, scan the QR code to join the WeChat group (微信):

<img src="https://github.com/walkinglabs/.github/raw/main/profile/wechat.png" alt="WeChat Group" width="300" />

## Citation

If you use this course in teaching materials, study notes, or derivative non-commercial educational work, please cite the repository:

```bibtex
@misc{hands_on_modern_rl,
  title        = {Hands-On Modern RL: Practice-first reinforcement learning from CartPole to LLM post-training and agentic systems},
  author       = {WalkingLabs},
  year         = {2026},
  howpublished = {\url{https://github.com/walkinglabs/hands-on-modern-rl}},
  note         = {Open courseware repository}
}
```

## License

This course is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).

You may share and adapt the material for non-commercial purposes, provided that you give appropriate credit and distribute derivative works under the same license.

## Star History

<a href="https://star-history.com/#walkinglabs/hands-on-modern-rl&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=walkinglabs/hands-on-modern-rl&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=walkinglabs/hands-on-modern-rl&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=walkinglabs/hands-on-modern-rl&type=Date" />
  </picture>
</a>

---

<div align="center">
  <sub>Maintained by WalkingLabs and contributors.</sub>
</div>
