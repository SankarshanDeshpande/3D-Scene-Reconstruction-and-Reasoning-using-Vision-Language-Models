Positioning Note — 3D Scene Reconstruction and Reasoning using Vision-Language Models
Sankarshan Deshpande (P25DS013) · Guide: Prof. M.A. Zaveri · SVNIT Surat · October 2026

Revised Problem Statement

Open-vocabulary 3D scene graph construction and language-grounded spatial reasoning systems for indoor environments are built on the assumption of sensor-measured depth and known camera poses. This dissertation investigates whether the geometry predicted by a feed-forward, RGB-only multi-view reconstruction model (VGGT) can substitute for that sensed geometry without unacceptable degradation of downstream semantic and reasoning performance. The work quantifies where and why that substitution degrades, relates the observed degradation to the reconstruction model’s own predicted confidence, and evaluates the resulting representation on spatial question answering over five ReplicaOcc indoor scenes.

Why the Original Framing Required Revision

The original problem statement framed the work as building a reconstruction pipeline that operates without a depth sensor. This describes what VGGT itself does — a solved problem, published at CVPR 2025. A pipeline reproduction is not a research contribution. The revised framing makes the completed reconstruction work the independent variable rather than the contribution, and places the contribution in the characterisation of its downstream effect.

Four Contribution Claims
Scale-propagation characterisation. The first systematic study of how scale ambiguity in feed-forward predicted depth (measured mean scale factor: 2.44×, range 1.98–3.02× across five scenes) propagates into open-vocabulary 3D scene graph construction, where ConceptGraphs uses absolute metric thresholds for object association.
Confidence-guided diagnosis and mitigation. A demonstration that VGGT’s own per-pixel predicted confidence is a usable predictor of downstream scene-graph node failure, with a confidence-gated fusion variant that measurably improves node precision.
Failure-mode taxonomy. A taxonomy linking reconstruction-level failures (soft surfaces, reflective regions, small-object merging, boundary error) to scene-graph and QA-level failures.
Space3D-Bench evaluation. Evaluation on spatial question answering over five Replica scenes, comparing sensed-depth, predicted-depth, and confidence-gated predicted-depth conditions.
Relation to the Closest Existing Work

arXiv 2605.18197 (May 2026) performs a closely related substitution experiment using MapAnything (a metric-scale backbone) instead of VGGT (scale-ambiguous). This work is differentiated on five grounds: (1) VGGT’s scale ambiguity creates phenomena that a metric backbone cannot exhibit; (2) confidence-guided analysis is absent from existing work; (3) edge-level and QA-level evaluation are not reported in the existing work; (4) a failure-mode taxonomy connecting reconstruction to scene-graph degradation does not exist in the existing work; (5) scale-alignment ablation is impossible in a metric-backbone setting.

What Is Not Claimed
VGGT is not claimed to be the current state of the art in feed-forward reconstruction (MapAnything, Pi3, and VGGT-Omega have since been published).
ConceptGraphs is not claimed to be the current state of the art in scene graph construction (Sparse3DPR reports 28.7% EM@1 improvement over it). It is used as the established baseline the substitution experiment requires.
Scene-LLM is not used for the reasoning stage; its training requirements are infeasible in the remaining timeline and its validation benchmarks (ScanQA, SQA3D) have been challenged by Real-3DQA (ICLR 2026).
