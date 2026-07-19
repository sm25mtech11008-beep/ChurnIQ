# ChurnIQ — Literature Survey & Concept Primer
### Everything we'll use, explained from intuition up. Read in order; each section ends with self-check questions — if you can answer them aloud, you're interview-ready for that layer.

---

## 1. The problem: churn prediction as supervised classification

Churn = a customer leaving. Businesses care because acquiring a new customer costs 5–7× more than retaining one, so predicting *who is about to leave* lets them intervene (discounts, support calls) before it happens.

Formally: given features X about a customer (tenure, contract type, monthly charges, services), predict a binary label y ∈ {stay, churn}. This is **supervised binary classification**. But note the subtlety that separates juniors from seniors: the business doesn't actually want a label — it wants a **probability** (how likely, so we can rank customers) and a **reason** (so the retention team knows what offer to make). That's why our project includes calibration and explainability, not just a classifier.

**Key framing from the literature:** churn modeling is a cost-sensitive problem. A missed churner costs the customer's lifetime value (CLV); a false alarm costs one retention offer. Those costs are wildly asymmetric, which drives everything in Section 5.

*Self-check: Why is predicting a probability more useful than predicting a label here? What are the two error types and their business costs?*

---

## 2. Data fundamentals: EDA, leakage, and pipelines

**EDA (exploratory data analysis)** is hypothesis generation: you look at distributions and relationships to form beliefs ("month-to-month customers churn more") that guide feature engineering. It is not chart decoration.

**Data leakage** is the most important concept in applied ML. It means information from the future (or from the test set) sneaks into training, producing beautiful offline metrics and a useless production model. Classic forms:
- *Target leakage*: a feature that's a consequence of the target (e.g., "account_closed_date" predicting churn — trivially true, uselessly late).
- *Train–test contamination*: fitting preprocessing (scalers, imputers, encoders) on ALL data before splitting. The test set's statistics leak into training.

The cure for the second form is the **scikit-learn Pipeline**: you chain preprocessing + model into one object, and cross-validation fits the *entire chain* only on training folds. A `ColumnTransformer` routes numeric columns to scaling/imputation and categorical columns to encoding, inside that pipeline. When an interviewer asks "how did you prevent leakage," the answer is: "all preprocessing lives inside the pipeline, so it's re-fit per fold."

**Encoding categoricals:** models need numbers. One-hot encoding creates a binary column per category (safe, standard for low-cardinality features like contract type). Ordinal encoding assigns integers (only when order is real). Target encoding uses churn rates per category (powerful but a leakage trap — needs out-of-fold computation).

*Self-check: Why must a scaler be fit only on the training fold? Give one example of target leakage in a churn dataset.*

---

## 3. Models: from logistic regression to XGBoost

**Logistic regression** is the canonical baseline: a linear model passed through a sigmoid to output probabilities. Strengths: fast, interpretable coefficients, well-calibrated by nature. Weakness: only linear decision boundaries (unless you hand-craft interactions). We use it as the baseline every fancier model must beat — a discipline from empirical ML research.

**Decision trees** split the feature space with if-else rules learned greedily (choose the split that maximizes purity gain — Gini or entropy). They capture non-linearities and interactions automatically but overfit badly alone (high variance).

**Ensembles** fix that in two ways:
- **Bagging (Random Forest, Breiman 2001):** train many deep trees on bootstrap samples with random feature subsets; average them. Variance drops because errors decorrelate.
- **Boosting (gradient boosting, Friedman 2001):** train shallow trees *sequentially*, each one fitting the errors (more precisely, the gradient of the loss) of the ensemble so far. Bias drops step by step. Intuition: it's gradient descent, but each "step" is a small tree.

**XGBoost (Chen & Guestrin, 2016)** is gradient boosting engineered hard: second-order gradients, built-in regularization (the part of the objective penalizing tree complexity), clever handling of missing values, and system-level speed. **LightGBM (Ke et al., 2017)** is its faster sibling using histogram-based splits and leaf-wise growth. For ~15 years this family has been the default winner on tabular data — a finding repeatedly confirmed in the literature ("Why do tree-based models still outperform deep learning on tabular data?", Grinsztajn et al., 2022 — remember this author name; he reappears in Section 7 on the *other* side of the debate).

Key hyperparameters you must be able to explain: `n_estimators` (number of trees), `learning_rate` (shrinkage per step — lower = more trees needed but better generalization), `max_depth` (interaction complexity), `subsample`/`colsample_bytree` (randomness that fights overfitting), `scale_pos_weight` (Section 4).

*Self-check: Bagging vs boosting — which reduces variance, which reduces bias, and why? What does learning_rate trade off against?*

---

## 4. Class imbalance

Our dataset is roughly 73% stay / 27% churn. Consequences: accuracy becomes misleading (predict "stay" always → 73% accuracy, zero value), and the model may under-learn the minority class.

Approaches in the literature:
1. **Cost/weight-based:** give minority-class errors more weight in the loss (`class_weight="balanced"`, XGBoost's `scale_pos_weight`). Clean, no data distortion. Our default.
2. **Resampling:** oversample the minority (or undersample the majority). **SMOTE (Chawla et al., 2002)** synthesizes new minority points by interpolating between neighbors.
3. **Threshold moving:** train normally, then choose a decision threshold ≠ 0.5 (Section 5).

The senior take you should hold: SMOTE is often applied blindly and can distort the feature distribution and *wreck probability calibration* — and recent empirical work argues that for strong learners with proper thresholding, weighting + threshold tuning usually matches or beats resampling. Also, 27% is only *mildly* imbalanced. So: we use class weights + business-driven thresholds, and we can cite having *considered* SMOTE and rejected it with reasons. That answer wins interviews.

*Self-check: Why is accuracy misleading at 73/27? Why can SMOTE hurt calibration?*

---

## 5. Evaluation: metrics, curves, calibration, and the business threshold

**Confusion matrix vocabulary:** TP (predicted churn, did churn), FP (false alarm), FN (missed churner), TN.
- **Precision** = TP/(TP+FP): of those we flagged, how many really churn? Governs wasted retention offers.
- **Recall** = TP/(TP+FN): of real churners, how many did we catch? Governs lost revenue.
- **F1** = harmonic mean; a compromise when you refuse to choose.

**ROC curve** plots TPR vs FPR across all thresholds; **ROC-AUC** is the probability the model ranks a random churner above a random non-churner. **PR curve** (precision vs recall) is more informative under imbalance because it ignores the flood of easy TNs. Report both; emphasize PR-AUC.

**Cross-validation:** k-fold CV (stratified, to preserve class ratios per fold) gives a variance-aware estimate of generalization instead of a single lucky split. All model comparisons in our benchmark use the *same* folds — that's what makes comparisons fair.

**Calibration** asks: when the model says 0.8, do 80% of such customers actually churn? Check with a reliability diagram (predicted prob vs observed frequency) and **Brier score** (mean squared error of probabilities). Boosted trees are often miscalibrated (overconfident); fixes are **Platt scaling** (fit a logistic on the outputs) or **isotonic regression**, both applied on held-out data (`CalibratedClassifierCV`). Calibration matters because Section 5's final step multiplies probabilities by money — garbage probabilities → garbage money math.

**The business threshold (our capstone idea):** define
Expected value of targeting a customer = p(churn) × CLV × offer_success_rate − offer_cost.
Sweep the threshold, compute total expected savings at each, pick the maximizer. Now your "model metric" is *rupees saved*, not F1 — and you've connected ML to business, which is precisely what "data science engineer" means.

*Self-check: Why prefer PR-AUC under imbalance? What does a point below the diagonal on a reliability diagram mean? Why does the optimal threshold depend on CLV?*

---

## 6. Explainability: Shapley values and SHAP

Retention teams won't act on a black box, and regulators increasingly require reasons. **SHAP (Lundberg & Lee, 2017)** grounds feature attribution in **Shapley values** from cooperative game theory (Shapley, 1953): a feature's contribution to a prediction is its average marginal contribution across all possible orderings in which features could be "added" to the model. It's the unique attribution scheme satisfying fairness axioms (efficiency: contributions sum to prediction-minus-baseline; symmetry; etc.).

Exact Shapley values are exponential to compute, but **TreeSHAP** exploits tree structure to compute them exactly in polynomial time — which is why SHAP + XGBoost is the industry-standard pairing.

Two views you'll produce:
- **Global**: mean |SHAP| per feature = which features drive churn overall (summary/beeswarm plot).
- **Local**: one customer's waterfall plot = why *this* person is at risk. This feeds both the API's `/explain` endpoint and the LLM assistant.

Caveat to know: SHAP explains the *model*, not causality. "Low tenure has high SHAP" ≠ "increasing tenure causes retention." Saying this unprompted in an interview is a flex.

*Self-check: What fairness property do Shapley values satisfy? Why is TreeSHAP fast? SHAP vs causality — what's the trap?*

---

## 7. Tabular foundation models — the 2026 research frontier

Background arc (this is your actual literature survey story):
1. 2016–2022: GBDTs rule tabular ML; deep learning attempts (TabNet, FT-Transformer) mostly fail to beat them (Grinsztajn et al., 2022).
2. **TabPFN (Hollmann et al., 2022)** flips the paradigm: a transformer **pre-trained once on millions of *synthetic* datasets** sampled from structural causal models. At use time it does **in-context learning (ICL)**: your entire training set is fed in as context, and predictions for test rows come out in a single forward pass. `.fit()` stores data; **no gradients, no training on your data**. Conceptually it approximates Bayesian inference: the synthetic pre-training defines a prior over "plausible tabular worlds," and the forward pass computes a posterior predictive.
3. **TabPFN v2 (Hollmann et al., Nature, Jan 2025)** made it practical (thousands of rows, mixed types, missing values) — the moment tabular foundation models (TFMs) became serious contenders.
4. 2025–2026: rapid ecosystem growth — TabPFN-2.5, TabICL/TabICL v2 (scaling ICL), CARTE (graph-based, uses column semantics), community benchmark **TabArena** where TFMs now trade blows with tuned GBDT ensembles.
5. **June 30, 2026 — Google releases TabFM**, a zero-shot TFM with an sklearn-style API, hybrid architecture, synthetic pre-training, and hard limits (~10 classes, ~500 features). Big-tech entry = the field went mainstream.

Why TFMs can work with no training: the transformer's attention lets test rows "attend" to training rows — think of it as a *learned, highly non-linear nearest-neighbor + reasoning machine* whose distance function and inference procedure were meta-learned from millions of synthetic tasks. That's the same mechanism as LLM few-shot prompting, applied to rows and columns.

Known trade-offs (what our benchmark measures): TFMs shine on **small-to-medium data** (hundreds to a few thousand rows — often matching tuned GBDTs with far less data), tend to produce **well-calibrated distributional outputs by design**, but cost more at **inference** (the whole training set rides along in every forward pass), have context-size limits, and lack native SHAP-style explainability. GBDTs remain cheap, scalable, and interpretable. **Whether TFMs beat tuned GBDTs, and when, is an open research question in 2026 — our project contributes a careful case study on churn data.**

*Self-check: What is the model pre-trained on and why synthetic? Why does `.fit()` train nothing? Name two regimes where XGBoost still wins. What is the Bayesian interpretation of the forward pass?*

---

## 8. The MLOps layer: tracking, serving, containers, CI

**Experiment tracking (MLflow):** research needs reproducibility; without tracking you get "best_model_final_v3_REAL.pkl". MLflow logs, per run: parameters, metrics, artifacts (plots, models), and code version. Concepts: a *run* (one training execution), an *experiment* (group of runs), the *model registry* (versioned promoted models). Our benchmark = many tagged runs you can sort in a table.

**Serving (FastAPI):** a model is only useful behind an interface. FastAPI is a Python web framework where you define endpoints (`POST /predict`); **Pydantic** models declare the request schema, giving automatic validation (bad input → clean 422 error, not a crash) and auto-generated interactive docs (Swagger at `/docs`). Concepts to learn: HTTP methods, JSON request/response, status codes, why the model loads once at startup (not per request).

**Docker:** packages your app + Python + libraries + OS deps into an *image*; a running image is a *container*. Solves "works on my machine" permanently and is the deployment unit everywhere in industry. Concepts: Dockerfile (build recipe, layer caching), image vs container, ports (`-p 8000:8000`), docker-compose (run API + database together, one command).

**Testing + CI (pytest, GitHub Actions):** unit tests pin down behavior (pipeline output shape, API contract, "no NaN survives preprocessing"). CI runs those tests automatically on every push — a robot reviewer that never sleeps. Even 8 good tests + a green badge signals engineering maturity that 95% of student projects lack.

*Self-check: What exactly does MLflow store per run? What does Pydantic buy you? Image vs container? What happens on git push with CI configured?*

---

## 9. The analytics layer: dimensional modeling and DAX

Power BI rewards proper data modeling. **Star schema (Kimball's dimensional modeling):** one central **fact table** (events/measurements — our predictions: customer_id, probability, risk tier, revenue at risk) surrounded by **dimension tables** (descriptive context — customer attributes, risk-tier definitions, dates). Why not one flat table? Slicers and filters work naturally along dimensions, measures aggregate correctly, the model stays small and fast, and it's the vocabulary BI interviewers expect.

**DAX** is Power BI's formula language. The one distinction that matters: a **calculated column** is computed row-by-row at refresh and stored; a **measure** is computed at query time *within the current filter context* (whatever slicers the user has clicked). "Revenue at risk" must be a measure so it responds to filters. `CALCULATE()` is DAX's power tool — it evaluates an expression under a *modified* filter context (e.g., churn rate *for month-to-month customers only*). Add a **What-If parameter** (a slider feeding a measure) to make savings projections interactive.

*Self-check: Fact vs dimension table? Measure vs calculated column? What does CALCULATE change?*

---

## 10. The GenAI layer: LLM tool calling and grounding

An LLM alone would happily invent a customer's churn probability — **hallucination**. The fix is **tool calling (function calling)**: you register functions with schemas (e.g., `get_customer_risk(customer_id)` → returns prediction + SHAP drivers from our store); the LLM decides *when* to call them, your code executes them, and the LLM composes its answer **only from returned data**. This "grounding via tools" pattern is the backbone of 2026 agentic systems, and our assistant is a clean minimal example: two tools (customer lookup, aggregate SQL stats), strict rule that every number in an answer must come from a tool result. LangChain is optional scaffolding; the concept is what matters.

*Self-check: Why can't we let the LLM answer from its weights? Walk through the message flow of one tool call.*

---

## 11. Reading list (in order of payoff)

**Must-read (core of your survey):**
1. Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System* (KDD 2016) — read §2 (the objective with regularization); skim the systems parts.
2. Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (NeurIPS 2017) — SHAP; focus on the axioms and TreeSHAP idea.
3. Hollmann et al., *Accurate predictions on small data with a tabular foundation model* (Nature 2025) — TabPFN v2; the paradigm-shift paper for your research module.
4. Grinsztajn et al., *Why do tree-based models still outperform deep learning on tabular data?* (NeurIPS 2022) — the "before" picture that makes #3 meaningful.

**Skim / reference:**
5. TabPFN-2.5 report (arXiv 2511.08667, 2025–26) and Google's TabFM release notes (June 2026) — for current capabilities and limits.
6. Friedman, *Greedy Function Approximation* (2001) — origin of gradient boosting; read for the "gradient descent in function space" intuition only.
7. Chawla et al., *SMOTE* (2002) — so you can critique it credibly.
8. Kimball's star-schema chapter summary (any online primer) + MLflow "Concepts" docs page + FastAPI tutorial first 3 sections + Docker "Get Started" part 1–2.

**How to survey like a researcher (you know this from your thesis):** for each paper capture in your own words — problem, key idea, what it replaced, limitation, and one question you'd ask the authors. Keep a `docs/literature_notes.md` in the repo; it doubles as interview prep and proves scholarly depth.

---

## 12. The one-paragraph synthesis (memorize the shape of this)

> Churn prediction is cost-sensitive binary classification. We establish a leakage-free sklearn pipeline and a logistic baseline, beat it with weighted XGBoost, choose the operating threshold by expected business value rather than F1, verify the probabilities via calibration, and explain predictions with TreeSHAP globally and locally. We then benchmark this classical stack against 2026's tabular foundation models — which replace training with in-context learning over synthetic-data-derived priors — across accuracy, calibration, data efficiency, and inference cost. The winning model ships behind a validated FastAPI service in Docker with CI, its predictions land in a star-schema warehouse powering DAX-driven Power BI dashboards, and a tool-grounded LLM assistant turns per-customer SHAP drivers into retention actions.

When you can say that paragraph *and defend every clause*, the project is yours in a way no copy-paste ever is.
