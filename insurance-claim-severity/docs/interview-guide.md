# Understand the project before presenting it

This is an AI-assisted implementation created in November 2026. Describe the parts you have run, inspected, changed and understood. Do not describe synthetic evaluation as a result on 188,318 real claims, or repeat the earlier 23% claim without a measured real-data run.

## One-minute explanation

“This project estimates the cost of an insurance claim. The workflow checks the data, stores it in a SQL database, and separates it into training, validation and test sets. Categorical inputs use frequency encoding, and numerical inputs use imputation and a statistical transformation. I compare linear regression with XGBoost and measure MAE in the original claim-cost units. Finally, the model's predicted costs define three severity bands. The current reproducible demonstration uses synthetic data; the real Allstate evaluation is pending.”

Adapt this to what you personally completed. Reading the code is different from having designed or implemented it independently.

## Questions to be able to answer

**What is the target?** `loss`, a continuous claim amount. This is regression, not classification. The bands are a separate post-processing step.

**Why SQL?** It gives a clear data contract, keys connecting claims to features and source batches, transaction safety, and reusable queries. This implementation uses SQLite, not PostgreSQL.

**What is frequency encoding?** If A occurs in 60 of 100 training rows, A becomes 0.60. It avoids a large one-hot matrix but loses identity when categories have equal frequency. A category absent from training becomes zero.

**Why not encode before splitting?** Even feature-only statistics would then include the evaluation distribution. Every learned preprocessing step belongs inside a training-fitted pipeline.

**Why transform loss?** Very large costs produce a right tail. `log1p` compresses that tail and allows zero. Invert using `expm1` before calculating monetary-unit error. This changes the learning objective; it is not automatically an improvement.

**What is XGBoost?** A sequence of decision trees that together improve predictions. Later trees optimize the remaining objective error. Depth, learning rate, number of trees, regularization and sampling control complexity.

**Why a baseline?** A stronger model is useful only if it improves on something simple on unseen data. Both a median predictor and linear regression are included.

**How was model selection done?** Two XGBoost depths, 3 and 5, are compared using validation MAE. The selected model is evaluated on a separate test set. This is a small search, not exhaustive tuning or cross-validation.

**What does 23% lower MAE mean?** For example, moving from 1,000 to 770 gives a 23% reduction. These numbers are an illustration, not this project's result. Use only the measured report and state which data produced it.

**How are risk groups made?** The one-third and two-thirds quantiles of validation predicted cost become fixed cutoffs. New predictions are classified by those cutoffs. They are severity bands, not estimated claim probabilities or independently trained risk classes.

**What would you improve?** Compare one-hot encoding, validate with policy/time groups if available, inspect tail errors, compare direct MAE objectives, and choose thresholds based on an actual operational use case.

## Practice tasks

1. Run the demo and explain one SQL row, one encoded category and one prediction.
2. Change the declared depth candidates and compare validation results in a new output directory; don't tune against the test results repeatedly.
3. Obtain the official dataset, run the pipeline, and replace unsupported CV statistics with the recorded dataset size and measured evaluation.
