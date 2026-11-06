-- Run against an experiment's outputs/.../claims.sqlite database.
-- Batch quality and dataset provenance.
SELECT batch_id, source, row_count, sha256, created_at FROM ingestion_batches;

-- Claim costs are in the dataset's original units; do not assume INR.
SELECT batch_id, COUNT(*) AS claims, AVG(loss) AS mean_loss,
       MIN(loss) AS minimum_loss, MAX(loss) AS maximum_loss
FROM claims GROUP BY batch_id;

-- Example relational join: category-level severity, not causal attribution.
SELECT c.batch_id, f.cat1, COUNT(*) AS claims, AVG(c.loss) AS mean_loss
FROM claims c JOIN claim_features f
  ON c.batch_id = f.batch_id AND c.claim_id = f.claim_id
GROUP BY c.batch_id, f.cat1 ORDER BY c.batch_id, mean_loss DESC;
