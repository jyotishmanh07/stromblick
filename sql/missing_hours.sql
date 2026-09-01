-- Data-quality view: hours absent from the index, and hours indexed but unpublished.
-- Gaps are reported, never imputed. generate_series includes both endpoints.
WITH bounds AS (
    SELECT min(timestamp) AS lo, max(timestamp) AS hi FROM fact_demand
),
expected AS (
    SELECT unnest(generate_series(lo, hi, INTERVAL 1 HOUR)) AS timestamp
    FROM bounds
)
SELECT
    e.timestamp,
    CASE WHEN d.timestamp IS NULL THEN 'absent from index' ELSE 'indexed, no value' END AS gap_type
FROM expected e
LEFT JOIN fact_demand d USING (timestamp)
WHERE d.timestamp IS NULL OR d.demand_mw IS NULL
ORDER BY e.timestamp
