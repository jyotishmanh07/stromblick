-- Headline benchmark table straight from the stored per-hour forecasts.
-- Per-origin MAE is averaged across origins (not pooled over hours) so each 24-hour
-- window counts once -- the same sampling unit the significance tests use.
WITH per_origin AS (
    SELECT
        model,
        origin,
        avg(abs(demand_mw - predicted))              AS mae,
        sqrt(avg(pow(demand_mw - predicted, 2)))     AS rmse,
        avg(200 * abs(demand_mw - predicted)
            / nullif(abs(demand_mw) + abs(predicted), 0)) AS smape
    FROM fact_forecast
    WHERE demand_mw IS NOT NULL
    GROUP BY model, origin
)
SELECT
    model,
    count(*)        AS origins,
    avg(mae)        AS mae_mean,
    stddev_samp(mae) AS mae_std,
    avg(rmse)       AS rmse_mean,
    avg(smape)      AS smape_mean
FROM per_origin
GROUP BY model
ORDER BY mae_mean
