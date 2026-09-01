-- Demand level on public holidays against ordinary days.
-- The forecasting counterpart (holiday error penalty) lives in error_slices.sql.
SELECT
    CASE WHEN c.is_public_holiday = 1 THEN 'Public holiday' ELSE 'Ordinary day' END AS day_class,
    count(*)              AS hours,
    avg(d.demand_mw)      AS mean_demand_mw,
    median(d.demand_mw)   AS median_demand_mw,
    min(d.demand_mw)      AS min_demand_mw,
    max(d.demand_mw)      AS max_demand_mw
FROM fact_demand d
JOIN dim_calendar c USING (timestamp)
WHERE d.demand_mw IS NOT NULL
GROUP BY day_class
ORDER BY day_class
