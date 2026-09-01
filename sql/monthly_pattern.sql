-- Mean demand for each calendar month of the snapshot (local time).
SELECT
    c.year_month,
    avg(d.demand_mw) AS demand_mw
FROM fact_demand d
JOIN dim_calendar c USING (timestamp)
WHERE d.demand_mw IS NOT NULL
GROUP BY c.year_month
ORDER BY c.year_month
