-- Mean demand by local weekday, ordered Monday-first rather than alphabetically.
SELECT
    c.weekday_name,
    avg(d.demand_mw) AS demand_mw
FROM fact_demand d
JOIN dim_calendar c USING (timestamp)
WHERE d.demand_mw IS NOT NULL
GROUP BY c.weekday_name, c.weekday
ORDER BY c.weekday
