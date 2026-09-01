-- Mean demand for each local hour, split by weekday / weekend / public holiday.
-- Feeds reports/figures/daily_profile_by_daytype.png.
SELECT
    c.day_type,
    c.hour,
    avg(d.demand_mw) AS demand_mw
FROM fact_demand d
JOIN dim_calendar c USING (timestamp)
WHERE d.demand_mw IS NOT NULL
GROUP BY c.day_type, c.hour
ORDER BY c.day_type, c.hour
