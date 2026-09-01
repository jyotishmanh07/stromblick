-- Champion forecast error sliced by local hour, weekday, month and holiday status.
-- The pandas equivalent is evaluation.error_slices(); tests assert they agree.
-- One long table rather than four wide ones, so a single query answers "where does it err".
WITH scored AS (
    SELECT
        c.hour, c.weekday, c.weekday_name, c.month, c.is_public_holiday,
        abs(f.demand_mw - f.predicted) AS absolute_error
    FROM fact_forecast f
    JOIN dim_calendar c USING (timestamp)
    WHERE f.model = ? AND f.demand_mw IS NOT NULL
)
SELECT 'hour'    AS slice, CAST(hour AS VARCHAR) AS bucket,
       avg(absolute_error) AS absolute_error, count(*) AS hours,
       CAST(hour AS INTEGER) AS sort_key
FROM scored GROUP BY hour
UNION ALL
SELECT 'weekday', weekday_name, avg(absolute_error), count(*), CAST(weekday AS INTEGER)
FROM scored GROUP BY weekday_name, weekday
UNION ALL
SELECT 'month', CAST(month AS VARCHAR), avg(absolute_error), count(*), CAST(month AS INTEGER)
FROM scored GROUP BY month
UNION ALL
SELECT 'holiday', CAST(is_public_holiday AS VARCHAR), avg(absolute_error), count(*),
       CAST(is_public_holiday AS INTEGER)
FROM scored GROUP BY is_public_holiday
ORDER BY slice, sort_key
