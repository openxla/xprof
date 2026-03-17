# Example PerfettoSQL Queries for Megascale Viewer

## recv-done ops

```sql
SELECT
  s.id,
  s.name,
  s.ts,
  s.dur,
  EXTRACT_ARG(s.arg_set_id, 'debug.run_id') AS run_id,
  ppt.name AS device,
  s.track_id,
  s.slice_id
FROM slice AS s
JOIN track AS pt
  ON s.track_id = pt.id
JOIN track AS ppt
  ON pt.parent_id = ppt.id
WHERE LOWER(pt.name) GLOB LOWER('*XLA Ops*')
AND s.name REGEXP 'recv-done.\d+$';
```

This query will return a list of all recv-done ops in the trace.

## recv-done stats

```sql
SELECT
  s.name AS name,
  pt.name AS device,
  ROUND(PERCENTILE(s.dur, 50), 2) AS dur_ns_p50,
  ROUND(PERCENTILE(s.dur, 90), 2) AS dur_ns_p90,
  ROUND(PERCENTILE(s.dur, 99), 2) AS dur_ns_p99,
  ROUND(AVG(s.dur), 2) AS dur_ns_mean,
  COUNT(*) AS count,
  SUM(s.dur) AS dur_ns_sum,
  ROUND(PERCENTILE(s.dur, 99) / AVG(s.dur), 2) AS p99_over_mean
FROM slice AS s
JOIN track AS t
  ON s.track_id = t.id
JOIN track AS pt
  ON t.parent_id = pt.id
WHERE LOWER(t.name) GLOB LOWER('*XLA Ops*')
AND s.name REGEXP 'recv-done.\d+$'
GROUP BY s.name, device;
```

This query will return statistics for all recv-done ops in the trace. It is
helpful for understanding the distribution of recv-done durations.

## NetworkReceive Actions

```sql
SELECT
  s.id AS id,
  s.name AS name,
  s.ts AS ts,
  s.dur AS dur,
  EXTRACT_ARG(s.arg_set_id, 'debug.network_transport_latency_us') AS network_latency_us,
  EXTRACT_ARG(s.arg_set_id, 'debug.action_duration_ns') AS action_duration_ns,
  EXTRACT_ARG(s.arg_set_id, 'debug.buffer_sizes') AS buffer_sizes,
  EXTRACT_ARG(s.arg_set_id, 'debug.run_id') AS run_id,
  ppt.name AS device,
  s.slice_id AS slice_id,
  s.track_id AS track_id
FROM slice AS s
JOIN track AS t
  ON s.track_id = t.id
JOIN track AS pt
  ON t.parent_id = pt.id
JOIN track AS ppt
  ON pt.parent_id = ppt.id
WHERE pt.name = 'Megascale'
AND s.name = 'NetworkReceive END';
```

This query will return a list of all megascale NetworkReceive events in the
trace. It is helpful for understanding action durations and their network
latency.
