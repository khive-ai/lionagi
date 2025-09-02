# Event 007: Metrics Collection

## Event Definition

```yaml
event:
  id: "007"
  name: "metrics.collection"
  description: "Post-implementation metrics gathering for ROI analysis"
  triggers:
    - "Implementation completed"
    - "Label changed to 'stage:metrics.collection'"
    - "Stabilization period ended"
  stage: "measurement"
  parallelizable: true # Can collect from multiple sources
```

## Trigger Conditions

- Has label: `stage:metrics.collection`
- Implementation marked complete
- Baseline metrics documented
- Collection period defined (typically 30-90 days)

## Collection Context

```yaml
collection_context:
  implementation_id: "Reference to implementation"
  baseline_metrics: "Pre-implementation values"
  target_metrics: "Expected values"
  collection_period: "Duration for gathering"
  data_sources: "Where metrics come from"
```

## Agent Assignment

- **Primary**: ROI Analysis Agent
- **Support**: Implementation Tracker Agent
- **Validation**: Critic Agent

## Collection Strategy

```yaml
metrics_categories:
  performance:
    metrics: ["latency", "throughput", "error_rate"]
    frequency: "Every 15 minutes"
    aggregation: "P50, P95, P99"

  business:
    metrics: ["conversion", "revenue", "user_satisfaction"]
    frequency: "Daily"
    aggregation: "Daily average, weekly trend"

  operational:
    metrics: ["incidents", "maintenance_time", "resource_usage"]
    frequency: "Weekly"
    aggregation: "Count, sum, average"

  quality:
    metrics: ["bug_count", "tech_debt", "test_coverage"]
    frequency: "Per release"
    aggregation: "Delta from baseline"
```

## State Transitions

- **Collection Active**: → Gathering data
- **Milestone Reached**: → Interim analysis
- **Period Complete**: → Event 008 (metrics.review)
- **Anomaly Detected**: → Investigation required

## GitHub Label Updates

```yaml
on_start:
  add: ["status:measuring", "metrics:collecting"]
  remove: ["status:complete"]

on_milestone:
  add: ["metrics:week-{n}"]

on_complete:
  add: ["stage:metrics.review", "metrics:ready"]
  remove: ["stage:metrics.collection", "status:measuring"]
```

## Collection Start Comment

````markdown
[ROI-ANALYSIS-{timestamp}] Metrics Collection Started 📊

**Implementation**: {implementation_name} **Collection Period**: {start_date} to
{end_date} **Review Checkpoints**: {checkpoint_schedule}

### Baseline Metrics (Pre-Implementation)

```yaml
baseline:
  performance:
    p95_latency: 45ms
    error_rate: 0.5%
  business:
    daily_revenue: $10,000
    conversion_rate: 2.5%
  operational:
    incidents_per_week: 3
    deployment_time: 2h
```
````

### Target Metrics

```yaml
targets:
  performance:
    p95_latency: "<50ms"
    error_rate: "<0.5%"
  business:
    daily_revenue: ">$12,000"
    conversion_rate: ">3.0%"
  operational:
    incidents_per_week: "<2"
    deployment_time: "<1h"
```

### Collection Schedule

- **Performance**: Continuous (15-min intervals)
- **Business**: Daily at 00:00 UTC
- **Operational**: Weekly on Mondays
- **Quality**: Per release

First checkpoint: {first_checkpoint_date}

````
## Weekly Collection Update
```markdown
[ROI-ANALYSIS-{timestamp}] Week {n} Metrics Update

**Status**: 📈 Collecting | ⚠️ Anomaly Detected
**Data Quality**: {percentage}% complete

### Performance Metrics Trend
````

Week 1: ████████ 43ms (↓4.4%) Week 2: ███████░ 41ms (↓8.9%) Week 3: ██████░░
39ms (↓13.3%) Target: ████████ <50ms ✅

```
### Business Metrics Summary
| Metric | Baseline | Current | Change | Target | Status |
|--------|----------|---------|---------|---------|---------|
| Revenue | $10,000 | $11,500 | +15% | >$12,000 | 🟡 |
| Conversion | 2.5% | 2.9% | +16% | >3.0% | 🟡 |

### Anomalies & Insights
{anomaly_analysis}

**Next Update**: {next_week}
```

## Automated Collection Events

```yaml
event_emissions:
  on_threshold_breach:
    event: "metrics.threshold_breached"
    payload:
      metric: "{metric_name}"
      value: "{current_value}"
      threshold: "{threshold_value}"
      direction: "above|below"

  on_milestone:
    event: "metrics.milestone_reached"
    payload:
      milestone: "week_{n}|month_{n}"
      summary: "{metrics_summary}"

  on_anomaly:
    event: "metrics.anomaly_detected"
    payload:
      metric: "{metric_name}"
      expected_range: "{range}"
      actual_value: "{value}"
      severity: "low|medium|high"
```

## Data Quality Monitoring

```yaml
quality_checks:
  completeness:
    target: ">95% data points collected"
    action_if_low: "Investigate collection issues"

  accuracy:
    validation: "Cross-reference multiple sources"
    outlier_detection: "Statistical analysis"

  timeliness:
    requirement: "Data available within SLA"
    alert_on_delay: true
```

## Interim Analysis Format

```markdown
[ROI-ANALYSIS-{timestamp}] Interim Analysis - {checkpoint_name}

**Period Analyzed**: {period} **Confidence Level**: {high|medium|low}

### Key Findings

1. **Performance**: {summary_of_performance_changes}
2. **Business Impact**: {summary_of_business_metrics}
3. **Operational**: {summary_of_operational_improvements}

### Projected ROI

Based on {n} weeks of data:

- **Investment**: ${total_investment}
- **Returns to Date**: ${returns_so_far}
- **Projected Annual**: ${projected_annual}
- **Estimated ROI**: {percentage}%

### Recommendations

{any_course_corrections_needed}

Full review scheduled: {final_review_date}
```

## Collection Completion

```markdown
[ROI-ANALYSIS-{timestamp}] Metrics Collection Complete ✅

**Collection Period**: {full_period} **Data Points Collected**: {total_count}
**Data Quality Score**: {percentage}%

### Final Collection Summary

{high_level_metrics_summary}

### Preparing for Review

All metrics have been collected and validated. The comprehensive metrics review
will:

- Calculate final ROI
- Compare predictions vs actuals
- Extract lessons learned
- Generate recommendations

**Metrics Review Scheduled**: {review_date} **Review Owner**: @{reviewer}
```

## Machine-Readable Metrics Format

```yaml
# Standardized format for automated processing
metrics_snapshot:
  timestamp: "{ISO_8601}"
  implementation_id: "{impl_id}"
  period: "{start_date} to {end_date}"
  categories:
    performance:
      p95_latency:
        baseline: 45
        current: 39
        unit: "ms"
        change_percent: -13.3
        meets_target: true
    business:
      daily_revenue:
        baseline: 10000
        current: 11500
        unit: "USD"
        change_percent: 15
        meets_target: false
```

## Integration Points

- **Dashboards**: Real-time metric feeds
- **Alerts**: Threshold breach notifications
- **Reports**: Weekly automated summaries
- **Knowledge Graph**: Pattern extraction

## Monitoring Metrics

- Collection completeness: >95%
- Data quality score: >90%
- Anomaly detection rate: <5% false positives
- Stakeholder engagement: >80% read reports
- Time to insight: <48h from collection
