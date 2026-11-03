# ERFA numerical reference

`eflow-erfa-r-oracle.csv` contains 210 independently calculated R values: all 70 indicators with
January, April and October water-year starts. Generated on 2026-10-21 with R 4.6.1 from the clean
`eflow-risk-tool` checkout at commit `16ad6f424bfa5425ce0c0403471482b5e7a5cdb4`.
The source package is `eflowrisk`; its `R/annual.R`, `R/metrics.R`, `R/utils.R` and `R/summarize.R`
define the calculation. No R package or external checkout is needed to run the Python test.

The deterministic input is `oracle_input()` in `tests/test_eflow_erfa_oracle.py`: five calendar years
including leap days, tied extremes, zero-flow sequences, missing February observations and a year
that fails completeness. The test compares every registered scalar and its independent checker,
with `rtol=1e-10`, `atol=1e-12`, and equal undefined values.

To regenerate, export `oracle_input().rename_axis("date").to_csv(input_path)`, then run the following
in the source package's selected R environment, substituting explicit input/output paths and source root:

```r
for (name in c("utils", "metrics", "annual", "summarize")) {
  source(file.path(source_root, "R", paste0(name, ".R")))
}
input <- read.csv(input_path)
results <- lapply(c(1L, 4L, 10L), function(start) {
  annual <- calculate_flow_metrics(input$q, as.Date(input$date), water_year_start = start)
  summary <- summarize_flow_metrics(annual)
  data.frame(
    anchor = c("YE-DEC", "YE-MAR", "YE-SEP")[match(start, c(1L, 4L, 10L))],
    indicator = paste0("q_", summary$metric_key, "_", summary$stat),
    value = summary$value
  )
})
write.csv(do.call(rbind, results), output_path, row.names = FALSE, na = "NA")
```

Inspect changes to source revision, defaults and reference values before replacing this fixture.
