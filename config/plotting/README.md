# Plotting setup

These toolbox-wide defaults apply to Python plots in every project using this checkout.

| File | Owns |
|---|---|
| `colors.yml` | Categorical, comparison, scenario, climate, anomaly and map colours |
| `style.yml` | Page/export defaults, typography and shared chart/evaluation appearance |

Edit the YAML and restart any interactive plotting process. Workflow renderers read the setup in each new
process and track both files by content digest. The historical display-reference producer also tracks
`colors.yml`, because that record retains map colours. Neither file enters scenario or simulation identity.

For loaders and adapter boundaries, use the
[technical configuration reference](../../docs/technical-manual/configuration.md).

Quote hex colours. Named dataset/evaluation entries reference `default_cycle` names; SSP entries remain
independent. Neutral and map-feature inks also accept quoted grayscale fractions from 0 to 1.
Style lengths are millimetres and font/line sizes are points; alpha is a fraction.

Classification, normalization, completeness, statistical methods and quantity-specific thresholds remain
in Python. Geometry and artist placement specific to one figure also remain with that renderer.
R/weathergenr plots retain their own setup. The agreed appearance and exceptions live in
[plot style](../../dev/reference/toolbox-plot-style.md); user choices are in
[configuration](../../docs/technical-manual/configuration-and-execution.md#plot-annotations).
