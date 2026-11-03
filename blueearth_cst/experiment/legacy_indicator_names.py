"""Read-only vocabulary bridge for publications saved before the indicator rename.

Only contract keys and schema identifiers change in memory. Authored names,
paths, values, digests and files remain exactly as saved. Callers must validate
original identities before normalizing a record.
"""

LEGACY_KEYS = {
    "metric": "indicator",
    "metric_set_id": "indicator_set_id",
    "metrics_manifest_sha256": "indicators_manifest_sha256",
    "metric_manifest": "indicator_manifest",
    "metric_definition": "indicator_definition",
    "metric_definition_sha256": "indicator_definition_sha256",
    "metric_environment": "indicator_environment",
    "metric_environment_sha256": "indicator_environment_sha256",
}


def indicator_names(value):
    """Normalize a validated legacy record without rewriting stored references."""
    if isinstance(value, list):
        return [indicator_names(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        name = LEGACY_KEYS.get(key, key)
        if name in result:
            raise ValueError(f"ambiguous legacy indicator field: {name}")
        if key == "schema_version" and isinstance(item, str):
            item = item.replace("metric", "indicator")
        result[name] = indicator_names(item)
    return result


def saved_set_markers(root):
    """Discover both generations; never move or edit a retained publication."""
    return sorted(
        [
            *(root / "_engine/indicator_sets").glob("*/indicators.json"),
            *(root / "_engine/metric_sets").glob("*/metrics.json"),
        ]
    )
