"""Identical shared numerical producers in the WF0 and WF1 DAGs."""

from blueearth_cst.shared.map_preparation import source_fields_path, source_map_inputs, source_spec_dict

for _map_spec in MAP_REFERENCE.source_specs:
    rule:
        name: f"prepare_historical_map_fields_{_map_spec.source}"
        message: PREPARE_SOURCE_MAPS.banner(part=_map_spec.source)
        input: **source_map_inputs(_map_spec),
        output: fields=source_fields_path(_map_spec).as_posix(),
        params:
            source_spec=source_spec_dict(_map_spec),
            m0=MAP_REFERENCE.params["m0"],
            completeness=MAP_REFERENCE.params["completeness"],
            preparation_code_digest=MAP_REFERENCE.params["preparation_code_digest"],
            environment=MAP_REFERENCE.params["environment"],
        log: PREPARE_SOURCE_MAPS.log(_map_spec.source),
        benchmark: PREPARE_SOURCE_MAPS.benchmark(_map_spec.source),
        script: "../prepare_source_map_fields.py"
