Fixtures for tests/test_lab_gpu_thunder.py.

* `openapi_instances_trimmed.json`: the instance and utility paths of Thunder's public OpenAPI document (fetched 2026-11-01; Apache-2.0
  in the document), trimmed to what `gpu_thunder.py` calls.
* `pricing.json`: `GET /v2/pricing` as recorded on 2026-11-01 (public endpoint, no token).
* `status.json`, `list_*.json`, `create_201.json`, `error_400.json`: SYNTHESISED from the OpenAPI schemas and the row shape in
  docs notes (gpu-lifecycle.md 1.1); not recorded from a live create. The `/v2/status` value strings and the create response are
  UNVERIFIED against the real service. All uuids, addresses and keys are made up.
