"""移植 ``packages/ai/test/model-data-validation.test.ts``。

上游文件测试的是 ``packages/ai/scripts/model-data.ts``——一个未移植到 Python 的
生成器脚本。为避免削弱覆盖，本模块把该脚本的校验逻辑（上游测试涉及的部分）
移植进测试文件，运行相同的 fixture，并额外对随附的已生成数据做一次检查。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from pi_ai.models_generated import BUILTIN_PROVIDER_IDS
from pi_ai.providers.catalog import PROVIDER_CATALOG_DATA_DIR, available_provider_catalogs

MODEL_DATA_SCHEMA_VERSION = 6
MODEL_DATA_MANIFEST_FILE = ".manifest.json"
GENERATED_AT = "2026-07-23T10:00:00.000Z"

_IMPORT_RE = re.compile(r'from\s+"\./providers/([^"]+)\.models\.ts"')


# --------------------------------------------------------------------------------------
# 移植的校验辅助函数（对应 scripts/model-data.ts）
# --------------------------------------------------------------------------------------


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sorted_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in sorted(record)}


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _is_modality_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(entry in ("text", "image") for entry in value)


def _describe_difference(expected: list[str], actual: list[str]) -> str:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    parts = []
    if missing:
        parts.append(f"missing: {', '.join(missing)}")
    if extra:
        parts.append(f"extra: {', '.join(extra)}")
    return "; ".join(parts)


def assert_exact_model_ids(label: str, expected: list[str], actual: list[str]) -> None:
    expected_ids = sorted(set(expected))
    actual_ids = sorted(set(actual))
    if expected_ids == actual_ids:
        return
    if set(expected_ids) - set(actual_ids):
        raise AssertionError(f"{label} model IDs do not match (missing: {', '.join(sorted(set(expected_ids) - set(actual_ids)))})")
    raise AssertionError(f"{label} model IDs do not match (extra: {', '.join(sorted(set(actual_ids) - set(expected_ids)))})")


def read_model_data_provider_ids(package_root: Path) -> list[str]:
    aggregator = (package_root / "src" / "models.generated.ts").read_text(encoding="utf-8")
    provider_ids = sorted(match.group(1) for match in _IMPORT_RE.finditer(aggregator))
    if not provider_ids:
        raise AssertionError("No generated provider imports found")
    return provider_ids


def read_provider_structure(path: Path, provider_id: str) -> dict[str, str]:
    if not path.is_file():
        raise AssertionError(f"{provider_id}.json does not exist")
    groups = json.loads(path.read_text(encoding="utf-8"))
    models: dict[str, str] = {}
    for api, value in groups.items():
        if not _is_record(value):
            raise AssertionError(f"{path} API group {api!r} must be an object")
        for model_key in value:
            if model_key in models:
                raise AssertionError(f"{path} contains {model_key} in more than one API group")
            models[model_key] = api
    if not models:
        raise AssertionError(f"{path} contains no generated model data")
    return _sorted_record(models)


def read_model_data_structure(package_root: Path) -> dict[str, dict[str, str]]:
    providers_dir = package_root / "src" / "providers"
    data_dir = providers_dir / "data"
    provider_ids = read_model_data_provider_ids(package_root)
    expected_shards = sorted(f"{provider_id}.models.ts" for provider_id in provider_ids)
    actual_shards = sorted(path.name for path in providers_dir.glob("*.models.ts"))
    if expected_shards != actual_shards:
        raise AssertionError(
            "Generated model aggregator and provider shards do not match "
            f"({_describe_difference(expected_shards, actual_shards)})"
        )
    return _sorted_record(
        {provider_id: read_provider_structure(data_dir / f"{provider_id}.json", provider_id) for provider_id in provider_ids}
    )


def model_data_structure_hash(structure: dict[str, dict[str, str]]) -> str:
    normalized = _sorted_record({provider_id: _sorted_record(models) for provider_id, models in structure.items()})
    return sha256(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")))


def create_model_data_manifest(
    structure: dict[str, dict[str, str]], file_contents: dict[str, str], generated_at: str
) -> dict[str, Any]:
    return {
        "schemaVersion": MODEL_DATA_SCHEMA_VERSION,
        "generatedAt": generated_at,
        "structureHash": model_data_structure_hash(structure),
        "files": _sorted_record({name: sha256(content) for name, content in file_contents.items()}),
    }


def _validate_model_value(
    value: Any, provider_id: str, model_id: str, expected_api: str, errors: list[str]
) -> None:
    label = f"{provider_id}/{model_id}"
    if not _is_record(value):
        errors.append(f"{label} must be an object")
        return
    if value.get("id") != model_id:
        errors.append(f"{label} has id {value.get('id')!r}, expected {model_id!r}")
    if value.get("provider") != provider_id:
        errors.append(f"{label} has provider {value.get('provider')!r}, expected {provider_id!r}")
    if value.get("api") != expected_api:
        errors.append(f"{label} has api {value.get('api')!r}, expected {expected_api!r}")
    if not isinstance(value.get("name"), str) or not value.get("name"):
        errors.append(f"{label} has no model name")
    if not isinstance(value.get("baseUrl"), str):
        errors.append(f"{label} has no baseUrl string")
    if not _is_modality_list(value.get("input")):
        errors.append(f"{label} has invalid input modalities")
    if value.get("type") == "image":
        if not _is_modality_list(value.get("output")) or "image" not in (value.get("output") or []):
            errors.append(f"{label} has invalid output modalities")
    elif value.get("output") is not None:
        errors.append(f"{label} has unsupported output modalities")
    if value.get("type") == "chat":
        if not isinstance(value.get("reasoning"), bool):
            errors.append(f"{label} has no reasoning boolean")
        context_window = value.get("contextWindow")
        if not isinstance(context_window, (int, float)) or context_window <= 0:
            errors.append(f"{label} has invalid contextWindow")
        max_tokens = value.get("maxTokens")
        if not isinstance(max_tokens, (int, float)) or max_tokens <= 0:
            errors.append(f"{label} has invalid maxTokens")
    elif value.get("type") == "classifier":
        context_window = value.get("contextWindow")
        if not isinstance(context_window, (int, float)) or context_window <= 0:
            errors.append(f"{label} has invalid contextWindow")
    elif value.get("type") != "image":
        errors.append(
            f'{label} has type {value.get("type")!r}, expected "chat", "image", or "classifier"'
        )
    cost = value.get("cost")
    if not _is_record(cost):
        errors.append(f"{label} has invalid cost metadata")
    else:
        for field in ("input", "output", "cacheRead", "cacheWrite"):
            if not isinstance(cost.get(field), (int, float)):
                errors.append(f"{label} has invalid cost.{field}")


def validate_model_data_directory(structure: dict[str, dict[str, str]], data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise AssertionError(f"Generated model data directory does not exist: {data_dir}")

    errors: list[str] = []
    expected_files = sorted(f"{provider_id}.json" for provider_id in structure)
    actual_files = sorted(
        path.name for path in data_dir.glob("*.json") if path.name != MODEL_DATA_MANIFEST_FILE
    )
    if expected_files != actual_files:
        errors.append(
            "provider data files do not match the generated catalog "
            f"({_describe_difference(expected_files, actual_files)})"
        )

    manifest_path = data_dir / MODEL_DATA_MANIFEST_FILE
    manifest: Any = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError as error:  # noqa: PERF203
            errors.append(f"model data manifest is not valid JSON: {error}")
    else:
        errors.append("model data manifest is missing")

    if _is_record(manifest):
        if manifest.get("schemaVersion") != MODEL_DATA_SCHEMA_VERSION:
            errors.append(
                f"model data schema is {manifest.get('schemaVersion')!r}, expected {MODEL_DATA_SCHEMA_VERSION}"
            )
        generated_at = manifest.get("generatedAt")
        valid_timestamp = False
        if isinstance(generated_at, str):
            try:
                datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                valid_timestamp = True
            except ValueError:
                valid_timestamp = False
        if not valid_timestamp:
            errors.append("model data manifest has an invalid generation timestamp")
        if manifest.get("structureHash") != model_data_structure_hash(structure):
            errors.append("model data generation stamp does not match the generated catalog")
        manifest_files = manifest.get("files") if _is_record(manifest.get("files")) else None
        if manifest_files is None:
            errors.append("model data manifest has no file hashes")
        elif sorted(manifest_files) != expected_files:
            errors.append(
                "manifest file hashes do not match provider data files "
                f"({_describe_difference(expected_files, sorted(manifest_files))})"
            )
    else:
        manifest_files = None

    for provider_id, expected_models in structure.items():
        filename = f"{provider_id}.json"
        path = data_dir / filename
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        if manifest_files is not None and manifest_files.get(filename) != sha256(content):
            errors.append(f"{filename} does not match its manifest hash")
        try:
            groups = json.loads(content)
        except ValueError:
            errors.append(f"{filename} is not valid JSON")
            continue
        if not _is_record(groups):
            errors.append(f"{filename} must contain a JSON object")
            continue

        actual_models: dict[str, str] = {}
        for api, value in groups.items():
            if not _is_record(value):
                errors.append(f"{filename} API group {api!r} must be an object")
                continue
            for model_key, model in value.items():
                if model_key in actual_models:
                    errors.append(f"{provider_id}/{model_key} appears in more than one API group")
                    continue
                actual_models[model_key] = api
                separator = model_key.find(":")
                model_id = model_key[separator + 1 :] if separator >= 0 else model_key
                _validate_model_value(model, provider_id, model_id, api, errors)
                if _is_record(model) and model_key != f"{model.get('type')}:{model.get('id')}":
                    errors.append(f"{provider_id}/{model_key} has mismatched type/id identity")

        expected_model_ids = sorted(expected_models)
        actual_model_ids = sorted(actual_models)
        if expected_model_ids != actual_model_ids:
            errors.append(
                f"{filename} model IDs do not match the generated catalog "
                f"({_describe_difference(expected_model_ids, actual_model_ids)})"
            )
        for model_id, expected_api in expected_models.items():
            actual_api = actual_models.get(model_id)
            if actual_api is not None and actual_api != expected_api:
                errors.append(
                    f"{provider_id}/{model_id} is grouped under API {actual_api!r}, expected {expected_api!r}"
                )

    if errors:
        visible = errors[:30]
        suffix = f"\n  ... and {len(errors) - len(visible)} more" if len(errors) > len(visible) else ""
        raise AssertionError(
            "Invalid generated model data:\n" + "\n".join(f"  - {error}" for error in visible) + suffix
        )


# --------------------------------------------------------------------------------------
# Fixture（对应上游的 createFixture / writeFixtureData）
# --------------------------------------------------------------------------------------


def write_fixture_data(
    data_dir: Path,
    structure: dict[str, dict[str, str]],
    values: dict[str, Any],
    manifest_schema_version: int = MODEL_DATA_SCHEMA_VERSION,
    api_group: str = "openai-completions",
) -> None:
    filename = "test-provider.json"
    content = json.dumps({api_group: values}) + "\n"
    (data_dir / filename).write_text(content, encoding="utf-8")
    manifest = create_model_data_manifest(structure, {filename: content}, GENERATED_AT)
    manifest["schemaVersion"] = manifest_schema_version
    (data_dir / MODEL_DATA_MANIFEST_FILE).write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def create_fixture(tmp_path: Path) -> dict[str, Any]:
    package_root = tmp_path / "package"
    package_root.mkdir()
    providers_dir = package_root / "src" / "providers"
    data_dir = providers_dir / "data"
    data_dir.mkdir(parents=True)
    (package_root / "src" / "models.generated.ts").write_text(
        'import { TEST_PROVIDER_CLASSIFIER_MODELS, TEST_PROVIDER_IMAGE_MODELS, TEST_PROVIDER_MODELS } '
        'from "./providers/test-provider.models.ts";\n',
        encoding="utf-8",
    )
    (providers_dir / "test-provider.models.ts").write_text(
        'import values from "./data/test-provider.json" with { type: "json" };\n',
        encoding="utf-8",
    )

    structure = {"test-provider": {"chat:model-a": "openai-completions"}}
    values: dict[str, Any] = {
        "chat:model-a": {
            "type": "chat",
            "id": "model-a",
            "name": "Model A",
            "api": "openai-completions",
            "provider": "test-provider",
            "baseUrl": "https://example.test/v1",
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1000,
            "maxTokens": 100,
        }
    }
    write_fixture_data(data_dir, structure, values)
    return {"dataDir": data_dir, "packageRoot": package_root, "structure": structure, "values": values}


# --------------------------------------------------------------------------------------
# 移植的用例
# --------------------------------------------------------------------------------------


def test_rejects_a_missing_upstream_model_from_an_exact_generated_allowlist() -> None:
    with pytest.raises(AssertionError, match=r"qwen-token-plan-individual model IDs do not match \(missing: model-b\)"):
        assert_exact_model_ids("qwen-token-plan-individual", ["model-a", "model-b"], ["model-a"])


def test_rejects_an_unexpected_model_from_an_exact_generated_allowlist() -> None:
    with pytest.raises(AssertionError, match=r"test-provider model IDs do not match \(extra: model-b\)"):
        assert_exact_model_ids("test-provider", ["model-a"], ["model-a", "model-b"])


def test_reads_and_validates_api_grouped_model_data(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    assert read_model_data_structure(fixture["packageRoot"]) == fixture["structure"]
    validate_model_data_directory(fixture["structure"], fixture["dataDir"])


def test_rejects_a_missing_model_data_directory(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    shutil.rmtree(fixture["dataDir"])
    with pytest.raises(AssertionError, match="does not exist"):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])


@pytest.mark.parametrize(
    "field,value,expected_message",
    [("id", "wrong-id", "has id"), ("provider", "wrong-provider", "has provider"), ("api", "anthropic-messages", "has api")],
)
def test_rejects_a_wrong_model_field(
    tmp_path: Path, field: str, value: str, expected_message: str
) -> None:
    fixture = create_fixture(tmp_path)
    fixture["values"]["chat:model-a"][field] = value
    write_fixture_data(fixture["dataDir"], fixture["structure"], fixture["values"])
    with pytest.raises(AssertionError, match=expected_message):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])


def test_rejects_a_model_without_a_known_type(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    del fixture["values"]["chat:model-a"]["type"]
    write_fixture_data(fixture["dataDir"], fixture["structure"], fixture["values"])
    with pytest.raises(AssertionError, match='expected "chat", "image", or "classifier"'):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])


def test_validates_image_models_with_output_modalities_and_without_chat_limits(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    structure = {"test-provider": {"image:image-a": "test-images"}}
    image: dict[str, Any] = {
        "type": "image",
        "id": "image-a",
        "name": "Image A",
        "api": "test-images",
        "provider": "test-provider",
        "baseUrl": "https://example.test/v1",
        "input": ["text"],
        "output": ["image", "text"],
        "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
    }

    def validate() -> None:
        write_fixture_data(
            fixture["dataDir"],
            structure,
            {"image:image-a": image},
            MODEL_DATA_SCHEMA_VERSION,
            "test-images",
        )
        validate_model_data_directory(structure, fixture["dataDir"])

    validate()

    del image["output"]
    with pytest.raises(AssertionError, match="invalid output modalities"):
        validate()
    image["output"] = ["text"]
    with pytest.raises(AssertionError, match="invalid output modalities"):
        validate()


def test_rejects_output_modalities_on_chat_models(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    fixture["values"]["chat:model-a"]["output"] = ["text"]
    write_fixture_data(fixture["dataDir"], fixture["structure"], fixture["values"])
    with pytest.raises(AssertionError, match="unsupported output modalities"):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])


def test_validates_classifier_models_without_chat_output_limits(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    structure = {"test-provider": {"classifier:classifier-a": "test-classifier"}}
    classifier = {
        "type": "classifier",
        "id": "classifier-a",
        "name": "Classifier A",
        "api": "test-classifier",
        "provider": "test-provider",
        "baseUrl": "https://example.test/v1",
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1000,
    }
    write_fixture_data(
        fixture["dataDir"],
        structure,
        {"classifier:classifier-a": classifier},
        MODEL_DATA_SCHEMA_VERSION,
        "test-classifier",
    )
    validate_model_data_directory(structure, fixture["dataDir"])


def test_rejects_a_model_in_the_wrong_api_group(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    write_fixture_data(
        fixture["dataDir"],
        fixture["structure"],
        fixture["values"],
        MODEL_DATA_SCHEMA_VERSION,
        "anthropic-messages",
    )
    with pytest.raises(AssertionError, match="grouped under API"):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])


def test_rejects_duplicate_model_keys_across_api_groups(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    filename = "test-provider.json"
    content = json.dumps({"openai-completions": fixture["values"], "anthropic-messages": fixture["values"]}) + "\n"
    (fixture["dataDir"] / filename).write_text(content, encoding="utf-8")
    manifest = create_model_data_manifest(fixture["structure"], {filename: content}, GENERATED_AT)
    (fixture["dataDir"] / MODEL_DATA_MANIFEST_FILE).write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="more than one API group"):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])


def test_rejects_missing_model_ids_and_stale_file_hashes(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    (fixture["dataDir"] / "test-provider.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=r"manifest hash|model IDs"):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])


def test_rejects_incompatible_schema_and_generation_stamps(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    write_fixture_data(
        fixture["dataDir"], fixture["structure"], fixture["values"], MODEL_DATA_SCHEMA_VERSION + 1
    )
    with pytest.raises(AssertionError, match="model data schema"):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])

    manifest_path = fixture["dataDir"] / MODEL_DATA_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["structureHash"] = "stale"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="generation stamp"):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])


def test_rejects_an_invalid_generation_timestamp(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    manifest_path = fixture["dataDir"] / MODEL_DATA_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generatedAt"] = "invalid"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="generation timestamp"):
        validate_model_data_directory(fixture["structure"], fixture["dataDir"])


def test_rejects_missing_provider_shards_imported_by_the_aggregator(tmp_path: Path) -> None:
    fixture = create_fixture(tmp_path)
    (fixture["packageRoot"] / "src" / "models.generated.ts").write_text(
        'import { TEST_PROVIDER_CLASSIFIER_MODELS, TEST_PROVIDER_IMAGE_MODELS, TEST_PROVIDER_MODELS } '
        'from "./providers/test-provider.models.ts";\n'
        'import { MISSING_CLASSIFIER_MODELS, MISSING_IMAGE_MODELS, MISSING_MODELS } '
        'from "./providers/missing.models.ts";\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="aggregator and provider shards do not match"):
        read_model_data_structure(fixture["packageRoot"])


# --------------------------------------------------------------------------------------
# 附加：对随附的已生成数据验证同样的不变式
# --------------------------------------------------------------------------------------


def test_shipped_generated_model_data_satisfies_the_same_invariants() -> None:
    data_dir = PROVIDER_CATALOG_DATA_DIR
    structure = _sorted_record(
        {
            provider_id: _sorted_record(
                {
                    f"{model['type']}:{model['id']}": api
                    for api, models in json.loads((data_dir / f"{provider_id}.json").read_text(encoding="utf-8")).items()
                    for model in models.values()
                }
            )
            for provider_id in available_provider_catalogs()
        }
    )

    assert sorted(BUILTIN_PROVIDER_IDS) == sorted(available_provider_catalogs())
    validate_model_data_directory(structure, data_dir)

    manifest = json.loads((data_dir / MODEL_DATA_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["structureHash"] == model_data_structure_hash(structure)
    assert manifest["schemaVersion"] == MODEL_DATA_SCHEMA_VERSION
