"""Files whose quality an independent measurement placed above their quant type."""

from __future__ import annotations

import json

import pytest

from whichllm.constants import QUANT_QUALITY_PENALTY
from whichllm.data import DATA_DIR
from whichllm.engine.quantization import measured_ratings, quant_quality_penalty
from whichllm.models.types import GGUFVariant, ModelInfo

ISTA = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"


def _model(repo: str) -> ModelInfo:
    return ModelInfo(
        id=repo,
        family_id="qwen/qwen3.8-flash-next",
        name=repo.rsplit("/", 1)[-1],
        parameter_count=180_000_000_000,
    )


def test_every_entry_names_a_known_type_and_a_source():
    data = json.loads((DATA_DIR / "quant_quality.json").read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    for entry in data["entries"]:
        assert entry["rated_as"] in QUANT_QUALITY_PENALTY
        assert entry["source"].startswith("https://")
        assert entry["measured"]
        # A rating only ever moves a file to a type with a smaller penalty.
        assert (
            QUANT_QUALITY_PENALTY[entry["rated_as"]]
            < QUANT_QUALITY_PENALTY[entry["quant"]]
        )


@pytest.mark.parametrize(
    ("quant", "rated"), [("IQ3_S", "IQ4_XS"), ("IQ3_XXS", "IQ4_XS")]
)
def test_ista_daslab_packs_rate_as_the_type_they_beat(quant, rated):
    variant = GGUFVariant(f"{quant}/x.gguf", quant, 1)

    assert quant_quality_penalty(_model(ISTA), variant) == QUANT_QUALITY_PENALTY[rated]


def test_a_file_without_a_measurement_keeps_its_type():
    iq3 = GGUFVariant("x-IQ3_S.gguf", "IQ3_S", 1)
    q2 = GGUFVariant("Q2_0/x.gguf", "Q2_0", 1)

    assert quant_quality_penalty(_model("someone/Other-GGUF"), iq3) == 0.17
    # Q2_0 measured +10% perplexity, below anything it could be rated as.
    assert quant_quality_penalty(_model(ISTA), q2) == QUANT_QUALITY_PENALTY["Q2_0"]


def test_a_base_model_takes_the_rating_of_the_repo_its_file_comes_from():
    variant = GGUFVariant("IQ3_S/x.gguf", "IQ3_S", 1)
    base = _model("Qwen/Qwen3.8-Flash-Next")

    assert quant_quality_penalty(base, variant) == 0.17
    assert quant_quality_penalty(base, variant, ISTA) == QUANT_QUALITY_PENALTY["IQ4_XS"]


def test_ratings_are_keyed_case_insensitively():
    assert measured_ratings()[(ISTA.casefold(), "IQ3_S")] == "IQ4_XS"
