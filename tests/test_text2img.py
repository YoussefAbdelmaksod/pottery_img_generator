import pytest
from pipeline.text2img import generate_pottery_image
import os

def test_generate_pottery_image(tmp_path):
    output_path = tmp_path / "test.png"
    generate_pottery_image(
        prompt="A test pottery vase",
        output_path=str(output_path),
        style="Greek",
        material="earthenware",
        perspective="side view",
        seed=123
    )
    assert os.path.exists(output_path)
