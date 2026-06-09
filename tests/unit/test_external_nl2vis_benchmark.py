from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.external.adapters import DataFormulatorCommandAdapter, extract_vegalite_spec


def test_extract_vegalite_spec_from_nl4dv_vis_list() -> None:
    spec = {"mark": "bar", "encoding": {"x": {"field": "a"}}}
    payload = {"visList": [{"vlSpec": spec}]}

    assert extract_vegalite_spec(payload) == spec


def test_data_formulator_command_adapter_reads_output_json(tmp_path: Path) -> None:
    script = tmp_path / "adapter.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[sys.argv.index('--input') + 1])
output_path = Path(sys.argv[sys.argv.index('--output') + 1])
payload = json.loads(input_path.read_text(encoding='utf-8'))
output_path.write_text(json.dumps({
    'generated_spec': {
        'mark': 'bar',
        'encoding': {'x': {'field': 'category'}, 'y': {'field': 'value'}},
    },
    'case_id': payload['case_id'],
}), encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )
    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\n", encoding="utf-8")
    adapter = DataFormulatorCommandAdapter(
        command_template=f"python {script.as_posix()} --input {{input_json}} --output {{output_json}}",
    )

    result = adapter.generate(query="show value by category", data_path=data_path, case_id="case-a")

    assert result.generated_spec["mark"] == "bar"
    assert result.raw_output["case_id"] == "case-a"
