
import json
from unittest.mock import patch

import polars as pl
import pytest

from pyrunner_lib.pyrunner_lib import (
    transform,
)


@pytest.fixture
def mock_env(tmp_path):
    config_file = tmp_path / "_config.json"
    connections_file = tmp_path / "_connections.json"
    
    with open(config_file, 'w') as f:
        json.dump({
            "python": "3.8",
            "dfType": "polars",
            "packages": ["polars"]
        }, f)
        
    with patch('pyrunner_lib.pyrunner_lib.CONFIG_FILE', str(config_file)), \
         patch('pyrunner_lib.pyrunner_lib.CONNECTIONS_FILE', str(connections_file)):
        yield tmp_path

class TestTransformAnnotations:
    
    def test_case_1_annotation(self, mock_env):
        """Case 1: def transform(input: Transform('rid-xxx'))"""
        transform_file = mock_env / "case1.py"
        with open(transform_file, 'w') as f:
            f.write("""
from pyrunner_lib import Transform
def transform(input_df: Transform("rid-xxx")):
    return input_df
""")
        
        connections = [
            {
                "id": "rid-xxx",
                "path": str(mock_env / "input_source.py"),
                "inputs": []
            },
            {
                "id": "case1",
                "path": str(transform_file),
                "inputs": ["rid-xxx"]
            }
        ]
        with open(mock_env / "_connections.json", 'w') as f:
            json.dump(connections, f)
            
        with patch('pyrunner_lib.pyrunner_lib.read_parquet_files') as mock_read:
            mock_read.return_value = {"input_df": pl.LazyFrame({"a": [1]})}
            with patch('pyrunner_lib.pyrunner_lib.write_df_to_parquet'):
                transform("case1", base_path=str(mock_env))
                
                # Verify mapping
                mock_read.assert_called_once()
                args = mock_read.call_args[0]
                param_mapping = args[1]
                assert param_mapping["input_df"] == "rid-xxx"

    def test_case_2_mixed(self, mock_env):
        """Case 2: Mixed parameters and annotations."""
        transform_file = mock_env / "case2.py"
        with open(transform_file, 'w') as f:
            f.write("""
from pyrunner_lib import Transform
def transform(input1, input2: Transform("rid-2")):
    return input1
""")
        
        connections = [
            {"id": "input1", "path": "p1.py", "inputs": []},
            {"id": "rid-2", "path": "p2.py", "inputs": []},
            {"id": "case2", "path": str(transform_file), "inputs": ["input1", "rid-2"]}
        ]
        with open(mock_env / "_connections.json", 'w') as f:
            json.dump(connections, f)
            
        with patch('pyrunner_lib.pyrunner_lib.read_parquet_files') as mock_read:
            mock_read.return_value = {"input1": pl.LazyFrame(), "input2": pl.LazyFrame()}
            with patch('pyrunner_lib.pyrunner_lib.write_df_to_parquet'):
                transform("case2", base_path=str(mock_env))
                
                param_mapping = mock_read.call_args[0][1]
                assert param_mapping["input1"] == "input1"
                assert param_mapping["input2"] == "rid-2"

    def test_case_3_decorator_simple(self, mock_env):
        """Case 3: @transform simple"""
        transform_file = mock_env / "case3.py"
        with open(transform_file, 'w') as f:
            f.write("""
from pyrunner_lib import transform
@transform
def myfunc(input1):
    return input1
""")
        
        connections = [
            {"id": "input1", "path": "p1.py", "inputs": []},
            {"id": "case3", "path": str(transform_file), "inputs": ["input1"]}
        ]
        with open(mock_env / "_connections.json", 'w') as f:
            json.dump(connections, f)
            
        with patch('pyrunner_lib.pyrunner_lib.read_parquet_files') as mock_read:
            mock_read.return_value = {"input1": pl.LazyFrame()}
            with patch('pyrunner_lib.pyrunner_lib.write_df_to_parquet'):
                transform("case3", base_path=str(mock_env))
                
                param_mapping = mock_read.call_args[0][1]
                assert param_mapping["input1"] == "input1"

    def test_case_4_decorator_kwargs(self, mock_env):
        """Case 4: @transform(input=Transform('rid-xxx'))"""
        transform_file = mock_env / "case4.py"
        with open(transform_file, 'w') as f:
            f.write("""
from pyrunner_lib import transform, Transform
@transform(input1=Transform("rid-xxx"))
def myfunc(input1):
    return input1
""")
        
        connections = [
            {"id": "rid-xxx", "path": "p1.py", "inputs": []},
            {"id": "case4", "path": str(transform_file), "inputs": ["rid-xxx"]}
        ]
        with open(mock_env / "_connections.json", 'w') as f:
            json.dump(connections, f)
            
        with patch('pyrunner_lib.pyrunner_lib.read_parquet_files') as mock_read:
            mock_read.return_value = {"input1": pl.LazyFrame()}
            with patch('pyrunner_lib.pyrunner_lib.write_df_to_parquet'):
                transform("case4", base_path=str(mock_env))
                
                param_mapping = mock_read.call_args[0][1]
                assert param_mapping["input1"] == "rid-xxx"

    def test_case_5_multiple_decorated(self, mock_env):
        """Case 5: Multiple @transform functions."""
        transform_file = mock_env / "case5.py"
        with open(transform_file, 'w') as f:
            f.write("""
from pyrunner_lib import transform
@transform
def func1(input1):
    return input1

@transform
def func2(input1):
    return input1
""")
        
        connections = [
            {"id": "input1", "path": "p1.py", "inputs": []},
            {"id": "case5", "path": str(transform_file), "inputs": ["input1"]}
        ]
        with open(mock_env / "_connections.json", 'w') as f:
            json.dump(connections, f)
            
        with patch('pyrunner_lib.pyrunner_lib.read_parquet_files') as mock_read:
            mock_read.return_value = {"input1": pl.LazyFrame()}
            with patch('pyrunner_lib.pyrunner_lib.write_df_to_parquet'):
                # Should pick the first one (func1)
                transform("case5", base_path=str(mock_env))
                mock_read.assert_called_once()

    def test_case_6_clash(self, mock_env):
        """Case 6: Clash between @transform and def transform."""
        transform_file = mock_env / "case6.py"
        # Since 'transform' from library is imported as 'transform', 
        # defining 'def transform' will overwrite it in the module scope.
        with open(transform_file, 'w') as f:
            f.write("""
from pyrunner_lib import transform, Transform
@transform
def func1(input1):
    return input1

def transform(input2):
    return input2
""")
        
        connections = [
            {"id": "input1", "path": "p1.py", "inputs": []},
            {"id": "input2", "path": "p2.py", "inputs": []},
            {"id": "case6", "path": str(transform_file), "inputs": ["input1", "input2"]}
        ]
        with open(mock_env / "_connections.json", 'w') as f:
            json.dump(connections, f)
            
        with patch('pyrunner_lib.pyrunner_lib.read_parquet_files') as mock_read:
            def side_effect(connections, param_mapping, base_path, transform_id, explicit_params=None):
                return {k: pl.LazyFrame() for k in param_mapping}
            mock_read.side_effect = side_effect
            with patch('pyrunner_lib.pyrunner_lib.write_df_to_parquet'):
                # Should prefer 'transform' function name
                transform("case6", base_path=str(mock_env))
                
                param_mapping = mock_read.call_args[0][1]
                assert "input2" in param_mapping
                assert "input1" not in param_mapping

    # DataSpace reads these annotations from the source, so the runner has to bind them the same way
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            (
                (
                    "from __future__ import annotations\nfrom pyrunner_lib import Transform\n"
                    "def transform(input_df: Transform('rid-xxx')):\n    return input_df\n"
                ),
                "rid-xxx",
            ),
            (
                (
                    "from typing import Annotated\nimport polars as pl\nfrom pyrunner_lib import Transform\n"
                    "def transform(input_df: Annotated[pl.LazyFrame, Transform('rid-xxx')]):\n    return input_df\n"
                ),
                "rid-xxx",
            ),
            (
                (
                    "from __future__ import annotations\nfrom typing import TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n    import polars as pl\n"
                    "def transform(input_df: pl.LazyFrame):\n    return input_df\n"
                ),
                "input_df",
            ),
            (
                (
                    "from __future__ import annotations\nfrom typing import TYPE_CHECKING, Annotated\n"
                    "from pyrunner_lib import Transform\n"
                    "if TYPE_CHECKING:\n    import polars as pl\n"
                    "def transform(input_df: Annotated[pl.LazyFrame, Transform('rid-xxx')]):\n    return input_df\n"
                ),
                "rid-xxx",
            ),
        ],
        ids=["postponed", "annotated", "type-checking-only", "postponed-annotated-type-checking-only"],
    )
    def test_annotation_forms(self, mock_env, source, expected):
        transform_file = mock_env / "forms.py"
        transform_file.write_text(source)
        with open(mock_env / "_connections.json", 'w') as f:
            json.dump([{"id": "forms", "path": str(transform_file), "inputs": []}], f)

        with patch('pyrunner_lib.pyrunner_lib.read_parquet_files') as mock_read:
            mock_read.return_value = {"input_df": pl.LazyFrame({"a": [1]})}
            with patch('pyrunner_lib.pyrunner_lib.write_df_to_parquet'):
                transform("forms", base_path=str(mock_env))

        assert mock_read.call_args[0][1]["input_df"] == expected

    def test_error_in_the_transform_is_not_wrapped(self, mock_env):
        """The user's own error reaches the caller unchanged, so the build log shows its traceback."""
        transform_file = mock_env / "broken.py"
        transform_file.write_text("def transform():\n    raise KeyError('missing column')\n")
        with open(mock_env / "_connections.json", 'w') as f:
            json.dump([{"id": "broken", "path": str(transform_file), "inputs": []}], f)

        with pytest.raises(KeyError, match="missing column"):
            transform("broken", base_path=str(mock_env))
