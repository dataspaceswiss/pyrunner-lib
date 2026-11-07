"""Tests for __main__ module."""

from unittest.mock import patch

import pytest

from pyrunner_lib.__main__ import main
from pyrunner_lib.pyrunner_lib import (
    ConfigurationError,
    DataLoadError,
    DataWriteError,
    ModuleLoadError,
    PyrunnerError,
    TransformNotFoundError,
)


class TestMain:
    """Test main function."""
    
    def test_main_missing_arguments(self):
        """Test main with missing arguments."""
        with patch('sys.argv', ['pyrunner']), pytest.raises(SystemExit) as exit_info:
            main()
        assert exit_info.value.code == 1
    
    def test_main_success(self):
        """Test successful main execution."""
        with patch('sys.argv', ['pyrunner', 'test_transform', '/path/to/base']), patch('pyrunner_lib.__main__.transform') as mock_transform:
            main()
            mock_transform.assert_called_once_with('test_transform', '/path/to/base')
    
    def test_main_without_base_path(self):
        """Test main without base path."""
        with patch('sys.argv', ['pyrunner', 'test_transform']), patch('pyrunner_lib.__main__.transform') as mock_transform:
            main()
            mock_transform.assert_called_once_with('test_transform', '')
    
    def test_main_configuration_error(self):
        """Test main with configuration error."""
        with patch('sys.argv', ['pyrunner', 'test_transform']), patch('pyrunner_lib.__main__.transform') as mock_transform:
            mock_transform.side_effect = ConfigurationError("Config error")
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_transform_not_found_error(self):
        """Test main with transform not found error."""
        with patch('sys.argv', ['pyrunner', 'test_transform']), patch('pyrunner_lib.__main__.transform') as mock_transform:
            mock_transform.side_effect = TransformNotFoundError("Transform not found")
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_module_load_error(self):
        """Test main with module load error."""
        with patch('sys.argv', ['pyrunner', 'test_transform']), patch('pyrunner_lib.__main__.transform') as mock_transform:
            mock_transform.side_effect = ModuleLoadError("Module load error")
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_data_load_error(self):
        """Test main with data load error."""
        with patch('sys.argv', ['pyrunner', 'test_transform']), patch('pyrunner_lib.__main__.transform') as mock_transform:
            mock_transform.side_effect = DataLoadError("Data load error")
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_data_write_error(self):
        """Test main with data write error."""
        with patch('sys.argv', ['pyrunner', 'test_transform']), patch('pyrunner_lib.__main__.transform') as mock_transform:
            mock_transform.side_effect = DataWriteError("Data write error")
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_pyrunner_error(self):
        """Test main with pyrunner error."""
        with patch('sys.argv', ['pyrunner', 'test_transform']), patch('pyrunner_lib.__main__.transform') as mock_transform:
            mock_transform.side_effect = PyrunnerError("Pyrunner error")
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_unexpected_error(self):
        """An error outside the library's own errors is not hidden: it ends the run with its traceback."""
        with patch('sys.argv', ['pyrunner', 'test_transform']), patch('pyrunner_lib.__main__.transform') as mock_transform:
            mock_transform.side_effect = RuntimeError("Unexpected error")
            with pytest.raises(RuntimeError):
                main()
