"""Tests for health_check module."""

import json
import os
import tempfile
import pytest
import polars as pl

from pyrunner_lib.health_check import Check, HealthCheckFailure, run_health_checks


class TestCheck:
    """Test the Check class and its methods."""
    
    def test_check_init(self):
        """Test Check initialization."""
        check = Check("test_col")
        assert check.column == "test_col"
        assert check.checks == []
    
    def test_check_build(self):
        """Test Check.build() returns column name and checks."""
        check = Check("col1").no_nulls()
        col, checks = check.build()
        assert col == "col1"
        assert len(checks) == 1
        assert checks[0][0] == "no_nulls"
    
    def test_check_chaining(self):
        """Test that checks can be chained."""
        check = Check("col1").no_nulls().unique().valid_range(min_val=0)
        col, checks = check.build()
        assert len(checks) == 3
        assert checks[0][0] == "no_nulls"
        assert checks[1][0] == "unique"
        assert checks[2][0] == "valid_range"


class TestNoNulls:
    """Test no_nulls check."""
    
    def test_no_nulls_pass(self):
        """Test no_nulls passes with no null values."""
        df = pl.LazyFrame({"col": [1, 2, 3]})
        check = Check("col").no_nulls()
        _, checks = check.build()
        name, severity, condition = checks[0]
        
        # Should not raise
        condition(df, "col")
    
    def test_no_nulls_fail(self):
        """Test no_nulls fails with null values."""
        df = pl.LazyFrame({"col": [1, None, 3]})
        check = Check("col").no_nulls()
        _, checks = check.build()
        name, severity, condition = checks[0]
        
        with pytest.raises(ValueError, match="null values found"):
            condition(df, "col")
    
    def test_no_nulls_default_severity(self):
        """Test no_nulls has default severity."""
        check = Check("col").no_nulls()
        _, checks = check.build()
        assert checks[0][1] == "warn"
    
    def test_no_nulls_custom_severity(self):
        """Test no_nulls with custom severity."""
        check = Check("col").no_nulls(severity="fail")
        _, checks = check.build()
        assert checks[0][1] == "fail"


class TestNonEmptyStrings:
    """Test non_empty_strings check."""
    
    def test_non_empty_strings_pass(self):
        """Test non_empty_strings passes with non-empty strings."""
        df = pl.LazyFrame({"col": ["a", "b", "c"]})
        check = Check("col").non_empty_strings()
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_non_empty_strings_fail_empty(self):
        """Test non_empty_strings fails with empty strings."""
        df = pl.LazyFrame({"col": ["a", "", "c"]})
        check = Check("col").non_empty_strings()
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="empty strings found"):
            condition(df, "col")
    
    def test_non_empty_strings_fail_whitespace(self):
        """Test non_empty_strings fails with whitespace-only strings."""
        df = pl.LazyFrame({"col": ["a", "   ", "c"]})
        check = Check("col").non_empty_strings()
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="empty strings found"):
            condition(df, "col")


class TestUnique:
    """Test unique check."""
    
    def test_unique_pass(self):
        """Test unique passes with all unique values."""
        df = pl.LazyFrame({"col": [1, 2, 3]})
        check = Check("col").unique()
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_unique_fail(self):
        """Test unique fails with duplicate values."""
        df = pl.LazyFrame({"col": [1, 2, 2, 3]})
        check = Check("col").unique()
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="Duplicate values found"):
            condition(df, "col")


class TestValidRange:
    """Test valid_range check."""
    
    def test_valid_range_pass(self):
        """Test valid_range passes when all values in range."""
        df = pl.LazyFrame({"col": [1, 5, 10]})
        check = Check("col").valid_range(min_val=0, max_val=10)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_valid_range_fail_below_min(self):
        """Test valid_range fails when value below min."""
        df = pl.LazyFrame({"col": [-1, 5, 10]})
        check = Check("col").valid_range(min_val=0, max_val=10)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="values out of range"):
            condition(df, "col")
    
    def test_valid_range_fail_above_max(self):
        """Test valid_range fails when value above max."""
        df = pl.LazyFrame({"col": [1, 5, 100]})
        check = Check("col").valid_range(min_val=0, max_val=10)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="values out of range"):
            condition(df, "col")
    
    def test_valid_range_min_only(self):
        """Test valid_range with only min specified."""
        df = pl.LazyFrame({"col": [1, 5, 100]})
        check = Check("col").valid_range(min_val=0)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_valid_range_max_only(self):
        """Test valid_range with only max specified."""
        df = pl.LazyFrame({"col": [-100, 5, 10]})
        check = Check("col").valid_range(max_val=10)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")


class TestInValues:
    """Test in_values check."""
    
    def test_in_values_pass(self):
        """Test in_values passes when all values allowed."""
        df = pl.LazyFrame({"col": ["A", "B", "C"]})
        check = Check("col").in_values(["A", "B", "C", "D"])
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_in_values_fail(self):
        """Test in_values fails with invalid value."""
        df = pl.LazyFrame({"col": ["A", "B", "X"]})
        check = Check("col").in_values(["A", "B", "C"])
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="invalid values"):
            condition(df, "col")
    
    def test_in_values_ignore_case(self):
        """Test in_values with ignore_case=True."""
        df = pl.LazyFrame({"col": ["a", "B", "c"]})
        check = Check("col").in_values(["A", "B", "C"], ignore_case=True)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")


class TestRegexMatch:
    """Test regex_match check."""
    
    def test_regex_match_pass(self):
        """Test regex_match passes when all values match."""
        df = pl.LazyFrame({"col": ["abc123", "def456", "ghi789"]})
        check = Check("col").regex_match(r"[a-z]+[0-9]+")
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_regex_match_fail(self):
        """Test regex_match fails when value doesn't match."""
        df = pl.LazyFrame({"col": ["abc123", "!!!", "ghi789"]})
        check = Check("col").regex_match(r"^[a-z]+[0-9]+$")
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="do not match regex"):
            condition(df, "col")
    
    def test_regex_match_non_string_column(self):
        """Test regex_match raises error for non-string columns."""
        df = pl.LazyFrame({"col": [1, 2, 3]})
        check = Check("col").regex_match(r"^[0-9]+$")
        _, checks = check.build()
        _, _, condition = checks[0]
        
        # Polars raises InvalidOperationError for type mismatch during lazy eval
        with pytest.raises(pl.exceptions.InvalidOperationError):
            condition(df, "col")


class TestNullPercentage:
    """Test null_percentage check."""
    
    def test_null_percentage_pass(self):
        """Test null_percentage passes when under threshold."""
        df = pl.LazyFrame({"col": [1, None, 3, 4, 5]})  # 20% nulls
        check = Check("col").null_percentage(max_pct=25.0)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_null_percentage_fail(self):
        """Test null_percentage fails when over threshold."""
        df = pl.LazyFrame({"col": [1, None, 3, 4, 5]})  # 20% nulls
        check = Check("col").null_percentage(max_pct=10.0)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="nulls in"):
            condition(df, "col")
    
    def test_null_percentage_zero_allowed(self):
        """Test null_percentage with 0% allowed."""
        df = pl.LazyFrame({"col": [1, 2, 3]})  # 0% nulls
        check = Check("col").null_percentage(max_pct=0.0)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")


class TestNumericCheck:
    """Test numeric_check."""
    
    def test_numeric_check_gt_pass(self):
        """Test numeric_check gt passes."""
        df = pl.LazyFrame({"col": [10, 20, 30]})
        check = Check("col").numeric_check(gt=5)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_numeric_check_gt_fail(self):
        """Test numeric_check gt fails."""
        df = pl.LazyFrame({"col": [1, 20, 30]})
        check = Check("col").numeric_check(gt=5)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="values failed numeric check"):
            condition(df, "col")
    
    def test_numeric_check_sum_eq_pass(self):
        """Test numeric_check sum_eq passes."""
        df = pl.LazyFrame({"col": [10, 20, 30]})
        check = Check("col").numeric_check(sum_eq=60)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_numeric_check_sum_eq_fail(self):
        """Test numeric_check sum_eq fails."""
        df = pl.LazyFrame({"col": [10, 20, 30]})
        check = Check("col").numeric_check(sum_eq=100)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="Sum of"):
            condition(df, "col")
    
    def test_numeric_check_non_numeric_column(self):
        """Test numeric_check raises error for non-numeric columns."""
        df = pl.LazyFrame({"col": ["a", "b", "c"]})
        check = Check("col").numeric_check(gt=0)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        # Polars raises ComputeError for type mismatch during lazy eval
        with pytest.raises(pl.exceptions.ComputeError):
            condition(df, "col")


class TestDistinctCount:
    """Test distinct_count check."""
    
    def test_distinct_count_pass(self):
        """Test distinct_count passes with expected count."""
        df = pl.LazyFrame({"col": ["A", "B", "C", "A"]})
        check = Check("col").distinct_count(expected_count=3)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        condition(df, "col")
    
    def test_distinct_count_fail(self):
        """Test distinct_count fails with wrong count."""
        df = pl.LazyFrame({"col": ["A", "B", "C"]})
        check = Check("col").distinct_count(expected_count=5)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="distinct values"):
            condition(df, "col")


class TestCustomCheck:
    """Test custom_check."""
    
    def test_custom_check_pass(self):
        """Test custom_check passes."""
        def my_check(lf, col):
            # Use lazy-compatible aggregate
            result = lf.select(pl.col(col).sum().alias("total")).collect()
            if result["total"][0] < 100:
                raise ValueError("Sum too low")
        
        df = pl.LazyFrame({"col": [50, 60, 70]})
        check = Check("col").custom_check("sum_check", "fail", my_check)
        _, checks = check.build()
        name, severity, condition = checks[0]
        
        assert name == "sum_check"
        assert severity == "fail"
        condition(df, "col")
    
    def test_custom_check_fail(self):
        """Test custom_check fails."""
        def my_check(lf, col):
            # Use lazy-compatible aggregate
            result = lf.select(pl.col(col).sum().alias("total")).collect()
            if result["total"][0] < 100:
                raise ValueError("Sum too low")
        
        df = pl.LazyFrame({"col": [10, 20, 30]})
        check = Check("col").custom_check("sum_check", "warn", my_check)
        _, checks = check.build()
        _, _, condition = checks[0]
        
        with pytest.raises(ValueError, match="Sum too low"):
            condition(df, "col")


class TestRunHealthChecks:
    """Test run_health_checks function."""
    
    def test_run_health_checks_all_pass(self):
        """Test run_health_checks when all checks pass."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["META_FOLDER"] = temp_dir
            
            lf = pl.LazyFrame({
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"]
            })
            
            checks = [
                Check("id").no_nulls(),
                Check("name").no_nulls()
            ]
            
            report = run_health_checks(lf, checks)
            
            # Check report structure
            assert "timestamp" in report
            assert "summary" in report
            assert "columns" in report
            
            # Check summary
            assert report["summary"]["total_columns"] == 2
            assert report["summary"]["total_checks"] == 2
            assert report["summary"]["passed"] == 2
            assert report["summary"]["failed"] == 0
            assert report["summary"]["overall_pass_percentage"] == 100.0
            
            # Check JSON file was created
            json_path = os.path.join(temp_dir, "health_report.json")
            assert os.path.exists(json_path)
    
    def test_run_health_checks_warn_severity_passes(self):
        """Test run_health_checks doesn't raise for warn-severity failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["META_FOLDER"] = temp_dir
            
            lf = pl.LazyFrame({
                "id": [1, 2, 2]  # Has duplicates - will fail unique check
            })
            
            checks = [
                Check("id").unique(severity="warn")  # Default is warn
            ]
            
            # Should not raise even though check fails
            report = run_health_checks(lf, checks)
            
            assert report["summary"]["failed"] == 1
            assert report["columns"]["id"]["checks"][0]["passed"] is False
    
    def test_run_health_checks_fail_severity_raises(self):
        """Test run_health_checks raises for fail-severity failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["META_FOLDER"] = temp_dir
            
            lf = pl.LazyFrame({
                "id": [1, None, 3]  # Has nulls
            })
            
            checks = [
                Check("id").no_nulls(severity="fail")
            ]
            
            with pytest.raises(HealthCheckFailure, match="fail'-severity checks failed"):
                run_health_checks(lf, checks)
    
    def test_run_health_checks_mixed_results(self):
        """Test run_health_checks with some passing and some failing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["META_FOLDER"] = temp_dir
            
            lf = pl.LazyFrame({
                "id": [1, 2, 3],
                "value": [10, 20, None]  # Has null
            })
            
            checks = [
                Check("id").no_nulls(severity="warn"),  # Passes
                Check("value").no_nulls(severity="warn")  # Fails
            ]
            
            report = run_health_checks(lf, checks)
            
            # Both checks run, one passes, one fails
            assert report["summary"]["total_checks"] == 2
            assert report["summary"]["failed"] == 1
            assert report["summary"]["overall_pass_percentage"] == 50.0
    
    def test_run_health_checks_report_structure(self):
        """Test the detailed structure of the health report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["META_FOLDER"] = temp_dir
            
            lf = pl.LazyFrame({"col": [1, 2, 3]})
            
            checks = [
                Check("col").no_nulls().valid_range(min_val=0, max_val=10)
            ]
            
            report = run_health_checks(lf, checks)
            
            # Check column report structure
            col_report = report["columns"]["col"]
            assert "pass_percentage" in col_report
            assert "checks" in col_report
            
            # Check individual check results
            check_results = col_report["checks"]
            assert len(check_results) == 2
            
            for check in check_results:
                assert "check" in check
                assert "severity" in check
                assert "passed" in check
                assert "error" in check
