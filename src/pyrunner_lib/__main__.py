import sys

from .health_check import HealthCheckFailure
from .pyrunner_lib import (
    ConfigurationError,
    DataLoadError,
    DataWriteError,
    ModuleLoadError,
    PyrunnerError,
    TransformNotFoundError,
    transform,
)


def main():
    """Main entry point for the pyrunner CLI."""
    if len(sys.argv) < 2:
        print("Error: Missing arguments: <transform_id> <optional_base_path>", file=sys.stderr)
        print("Usage: pyrunner <transform_id> [base_path]", file=sys.stderr)
        sys.exit(1)
    
    try:
        transform_id = sys.argv[1]
        base_path = sys.argv[2] if len(sys.argv) > 2 else ""
        transform(transform_id, base_path)
    except ConfigurationError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except TransformNotFoundError as e:
        print(f"Transform Not Found: {e}", file=sys.stderr)
        sys.exit(1)
    except ModuleLoadError as e:
        print(f"Module Load Error: {e}", file=sys.stderr)
        sys.exit(1)
    except DataLoadError as e:
        print(f"Data Load Error: {e}", file=sys.stderr)
        sys.exit(1)
    except DataWriteError as e:
        print(f"Data Write Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PyrunnerError as e:
        print(f"Pyrunner Error: {e}", file=sys.stderr)
        sys.exit(1)
    # The report says which checks failed; a traceback would add nothing
    except HealthCheckFailure as e:
        print(f"Health Check Failure: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
