import sys
from .pyrunner_lib import transform

def main():
    if len(sys.argv) < 2:
        print("Error: Missing arguments: <transform_id> <optional_base_path>", file=sys.stderr)
        sys.exit(1)
    transform(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")

if __name__ == "__main__":
    main()
