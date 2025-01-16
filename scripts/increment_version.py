import toml
import sys

def increment_version(file_path, release_type):
    with open(file_path, "r") as f:
        pyproject_data = toml.load(f)

    if "tool" not in pyproject_data or "poetry" not in pyproject_data["tool"]:
        raise KeyError("The file does not contain the [tool.poetry] section.")

    version = pyproject_data["tool"]["poetry"]["version"]
    major, minor, patch = map(int, version.split("."))

    if release_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif release_type == "minor":
        minor += 1
        patch = 0
    elif release_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid release type: {release_type}")

    new_version = f"{major}.{minor}.{patch}"
    pyproject_data["tool"]["poetry"]["version"] = new_version

    with open(file_path, "w") as f:
        toml.dump(pyproject_data, f)

    # Save the version to a temporary file for tagging
    with open(".version", "w") as f:
        f.write(new_version)

    print(f"Version updated: {version} -> {new_version}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: increment_version.py <file_path> <release_type>")
        sys.exit(1)

    file_path = sys.argv[1]
    release_type = sys.argv[2]

    increment_version(file_path, release_type)
