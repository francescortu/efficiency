import toml

def increment_patch_version(file_path):
    # Read the pyproject.toml file
    with open(file_path, "r") as f:
        pyproject_data = toml.load(f)

    # Access the version field under [tool.poetry]
    if "tool" not in pyproject_data or "poetry" not in pyproject_data["tool"]:
        raise KeyError("The file does not contain the [tool.poetry] section.")

    version = pyproject_data["tool"]["poetry"]["version"]
    major, minor, patch = map(int, version.split("."))

    # Increment the patch version
    new_version = f"{major}.{minor}.{patch + 1}"
    pyproject_data["tool"]["poetry"]["version"] = new_version

    # Write the updated version back to pyproject.toml
    with open(file_path, "w") as f:
        toml.dump(pyproject_data, f)

    print(f"Version updated: {version} -> {new_version}")

# Specify the pyproject.toml file path
increment_patch_version("pyproject.toml")
