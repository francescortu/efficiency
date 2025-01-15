import toml

def increment_patch_version(file_path):
    # Read the pyproject.toml file
    with open(file_path, "r") as f:
        pyproject_data = toml.load(f)

    # Access the version field
    version = pyproject_data["project"]["version"]
    major, minor, patch = map(int, version.split("."))

    # Increment the patch version
    new_version = f"{major}.{minor}.{patch + 1}"
    pyproject_data["project"]["version"] = new_version

    # Write the updated version back to pyproject.toml
    with open(file_path, "w") as f:
        toml.dump(pyproject_data, f)

    print(f"Version updated: {version} -> {new_version}")

# Specify the pyproject.toml file path
increment_patch_version("pyproject.toml")
