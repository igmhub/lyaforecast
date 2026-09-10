"""Check that the wheel and source distribution contain the supported inputs."""
from pathlib import Path
import tarfile
import zipfile


root = Path(__file__).resolve().parents[1]
wheel, = (root / "dist").glob("*.whl")
sdist, = (root / "dist").glob("*.tar.gz")
resources = {p.relative_to(root).as_posix() for p in (root / "lyaforecast/resources").rglob("*")
             if p.is_file() and p.suffix in {".ini", ".dat", ".txt", ".md"}}
templates = sorted((root / "lyaforecast/resources/default_configs").glob("*.ini"))
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    assert resources <= names, resources - names
    assert not any(name.startswith(("tests/", "examples/", ".validation/")) for name in names)
    assert "lyaforecast/forecast.py" not in names
    for path in templates:
        assert archive.read(path.relative_to(root).as_posix()) == path.read_bytes()
with tarfile.open(sdist) as archive:
    names = {name.split("/", 1)[-1] for name in archive.getnames()}
    assert resources <= names, resources - names
    for path in (root / "examples").rglob("*"):
        if path.is_file() and path.suffix in {".ini", ".py", ".ipynb", ".md"}:
            assert path.relative_to(root).as_posix() in names
    assert not any(".validation" in name for name in names)
    members = {member.name.split("/", 1)[-1]: member for member in archive.getmembers()}
    for path in templates:
        member = members[path.relative_to(root).as_posix()]
        assert archive.extractfile(member).read() == path.read_bytes()
print("Wheel resources and source-distribution examples verified.")
