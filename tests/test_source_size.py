from pathlib import Path


def test_starrygl_modules_do_not_exceed_500_lines() -> None:
    source = Path(__file__).parents[1] / "src" / "starrygl"
    oversized = {
        str(path.relative_to(source)): count
        for path in source.rglob("*.py")
        if (count := len(path.read_text(encoding="utf-8").splitlines())) > 500
    }
    assert not oversized, f"production modules exceed 500 lines: {oversized}"
