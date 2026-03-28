import yaml
from pathlib import Path
from typing import Any

def load_yaml(path: Path) -> dict[str, Any]:
    """ Carrega um arquivo YAML e retorna um dicionário.

    Args:
        path (Path): recebe o caminho do arquivo YAML

    Returns:
        dict[str, Any]: retorna um dicionário com os dados do arquivo YAML
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
            f"Expected: {path.resolve()}"
        )

    with path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}