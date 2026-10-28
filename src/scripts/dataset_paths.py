"""Locate consolidated inputs while retaining the hashes in historical manifests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELOCATIONS = (
    ('new-data/output-v3', 'dataset/processed/multimodal'),
    ('data/processed', 'dataset/processed/original'),
    ('data/raw', 'dataset/raw/original'),
    ('MRI-Data', 'dataset/raw/mri'),
    ('CSF', 'dataset/raw/csf'),
    ('new-data/config.json', 'config/prepare_multimodal.json'),
)


def relocated(value):
    if value == '':
        return value
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Use project-relative paths without parent traversal')
    name = path.as_posix()
    if len(path.parts) == 1 and path.suffix == '.csv':
        if name.startswith('MCI_AD_prediction_'):
            return 'dataset/raw/original/tables/' + name
        if name.startswith('DATADIC_'):
            return 'dataset/raw/original/dictionaries/' + name
    if name == 'reports':
        return 'dataset/processed/original/reports'
    if len(path.parts) == 2 and path.parts[0] == 'reports':
        if path.suffix in {'.csv', '.json'} or path.name == 'data_quality_report.md':
            return 'dataset/processed/original/reports/' + path.name
    for old, new in RELOCATIONS:
        if name == old or name.startswith(old + '/'):
            return new + name[len(old):]
    return name


def project_path(value):
    path = (ROOT / relocated(value)).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Path resolves outside the project')
    return path


def recorded_hash(hashes, path):
    """Match a moved file to its original manifest key; never relax its checksum."""
    key = relocated(Path(path).relative_to(ROOT))
    matches = {digest for name, digest in hashes.items() if relocated(name) == key}
    if len(matches) > 1:
        raise ValueError('Conflicting recorded hashes for relocated input: ' + key)
    return next(iter(matches), None)


def comparable_config(value):
    """Normalize locations when comparing an archived configuration with a current one."""
    if isinstance(value, dict):
        return {key: comparable_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [comparable_config(item) for item in value if item != 'dummy']
    if isinstance(value, str):
        return relocated(value)
    return value
