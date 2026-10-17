import hashlib
import json
import shutil
from pathlib import Path


def data_directory(output):
    output = output.resolve()
    workspace = Path(__file__).resolve().parents[1]
    try:
        relative = output.relative_to(workspace / 'results')
    except ValueError:
        return output.parent / 'data' / output.name
    return workspace / 'data' / relative


def report_directory(data):
    data = data.resolve()
    workspace = Path(__file__).resolve().parents[1]
    try:
        relative = data.relative_to(workspace / 'data')
    except ValueError:
        return data.parent.parent / data.name if data.parent.name == 'data' else data / 'report'
    return workspace / 'results' / relative


def export_results(data, output):
    output.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((data / name).read_text(encoding='utf-8'))
    write_json(output / 'config.json', read('manifest.json'))
    write_json(output / 'statistics.json', {
        'groups': read('group_estimates.json'),
        'comparisons': read('paired_comparisons.json'),
        'phases': read('path_audit_estimates.json')})
    shutil.copy2(data / '逐种子统计.csv', output / 'statistics.csv')


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def source_digest():
    workspace = Path(__file__).resolve().parents[1]
    paths = [*sorted((workspace / 'market').glob('*.py')),
             workspace / 'main.py', workspace / 'pyproject.toml', workspace / 'uv.lock']
    digest = hashlib.sha256()
    for path in paths:
        content = path.read_bytes()
        name = path.relative_to(workspace).as_posix()
        digest.update(name.encode() + b'\0' + content)
    return digest.hexdigest()
