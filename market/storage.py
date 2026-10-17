"""保存实验记录及对应的代码版本。"""

import hashlib
import json
from pathlib import Path
import zipfile


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def snapshot(root):
    workspace = Path(__file__).resolve().parents[1]
    paths = [*sorted((workspace / 'market').glob('*.py')),
             workspace / 'main.py', workspace / 'pyproject.toml', workspace / 'uv.lock']
    digest = hashlib.sha256()
    with zipfile.ZipFile(root / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            content = path.read_bytes()
            name = path.relative_to(workspace).as_posix()
            archive.writestr(name, content)
            digest.update(name.encode() + b'\0' + content)
    return digest.hexdigest()
