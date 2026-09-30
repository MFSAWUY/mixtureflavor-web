"""Build a portable release with an explicit allowlist, never include submission data or secrets."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ['backend', 'frontend/src', 'frontend/tests', 'frontend/public', 'frontend/dist', 'vendor', 'examples', 'deploy', 'docs', 'scripts', 'tests']
FILES = ['README.md', 'START_LOCAL.ps1', 'compose.yaml', '.env.example', '.dockerignore', '.gitignore', '.gitattributes',
         'frontend/package.json', 'frontend/package-lock.json', 'frontend/tsconfig.json', 'frontend/vite.config.ts', 'frontend/index.html']


def main():
    if not (ROOT / 'frontend/dist/index.html').is_file():
        raise SystemExit('Build the frontend before creating a release.')
    paths = [ROOT / p for p in FILES]
    for directory in DIRECTORIES:
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f'Unexpected external path: {path}')
        if path.name == '.env' or path.suffix in {'.sqlite3', '.db'} or 'runtime' in path.relative_to(ROOT).parts:
            raise RuntimeError(f'Private file cannot be packaged: {path.name}')
    release = ROOT / 'release'
    release.mkdir(exist_ok=True)
    output = release / 'MixtureFlavor-web-v1.0.0.zip'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(paths)):
            archive.write(path, 'mixtureflavor-web/' + path.relative_to(ROOT).as_posix())
        archive.writestr('mixtureflavor-web/runtime/.gitkeep', '')
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(checksum + '  ' + output.name + '\n', encoding='ascii')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        count = len(archive.namelist())
    print(json.dumps({'file': str(output), 'files': count, 'bytes': output.stat().st_size, 'sha256': checksum}, indent=2))


if __name__ == '__main__':
    main()
