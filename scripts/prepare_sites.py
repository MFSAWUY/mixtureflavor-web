"""Copy only the public frontend into the nested Sites checkout; preserve its hosting identity."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'sites-preview'


def main():
    if TARGET.is_symlink() or not TARGET.resolve().is_relative_to(ROOT):
        raise SystemExit('Sites checkout must be an ordinary directory inside this project.')
    TARGET.mkdir(exist_ok=True)
    frontend = ROOT / 'frontend'
    for name in ['src', 'public', 'tests']:
        for source in (frontend / name).rglob('*'):
            if source.is_file():
                if source.is_symlink():
                    raise SystemExit(f'Symlink not allowed: {source}')
                destination = TARGET / source.relative_to(frontend)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    for name in ['package.json', 'package-lock.json', 'index.html', 'tsconfig.json', 'vite.config.ts']:
        shutil.copyfile(frontend / name, TARGET / name)
    package_path = TARGET / 'package.json'
    package = json.loads(package_path.read_text(encoding='utf-8'))
    package['scripts']['build'] = package['scripts']['build:showcase']
    package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (TARGET / '.gitignore').write_text('node_modules/\ndist/\ndist-showcase/\n*.tsbuildinfo\n.sites-runtime/\n.env*\n', encoding='utf-8')
    (TARGET / '.gitattributes').write_text('* text=auto eol=lf\n', encoding='utf-8')
    manifest = TARGET / '.openai/hosting.json'
    manifest.parent.mkdir(exist_ok=True)
    if not manifest.exists():
        manifest.write_text(json.dumps({'static': {'directory': 'dist'}}) + '\n', encoding='utf-8')
    (TARGET / 'README.md').write_text('# MixtureFlavor showcase\n\nPublic frontend only. Build: `npm run build:showcase`.\nNo model weights, user submissions or credentials are included.\n', encoding='utf-8')
    print(json.dumps({'checkout': str(TARGET), 'hosting_identity_preserved': True}))


if __name__ == '__main__':
    main()
