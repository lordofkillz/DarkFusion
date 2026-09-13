"""Prepare versioned runtime assets for the single-EXE Windows installer."""
from pathlib import Path
import argparse
import hashlib
import json
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distribution', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release-tag', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.release_tag):
        parser.error('Use a release tag containing letters, numbers, dots, underscores or hyphens.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be a new or empty folder.')
    payload = json.loads((args.distribution / 'payload.json').read_text(encoding='utf-8'))
    archive = args.distribution / 'payload.zip'
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != payload['sha256']:
            parser.error('The distribution archive does not match its manifest.')
    args.output.mkdir(parents=True, exist_ok=True)
    parts = []
    with archive.open('rb') as source:
        while source.tell() < archive.stat().st_size:
            name = f'DarkFusion-runtime.{len(parts) + 1:03d}.bin'
            target = args.output / name
            digest = hashlib.sha256()
            remaining = min(1600 * 1024 * 1024, archive.stat().st_size - source.tell())
            with target.open('wb') as output:
                while remaining:
                    data = source.read(min(8 * 1024 * 1024, remaining))
                    if not data:
                        raise RuntimeError('The payload ended unexpectedly.')
                    digest.update(data)
                    output.write(data)
                    remaining -= len(data)
            parts.append({'name': name, 'size_bytes': target.stat().st_size,
                          'sha256': digest.hexdigest(),
                          'url': f'https://github.com/lordofkillz/DarkFusion/releases/download/{args.release_tag}/{name}'})
            print(f'Prepared {name}', flush=True)
    models = json.loads((Path(__file__).parent / 'windows' / 'model-bundle.json').read_text(encoding='utf-8'))
    manifest = {'schema_version': 1, 'product': 'DarkFusion', 'payload': payload, 'parts': parts, 'models': models}
    (args.output / 'download.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Build DarkFusionSetup.exe with -OnlineManifest pointing to download.json.', flush=True)


if __name__ == '__main__':
    main()
