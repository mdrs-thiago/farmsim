"""
fetch_assets.py

Downloads each field's large files (too big for git) into farm/fields/<date>/:
the photogrammetry Gazebo model (model/) and the top-down orthophoto
(topdown.png, used by select_path previews, the alignment check and the
raster tools). Links live in cfg/assets.yaml, one zip per field; `pack`
builds those zips from a local copy, for uploading.

    python -m farmsim_utils.fetch_assets fetch [--field 20250425]
    python -m farmsim_utils.fetch_assets pack --out outputs/asset_zips
"""

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from farmsim_utils.config import load_yaml
from farmsim_utils.paths import FIELDS_DIR


def _download(url, dest):
    if 'drive.google.com' in url:
        try:
            import gdown
        except ImportError as exc:
            raise SystemExit('Google Drive links need `pip install gdown`') from exc
        gdown.download(url, str(dest), quiet=False, fuzzy=True)
    else:
        urllib.request.urlretrieve(url, dest)


def fetch(fields=None, force=False):
    assets = load_yaml('assets.yaml').get('field_models', {})
    for date, url in assets.items():
        if fields and str(date) not in fields:
            continue
        field = FIELDS_DIR / str(date)
        target = field / 'model'
        if target.is_dir() and any(target.iterdir()) and not force:
            print(f'{date}: present, skipping (use --force to re-download)')
            continue
        if not url or 'TODO' in str(url):
            print(f'{date}: no link set in cfg/assets.yaml, skipping')
            continue
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / f'{date}.zip'
            print(f'{date}: downloading {url}')
            _download(url, zpath)
            if target.exists():
                shutil.rmtree(target)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(field)
        print(f'{date}: extracted into {field}')


def pack(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in sorted(p for p in FIELDS_DIR.iterdir() if (p / 'model').is_dir()):
        zpath = out_dir / f'farmsim_field_{field.name}.zip'
        files = [f for f in sorted((field / 'model').rglob('*')) if f.is_file()]
        if (field / 'topdown.png').is_file():
            files.append(field / 'topdown.png')
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, f.relative_to(field))
        print(f'{zpath} ({zpath.stat().st_size / 1e6:.0f} MB)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('--field', action='append')
    f.add_argument('--force', action='store_true')
    p = sub.add_parser('pack')
    p.add_argument('--out', default='outputs/asset_zips')
    args = ap.parse_args()
    if args.cmd == 'fetch':
        fetch(args.field, args.force)
    else:
        pack(args.out)


if __name__ == '__main__':
    main()
