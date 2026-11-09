"""Download checksummed best-checkpoint bundles from this project's Release."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile

from .paths import ROOT
from .support.contract import sha256


def fetch(names):
    manifest=json.loads((ROOT/'models/manifest.json').read_text())
    for name in names:
        record=manifest['models'][name]
        destination=ROOT/'models'/name
        current=destination/'best.pt'
        if current.is_file():
            if sha256(current)!=record['checkpoint_sha256']:
                raise FileExistsError(f'Preserve existing different checkpoint: {current}')
            print(f'ALREADY_VERIFIED={current}',flush=True)
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.download-',dir=destination.parent) as temporary:
            temporary=Path(temporary)
            archive=temporary/'bundle.zip'
            request=urllib.request.Request(record['url'],headers={'User-Agent':'godot-supermimic'})
            print(f'DOWNLOAD={record["url"]}',flush=True)
            with urllib.request.urlopen(request,timeout=90) as response,archive.open('wb') as out:
                shutil.copyfileobj(response,out,1024*1024)
            if sha256(archive)!=record['archive_sha256']:
                raise ValueError(f'Model archive checksum differs: {name}')
            unpack=temporary/'unpack'
            unpack.mkdir()
            allowed={'best.pt','run_manifest.json','best.json','MODEL_CARD.md'}
            with zipfile.ZipFile(archive) as bundle:
                if set(bundle.namelist())!=allowed or len(bundle.namelist())!=len(allowed):
                    raise ValueError('Unexpected model bundle members')
                for member in allowed:
                    with bundle.open(member) as source,(unpack/member).open('wb') as target:
                        shutil.copyfileobj(source,target)
            if sha256(unpack/'best.pt')!=record['checkpoint_sha256']:
                raise ValueError('Checkpoint checksum differs')
            if destination.exists():
                raise FileExistsError(f'Preserve existing model directory: {destination}')
            unpack.replace(destination)
        print(f'INSTALLED={destination}',flush=True)


def add_parser(sub):
    parser=sub.add_parser('models',help='Download verified best models from this project Release')
    parser.add_argument('--only',nargs='+',choices=('lafan1','skills'),default=['lafan1','skills'])
    return parser
