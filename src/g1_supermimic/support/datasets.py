"""LAFAN1 公共数据准备：固定 HF 版本、校验下载，再调用上游 CSV 转换器。

只下载 Godot-SuperMimic README 指定的 G1 40 条 CSV，不拉取其他机器人。
原始数据和转换结果均在 Git 忽略的 motions/ 下，不改上游源码或原运行环境。
"""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from .contract import sha256, write_json

from ..paths import ROOT
DATASET = 'lvhaidong/LAFAN1_Retargeting_Dataset'
RAW = ROOT / 'data/raw/lafan1'
STANDARD = ROOT / 'data/lafan1'


def request_bytes(url):
    """处理本机代理偶发 TLS/网络断开；完整失败时保留已校验文件以便续跑。"""
    error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'godot-supermimic'})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            error = exc
            if attempt < 3:
                time.sleep(min(attempt + 1, 3))
    raise RuntimeError(f'Download failed after retries: {url}: {error}') from error


def git_blob_sha1(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def download_lafan1(workers=2):
    """缓存固定 revision；后续不悄悄切换到新版数据，逐文件校验 HF git oid。"""
    index = RAW / 'source_manifest.json'
    if index.is_file():
        manifest = json.loads(index.read_text())
    elif (ROOT/'data/lafan1_source.json').is_file():
        manifest = json.loads((ROOT/'data/lafan1_source.json').read_text())
        write_json(index, manifest)
    else:
        metadata = json.loads(request_bytes(f'https://huggingface.co/api/datasets/{DATASET}'))
        revision = metadata['sha']
        entries = json.loads(request_bytes(
            f'https://huggingface.co/api/datasets/{DATASET}/tree/{revision}/g1?recursive=true&expand=false'))
        entries = [entry for entry in entries if entry['type'] == 'file' and entry['path'].endswith('.csv')]
        if len(entries) != 40:
            raise ValueError(f'Expected upstream LAFAN1 40 G1 CSV clips, got {len(entries)}')
        manifest = dict(dataset=DATASET, revision=revision, input_fps=30, files=entries)
        write_json(index, manifest)

    def fetch(entry):
        relative = Path(entry['path'])
        if relative.parts[0] != 'g1' or len(relative.parts) != 2:
            raise ValueError(f'Unexpected dataset path: {relative}')
        destination = RAW / relative
        if destination.is_file() and git_blob_sha1(destination.read_bytes()) == entry['oid']:
            return destination
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{manifest["revision"]}/{relative.as_posix()}'
        payload = request_bytes(url)
        if len(payload) != entry['size'] or git_blob_sha1(payload) != entry['oid']:
            raise ValueError(f'Download differs from pinned HF object: {relative}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.csv.download')
        temporary.write_bytes(payload)
        temporary.replace(destination)
        print(f'DOWNLOADED {destination.name}', flush=True)
        return destination

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        paths = list(pool.map(fetch, manifest['files']))
    # 保存数据卡说明，避免数据脱离来源与使用限制；数据本身不重新分发。
    card = RAW / 'DATASET_README.md'
    if not card.is_file():
        card.write_bytes(request_bytes(
            f'https://huggingface.co/datasets/{DATASET}/resolve/{manifest["revision"]}/README.md'))
    return paths, manifest


def prepare_lafan1(workers=2):
    """逐条转换限制峰值内存；CSV→50 Hz 的插值/FK 完全调用固定上游实现。"""
    paths, source = download_lafan1(workers)
    from g1_supermimic.runtime import activate
    activate()
    from g1_supermimic.core.data.motion_convert import convert_csv_to_npz, load_g1_model
    from .motion import inspect
    model = load_g1_model()
    STANDARD.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_file in paths:
        destination = STANDARD / f'{source_file.stem}.npz'
        metadata = destination.with_suffix('.json')
        source_hash = sha256(source_file)
        cached = json.loads(metadata.read_text()) if metadata.is_file() else {}
        if not (destination.is_file() and cached.get('source_sha256') == source_hash
                and cached.get('output_sha256') == sha256(destination)):
            # np.savez 自动追加扩展名，所以临时路径仍以 .npz 结尾。
            temporary = destination.with_name(destination.stem + '.converting.npz')
            info = convert_csv_to_npz(source_file, temporary, 30., 50., model=model)
            temporary.replace(destination)
            info.update(source_sha256=source_hash, output_sha256=sha256(destination),
                        converter='Godot-SuperMimic pinned motion_convert.py', input_fps=30, output_fps=50)
            write_json(metadata, info)
            print(f'CONVERTED {source_file.stem}: {info["output_frames"]} frames', flush=True)
        report = inspect(destination, expected_fps=50.)
        rows.append(dict(name=destination.stem, frames=report['frames'], fps=50.,
                         duration_s=(report['frames'] - 1) / 50., input_fps=30.))
    with (STANDARD / 'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'frames', 'fps', 'duration_s', 'input_fps'])
        writer.writeheader()
        writer.writerows(rows)
    from ..configuration import digest
    result = dict(dataset=DATASET, revision=source['revision'], clips=len(rows),
                  duration_hours=sum(row['duration_s'] for row in rows) / 3600,
                  frames=sum(row['frames'] for row in rows), npz_dir=str(STANDARD),
                  conversion='upstream 30 Hz CSV -> 50 Hz NPZ, CPU FK',
                  data_hash='motion_arrays_v1',
                  outputs={f'{row["name"]}.npz': digest(STANDARD/f'{row["name"]}.npz') for row in rows})
    write_json(STANDARD / 'dataset_manifest.json', result)
    return result
