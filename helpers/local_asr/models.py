"""Download pinned model files into a shared, user-local cache."""

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MANIFEST = json.loads(
    Path(__file__).with_name("models.json").read_text(encoding="utf-8")
)


def cache_root():
    override = os.environ.get("VIDEO_USE_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "video-use" / "models"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verified(path, file):
    return (
        path.is_file()
        and path.stat().st_size == file["bytes"]
        and digest(path) == file["sha256"]
    )


def download(url, path):
    # Public model assets require no credentials, including after redirects.
    with requests.Session() as session:
        session.auth = lambda request: request
        retries = Retry(
            total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        try:
            with session.get(url, stream=True, timeout=(20, 180)) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Model download failed with HTTP {response.status_code}: {url}"
                    )
                with path.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
        except requests.RequestException as error:
            raise RuntimeError(
                f"Model download failed: {url} ({type(error).__name__})"
            ) from None


def model_dir(key, offline=False):
    spec = MANIFEST[key]
    # Content lookup validates a complete pinned revision, including small configs.
    directory = cache_root() / key
    missing = {
        name: file
        for name, file in spec["files"].items()
        if not verified(directory / name, file)
    }
    if not missing:
        return directory
    if offline:
        raise RuntimeError(f"Model {key} is missing or corrupt in {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    print(f"  downloading local model: {key}", flush=True)
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        staging = Path(temp)
        if "archive" in spec:
            archive_path = staging / "source.tar.bz2"
            download(spec["archive"], archive_path)
            if digest(archive_path) != spec["sha256"]:
                raise RuntimeError(f"Model archive checksum mismatch: {key}")
            with tarfile.open(archive_path, mode="r:bz2") as archive:
                for name, file in missing.items():
                    member = archive.getmember(file["member"])
                    if not member.isfile() or member.size != file["bytes"]:
                        raise RuntimeError(
                            f"Unexpected model archive member: {member.name}"
                        )
                    with (
                        archive.extractfile(member) as source,
                        (staging / name).open("wb") as dest,
                    ):
                        shutil.copyfileobj(source, dest)
        for name, file in missing.items():
            if "archive" not in spec:
                download(file.get("url") or spec["source"] + name, staging / name)
            if not verified(staging / name, file):
                raise RuntimeError(f"Model checksum mismatch: {key}/{name}")
            (staging / name).replace(directory / name)
    return directory
