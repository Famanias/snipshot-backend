"""ModelWrapper — base class for downloadable model inference."""

import os
import stat
import sys
import re
import tempfile
import shutil
import filecmp
from abc import ABC, abstractmethod
from functools import cached_property

from .generic import BASE_PATH, download_url_with_progressbar, get_digest
from .log import get_logger


class ModelVerificationException(Exception):
    pass


class InvalidModelMappingException(ValueError):
    def __init__(self, cls: str, map_key: str, error_msg: str):
        super().__init__(f"[{cls}->{map_key}] Invalid _MODEL_MAPPING - {error_msg}")


def _get_filename_from_url(url: str, fallback: str) -> str:
    from urllib.parse import urlparse
    path = urlparse(url).path
    name = os.path.basename(path)
    return name if name else fallback


def _replace_prefix(s: str, old: str, new: str) -> str:
    if s.startswith(old):
        return new + s[len(old):]
    return s


class ModelWrapper(ABC):
    """Unified download + load + infer lifecycle for a single model."""

    _MODEL_DIR = os.path.join(BASE_PATH, "models")
    _MODEL_SUB_DIR = ""
    _MODEL_MAPPING: dict = {}
    _KEY = ""

    def __init__(self):
        os.makedirs(self.model_dir, exist_ok=True)
        self._key = self._KEY or self.__class__.__name__
        self._loaded = False
        self.logger = get_logger(self._key)
        self._check_for_malformed_model_mapping()
        self._downloaded = self._check_downloaded()

    def is_loaded(self) -> bool:
        return self._loaded

    def is_downloaded(self) -> bool:
        return self._downloaded

    @property
    def model_dir(self):
        return os.path.join(self._MODEL_DIR, self._MODEL_SUB_DIR)

    def _get_file_path(self, *args) -> str:
        return os.path.join(self.model_dir, *args)

    # ── mapping validation ───────────────────────────────────────────

    def _check_for_malformed_model_mapping(self):
        for map_key, mapping in self._MODEL_MAPPING.items():
            if "url" not in mapping:
                raise InvalidModelMappingException(self._key, map_key, "Missing url")
            if not re.search(r"^https?://", mapping["url"]):
                raise InvalidModelMappingException(self._key, map_key, f'Malformed url: "{mapping["url"]}"')
            if "file" not in mapping and "archive" not in mapping:
                mapping["file"] = "."
            elif "file" in mapping and "archive" in mapping:
                raise InvalidModelMappingException(self._key, map_key, "file and archive are mutually exclusive")

    # ── download ─────────────────────────────────────────────────────

    async def download(self, force=False):
        if force or not self.is_downloaded():
            await self._download()
            self._downloaded = True

    async def _download(self):
        self.logger.info("Downloading models into %s", self.model_dir)
        for map_key, mapping in self._MODEL_MAPPING.items():
            if self._check_downloaded_map(map_key):
                self.logger.info("Skipping %s (already downloaded)", map_key)
                continue

            is_archive = "archive" in mapping
            if is_archive:
                download_path = os.path.join(self._temp_working_directory, map_key, "")
            else:
                download_path = self._get_file_path(mapping["file"])

            if not os.path.basename(download_path):
                os.makedirs(download_path, exist_ok=True)
            if os.path.basename(download_path) in ("", "."):
                download_path = os.path.join(download_path, _get_filename_from_url(mapping["url"], map_key))
            if not is_archive:
                download_path += ".part"

            if "hash" in mapping:
                downloaded = False
                if os.path.isfile(download_path):
                    try:
                        self._verify_file(mapping["hash"], download_path)
                        downloaded = True
                    except ModelVerificationException:
                        self.logger.info("Resuming interrupted download")
                if not downloaded:
                    download_url_with_progressbar(mapping["url"], download_path)
                    self._verify_file(mapping["hash"], download_path)
            else:
                download_url_with_progressbar(mapping["url"], download_path)

            if download_path.endswith(".part"):
                p = download_path[:-5]
                shutil.move(download_path, p)
                download_path = p

            if is_archive:
                extracted_path = os.path.join(os.path.dirname(download_path), "extracted")
                shutil.unpack_archive(download_path, extracted_path)
                for orig, dest in mapping["archive"].items():
                    p1 = os.path.join(extracted_path, orig)
                    if not os.path.exists(p1):
                        raise InvalidModelMappingException(self._key, map_key, f'File "{orig}" not in archive')
                    p2 = self._get_file_path(dest)
                    if os.path.basename(p2) in ("", "."):
                        p2 = os.path.join(p2, os.path.basename(p1))
                    if os.path.isfile(p2) and not filecmp.cmp(p1, p2):
                        raise InvalidModelMappingException(self._key, map_key, f'File "{orig}" already exists at "{dest}"')
                    os.makedirs(os.path.dirname(p2), exist_ok=True)
                    shutil.move(p1, p2)
                try:
                    os.remove(download_path)
                    shutil.rmtree(extracted_path)
                except Exception:
                    pass

    def _verify_file(self, expected_hash: str, path: str):
        actual = get_digest(path).lower()
        if actual != expected_hash.lower():
            raise ModelVerificationException(f"Hash mismatch: {actual} != {expected_hash}")

    @cached_property
    def _temp_working_directory(self):
        p = os.path.join(tempfile.gettempdir(), "snipshot-engine", self._key.lower())
        os.makedirs(p, exist_ok=True)
        return p

    # ── check downloaded ─────────────────────────────────────────────

    def _check_downloaded(self) -> bool:
        for map_key in self._MODEL_MAPPING:
            if not self._check_downloaded_map(map_key):
                return False
        return True

    def _check_downloaded_map(self, map_key: str) -> bool:
        mapping = self._MODEL_MAPPING[map_key]
        if "file" in mapping:
            path = mapping["file"]
            if os.path.basename(path) in (".", ""):
                path = os.path.join(path, _get_filename_from_url(mapping["url"], map_key))
            if not os.path.exists(self._get_file_path(path)):
                return False
        elif "archive" in mapping:
            for orig, dest in mapping["archive"].items():
                if os.path.basename(dest) in ("", "."):
                    dest = os.path.join(dest, os.path.basename(orig.rstrip("/")))
                if not os.path.exists(self._get_file_path(dest)):
                    return False
        return True

    # ── load / unload / infer lifecycle ──────────────────────────────

    async def load(self, device: str, *args, **kwargs):
        if not self.is_downloaded():
            await self.download()
        if not self.is_loaded():
            await self._load(*args, device=device, **kwargs)
            self._loaded = True

    async def unload(self):
        if self.is_loaded():
            await self._unload()
            self._loaded = False

    async def reload(self, device: str, *args, **kwargs):
        await self.unload()
        await self.load(device, *args, **kwargs)

    async def infer(self, *args, **kwargs):
        if not self.is_loaded():
            raise RuntimeError(f"{self._key}: Model not loaded.")
        return await self._infer(*args, **kwargs)

    @abstractmethod
    async def _load(self, device: str, *args, **kwargs):
        pass

    @abstractmethod
    async def _unload(self):
        pass

    @abstractmethod
    async def _infer(self, *args, **kwargs):
        pass
