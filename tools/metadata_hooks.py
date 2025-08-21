# Hatch metadata hook to populate dependencies from requirements.txt,
# infer authors/URLs from LICENSE and .git/config, and append a license classifier.
# This satisfies: migrate requirements, add authors/URLs, and license classifier.

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

try:
    from hatchling.metadata.plugin.interface import MetadataHookInterface
except Exception as e:  # pragma: no cover
    # Allow basic linting without hatchling installed
    class MetadataHookInterface:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.root = os.getcwd()


LICENSE_CLASSIFIERS = {
    "mit": "License :: OSI Approved :: MIT License",
    "apache-2.0": "License :: OSI Approved :: Apache Software License",
    "bsd": "License :: OSI Approved :: BSD License",
    "gpl-3.0": "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "gpl-2.0": "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
    "lgpl-3.0": "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "lgpl-2.1": "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
    "mpl-2.0": "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
    "unlicense": "License :: OSI Approved :: The Unlicense (Unlicense)",
    "cc0": "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    "proprietary": "License :: Other/Proprietary License",
}


class VoidkitMetadataHook(MetadataHookInterface):
    def update(self, metadata: Dict) -> None:
        root = self.root

        # 1) dependencies from requirements.txt
        req_path = os.path.join(root, "requirements.txt")
        deps = self._read_requirements(req_path)
        if deps:
            metadata["dependencies"] = deps

        # 2) authors from LICENSE (name/email heuristics)
        lic_path = os.path.join(root, "LICENSE")
        authors = self._detect_authors(lic_path)
        if authors:
            metadata["authors"] = []
            for name, email in authors:
                entry = {"name": name}
                if email:
                    entry["email"] = email
                metadata["authors"].append(entry)

        # 3) urls from .git/config
        urls = self._detect_urls(os.path.join(root, ".git", "config"))
        if urls:
            metadata["urls"] = urls

        # 4) license classifier inferred from LICENSE
        license_classifier = self._detect_license_classifier(lic_path)
        if license_classifier:
            existing = metadata.get("classifiers", [])
            if license_classifier not in existing:
                metadata["classifiers"] = existing + [license_classifier]

    # -------- helpers --------

    def _read_requirements(self, path: str) -> List[str]:
        if not os.path.isfile(path):
            return []
        reqs: List[str] = []
        # Common dev-only packages to exclude from runtime dependencies
        dev_markers = {"pytest", "pytest-cov", "black", "ruff", "mypy", "tox", "pre-commit"}
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # remove inline comments: "pkg==1.2  # note"
                if " #" in line:
                    line = line.split(" #", 1)[0].strip()
                # skip include/option/editable lines
                if line.startswith(("-r", "--requirement", "-e", "--editable", "--", "-c", "--constraint")):
                    continue
                if line.startswith(("-", "--")):
                    continue
                # simple dev filter: drop obvious dev tools from runtime
                name = re.split(r"[<>=!~\[\];\s]", line, maxsplit=1)[0].lower()
                if name in dev_markers:
                    continue
                reqs.append(line)
        return reqs

    def _detect_authors(self, license_path: str) -> List[tuple]:
        authors: List[tuple] = []
        if not os.path.isfile(license_path):
            return authors
        try:
            text = self._read_text(license_path)
        except Exception:
            return authors

        # Try "Author:" lines
        for m in re.finditer(r"^[ \t]*Author[s]?:[ \t]*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE):
            name_email = m.group(1).strip()
            name, email = self._split_name_email(name_email)
            if name:
                authors.append((name, email))

        # Try copyright lines
        if not authors:
            for m in re.finditer(
                r"copyright\s*\(c\)\s*\d{2,4}[^A-Za-z0-9]*([^\r\n<]+?)(?:\s*<([^>]+)>)?\s*$",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            ):
                name = m.group(1).strip().strip(",")
                email = (m.group(2) or "").strip()
                if name:
                    authors.append((name, email or None))

        # Deduplicate preserving order
        seen = set()
        deduped: List[tuple] = []
        for pair in authors:
            if pair not in seen:
                deduped.append(pair)
                seen.add(pair)
        return deduped

    def _split_name_email(self, s: str) -> tuple:
        s = s.strip()
        m = re.match(r"(.+?)\s*<([^>]+)>", s)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return s, None

    def _detect_urls(self, git_config_path: str) -> Optional[Dict[str, str]]:
        if not os.path.isfile(git_config_path):
            return None
        try:
            with open(git_config_path, "r", encoding="utf-8") as f:
                cfg = f.read()
        except Exception:
            return None

        # Extract origin url
        m = re.search(r'\[remote\s+"origin"\][^\[]*?^\s*url\s*=\s*(.+)$', cfg, flags=re.MULTILINE)
        if not m:
            return None
        raw_url = m.group(1).strip()

        clean_url = self._normalize_git_url(raw_url)
        urls: Dict[str, str] = {"Repository": clean_url, "Homepage": clean_url}
        if "github.com" in clean_url.lower():
            urls["Issues"] = clean_url.rstrip("/") + "/issues"
        elif "gitlab.com" in clean_url.lower():
            urls["Issues"] = clean_url.rstrip("/") + "/-/issues"
        return urls

    def _normalize_git_url(self, url: str) -> str:
        url = url.strip()
        if url.endswith(".git"):
            url = url[:-4]
        # git@github.com:user/repo -> https://github.com/user/repo
        ssh_match = re.match(r"git@([^:]+):(.+)", url)
        if ssh_match:
            host = ssh_match.group(1)
            path = ssh_match.group(2)
            return f"https://{host}/{path}".rstrip("/")
        return url.rstrip("/")

    def _detect_license_classifier(self, license_path: str) -> Optional[str]:
        if not os.path.isfile(license_path):
            return None
        try:
            text = self._read_text(license_path).lower()
        except Exception:
            return None

        def has(*terms: str) -> bool:
            return all(t in text for t in terms)

        # Heuristic checks
        if has("mit license") and has("permission is hereby granted"):
            return LICENSE_CLASSIFIERS["mit"]
        if has("apache license") and ("version 2.0" in text or "apache license, version 2.0" in text):
            return LICENSE_CLASSIFIERS["apache-2.0"]
        if has("redistribution and use in source and binary forms"):
            return LICENSE_CLASSIFIERS["bsd"]
        if has("gnu general public license") and "version 3" in text:
            return LICENSE_CLASSIFIERS["gpl-3.0"]
        if has("gnu general public license") and "version 2" in text:
            return LICENSE_CLASSIFIERS["gpl-2.0"]
        if has("gnu lesser general public license") and "version 3" in text:
            return LICENSE_CLASSIFIERS["lgpl-3.0"]
        if has("gnu lesser general public license") and "version 2.1" in text:
            return LICENSE_CLASSIFIERS["lgpl-2.1"]
        if has("mozilla public license") and "2.0" in text:
            return LICENSE_CLASSIFIERS["mpl-2.0"]
        if has("this is free and unencumbered software released into the public domain") or "the unlicense" in text:
            return LICENSE_CLASSIFIERS["unlicense"]
        if has("creative commons") and "cc0" in text:
            return LICENSE_CLASSIFIERS["cc0"]
        # Fallback
        return LICENSE_CLASSIFIERS["proprietary"]

    def _read_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()