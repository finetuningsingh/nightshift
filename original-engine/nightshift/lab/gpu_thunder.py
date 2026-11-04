"""``GpuProvider`` for Thunder Compute over its public REST API (standard library only).

Why REST and not the ``tnr`` CLI: ``POST /instances/create`` accepts our own ``public_key``, and every list row returns it in
``sshPublicKeys``, so ``owner_tag_of_key(public_key)`` is an unforgeable ownership tag (the CLI cannot name or tag instances). The
Thunder account is shared with the Mac, ax41 and other projects; this class therefore never touches a row whose tag is not ours.

Run live on 2026-11-02 (one A100 80GB, 77 minutes, $1.44): list, pricing, create with our own key, ready polling, ssh, and delete by list position
with the ownership tag checked first all worked; the create needs ``gpu_type`` ``a100xl`` for the 80 GB A100 (``DEFAULT_GPU_TYPE_MAP``).
Still UNVERIFIED: the response of ``/v2/status``, whether ``POST /instances/{id}/delete`` accepts the 8-character uuid (``delete_by``
defaults to the position, which is what ``tnr`` does and what worked), and the other GPU type names.

Failure rules (docs notes 2.2): ``list_instances`` raises ``ProviderUnavailable`` on ANY error or malformed output and never returns
``[]`` for a failure; ``create`` is never retried, a definite refusal (4xx) is ``CreateRejected`` and anything that may have reached
the service (timeout, connection loss, 5xx, an unparseable success) is ``CreateAmbiguous``; ``delete`` re-lists, checks the tag,
addresses the CURRENT position, all under one host-wide lock
so two deleters on this host cannot shift the positions between each other's check and request (a deleter on another host can: see the comment in ``__init__``). The token is read from the environment on every call, is sent only over https (or
loopback http for tests), is never followed across a redirect, and never appears in an exception message.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Mapping, Optional

from nightshift.lab.contracts import (
    CreateAmbiguous,
    CreateReceipt,
    CreateRejected,
    GpuInstance,
    OwnershipError,
    ProviderUnavailable,
    Quote,
    Shape,
    owner_tag_of_key,
)

DEFAULT_BASE_URL = "https://api.thundercompute.com:8443"
USER_AGENT = "nightshift-lab-gpu/1.0"  # the service answers 403 to the default Python-urllib agent
MAX_BODY_BYTES = 4 << 20

# Shape.gpu -> key prefix of /v2/pricing and /v2/status ("a100" is the 80 GB A100, priced as a100xl).
PRICE_KEY = {"a100": "a100xl", "a100xl": "a100xl", "h100": "h100", "l40": "l40", "a6000": "a6000"}
# vCPUs included in the base price of a one-GPU instance (pricing page, 2026-11-01). Unknown means none included (an overcount).
INCLUDED_VCPUS = {"a100xl": 8, "h100": 4, "l40": 6, "a6000": 6}
INCLUDED_DISK_GB_PER_GPU = 100
# Shape.gpu -> the string POST /instances/create accepts. LIVE 2026-11-02: "a100" is refused (HTTP 400 "invalid GPU type"); "a100xl" (the 80 GB A100,
# the same key /v2/pricing uses) created an NVIDIA A100-SXM4-80GB. The other names are untested.
DEFAULT_GPU_TYPE_MAP = {"a100": "a100xl"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


class _Transport(Exception):
    """The request may or may not have reached the service."""


def _parse_epoch(text: Any) -> float:
    if not isinstance(text, str):
        raise ValueError("createdAt is not a string")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


class ThunderProvider:
    name = "thunder"

    def __init__(
        self,
        api_token_env: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 30.0,
        delete_by: str = "index",
        gpu_type_map: Optional[Mapping[str, str]] = None,
        delete_lock: Optional[str] = None,
    ) -> None:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", api_token_env or ""):
            raise ProviderUnavailable("api_token_env must be the NAME of an environment variable")
        parsed = urllib.parse.urlparse(base_url)
        loopback = parsed.hostname in ("127.0.0.1", "localhost", "::1")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
            raise ProviderUnavailable("base_url must be https (http only for a loopback test server): the token is sent in a header")
        if delete_by not in ("index", "uuid"):
            raise ProviderUnavailable("delete_by must be 'index' or 'uuid'")
        self.api_token_env = api_token_env
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.delete_by = delete_by
        self.gpu_type_map = {**DEFAULT_GPU_TYPE_MAP, **dict(gpu_type_map or {})}
        # Deleting by list position is only safe while the list does not change between the ownership check and the POST. Every delete on this
        # host (coordinator, watchdogs) therefore runs list -> verify -> POST under one exclusive flock. The flock cannot serialise ANOTHER host
        # (the reaper's): a delete there can shift the positions inside that window. Only ``delete_by="uuid"`` closes it, and whether Thunder
        # accepts a uuid is UNVERIFIED; until then the manager's post-delete check (other instances vanished) is the detector, not a guard.
        self._delete_lock = delete_lock or os.path.join(tempfile.gettempdir(), f"lab-thunder-delete-{os.getuid()}.lock")
        self._opener = urllib.request.build_opener(_NoRedirect)

    def __repr__(self) -> str:
        return f"ThunderProvider(token_env={self.api_token_env!r}, base_url={self.base_url!r})"

    # HTTP ----------------------------------------------------------------------------------------
    def _token(self) -> str:
        tok = os.environ.get(self.api_token_env, "").strip()
        if not tok:
            raise ProviderUnavailable(f"environment variable {self.api_token_env} is not set")
        return tok

    def _call(self, method: str, path: str, body: Optional[dict] = None, *, auth: bool = True) -> tuple[int, Any]:
        """Returns (status, parsed JSON or None). HTTP error statuses are returned, not raised. Raises _Transport when the outcome is unknown."""
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        token = ""
        if auth:
            token = self._token()
            headers["Authorization"] = f"Bearer {token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                status, raw = resp.status, resp.read(MAX_BODY_BYTES + 1)
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read(MAX_BODY_BYTES + 1)
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            reason = str(getattr(exc, "reason", exc))
            raise _Transport(reason.replace(token, "***") if token else reason) from None
        if len(raw) > MAX_BODY_BYTES:
            raise _Transport("response body too large")
        try:
            doc = json.loads(raw) if raw.strip() else None
        except ValueError:
            doc = None
        return status, doc

    def _message(self, doc: Any, status: int) -> str:
        if isinstance(doc, dict):
            text = str(doc.get("message") or doc.get("error") or "")
            token = os.environ.get(self.api_token_env, "").strip()
            if token:
                text = text.replace(token, "***")
            return f"HTTP {status}: {text[:200]}"
        return f"HTTP {status}"

    # reads ---------------------------------------------------------------------------------------
    def quote(self, shape: Shape) -> Quote:
        key = PRICE_KEY.get(shape.gpu.lower())
        if key is None:
            raise ProviderUnavailable(f"no price key is known for gpu {shape.gpu!r} (known: {sorted(PRICE_KEY)})")
        try:
            status, doc = self._call("GET", "/v2/pricing", auth=False)
        except _Transport as exc:
            raise ProviderUnavailable(f"pricing request failed: {exc}") from None
        pricing = doc.get("pricing") if isinstance(doc, dict) else None
        spec_key = f"{key}_x{shape.num_gpus}"
        if status != 200 or not isinstance(pricing, dict) or not isinstance(pricing.get(spec_key), (int, float)):
            raise ProviderUnavailable(f"pricing for {spec_key} is not available ({self._message(doc, status)})")
        extras = pricing.get("additional_vcpus"), pricing.get("disk_gb")
        if not all(isinstance(x, (int, float)) for x in extras):
            raise ProviderUnavailable("pricing response lacks additional_vcpus or disk_gb")
        included = INCLUDED_VCPUS.get(key, 0) if shape.num_gpus == 1 else 0
        usd = float(pricing[spec_key])
        usd += max(0, shape.vcpus - included) * float(extras[0])
        usd += max(0, shape.disk_gb - INCLUDED_DISK_GB_PER_GPU * shape.num_gpus) * float(extras[1])
        available: Optional[bool] = None
        try:
            st, sdoc = self._call("GET", "/v2/status", auth=False)
            specs = sdoc.get("specs") if isinstance(sdoc, dict) else None
            if st == 200 and isinstance(specs, dict) and isinstance(specs.get(spec_key), str):
                word = specs[spec_key].strip().lower()
                available = True if word == "available" else False if word in ("unavailable", "sold_out", "soldout") else None
        except _Transport:
            available = None
        return Quote(round(usd, 6), 0.0, available, f"thunder /v2/pricing {spec_key} + vcpu/disk extras")

    def _rows(self) -> list[dict]:
        try:
            status, doc = self._call("GET", "/instances/list")
        except _Transport as exc:
            raise ProviderUnavailable(f"list request failed: {exc}") from None
        if status != 200 or not isinstance(doc, dict):
            raise ProviderUnavailable(f"list failed ({self._message(doc, status)})")
        rows = []
        for index, item in doc.items():
            if not isinstance(item, dict) or not isinstance(item.get("uuid"), str) or not item["uuid"]:
                raise ProviderUnavailable("list returned a malformed row (no uuid)")
            row = dict(item)
            row["_index"] = str(index)
            rows.append(row)
        return rows

    @staticmethod
    def _tags(row: dict) -> list[str]:
        keys = row.get("sshPublicKeys") or []
        if not isinstance(keys, list):
            raise ProviderUnavailable("malformed sshPublicKeys")
        out = []
        for k in keys:
            try:
                out.append(owner_tag_of_key(k))
            except Exception:  # a key the contracts cannot tag simply carries no tag
                continue
        return out

    def _instance(self, row: dict) -> GpuInstance:
        shape = None
        try:
            shape = Shape(str(row["gpuType"]).lower(), int(row["numGpus"]), int(row["cpuCores"]), int(row["storage"]), str(row.get("template") or "base"))
        except Exception:  # noqa: BLE001 - an unreadable shape is unknown, not fatal
            shape = None
        try:
            created = _parse_epoch(row.get("createdAt"))
        except ValueError:
            raise ProviderUnavailable(f"row {row['uuid']}: createdAt is not a timestamp") from None
        tags = self._tags(row)
        ip, port = row.get("ip"), row.get("port")
        return GpuInstance(
            row["uuid"], str(row.get("status", "")).upper(), shape, created, tags[0] if tags else None,
            ip if isinstance(ip, str) and ip else None, port if isinstance(port, int) and not isinstance(port, bool) and port else None, "ubuntu",
        )  # fmt: skip

    def list_instances(self) -> list[GpuInstance]:
        return [self._instance(r) for r in self._rows()]

    def exists(self, handle: str) -> Optional[bool]:
        try:
            return any(r["uuid"] == handle for r in self._rows())
        except ProviderUnavailable:
            return None

    @staticmethod
    def _row_matches_shape(row: dict, shape: Shape) -> bool:
        try:
            return (
                str(row["gpuType"]).lower().startswith(shape.gpu.lower()) and int(row["numGpus"]) == shape.num_gpus
                and int(row["cpuCores"]) == shape.vcpus and int(row["storage"]) == shape.disk_gb
            )  # fmt: skip
        except (KeyError, ValueError, TypeError):
            return False

    def find_created(self, before: set[str], shape: Shape, attempt_epoch: float, owner_tag: str) -> Optional[str]:
        hits = []
        for row in self._rows():
            if row["uuid"] in before or owner_tag not in self._tags(row) or not self._row_matches_shape(row, shape):
                continue
            try:
                if _parse_epoch(row.get("createdAt")) < attempt_epoch - 120:
                    continue
            except ValueError:
                continue
            hits.append(row["uuid"])
        return hits[0] if len(hits) == 1 else None

    # writes --------------------------------------------------------------------------------------
    def create(self, shape: Shape, ssh_public_key: str) -> CreateReceipt:
        owner_tag_of_key(ssh_public_key)  # refuse a key the tag cannot be derived from, before anything is sent
        body = {
            "cpu_cores": shape.vcpus, "disk_size_gb": shape.disk_gb, "gpu_type": self.gpu_type_map.get(shape.gpu, shape.gpu),
            "num_gpus": shape.num_gpus, "template": shape.template, "public_key": ssh_public_key,
        }  # fmt: skip
        attempt = time.time()
        try:
            status, doc = self._call("POST", "/instances/create", body)
        except ProviderUnavailable as exc:  # no token: nothing was sent
            raise CreateRejected(str(exc)) from None
        except _Transport as exc:
            raise CreateAmbiguous(f"create outcome unknown: {exc}") from None
        if status in (200, 201):
            uuid = doc.get("uuid") if isinstance(doc, dict) else None
            if isinstance(uuid, str) and uuid:
                return CreateReceipt(uuid, attempt)
            return CreateReceipt(None, attempt)  # accepted but no usable handle: the manager recovers it by tag
        if 400 <= status < 500 and status not in (408, 425, 429):
            raise CreateRejected(self._message(doc, status))
        raise CreateAmbiguous(f"create outcome unknown ({self._message(doc, status)})")

    def delete(self, handle: str, expect_tag: Optional[str]) -> None:
        """Re-list, verify the tag on the CURRENT row of ``handle``, address that row, all while holding the host-wide delete lock. No-op if the handle is absent."""
        with self._delete_serialised():
            self._delete_locked(handle, expect_tag)

    @contextlib.contextmanager
    def _delete_serialised(self):
        try:
            fd = os.open(self._delete_lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except OSError as exc:
            raise ProviderUnavailable(f"cannot open the delete lock: {exc}") from None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing releases the flock

    def _delete_locked(self, handle: str, expect_tag: Optional[str]) -> None:
        rows = self._rows()  # ProviderUnavailable: nothing is sent
        row = next((r for r in rows if r["uuid"] == handle), None)
        if row is None:
            return
        if expect_tag is None or expect_tag not in self._tags(row):
            raise OwnershipError(f"instance {handle} does not carry ownership tag {expect_tag!r}; refusing to delete it")
        ident = handle if self.delete_by == "uuid" else row["_index"]
        try:
            status, doc = self._call("POST", f"/instances/{urllib.parse.quote(ident, safe='')}/delete", {})
        except _Transport as exc:
            raise ProviderUnavailable(f"delete outcome unknown: {exc}") from None
        if status == 404:
            return  # already gone; the caller confirms with exists()
        if status not in (200, 202, 204):
            raise ProviderUnavailable(f"delete failed ({self._message(doc, status)})")
