from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import threading
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Collection, Protocol


FROZEN_MEMORYARENA_COMMIT = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
NATIVE_WEBSHOP_UPSTREAM_SCOPE = (
    "env/env_systems/web_shopping_env/runtime"
)


@dataclass(frozen=True)
class NativePurchase:
    asin: str
    price_cents: int
    selected_options: dict[str, str]


@dataclass(frozen=True)
class NativePage:
    observation: str
    url: str
    has_search_bar: bool
    clickables: tuple[str, ...]
    purchase: NativePurchase | None = None


class NativeWebShopBackend(Protocol):
    surface: str

    def open_session(self, session_token: str, instruction: str) -> NativePage:
        ...

    def step(self, session_token: str, action: str) -> NativePage:
        ...

    def close_session(self, session_token: str) -> None:
        ...

    def has_product(self, asin: str) -> bool:
        ...

    def product_asins(self) -> Collection[str]:
        ...

    def product_title(self, asin: str) -> str:
        ...

    def product_record(self, asin: str) -> dict[str, Any]:
        ...

    def product_price_cents(self, asin: str) -> int:
        ...

    def product_record_sha256(self, asin: str) -> str:
        ...

    def metadata(self) -> dict[str, Any]:
        ...

    def active_session_count(self) -> int:
        ...


class MemoryArenaNativeWebShopBackend:
    """Shared original MemoryArena WebShop runtime with isolated sessions.

    The expensive catalog, price table, and Lucene searcher are loaded once.
    Every outer rollout receives its own ``WebAgentTextEnv`` browser and native
    session. Imports are delayed so local contract tests do not require the
    MemoryArena runtime's torch/PyLucene dependencies.
    """

    surface = "memoryarena_webshop_native_v1"
    module_name = (
        "env.env_systems.web_shopping_env.runtime.service."
        "web_agent_site.envs.web_agent_text_env"
    )

    def __init__(
        self,
        *,
        memoryarena_root: str | Path,
        items_file: str | Path,
        attributes_file: str | Path,
        search_root: str | Path,
        java_home: str | Path,
        expected_memoryarena_commit: str = FROZEN_MEMORYARENA_COMMIT,
        price_seed: int = 233,
        limit_goals: int = 1,
    ) -> None:
        self.memoryarena_root = Path(memoryarena_root).expanduser().resolve()
        self.items_file = Path(items_file).expanduser().resolve()
        self.attributes_file = Path(attributes_file).expanduser().resolve()
        self.search_root = Path(search_root).expanduser().resolve()
        self.java_home = Path(java_home).expanduser().resolve()
        self.jvm_path = self.java_home / "lib" / "jvm" / "lib" / "server" / "libjvm.so"
        self.expected_memoryarena_commit = str(expected_memoryarena_commit)
        self.price_seed = int(price_seed)
        self.limit_goals = int(limit_goals)
        if self.limit_goals < 1:
            raise ValueError("limit_goals must be at least 1.")

        self._module: Any | None = None
        self._server: Any | None = None
        self._envs: dict[str, Any] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._lifecycle_lock = threading.RLock()
        self._price_table_sha256: str | None = None
        self._upstream_provenance: dict[str, Any] | None = None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._server is not None:
                return
            self._validate_paths()
            self._upstream_provenance = attest_native_webshop_upstream(
                self.memoryarena_root,
                expected_commit=self.expected_memoryarena_commit,
            )
            self._configure_runtime_paths()
            root_text = str(self.memoryarena_root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)

            module = importlib.import_module(self.module_name)
            module_path = Path(module.__file__).resolve()
            try:
                module_path.relative_to(self.memoryarena_root)
            except ValueError as exc:
                raise RuntimeError(
                    "Imported MemoryArena WebShop from the wrong source root: "
                    f"{module_path}"
                ) from exc
            engine_module = importlib.import_module(
                "env.env_systems.web_shopping_env.runtime.service."
                "web_agent_site.engine.engine"
            )
            imported_search_root = Path(engine_module.SEARCH_ENGINE_ROOT).expanduser().resolve()
            if imported_search_root != self.search_root:
                raise RuntimeError(
                    "Imported MemoryArena WebShop with the wrong Lucene root: "
                    f"expected {self.search_root}, observed {imported_search_root}."
                )
            imported_attr_file = Path(engine_module.DEFAULT_ATTR_PATH).expanduser().resolve()
            if imported_attr_file != self.attributes_file:
                raise RuntimeError(
                    "Imported MemoryArena WebShop with the wrong attributes file: "
                    f"expected {self.attributes_file}, observed {imported_attr_file}."
                )
            random_state = random.getstate()
            random.seed(self.price_seed)
            try:
                server = _build_external_task_server(
                    module,
                    base_url="http://127.0.0.1:3000",
                    file_path=str(self.items_file),
                    num_products=None,
                    human_goals=0,
                    show_attrs=False,
                )
            finally:
                random.setstate(random_state)

            self._module = module
            self._server = server
            for asin, product in self._server.product_item_dict.items():
                product["Price"] = f"${self._server.product_prices[asin]:.2f}"

    def open_session(self, session_token: str, instruction: str) -> NativePage:
        token = _validate_session_token(session_token)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string.")
        self.start()
        with self._lifecycle_lock:
            if token in self._envs:
                raise ValueError(f"Native session already exists: {token}")
            if self._module is None or self._server is None:
                raise RuntimeError("MemoryArena native backend failed to start.")
            env = self._module.WebAgentTextEnv(
                observation_mode="text",
                file_path=str(self.items_file),
                server=self._server,
                human_goals=0,
            )
            bootstrap_token = env.session
            env.reset(session=token, instruction_text=instruction)
            session = self._server.user_sessions[token]
            session["goal"] = copy.deepcopy(session["goal"])
            session["goal"]["instruction_text"] = instruction
            env.reset(session=token, instruction_text=instruction)
            if bootstrap_token != token:
                self._server.user_sessions.pop(bootstrap_token, None)
            self._envs[token] = env
            self._locks[token] = threading.RLock()
            return self._page(token)

    def step(self, session_token: str, action: str) -> NativePage:
        token = _validate_session_token(session_token)
        lock = self._require_lock(token)
        with lock:
            env = self._require_env(token)
            _, _, done, _ = env.step(action)
            purchase = self._purchase(token) if done else None
            return self._page(token, purchase=purchase)

    def close_session(self, session_token: str) -> None:
        token = _validate_session_token(session_token)
        with self._lifecycle_lock:
            env = self._envs.pop(token, None)
            self._locks.pop(token, None)
            if env is not None:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()
            if self._server is not None:
                self._server.user_sessions.pop(token, None)

    def close(self) -> None:
        with self._lifecycle_lock:
            tokens = tuple(self._envs)
        for token in tokens:
            self.close_session(token)

    def active_session_count(self) -> int:
        with self._lifecycle_lock:
            return len(self._envs)

    def has_product(self, asin: str) -> bool:
        self.start()
        if self._server is None:
            return False
        return str(asin).upper() in self._server.product_item_dict

    def product_asins(self) -> Collection[str]:
        self.start()
        if self._server is None:
            raise RuntimeError("MemoryArena native backend failed to start.")
        return self._server.product_item_dict

    def product_title(self, asin: str) -> str:
        self.start()
        if self._server is None:
            raise RuntimeError("MemoryArena native backend failed to start.")
        normalized = str(asin).upper()
        try:
            product = self._server.product_item_dict[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown native WebShop ASIN: {normalized}") from exc
        return str(product["Title"])

    def product_record(self, asin: str) -> dict[str, Any]:
        """Return the frozen catalog fields used for rule classification."""

        self.start()
        if self._server is None:
            raise RuntimeError("MemoryArena native backend failed to start.")
        normalized = str(asin).upper()
        try:
            product = self._server.product_item_dict[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown native WebShop ASIN: {normalized}") from exc
        return {
            key: copy.deepcopy(product.get(key))
            for key in ("Title", "category", "query", "product_category")
        }

    def product_price_cents(self, asin: str) -> int:
        self.start()
        if self._server is None:
            raise RuntimeError("MemoryArena native backend failed to start.")
        normalized = str(asin).upper()
        try:
            price = self._server.product_prices[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown native WebShop ASIN: {normalized}") from exc
        return _price_to_cents(price)

    def product_record_sha256(self, asin: str) -> str:
        """Hash the exact normalized product record used by the native runtime."""

        self.start()
        if self._server is None:
            raise RuntimeError("MemoryArena native backend failed to start.")
        normalized = str(asin).upper()
        try:
            product = self._server.product_item_dict[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown native WebShop ASIN: {normalized}") from exc
        payload = json.dumps(
            product,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def metadata(self) -> dict[str, Any]:
        self.start()
        if self._server is None:
            raise RuntimeError("MemoryArena native backend failed to start.")
        return {
            "surface": self.surface,
            "memoryarena_root": str(self.memoryarena_root),
            "items_file": str(self.items_file),
            "attributes_file": str(self.attributes_file),
            "search_root": str(self.search_root),
            "java_home": str(self.java_home),
            "jvm_path": str(self.jvm_path),
            "price_seed": self.price_seed,
            "product_count": len(self._server.product_item_dict),
            "active_session_count": self.active_session_count(),
            "price_table_sha256": self.price_table_sha256(),
            "upstream_provenance": dict(self._upstream_provenance or {}),
        }

    def price_table_sha256(self) -> str:
        self.start()
        if self._server is None:
            raise RuntimeError("MemoryArena native backend failed to start.")
        if self._price_table_sha256 is None:
            digest = hashlib.sha256()
            for asin, price in sorted(self._server.product_prices.items()):
                row = json.dumps(
                    [str(asin).upper(), _price_to_cents(price)],
                    separators=(",", ":"),
                )
                digest.update(row.encode("utf-8"))
                digest.update(b"\n")
            self._price_table_sha256 = digest.hexdigest()
        return self._price_table_sha256

    def _page(
        self,
        session_token: str,
        *,
        purchase: NativePurchase | None = None,
    ) -> NativePage:
        env = self._require_env(session_token)
        actions = env.get_available_actions()
        state = env.state
        return NativePage(
            observation=env.observation,
            url=str(state["url"]),
            has_search_bar=bool(actions["has_search_bar"]),
            clickables=tuple(str(item) for item in actions["clickables"]),
            purchase=purchase,
        )

    def _purchase(self, session_token: str) -> NativePurchase:
        if self._server is None:
            raise RuntimeError("MemoryArena native backend is not started.")
        session = self._server.user_sessions[session_token]
        asin = str(session.get("asin") or "").upper()
        if not asin:
            raise RuntimeError("Native WebShop purchase is missing an ASIN.")
        price = self._server.product_prices.get(asin)
        if price is None:
            raise RuntimeError(f"Native WebShop purchase is missing price for {asin}.")
        options = {
            str(key): str(value)
            for key, value in dict(session.get("options") or {}).items()
        }
        return NativePurchase(
            asin=asin,
            price_cents=_price_to_cents(price),
            selected_options=options,
        )

    def _require_env(self, session_token: str) -> Any:
        try:
            return self._envs[session_token]
        except KeyError as exc:
            raise KeyError(f"Unknown native WebShop session: {session_token}") from exc

    def _require_lock(self, session_token: str) -> threading.RLock:
        try:
            return self._locks[session_token]
        except KeyError as exc:
            raise KeyError(f"Unknown native WebShop session: {session_token}") from exc

    def _validate_paths(self) -> None:
        required = {
            "MemoryArena root": self.memoryarena_root,
            "MemoryArena items file": self.items_file,
            "MemoryArena attributes file": self.attributes_file,
            "MemoryArena Lucene search root": self.search_root,
            "Java 11 home": self.java_home,
            "Java 11 libjvm": self.jvm_path,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing native WebShop runtime paths:\n" + "\n".join(missing))
        expected_index = self.search_root / "indexes-full"
        if not expected_index.is_dir():
            raise FileNotFoundError(
                "MemoryArena Lucene root must be the parent directory containing "
                f"indexes-full; missing {expected_index}."
            )

    def _configure_runtime_paths(self) -> None:
        os.environ["MEMORYARENA_WEBSHOP_ITEMS_FILE"] = str(self.items_file)
        os.environ["MEMORYARENA_WEBSHOP_ATTR_FILE"] = str(self.attributes_file)
        os.environ["MEMORYARENA_WEBSHOP_SEARCH_ROOT"] = str(self.search_root)
        os.environ["JAVA_HOME"] = str(self.java_home)
        os.environ["JVM_PATH"] = str(self.jvm_path)
        java_bin = str(self.java_home / "bin")
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if java_bin not in path_parts:
            os.environ["PATH"] = os.pathsep.join([java_bin, *path_parts])


def _price_to_cents(value: Any) -> int:
    try:
        amount = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception as exc:
        raise ValueError(f"Invalid native WebShop price: {value!r}") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"Invalid native WebShop price: {value!r}")
    return int(amount * 100)


def _validate_session_token(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("session_token must be a non-empty string.")
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in value):
        raise ValueError("session_token may contain only ASCII letters, digits, '-' and '_'.")
    return value


def attest_native_webshop_upstream(
    memoryarena_root: str | Path,
    *,
    expected_commit: str = FROZEN_MEMORYARENA_COMMIT,
) -> dict[str, Any]:
    """Require the imported WebShop runtime to match the pinned upstream tree."""

    root = Path(memoryarena_root).expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"MemoryArena root is not a git worktree: {root}")
    commit = _git(root, "rev-parse", "HEAD").strip()
    if commit != expected_commit:
        raise RuntimeError(
            "MemoryArena commit mismatch for native WebShop: "
            f"expected {expected_commit}, observed {commit}"
        )
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        NATIVE_WEBSHOP_UPSTREAM_SCOPE,
    )
    if status.strip():
        raise RuntimeError(
            "MemoryArena native WebShop source is not pristine at the pinned commit:\n"
            + status.rstrip()
        )

    tracked_files = tuple(
        line
        for line in _git(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            NATIVE_WEBSHOP_UPSTREAM_SCOPE,
        ).splitlines()
        if line
    )
    if not tracked_files:
        raise RuntimeError("Pinned MemoryArena commit has no native WebShop runtime files.")
    tracked_python = {path for path in tracked_files if path.endswith(".py")}
    runtime_root = root / NATIVE_WEBSHOP_UPSTREAM_SCOPE
    filesystem_python = {
        path.relative_to(root).as_posix()
        for path in runtime_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if filesystem_python != tracked_python:
        raise RuntimeError(
            "MemoryArena native WebShop Python source set is not pristine: "
            f"untracked={sorted(filesystem_python - tracked_python)} "
            f"missing={sorted(tracked_python - filesystem_python)}"
        )

    files_sha256 = {}
    for relative_path in tracked_files:
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(f"Missing MemoryArena native WebShop file: {path}")
        files_sha256[relative_path] = _sha256_file(path)
    digest = hashlib.sha256(
        json.dumps(
            files_sha256,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "mode": "pinned_pristine_upstream",
        "memoryarena_commit": commit,
        "source_scope": NATIVE_WEBSHOP_UPSTREAM_SCOPE,
        "source_file_count": len(files_sha256),
        "source_bundle_sha256": digest,
    }


def _build_external_task_server(
    module: Any,
    *,
    base_url: str,
    file_path: str,
    num_products: int | None,
    human_goals: int,
    show_attrs: bool,
) -> Any:
    """Use the native browser/search engine with AMG-owned tasks and rewards."""

    class ExternalTaskSimServer(module.SimServer):
        def __init__(self) -> None:
            self.base_url = base_url
            (
                self.all_products,
                self.product_item_dict,
                self.product_prices,
                _,
            ) = module.load_products(
                filepath=file_path,
                num_products=num_products,
                human_goals=human_goals,
            )
            if not self.all_products:
                raise RuntimeError("MemoryArena WebShop catalog is empty.")
            self.search_engine = module.init_search_engine(num_products=num_products)
            product = self.all_products[0]
            self.goals = [_external_bootstrap_goal(product)]
            self.show_attrs = show_attrs
            self.weights = [1.0]
            self.cum_weights = [0.0, 1.0]
            self.user_sessions = {}
            self.search_time = 0.0
            self.render_time = 0.0
            self.sample_time = 0.0
            self.assigned_instruction_text = None

        def done(self, session_id, **kwargs):
            session = self.user_sessions[session_id]
            session["actions"]["purchase"] += 1
            session["done"] = True
            session["reward"] = 0.0
            session["verbose_info"] = {}
            url = (
                f"{self.base_url}/done/{session_id}/"
                f"{session['asin']}/{session['options']}"
            )
            html = module.map_action_to_html(
                f"click[{module.END_BUTTON}]",
                session_id=session_id,
                reward=0.0,
                asin=session["asin"],
                options=session["options"],
                instruction_text=session["goal"]["instruction_text"],
            )
            return html, url, 0.0

    return ExternalTaskSimServer()


def _external_bootstrap_goal(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "asin": str(product["asin"]),
        "category": str(product.get("category", "")),
        "query": str(product.get("query", "")),
        "name": str(product.get("name") or product.get("Title") or ""),
        "product_category": str(product.get("product_category", "")),
        "instruction_text": "",
        "attributes": [],
        "price_upper": 0.0,
        "goal_options": {},
        "weight": 1.0,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
        *args,
    ]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(
            f"Cannot attest MemoryArena native WebShop source at {root}: "
            f"{stderr.strip()}"
        ) from exc
