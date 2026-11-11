"""Persistent multi-GPU workers with bounded, synchronous failure recovery.

API compatible with the supplied WorkerPool. A failed command retires the whole
pool, including queues, and synchronously loads a fresh idle generation BEFORE
reporting the case failure. Failed cases are never retried automatically.
CPU/shared-memory task inputs only.
"""
import gc
import importlib
import logging
import os
import pickle
import queue
import sys
import threading
import time
import traceback
import uuid
from types import SimpleNamespace

logger = logging.getLogger(__name__)
SENTINEL = "__SENTINEL__"
OFFLOAD = "__OFFLOAD__"
RELOAD = "__RELOAD__"


class PoolRecoveryError(RuntimeError):
    """The current case failed AND the pool could not be made ready again.

    Catch this BEFORE Exception at a request boundary. Do not advertise ready
    or continue accepting cases as if recovery had succeeded.
    """
    def __init__(self, pool_name, command, failures, recovery_error):
        self.pool_name = pool_name
        self.command = command
        self.failures = failures
        self.recovery_error = str(recovery_error)
        super().__init__(
            f"[pool {pool_name}] RECOVERY_FAILED after {command}; "
            f"pool is NOT READY. Original failure: {failures}; "
            f"recovery error: {self.recovery_error}"
        )


def _cleanup_cuda():
    """Current process only. Do not initialize CUDA merely for cleanup."""
    gc.collect()
    import torch
    if torch.cuda.is_initialized():
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.exception("CUDA cache cleanup failed")


class WorkerPool:
    def __init__(self, name, world_size, model_params_dict, warmup_params_dict,
                 mp_ctx, device_ids=None, ready_timeout=1800,
                 offload_timeout=120, reload_timeout=300,
                 result_get_timeout=3600, *, poll_interval=0.2,
                 stop_timeout=3.0, _worker_target=None):
        if world_size < 1:
            raise ValueError("world_size must be positive")
        self.name, self.world_size = name, world_size
        self.model_params_dict = dict(model_params_dict)
        self.warmup_params_dict = dict(warmup_params_dict)
        self.mp_ctx = mp_ctx
        self.device_ids = list(range(world_size) if device_ids is None else device_ids)
        if len(self.device_ids) != world_size or len(set(self.device_ids)) != world_size:
            raise ValueError("device_ids must contain world_size unique device indices")
        for key, value in dict(ready_timeout=ready_timeout,
                               offload_timeout=offload_timeout,
                               reload_timeout=reload_timeout,
                               result_get_timeout=result_get_timeout,
                               poll_interval=poll_interval,
                               stop_timeout=stop_timeout).items():
            if value <= 0:
                raise ValueError(f"{key} must be positive")
            setattr(self, key, float(value))
        self._worker_target = _worker_target or persistent_worker_fn
        self._lock = threading.RLock()
        self._closing = threading.Event()
        self._aborting = threading.Event()
        self._generation = 0
        self._started = False
        self._idle_offloaded = False
        self.result_queue = None
        self.task_queues = [None] * world_size
        self.workers = [None] * world_size

    def _check_open(self):
        if self._closing.is_set():
            raise RuntimeError(f"[pool {self.name}] pool is shut down")
        if self._aborting.is_set():
            raise RuntimeError(f"[pool {self.name}] pool is being aborted")

    @staticmethod
    def _close_queue(q):
        if q is not None:
            # A killed consumer may leave a feeder blocked on a full pipe.
            # Do not wait for that feeder or reuse this queue after failure.
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass

    def _stop_all(self, graceful=False):
        """Call under _lock; use shared deadlines, not N full joins."""
        self._started = False
        self._idle_offloaded = False
        if graceful:
            for q in self.task_queues:
                if q is not None:
                    try:
                        q.put_nowait({"command": "stop", "generation": self._generation})
                    except Exception:
                        pass
            deadline = time.monotonic() + self.stop_timeout
            for p in self.workers:
                if p is not None:
                    p.join(max(0.0, deadline - time.monotonic()))
        for p in self.workers:
            if p is not None and p.is_alive():
                p.terminate()
        deadline = time.monotonic() + self.stop_timeout
        for p in self.workers:
            if p is not None:
                p.join(max(0.0, deadline - time.monotonic()))
        for p in self.workers:
            if p is not None and p.is_alive():
                p.kill()
        deadline = time.monotonic() + self.stop_timeout
        for p in self.workers:
            if p is not None:
                p.join(max(0.0, deadline - time.monotonic()))
        survivors = [p for p in self.workers if p is not None and p.is_alive()]
        for q in self.task_queues:
            self._close_queue(q)
        self._close_queue(self.result_queue)
        self.result_queue = None
        self.task_queues = [None] * self.world_size
        # Keep unkillable processes tracked; never start a duplicate GPU pool.
        self.workers = [None] * self.world_size if not survivors else self.workers
        if survivors:
            raise RuntimeError(
                f"[pool {self.name}] OS could not stop worker PIDs "
                f"{[p.pid for p in survivors]}; refusing to start duplicates"
            )

    def _spawn_worker(self, rank):
        q = self.mp_ctx.Queue()
        # Parent only writes tasks. On Python 3.11/3.12, closing its unused
        # reader BEFORE the first put lets feeders see EPIPE when a consumer
        # dies, instead of retaining a large shared input forever. Do not race
        # reader.close() with an already running QueueFeederThread at teardown.
        q._ignore_epipe = True
        self.task_queues[rank] = q
        p = self.mp_ctx.Process(
            target=self._worker_target,
            args=(rank, self.world_size, self.device_ids[rank],
                  self.model_params_dict, self.warmup_params_dict,
                  q, self.result_queue, self.name, self._generation),
        )
        p.daemon = False
        p.start()
        self.workers[rank] = p
        q._reader.close()

    def _wait_results(self, expected_status, timeout_per_get, expected_ranks=None,
                      *, request_id="init"):
        # timeout_per_get is retained for API compatibility. It is now the
        # TOTAL phase deadline and is never reset by a response/stale message.
        pending = set(range(self.world_size) if expected_ranks is None else expected_ranks)
        ok, failed = set(), {}
        deadline = time.monotonic() + timeout_per_get
        while pending:
            if self._closing.is_set() or self._aborting.is_set():
                for r in pending:
                    failed[r] = ("cancelled", "pool shutdown/abort requested")
                break
            for r in list(pending):
                p = self.workers[r]
                if p is None or not p.is_alive():
                    failed[r] = ("dead", f"exitcode={None if p is None else p.exitcode}")
                    pending.remove(r)
            if failed:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for r in pending:
                    failed[r] = ("timeout", f"{expected_status}: exceeded {timeout_per_get:g}s")
                break
            try:
                message = self.result_queue.get(timeout=min(self.poll_interval, remaining))
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError) as exc:
                for r in pending:
                    failed[r] = ("ipc_error", str(exc))
                break
            if not isinstance(message, dict):
                continue
            if (message.get("generation") != self._generation
                    or message.get("request_id") != request_id):
                continue
            rank = message.get("rank")
            if rank not in pending:
                continue
            pending.remove(rank)
            if message.get("status") == expected_status:
                ok.add(rank)
            else:
                failed[rank] = (message.get("status", "error"), message.get("error"))
                break
        if failed:
            for r in pending:
                failed.setdefault(r, ("cancelled", "another rank failed; this case is aborted"))
        return ok, failed

    def _start_locked(self):
        self._check_open()
        self._stop_all()
        self._generation += 1
        self.result_queue = self.mp_ctx.Queue()
        try:
            for rank in range(self.world_size):
                self._spawn_worker(rank)
            _, failed = self._wait_results("model_ready", self.ready_timeout)
            if failed:
                raise RuntimeError(f"[pool {self.name}] initialization failed: {failed}")
            self._started = True
            # Even an empty warmup list must leave model hooks offloaded.
            # Recovery startup must not recursively try another recovery.
            _, failed = self._rpc_locked("offload", None, "offloaded", self.offload_timeout,
                                         recover_on_failure=False)
            if failed:
                raise RuntimeError(f"[pool {self.name}] startup offload failed: {failed}")
            if not self.is_ready():
                raise RuntimeError(f"[pool {self.name}] workers died before idle readiness confirmation")
        except BaseException:
            self._stop_all()
            raise
        logger.info("[pool %s] generation %s ready; devices=%s",
                    self.name, self._generation, self.device_ids)

    def start(self):
        with self._lock:
            self._ensure_workers_alive()

    def _ensure_workers_alive(self):
        with self._lock:
            self._check_open()
            if self._started and all(p is not None and p.is_alive() for p in self.workers):
                return
            try:
                self._start_locked()
            except PoolRecoveryError:
                raise
            except Exception as exc:
                raise PoolRecoveryError(self.name, "initialization", {}, exc) from exc

    def is_ready(self):
        """Snapshot: initialized, all ranks alive, and idle cleanup acknowledged."""
        with self._lock:
            return (not self._closing.is_set() and not self._aborting.is_set()
                    and self._started and self._idle_offloaded
                    and all(p is not None and p.is_alive() for p in self.workers))

    def _recover_locked(self, command, failures):
        """One synchronous rebuild; NEVER retry the failed case or recurse."""
        logger.warning("[pool %s] RECOVERING after %s; rebuilding all ranks before returning",
                       self.name, command)
        try:
            self._start_locked()
            if not self.is_ready():
                raise RuntimeError("new generation did not reach idle/offloaded readiness")
        except Exception as exc:
            raise PoolRecoveryError(self.name, command, failures, exc) from exc
        logger.warning("[pool %s] RECOVERY_READY generation=%s; all %s ranks initialized "
                       "and offloaded; current case remains FAILED; ready for NEXT case",
                       self.name, self._generation, self.world_size)

    def _rpc_locked(self, command, payload, expected_status, timeout, *, recover_on_failure=True):
        self._idle_offloaded = False
        request_id = uuid.uuid4().hex
        try:
            for rank, q in enumerate(self.task_queues):
                params = None
                if command == "infer":
                    params = dict(payload)
                    params.update(rank=rank, num_gpus=self.world_size,
                                  device_id=self.device_ids[rank], cuda_id=self.device_ids[rank])
                q.put({"generation": self._generation, "request_id": request_id,
                       "command": command, "payload": params})
            ok, failed = self._wait_results(expected_status, timeout, request_id=request_id)
        except Exception as exc:
            ok = set()
            failed = {r: ("dispatch_error", str(exc)) for r in range(self.world_size)}
        if failed:
            logger.error("[pool %s] %s failed; retiring generation %s: %s",
                         self.name, command, self._generation, failed)
            # Do not attempt a GPU cleanup command on a hung process. Its
            # context, hooks, queues and any collective state are discarded.
            try:
                self._stop_all()
            except Exception as exc:
                raise PoolRecoveryError(self.name, command, failed, exc) from exc
            if (recover_on_failure and not self._closing.is_set()
                    and not self._aborting.is_set()):
                self._recover_locked(command, failed)
        elif command == "offload":
            self._idle_offloaded = True
        return ok, failed

    def _reload_all(self):
        """Compatibility API. broadcast_infer prepares workers itself."""
        with self._lock:
            self._ensure_workers_alive()
            ok, failed = self._rpc_locked("reload", None, "reloaded", self.reload_timeout)
            if failed:
                raise RuntimeError(f"[pool {self.name}] reload failed: {failed}")
            return ok, failed

    def _offload_all_best_effort(self):
        with self._lock:
            if not self._started:
                return set(), {}
            # In model_cpu_offload mode this invokes the CPU offload hooks;
            # it does not remove/rebuild them or put all weights on the GPU.
            return self._rpc_locked("offload", None, "offloaded", self.offload_timeout)

    def broadcast_infer(self, infer_params_dict):
        # Single preparation entry: check/spawn here. The worker's infer
        # command restores excluded components and resets peak stats itself;
        # model_cpu_offload hooks handle the remaining onloads at forward.
        with self._lock:
            self._ensure_workers_alive()
            return self._rpc_locked("infer", infer_params_dict, "done", self.result_get_timeout)

    def abort(self):
        """Retire this generation, leaving the pool reusable on the NEXT case."""
        self._aborting.set()
        try:
            with self._lock:
                self._stop_all()
        finally:
            self._aborting.clear()

    def shutdown(self):
        """Final shutdown, including interruption of an in-flight wait."""
        self._closing.set()
        with self._lock:
            self._stop_all(graceful=True)


def _offload_pipeline(model):
    """For ordinary Diffusers/Accelerate enable_model_cpu_offload pipelines.

    Invoke ALL UserCpuOffloadHooks: an exception can leave a component active
    before the pipeline reaches its normal final-hook cleanup.
    """
    import torch
    pipe = model.pipeline
    hooks = list(getattr(pipe, "_all_hooks", None) or [])
    final_hook = getattr(pipe, "final_offload_hook", None)
    if final_hook is not None:
        hooks.append(final_hook)
    seen = set()
    for hook in hooks:
        if id(hook) not in seen:
            seen.add(id(hook))
            hook.offload()
    # Components excluded from the hook sequence (e.g. a custom LightVAE)
    # may also be ordinary GPU modules. Keep hooks installed for the next call.
    components = {}
    try:
        components.update(pipe.components)
    except (AttributeError, ValueError):
        pass
    for name in ("transformer", "transformer_2", "vae", "text_encoder",
                 "text_encoder_2", "image_encoder", "image_proj_model", "controlnet"):
        components.setdefault(name, getattr(pipe, name, None))
    restore = getattr(model, "_worker_pool_restore_components", None)
    if restore is None:
        restore = {}
        model._worker_pool_restore_components = restore
    seen.clear()
    for component in components.values():
        if not isinstance(component, torch.nn.Module) or id(component) in seen:
            continue
        seen.add(id(component))
        tensors = list(component.parameters()) + list(component.buffers())
        if any(t.device.type == "cuda" for t in tensors):
            if any(t.device.type == "meta" for t in tensors):
                raise RuntimeError("Mixed meta/CUDA component: this helper targets model_cpu_offload")
            cuda_devices = {t.device for t in tensors if t.device.type == "cuda"}
            if len(cuda_devices) != 1:
                raise RuntimeError("An unoffloaded component spans GPUs; custom restoration is required")
            # No hook loaded this component back automatically: remember the
            # original device for excluded modules and custom VAE components.
            restore[id(component)] = (component, next(iter(cuda_devices)))
            component.to("cpu")
    # Optional explicit cache release supplied by the custom Wan wrapper.
    release = getattr(model, "release_task_cache", None)
    if callable(release):
        release()


def _reload_pipeline(model):
    # Only restore components left outside the CPU offload hook sequence.
    # Hook-managed transformer/text-encoder/vae stay on CPU until forward.
    for component, device in getattr(model, "_worker_pool_restore_components", {}).values():
        component.to(device)


def _warmup(model, params, device):
    import torch
    sizes = params.get("sample_size_list", [])
    if not sizes:
        return  # Do not even load prompt embeddings when no warmup is requested.
    video_length = params["video_length"]
    if video_length < 1:
        raise ValueError("warmup video_length must be positive")
    embeds = torch.load(params["prompt_embed_path"], map_location="cpu")
    pe = [embeds["prompt"].to(device)]
    ne = [embeds["negative"].to(device)]
    del embeds
    with torch.inference_mode():
        for h, w in sizes:
            _warmup_one(model, video_length, h, w, pe, ne, device)


def _warmup_one(model, frames, h, w, pe, ne, device):
    import torch
    # A separate frame keeps warmup inputs/outputs out of the persistent loop.
    video = torch.rand(1, 3, frames, h, w, device=device, dtype=torch.bfloat16)
    mask = torch.full((1, 1, frames, h, w), 255., device=device, dtype=torch.bfloat16)
    mask[:, :, 0] = 0
    mask[:, :, min(80, frames - 1)] = 0
    ctrl = torch.rand_like(video)
    ref = torch.rand(1, 3, 1, h, w, device=device, dtype=torch.bfloat16)
    model.pipeline._ensure_mask_cache(mask, frames, h, w, 1, torch.bfloat16, device, 1)
    model.pipeline(
        prompt=None, negative_prompt=None, prompt_embeds=pe, negative_prompt_embeds=ne,
        num_frames=frames, height=h, width=w, num_inference_steps=1,
        guidance_scale=0., video=video, mask_video=mask, control_video=ctrl,
        control_camera_video=None, ref_image=ref,
        generator=torch.Generator(device=device).manual_seed(42), boundary=.875, shift=5,
    )


def _run_command(model, command, payload, device_id, *, offload_model=False):
    """Return strings only; exception traceback/param frames die before reply."""
    import torch
    try:
        if command == "offload":
            if offload_model:
                _offload_pipeline(model)
            _cleanup_cuda()
            return "offloaded", None
        if command == "reload":
            # model_cpu_offload hooks load each component on its next forward.
            # Moving transformer/vae to CUDA here would defeat CPU offload.
            _reload_pipeline(model)
            torch.cuda.reset_peak_memory_stats(device_id)
            return "reloaded", None
        if command != "infer":
            raise ValueError(f"Unknown command: {command}")
        _reload_pipeline(model)
        torch.cuda.reset_peak_memory_stats(device_id)
        with torch.inference_mode():
            model.infer(SimpleNamespace(**payload))
        torch.cuda.synchronize(device_id)
        print(f"[worker pid={os.getpid()} cuda:{device_id}] infer completed; "
              f"peak_alloc={torch.cuda.max_memory_allocated(device_id)/1024**3:.3f} GiB",
              flush=True)
        return "done", None
    except Exception as exc:
        status = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error"
        return status, traceback.format_exc()[-16000:]


def persistent_worker_fn(rank, world_size, device_id, model_params_dict,
                         warmup_params_dict, task_queue, result_queue,
                         pool_name="A", generation=0):
    parent_pid = os.getppid()  # Capture in the actual worker, not at module import.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    for key, path in {
        "TRITON_CACHE_DIR": f"~/.triton/cache_worker_{pool_name}_rank{rank}_cuda{device_id}",
        "TORCHINDUCTOR_CACHE_DIR": f"~/.cache/torchinductor_worker_{pool_name}_rank{rank}_cuda{device_id}",
    }.items():
        os.environ[key] = os.path.expanduser(path)
        os.makedirs(os.environ[key], exist_ok=True)

    def reply(request_id, status, error=None):
        if error:
            print(f"[pool {pool_name} rank={rank} pid={os.getpid()}] {status}:\n{error}",
                  file=sys.stderr, flush=True)
            error = error.encode("utf-8")[-2048:].decode("utf-8", errors="replace")
        response = dict(generation=generation, request_id=request_id,
                        rank=rank, status=status, error=error)
        # Linux status messages fit in PIPE_BUF (4096 bytes including framing).
        # Queue uses a single write for this size; a writer killed mid-response
        # cannot leave a large partial frame blocking the parent's recv.
        if len(pickle.dumps(response, protocol=4)) > 3500:
            response["error"] = "worker failed; see worker stderr for traceback"
        result_queue.put(response)

    model = None
    try:
        try:
            import torch
            torch.set_num_threads(1)
            torch.cuda.set_device(device_id)
            device = f"cuda:{device_id}"
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            module = importlib.import_module(model_params_dict.get(
                "module_name",
                "VideoX-Fun.predict_v2v_control_5b_separate_planning_all_views_lora_stream_sample",
            ))
            cls = getattr(module, model_params_dict.get("class_name", "Wan_5b"))
            model = cls(SimpleNamespace(**model_params_dict), device)
            _warmup(model, warmup_params_dict, device)
            # _offload_pipeline(model)
            _cleanup_cuda()
            reply("init", "model_ready")
        except Exception:
            reply("init", "init_failed", traceback.format_exc()[-16000:])
            return

        while os.getppid() == parent_pid:
            try:
                message = task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, OSError, KeyboardInterrupt):
                break
            if not isinstance(message, dict):
                continue
            if message.get("generation") != generation:
                continue
            command = message.get("command")
            if command == "stop":
                break
            status, error = _run_command(
                model, command, message.get("payload"), device_id,
                offload_model=model_params_dict.get("offload_model", False),
            )
            request_id = message.get("request_id")
            message = None  # Release this request's shared input references.
            reply(request_id, status, error)
            if status in {"oom", "error"}:
                # A fresh worker is safer than reusing a partially executed
                # model/offload/collective state after an inference exception.
                break
    finally:
        # Never block here on a CUDA synchronize after an error. The parent
        # imposes bounded terminate/kill deadlines if teardown gets stuck.
        model = None
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
