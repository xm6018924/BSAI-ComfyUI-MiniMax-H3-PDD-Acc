"""Compatibility layer for different ComfyUI versions.

This module isolates all ComfyUI-core version checks so the rest of the
plugin can import stable names. If a required core API is missing, a clear
error is raised at import time instead of an opaque AttributeError deep in
the execution path.
"""

import inspect
import logging
import sys

MIN_COMFY_VERSION = (0, 33, 0)
MIN_COMFY_VERSION_STR = "0.33.0"


def _parse_comfy_version():
    """Best-effort ComfyUI version parse from comfy.version.VERSION."""
    try:
        import comfy.version
        raw = getattr(comfy.version, "VERSION", "")
        # e.g. "0.34.0-59-g18ebc2afd" -> (0, 34, 0)
        parts = raw.split("-")[0].split(".")
        return tuple(int(p) for p in parts[:3])
    except Exception:
        return None


def check_core_version():
    """Raise a clear RuntimeError if ComfyUI is too old.

    Returns the detected version tuple (or None if undetectable).
    """
    ver = _parse_comfy_version()
    if ver is not None and ver < MIN_COMFY_VERSION:
        raise RuntimeError(
            f"MiniMax-H3-PDD-Acc requires ComfyUI >= {MIN_COMFY_VERSION_STR} "
            f"(detected {'.'.join(map(str, ver))}). This plugin relies on the "
            f"MiniMax-H3 carried-audio rework (comfyanonymous/ComfyUI#15243). "
            f"Please update ComfyUI."
        )
    return ver


def check_audio_mechanics():
    """Verify the MiniMaxH3Model forward has the audio_scale rework.

    This is the actual feature gate, more reliable than version strings
    (some forks backport features without bumping VERSION).
    """
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
        src = inspect.getsource(MiniMaxH3Model.forward)
    except Exception:
        return  # can't probe — don't block on source unavailability alone
    if "audio_scale" not in src:
        raise RuntimeError(
            "MiniMax-H3-PDD-Acc: this ComfyUI build predates the MiniMax-H3 "
            "audio-mechanics rework (comfyanonymous/ComfyUI#15243), so the "
            "PDD heads would mis-integrate audio. Update ComfyUI to "
            f"{MIN_COMFY_VERSION_STR} or newer."
        )


def get_nested_tensor_class():
    """Return comfy.nested_tensor.NestedTensor, with a clear error if absent."""
    try:
        import comfy.nested_tensor
        return comfy.nested_tensor.NestedTensor
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "MiniMax-H3-PDD-Acc: comfy.nested_tensor.NestedTensor is missing "
            f"from this ComfyUI build ({e}). Update ComfyUI to "
            f"{MIN_COMFY_VERSION_STR} or newer."
        )


def get_wrappers_mp():
    """Return comfy.patcher_extension.WrappersMP, with a clear error if absent."""
    try:
        import comfy.patcher_extension
        return comfy.patcher_extension.WrappersMP
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "MiniMax-H3-PDD-Acc: comfy.patcher_extension.WrappersMP is missing "
            f"from this ComfyUI build ({e}). Update ComfyUI to "
            f"{MIN_COMFY_VERSION_STR} or newer."
        )


# Run lightweight checks at import time so a broken environment fails fast
# with a readable message, rather than halfway through sampling.
try:
    check_core_version()
except RuntimeError as e:
    logging.warning(str(e))
    # Don't hard-fail at import: the node itself will re-check and surface
    # the error in the UI. Import-time hard fails can break the whole UI.
