import math


def validate_threshold(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(
            "Grounding threshold must be finite and strictly between 0 and 1"
        )
    return value


def checkpoint_threshold(embedded=None):
    return None if embedded is None else validate_threshold(embedded)


def resolve_threshold(configured, declared=None, *, allow_override=False):
    configured = validate_threshold(configured)
    if declared is not None:
        declared = validate_threshold(declared)
        if not math.isclose(configured, declared, abs_tol=1e-12) and not allow_override:
            raise ValueError(
                f"Checkpoint requires inference threshold {declared:g}, but the config uses "
                f"{configured:g}. Use configs/brepground-inference.yaml or set "
                f"model.inference.threshold={declared:g}. For an intentional calibration "
                "experiment, pass --allow-threshold-override."
            )
    return configured
