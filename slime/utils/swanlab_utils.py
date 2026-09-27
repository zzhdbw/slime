#slime\utils\swanlab_utils.py
"""SwanLab integration for experiment tracking.

Multi-process: same pattern as wandb — primary creates run and sets args.swanlab_run_id;
secondary (RolloutManager, train actors) call init with id=args.swanlab_run_id, resume=\"allow\"
to log to the same run.
"""
import logging
import os
from copy import deepcopy

logger = logging.getLogger(__name__)


def init_swanlab_primary(args):
    """Create a new SwanLab run in the current process (driver). Sets args.swanlab_run_id for workers."""
    if not getattr(args, "use_swanlab", False):
        return

    try:
        import swanlab
    except ImportError:
        logger.warning(
            "SwanLab requested but 'swanlab' is not installed. Install with: pip install swanlab"
        )
        return

    project = getattr(args, "swanlab_project", None) or getattr(args, "wandb_project", None) or "slime"
    experiment_name = (
        getattr(args, "swanlab_experiment_name", None)
        or getattr(args, "wandb_group", None)
        or _default_swanlab_run_name()
    )
    config = _compute_config_for_logging(args)
    mode = getattr(args, "swanlab_mode", None) or "cloud"

    run = swanlab.init(
        project=project,
        experiment_name=experiment_name,
        config=config,
        mode=mode,
    )
    _define_swanlab_metrics(swanlab)
    args.swanlab_run_id = run.id
    logger.info(
        "SwanLab initialized (primary). project=%s experiment_name=%s run_id=%s",
        project, experiment_name, run.id,
    )


def init_swanlab_secondary(args):
    """Join the existing SwanLab run in worker processes (same run id as primary)."""
    if not getattr(args, "use_swanlab", False):
        return
    swanlab_run_id = getattr(args, "swanlab_run_id", None)
    if swanlab_run_id is None:
        return

    try:
        import swanlab
    except ImportError:
        return

    project = getattr(args, "swanlab_project", None) or getattr(args, "wandb_project", None) or "slime"
    config = _compute_config_for_logging(args)
    mode = getattr(args, "swanlab_mode", None) or "cloud"

    swanlab.init(
        project=project,
        id=swanlab_run_id,
        resume="allow",
        config=config,
        mode=mode,
    )
    logger.info("SwanLab initialized (secondary), joined run_id=%s", swanlab_run_id)


def _define_swanlab_metrics(swanlab):
    """Configure separate X axes for train/rollout/eval metrics when supported."""
    define_metric = getattr(swanlab, "define_metric", None)
    if define_metric is None:
        return

    metric_groups = [
        ("train/*", "train/step"),
        ("rollout/*", "rollout/step"),
        ("multi_turn/*", "rollout/step"),
        ("passrate/*", "rollout/step"),
        ("perf/*", "rollout/step"),
        ("eval/*", "eval/step"),
    ]
    for key, x_axis in metric_groups:
        try:
            define_metric(key, x_axis=x_axis)
        except TypeError:
            # Older SwanLab versions accepted step_metric as a keyword alias.
            try:
                define_metric(key, step_metric=x_axis)
            except Exception:
                logger.warning("Failed to define SwanLab metric %s", key, exc_info=True)
        except Exception:
            logger.warning("Failed to define SwanLab metric %s", key, exc_info=True)


def _default_swanlab_run_name():
    try:
        from slime.utils.external_utils.command_utils import create_run_id
        return create_run_id()
    except Exception:
        import uuid
        return f"run-{uuid.uuid4().hex[:8]}"


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)
    whitelist_env_vars = ["SLURM_JOB_ID"]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}
    return output