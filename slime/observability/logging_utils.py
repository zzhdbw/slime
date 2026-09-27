import logging

import wandb

from slime.observability import wandb_utils
from slime.observability.tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _use_swanlab(args) -> bool:
    return bool(getattr(args, "use_swanlab", False))


def _get_swanlab_run():
    """Return the active SwanLab run, or None when SwanLab is unavailable/idle."""
    try:
        import swanlab

        try:
            return swanlab.get_run()
        except RuntimeError:
            return None
    except ImportError:
        return None


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
        if _use_swanlab(args):
            from slime.utils import swanlab_utils

            swanlab_utils.init_swanlab_primary(args)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)
        if _use_swanlab(args):
            from slime.utils import swanlab_utils

            swanlab_utils.init_swanlab_secondary(args)


def finish_tracking(args):
    if _use_swanlab(args):
        try:
            if _get_swanlab_run() is not None:
                import swanlab

                swanlab.finish()
        except Exception:
            logging.getLogger(__name__).exception("Failed to finish SwanLab run")

    if not getattr(args, "use_wandb", False):
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if getattr(args, "use_wandb", False):
        wandb.log(metrics)

    if getattr(args, "use_tensorboard", False):
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    if _use_swanlab(args):
        if _get_swanlab_run() is None:
            return
        import swanlab

        # Keep the step metric in the payload: ``swanlab.define_metric`` uses
        # e.g. ``train/step`` / ``rollout/step`` as the custom X series.
        swanlab_metrics = dict(metrics)
        step = metrics.get(step_key)
        if step is not None:
            swanlab.log(swanlab_metrics, step=int(step))
        else:
            swanlab.log(swanlab_metrics)
