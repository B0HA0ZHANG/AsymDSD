import os
from typing import Any, Literal, Union

from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers import WandbLogger as _WandbLogger
from wandb.sdk.lib import RunDisabled
from wandb.util import generate_id
from wandb.wandb_run import Run


class WandbLogger(_WandbLogger):
    def __init__(
        self,
        name: str | None = None,
        save_dir: _PATH = ".",
        version: str | None = None,
        offline: bool = False,
        dir: _PATH | None = None,
        id: str | None = None,
        anonymous: bool | None = None,
        project: str | None = None,
        log_model: bool | Literal["all"] = False,
        experiment: Run | RunDisabled | None = None,
        prefix: str = "",
        checkpoint_name: str | None = None,
        resume: Union[str, bool] = "allow",
        **kwargs: Any,
    ) -> None:
        # Always generates unique id if not given.
        if id is None:
            id = generate_id()

        os.environ["WANDB__SERVICE_WAIT"] = "300"

        super().__init__(
            name,
            save_dir,
            version,
            offline,
            dir,
            id,
            anonymous,
            project,
            log_model,
            experiment,
            prefix,
            checkpoint_name,
            resume=resume,
            **kwargs,
        )
