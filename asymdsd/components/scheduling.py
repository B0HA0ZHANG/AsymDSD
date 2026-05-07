import math
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar


class Schedule(ABC):
    """Base class for all schedules.

    A schedule is a function that maps a step to a value.
    """

    def __init__(
        self,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        steps_per_epoch: int = 1,
    ) -> None:
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.can_update_max_epochs = False

        if max_steps is None and max_epochs is not None:
            self.max_epochs = max_epochs

            if self.max_epochs == -1:
                self.can_update_max_epochs = True
            else:
                self.max_steps = max_epochs * steps_per_epoch

        elif max_steps is not None and max_epochs is None:
            self.max_steps = max_steps
            self.max_epochs = max_steps // steps_per_epoch

        else:
            raise ValueError(
                "Either max_steps or max_epochs must be given. "
                "Can give -1 for max_epochs to be initialized later."
            )

        self.max_steps: int = max_steps  # type: ignore
        self.max_epochs = max_epochs

    def update(self):
        if self.max_epochs is not None:
            self.max_steps = (
                0 if self.max_epochs < 0 else self.max_epochs * self.steps_per_epoch
            )

    def set_default_max_epochs(self, max_epochs: int) -> None:
        if self.can_update_max_epochs:
            self.max_epochs = max_epochs
            self.update()

    def set_steps_per_epoch(self, steps_per_epoch: int) -> None:
        self.steps_per_epoch = steps_per_epoch
        self.update()

    @abstractmethod
    def __call__(self, step: int) -> float:
        pass


class LinearWarmupSchedule(Schedule):
    def __init__(
        self,
        start_value: float,
        final_value: float,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        steps_per_epoch: int = 1,
    ) -> None:
        super().__init__(max_steps, max_epochs, steps_per_epoch)
        self.start_value = start_value
        self.final_value = final_value
        self.update()

    def __call__(self, step: int) -> float:
        if step < self.max_steps:
            return (
                self.start_value
                + (self.final_value - self.start_value) * step / self.max_steps
            )

        return self.final_value


class CosineAnnealingWarmupSchedule(Schedule):
    def __init__(
        self,
        base_value: float,
        final_value: float,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        steps_per_epoch: int = 1,
        warmup_epochs: int | None = None,
        warmup_steps: int | None = None,
        startup_value: float = 0.0,
    ) -> None:
        super().__init__(max_steps, max_epochs, steps_per_epoch)
        self.base_value = base_value
        self.final_value = final_value
        self.warmup_epochs = warmup_epochs
        self.warmup_steps: int = warmup_steps  # type: ignore
        if warmup_steps is not None and warmup_epochs is not None:
            raise ValueError("Only one of warmup_epochs and warmup_steps can be given.")
        if warmup_epochs is None and warmup_steps is None:
            self.warmup_steps = 0
        self.startup_value = startup_value
        self.update()

    def update(self):
        super().update()
        if self.warmup_epochs is not None:
            self.warmup_steps = self.warmup_epochs * self.steps_per_epoch

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return (
                self.startup_value
                + (self.base_value - self.startup_value) * step / self.warmup_steps
            )
        elif step < self.max_steps:
            return self.final_value + 0.5 * (self.base_value - self.final_value) * (
                1
                + math.cos(
                    math.pi
                    * (step - self.warmup_steps)
                    / (self.max_steps - self.warmup_steps)
                )
            )

        return self.final_value


class SequentialSchedule(Schedule):
    def __init__(
        self,
        schedules: list[Schedule],
        steps_per_epoch: int = 1,
    ) -> None:
        self.schedules = schedules
        self.set_steps_per_epoch(steps_per_epoch)

        self.current_schedule = 0

    def set_default_max_epochs(self, max_epochs: int) -> None:
        self.schedules[-1].set_default_max_epochs(
            max_epochs - self.steps_cumulative[-2] // self.steps_per_epoch
        )
        self.update()

    def set_steps_per_epoch(self, steps_per_epoch: int) -> None:
        self.steps_per_epoch = steps_per_epoch
        for schedule in self.schedules:
            schedule.set_steps_per_epoch(steps_per_epoch)
        self.update()

    def update(self):
        for schedule in self.schedules:
            schedule.update()

        self.steps_cumulative = [0]
        for schedule in self.schedules[:-1]:
            self.steps_cumulative.append(self.steps_cumulative[-1] + schedule.max_steps)

        if self.schedules[-1].max_steps is not None:
            self.steps_cumulative.append(
                self.steps_cumulative[-1] + self.schedules[-1].max_steps
            )
        else:
            self.steps_cumulative.append(-1)

    def __call__(self, step: int) -> float:
        if (
            step >= self.steps_cumulative[self.current_schedule + 1]
            and self.current_schedule < len(self.schedules) - 1
        ):
            self.current_schedule += 1

        schedule_step = step - self.steps_cumulative[self.current_schedule]

        return self.schedules[self.current_schedule](schedule_step)


# Requires 3.12
# class StateFullScheduler[T](nn.Module):
T = TypeVar("T")


class Scheduler:
    def __init__(self, **schedules: Any | Callable[[int], Any]) -> None:
        super().__init__()
        self.schedule: dict[str, Callable[[int], Any]] = {}

        def _wrap_callable(x):
            return x if callable(x) else lambda _: x

        for name, schedule in schedules.items():
            schedule_fn = _wrap_callable(schedule)
            self.schedule[name] = schedule_fn

        self._step = 0
        self._update_values()

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def value(self) -> dict[str, Any]:
        return self._value

    @staticmethod
    def step_forward(fn: Callable) -> Callable:
        def wrapper(self: Scheduler, *args, **kwargs):
            self.step()
            res = fn(self, *args, **kwargs)
            return res

        return wrapper

    def step(self) -> None:
        self._step += 1
        self._update_values()

    def _update_values(self) -> None:
        self._value = {
            name: schedule(self._step) for name, schedule in self.schedule.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "last_values": self._value,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._step = state_dict["step"]
        self._update_values()
        self._value.update(state_dict.get("last_values", {}))
