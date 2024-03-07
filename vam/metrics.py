import jax
import pdb
import jax.numpy as jnp
from jax import random
import numpy as np
import wandb
from clu import metrics
import flax

from .lba import sim_lba_rts


@flax.struct.dataclass
class MeanRT(metrics.Average):
    @classmethod
    def from_model_output(
        cls,
        *,
        rts: jnp.array,
        responses: jnp.array,
        targets: jnp.array,
        valid_idx: jnp.array,
        **kwargs,
    ):
        correct_idx = jnp.where(
            responses == targets, jnp.ones(targets.shape), jnp.zeros(targets.shape)
        )
        metric = super().from_model_output(
            values=rts, mask=(correct_idx * valid_idx).astype(bool), **kwargs
        )
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class FractionValid(metrics.Average):
    """Calculate the fraction of valid trials (at least one drift rate > 0)."""

    @classmethod
    def from_model_output(cls, *, valid_idx: jnp.array, **kwargs):
        metric = super().from_model_output(values=valid_idx, **kwargs)
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class TaskAccuracy(metrics.Average):
    @classmethod
    def from_model_output(
        cls, *, responses: jnp.array, targets: jnp.array, valid_idx: jnp.array, **kwargs
    ):
        correct_idx = jnp.where(
            responses == targets, jnp.ones(targets.shape), jnp.zeros(targets.shape)
        )
        metric = super().from_model_output(
            values=correct_idx, mask=(valid_idx).astype(bool), **kwargs
        )
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class MeanDrift(metrics.Average):
    """Calculate the average drift rate across all accumulators."""

    @classmethod
    def from_model_output(cls, *, drifts: jnp.array, valid_idx: jnp.array, **kwargs):
        metric = super().from_model_output(
            values=jnp.mean(drifts, axis=1), mask=(valid_idx).astype(bool), **kwargs
        )
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class TargetMeanDrift(metrics.Average):
    """Calculate the average drift rate for the target (all valid trials)."""

    @classmethod
    def from_model_output(
        cls, *, drifts: jnp.array, targets: jnp.array, valid_idx: jnp.array, **kwargs
    ):
        target_drifts = drifts[jnp.arange(len(targets)), targets]
        metric = super().from_model_output(
            values=target_drifts, mask=(valid_idx).astype(bool), **kwargs
        )
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class RTConEffect(metrics.Metric):
    """Calculate the RT congruency effect (correct trials only)."""

    con_count: jnp.array
    incon_count: jnp.array
    con_rt_sum: jnp.array
    incon_rt_sum: jnp.array

    @classmethod
    def empty(cls):
        return cls(
            con_count=jnp.zeros((1,), jnp.int32),
            incon_count=jnp.zeros((1,), jnp.int32),
            con_rt_sum=jnp.zeros((1,), jnp.float32),
            incon_rt_sum=jnp.zeros((1,), jnp.float32),
        )

    @classmethod
    def from_model_output(
        cls,
        *,
        rts: jnp.array,
        responses: jnp.array,
        targets: jnp.array,
        valid_idx: jnp.array,
        congruency: jnp.array,
        **_,
    ):
        correct_idx = jnp.where(
            responses == targets, jnp.ones(targets.shape), jnp.zeros(targets.shape)
        )
        con_idx = jnp.where(congruency == 0, jnp.ones(rts.shape), jnp.zeros(rts.shape))
        incon_idx = jnp.where(
            congruency == 1, jnp.ones(rts.shape), jnp.zeros(rts.shape)
        )

        con_rt_sum = jnp.sum(rts * con_idx * correct_idx * valid_idx)
        incon_rt_sum = jnp.sum(rts * incon_idx * correct_idx * valid_idx)
        con_count = jnp.sum(con_idx * correct_idx * valid_idx)
        incon_count = jnp.sum(incon_idx * correct_idx * valid_idx)

        return cls(
            con_count=con_count,
            incon_count=incon_count,
            con_rt_sum=con_rt_sum,
            incon_rt_sum=incon_rt_sum,
        )

    def merge(self, other):
        return type(self)(
            con_count=self.con_count + other.con_count,
            incon_count=self.incon_count + other.incon_count,
            con_rt_sum=self.con_rt_sum + other.con_rt_sum,
            incon_rt_sum=self.incon_rt_sum + other.incon_rt_sum,
        )

    def compute(self):
        return self.incon_rt_sum / self.incon_count - self.con_rt_sum / self.con_count


@flax.struct.dataclass
class AccConEffect(metrics.Metric):
    """Calculate the accuracy congruency effect."""

    con_count: jnp.array
    incon_count: jnp.array
    con_correct_sum: jnp.array
    incon_correct_sum: jnp.array

    @classmethod
    def empty(cls):
        return cls(
            con_count=jnp.zeros((1,), jnp.int32),
            incon_count=jnp.zeros((1,), jnp.int32),
            con_correct_sum=jnp.zeros((1,), jnp.float32),
            incon_correct_sum=jnp.zeros((1,), jnp.float32),
        )

    @classmethod
    def from_model_output(
        cls,
        *,
        responses: jnp.array,
        targets: jnp.array,
        valid_idx: jnp.array,
        congruency: jnp.array,
        **_,
    ):
        correct_idx = jnp.where(
            responses == targets, jnp.ones(targets.shape), jnp.zeros(targets.shape)
        )
        con_idx = jnp.where(
            congruency == 0, jnp.ones(targets.shape), jnp.zeros(targets.shape)
        )
        incon_idx = jnp.where(
            congruency == 1, jnp.ones(targets.shape), jnp.zeros(targets.shape)
        )

        con_correct_sum = jnp.sum(con_idx * correct_idx * valid_idx)
        incon_correct_sum = jnp.sum(incon_idx * correct_idx * valid_idx)
        con_count = jnp.sum(con_idx * valid_idx)
        incon_count = jnp.sum(incon_idx * valid_idx)

        return cls(
            con_count=con_count,
            incon_count=incon_count,
            con_correct_sum=con_correct_sum,
            incon_correct_sum=incon_correct_sum,
        )

    def merge(self, other):
        return type(self)(
            con_count=self.con_count + other.con_count,
            incon_count=self.incon_count + other.incon_count,
            con_correct_sum=self.con_correct_sum + other.con_correct_sum,
            incon_correct_sum=self.incon_correct_sum + other.incon_correct_sum,
        )

    def compute(self):
        return (
            self.con_correct_sum / self.con_count
            - self.incon_correct_sum / self.incon_count
        )


@flax.struct.dataclass
class ConTargetDrift(metrics.Average):
    """Calculate the average drift rate for the target (all congruent trials)."""

    @classmethod
    def from_model_output(
        cls,
        *,
        drifts: jnp.array,
        targets: jnp.array,
        valid_idx: jnp.array,
        congruency: jnp.array,
        **kwargs,
    ):
        target_drifts = drifts[jnp.arange(len(targets)), targets]
        metric = super().from_model_output(
            values=target_drifts,
            mask=((1 - congruency) * valid_idx).astype(bool),
            **kwargs,
        )
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class InconTargetDrift(metrics.Average):
    """Calculate the average drift rate for the target (all incongruent trials)."""

    @classmethod
    def from_model_output(
        cls,
        *,
        drifts: jnp.array,
        targets: jnp.array,
        valid_idx: jnp.array,
        congruency: jnp.array,
        **kwargs,
    ):
        target_drifts = drifts[jnp.arange(len(targets)), targets]
        metric = super().from_model_output(
            values=target_drifts, mask=(congruency * valid_idx).astype(bool), **kwargs
        )
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class InconDistractorDrift(metrics.Average):
    """Calculate the average drift rate for the distractor direction
    (all incongruent trials, for congruent trials this would be the same
     as ConTargetDrift)."""

    @classmethod
    def from_model_output(
        cls,
        *,
        drifts: jnp.array,
        distractors: jnp.array,
        valid_idx: jnp.array,
        congruency: jnp.array,
        **kwargs,
    ):
        distractor_drifts = drifts[jnp.arange(len(distractors)), distractors]
        metric = super().from_model_output(
            values=distractor_drifts,
            mask=(congruency * valid_idx).astype(bool),
            **kwargs,
        )
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class ConOtherDrift(metrics.Average):
    """Calculate the average drift rate for the non-target
    directions (all congruent trials)."""

    @classmethod
    def from_model_output(
        cls,
        *,
        drifts: jnp.array,
        targets: jnp.array,
        valid_idx: jnp.array,
        congruency: jnp.array,
        **kwargs,
    ):
        target_drifts = drifts[jnp.arange(len(targets)), targets]
        other_drifts = (jnp.sum(drifts, axis=1) - target_drifts) / 3
        metric = super().from_model_output(
            values=other_drifts,
            mask=((1 - congruency) * valid_idx).astype(bool),
            **kwargs,
        )
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class InconOtherDrift(metrics.Average):
    """Calculate the average drift rate for the non-target non-distraction
    directions (all incongruent trials)."""

    @classmethod
    def from_model_output(
        cls,
        *,
        drifts: jnp.array,
        targets: jnp.array,
        distractors: jnp.array,
        valid_idx: jnp.array,
        congruency: jnp.array,
        **kwargs,
    ):
        target_drifts = drifts[jnp.arange(len(targets)), targets]
        distractor_drifts = drifts[jnp.arange(len(distractors)), distractors]
        other_drifts = (jnp.sum(drifts, axis=1) - target_drifts - distractor_drifts) / 2
        metric = super().from_model_output(
            values=other_drifts, mask=(congruency * valid_idx).astype(bool), **kwargs
        )
        return cls(**vars(metric))  # cls(metrics) doesn't work for a dataclass


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


@flax.struct.dataclass
class TaskOptMetrics(metrics.Collection):
    # Star Search metrics (eval)
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy


@flax.struct.dataclass
class VAMMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    mean_rt: MeanRT
    accuracy: TaskAccuracy
    rt_con_effect: RTConEffect
    acc_con_effect: AccConEffect
    con_target_drift: ConTargetDrift
    incon_target_drift: InconTargetDrift
    incon_distractor_drift: InconDistractorDrift
    con_other_drift: ConOtherDrift
    incon_other_drift: InconOtherDrift
    fraction_valid: FractionValid


def get_vam_user_metrics(batch):
    user_metrics = None
    for b in batch:
        imgs, choices, rts, targets, distractors, congruency = (
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b[5],
        )

        metrics = VAMMetrics.single_from_model_output(
            loss=None,
            rts=rts,
            responses=choices,
            targets=targets,
            distractors=distractors,
            valid_idx=jnp.ones([len(rts)]),
            drifts=jnp.zeros((len(rts), 4)),  # dummy input to preserve API
            congruency=congruency,
        )

        if user_metrics is None:
            user_metrics = metrics
        else:
            user_metrics = user_metrics.merge(metrics)

    user_metrics = user_metrics.compute()

    return user_metrics


def log_metrics(
    train_metrics,
    eval_metrics,
    user_metrics,
    epoch,
    model_lba_params,
):
    for key, val in train_metrics.items():
        wandb.log({f"train/{key}": val}, step=epoch)

    for key, val in eval_metrics.items():
        wandb.log({f"val/{key}": val}, step=epoch)

    if model_lba_params is not None:
        for key, val in model_lba_params.items():
            wandb.log({f"model_{key}": val}, step=epoch)

    if user_metrics is not None:
        for key, val in user_metrics.items():
            if key == "loss":
                continue
            wandb.log({f"user/{key}": val}, step=epoch)
