"""
Actuator reference shaping: make real joints track the idealized second-order
reference dynamics the simulator assumed, instead of raising simulator fidelity
to match the real hardware.

Adapted from "Actuator Reality Shaping for Zero-Shot Sim-to-Real Robot Learning"
(arXiv:2607.02205). The paper equips each joint with a two-degree-of-freedom
(2-DOF) feedforward-feedback controller that shapes the *closed-loop* response
of a physical actuator to a prescribed second-order reference model, decoupling
reference-response shaping (feedforward) from robust stabilization (feedback).
Policies trained against that reference model then transfer zero-shot.

ManiSkill's :py:class:`~mani_skill.envs.sim2real_env.Sim2RealEnv` forwards the
simulator's raw drive targets to the real motors. A servo's own internal
position loop (e.g. the PID inside a Dynamixel / Feetech servo on the SO100 and
Koch arms) responds to those targets with hardware-dependent lag, backlash and
gain nonlinearities, so the real joint does not follow the trajectory the
policy was trained on. This module closes that gap on the real side: it runs
the idealized second-order reference model forward at the real control rate and
commands the position that the *ideal* actuator would have reached, using a
high-gain feedback term on the measured joint error to pin the real joint onto
that ideal trajectory.

Only the shaping law itself lives here. Anything that needs to talk to a motor
(stub friction, torque feedforward, saturation) belongs in a concrete
``BaseRealAgent`` that can actually command torques.
"""

from dataclasses import dataclass, field
import math
from typing import Optional

import torch

from mani_skill.utils import common
from mani_skill.utils.structs.types import Array

__all__ = ["ReferenceModelConfig", "ActuatorReferenceShaper"]


@dataclass
class ReferenceModelConfig:
    """Parameters of the idealized second-order reference dynamics.

    The reference model is the critically-damped-by-default second-order system
    used as the actuator model during policy training::

        ddot{q}_m = omega_n^2 * (q_ref - q_m) - 2 * zeta * omega_n * qdot_m

    A PDJointPos controller in ManiSkill drives joints with
    ``stiffness = omega_n^2`` and ``damping = 2 * zeta * omega_n`` (see e.g. the
    SO100 agent's ``stiffness=1e3, damping=1e2`` gains), so the natural
    frequency and damping ratio of the trained-on dynamics can be read straight
    off the agent's controller config.

    Args:
        natural_frequency (float): undamped natural frequency omega_n in rad/s.
        damping_ratio (float): damping ratio zeta; 1.0 is critically damped.
        control_dt (float): real control period in seconds. The reference model
            is integrated once per real control step.
    """

    natural_frequency: float = 31.6
    damping_ratio: float = 1.58
    control_dt: float = 1 / 30

    def __post_init__(self):
        if self.natural_frequency <= 0:
            raise ValueError(
                "natural_frequency must be positive, got "
                f"{self.natural_frequency}"
            )
        if self.control_dt <= 0:
            raise ValueError(
                f"control_dt must be positive, got {self.control_dt}"
            )

    @classmethod
    def from_pd_gains(cls, stiffness: float, damping: float, control_dt: float):
        """Derive the reference model from a ManiSkill PD joint controller.

        Args:
            stiffness: the controller config's stiffness (equals omega_n^2).
            damping: the controller config's damping (equals 2 * zeta * omega_n).
            control_dt: real control period in seconds.
        """
        omega_n = float(stiffness) ** 0.5
        zeta = float(damping) / (2 * omega_n)
        return cls(
            natural_frequency=omega_n, damping_ratio=zeta, control_dt=control_dt
        )


@dataclass
class ActuatorReferenceShaper:
    """2-DOF feedforward-feedback shaper for one bank of real joints.

    Given the joint targets the simulator produced, this computes what an
    ideal actuator obeying :py:class:`ReferenceModelConfig` would have done by
    now, then commands the real servo to that ideal state plus a feedback
    correction for the tracking error it has accumulated so far. The real
    closed loop is thereby shaped to match the trained-on dynamics regardless
    of the servo's own gains or load.

    Feedforward and feedback are decoupled by design: ``_integrate_reference``
    depends only on the command history and realizes the reference dynamics,
    while the ``feedback_gain`` term acts only on the measured error and
    stabilizes the real joint onto the reference trajectory. Raising
    ``feedback_gain`` makes the real joint track harder without changing the
    response shape the policy expects.

    Args:
        num_joints (int): number of joints to shape, M in the (1, M) qpos
            tensors this shaper exchanges with the real agent.
        config (Optional[ReferenceModelConfig]): the idealized reference
            dynamics. Defaults to the SO100 arm's PD gains
            (``stiffness=1e3, damping=1e2``) at a 30 Hz control rate.
        feedback_gain (float): dimensionless gain on the measured reference
            error ``q_m - q``. 0 disables feedback and leaves only the
            feedforward reference trajectory, which is the paper's ablation
            baseline.
        qpos_limits (Optional[torch.Tensor]): (2, M) lower/upper joint limits
            used to clamp commanded positions so a poorly tuned shaper cannot
            command a joint past its mechanical range.
        initial_qpos (Optional[Array]): (1, M) or (M,) joint positions to
            initialize the reference state from. If omitted, the reference
            state starts at zero and should be seeded via :py:meth:`reset`
            before the first command.
    """

    num_joints: int
    config: ReferenceModelConfig = field(default_factory=ReferenceModelConfig)
    feedback_gain: float = 1.0
    qpos_limits: Optional[torch.Tensor] = None
    initial_qpos: Optional[Array] = None

    def __post_init__(self):
        if self.num_joints <= 0:
            raise ValueError(
                f"num_joints must be positive, got {self.num_joints}"
            )
        if self.config.natural_frequency <= 0:
            raise ValueError(
                "natural_frequency must be positive, got "
                f"{self.config.natural_frequency}"
            )
        if self.config.control_dt <= 0:
            raise ValueError(
                f"control_dt must be positive, got {self.config.control_dt}"
            )
        if self.feedback_gain < 0:
            raise ValueError(
                f"feedback_gain must be non-negative, got {self.feedback_gain}"
            )
        if self.qpos_limits is not None:
            limits = common.to_tensor(self.qpos_limits).float()
            if limits.shape != (2, self.num_joints):
                raise ValueError(
                    "qpos_limits must have shape (2, num_joints), got "
                    f"{tuple(limits.shape)} for {self.num_joints} joints"
                )
            self.qpos_limits = limits

        # integrate the reference dynamics exactly rather than with a fixed
        # step, so the shaped response does not drift with the control rate
        wn = self.config.natural_frequency
        zeta = self.config.damping_ratio
        dt = self.config.control_dt
        # Phi = exp(A dt) for A = [[0, 1], [-wn^2, -2*zeta*wn]], written via
        # Cayley-Hamilton as alpha*I + beta*A. The trig branch covers zeta < 1
        # and the hyperbolic branch zeta >= 1; both reduce to the critically
        # damped form as the damped frequency goes to zero.
        wn_d = wn * abs(1 - zeta**2) ** 0.5
        decay = math.exp(-zeta * wn * dt)
        if zeta < 1:
            wave, ratio = math.cos, math.sin
        else:
            wave, ratio = math.cosh, math.sinh
        arg = wn_d * dt
        if wn_d == 0.0:
            # critically damped: the trig terms degenerate to their limits
            wave_val, ratio_val = 1.0, dt
        else:
            wave_val, ratio_val = wave(arg), ratio(arg) / wn_d
        self._phi_alpha = decay * (wave_val + zeta * wn * ratio_val)
        self._phi_beta = decay * ratio_val
        self.reset(self.initial_qpos)

    # --------------------------------------------------------------------- #
    # feedforward: the idealized reference model
    # --------------------------------------------------------------------- #
    def reset(self, qpos: Optional[Array] = None):
        """Seed the reference state.

        Args:
            qpos: (1, M) or (M,) measured joint positions to start the
                reference model from. Passing None resets to zeros.
        """
        device = None
        if qpos is not None:
            qpos = common.to_tensor(qpos).float().flatten()
            device = qpos.device
            if qpos.shape[0] != self.num_joints:
                raise ValueError(
                    f"expected {self.num_joints} joint positions, got "
                    f"{qpos.shape[0]}"
                )
        else:
            qpos = torch.zeros(self.num_joints)
        self._ref_qpos = qpos.clone()
        self._ref_qvel = torch.zeros(self.num_joints, device=device)
        self._last_measured = qpos.clone()
        # q_target the reference model is currently converging towards; None
        # until the first command so the first command is not fought by a
        # stale reference
        self._target = None
        return self

    def _integrate_reference(self, target: torch.Tensor):
        """Advance the idealized second-order model one control step.

        Integrates the closed-form discrete transition of the reference
        dynamics rather than with Euler steps, so the shaped response does not
        change character when the real control rate differs from the simulator
        control rate. The target is held constant over the step, which is what
        a zero-order-hold servo command physically does.
        """
        if self._target is None:
            self._target = target.clone()
        prev_target = self._target
        self._target = target.clone()

        # error coordinates relative to the target being tracked
        err = self._ref_qpos - prev_target
        err_vel = self._ref_qvel
        wn = self.config.natural_frequency
        zeta = self.config.damping_ratio
        alpha, beta = self._phi_alpha, self._phi_beta
        # Phi = alpha*I + beta*A applied to the error state, then shifted into
        # the new target's frame
        self._ref_qpos = target + alpha * err + beta * err_vel
        self._ref_qvel = -beta * wn * wn * err + (
            alpha - beta * 2 * zeta * wn
        ) * err_vel
        return self._ref_qpos

    # --------------------------------------------------------------------- #
    # the 2-DOF shaping law
    # --------------------------------------------------------------------- #
    def compute_command(self, target_qpos: Array, measured_qpos: Array):
        """Map a simulator drive target to a shaped real joint command.

        Args:
            target_qpos: (1, M) or (M,) joint targets produced by the
                simulator controller (what Sim2RealEnv would have sent raw).
            measured_qpos: (1, M) or (M,) current measured joint positions
                read back from the real robot.

        Returns:
            (1, M) tensor of shaped joint positions to command on the real
            robot, clamped to ``qpos_limits`` when provided.
        """
        target = common.to_tensor(target_qpos).float().flatten()
        measured = common.to_tensor(measured_qpos).float().flatten()
        if target.shape != measured.shape:
            raise ValueError(
                f"target_qpos shape {tuple(target.shape)} does not match "
                f"measured_qpos shape {tuple(measured.shape)}"
            )
        if target.shape[0] != self.num_joints:
            raise ValueError(
                f"expected {self.num_joints} joints, got {target.shape[0]}"
            )

        # feedforward: where the ideal actuator is now
        self._integrate_reference(target)
        # feedback: pull the real joint onto the ideal trajectory
        self._last_measured = measured
        command = self._ref_qpos + self.feedback_gain * (
            self._ref_qpos - measured
        )
        if self.qpos_limits is not None:
            command = command.clamp(
                min=self.qpos_limits[0].to(command.device),
                max=self.qpos_limits[1].to(command.device),
            )
        return command.unsqueeze(0)

    @property
    def reference_qpos(self):
        """(M,) current position of the idealized reference model."""
        return self._ref_qpos

    @property
    def reference_qvel(self):
        """(M,) current velocity of the idealized reference model."""
        return self._ref_qvel

    @property
    def tracking_error(self):
        """(M,) ideal reference position minus the last measured position.

        This is the sim-to-real tracking error the shaping is trying to drive
        to zero, and is the quantity to log when comparing shaped versus
        unshaped deployment on real hardware.
        """
        return self._ref_qpos - self._last_measured

    @property
    def rms_tracking_error(self):
        """Root-mean-square of :py:attr:`tracking_error` across joints, the
        reduction in which is the headline measurement of the paper."""
        err = self.tracking_error
        return torch.sqrt(torch.mean(err**2))
