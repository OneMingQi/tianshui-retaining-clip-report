"""Common encoder-driven withdrawal diagnostic for semiconductor retaining clips.

This is a necessary endpoint study, not compensated-wrench fullcycle certification.
No object truth is supplied to the withdrawal controller.
"""
import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = ['left_gripper_zjb_Joint', 'left_gripper_yjb_Joint']

class Withdrawal:
    def __init__(self, env, obs):
        self.origin = np.asarray(obs['ee_pos']).copy()
        self.command = self.origin.copy()
        self.finger_start = np.asarray([obs['joint_positions'][env.dof_names.index(n)] for n in FINGERS])
        self.finger_target = np.clip(self.finger_start + np.array([-.008,.008]), -.0249,.0249)
        self.k = 0
        self.velocity = 0.
        self.phase = 'endpoint_open'
        self.failure = None
        self.verify_frames = 0
        self.last_command = self.command.copy()

    def act(self, obs):
        self.k += 1
        measured = np.asarray(obs['ee_pos'])
        # Guards are for this common extension only; native history is audited
        # separately and cannot be retrospectively certified by these guards.
        if np.linalg.norm(np.asarray(obs['force'])) > 500:
            self.failure = 'extension_raw_force_limit'
        if np.linalg.norm(measured-self.last_command) > .005:
            self.failure = 'extension_tracking_limit'
        if self.failure:
            self.phase = 'endpoint_failed'
            return {'ee_pos':measured.copy(),'grip_targets':dict(zip(FINGERS,self.finger_target))},True
        if self.k <= 120:
            finger = self.finger_start + (self.finger_target-self.finger_start)*min(1.,self.k/120.)
        else:
            finger = self.finger_target.copy()
            self.phase = 'endpoint_withdraw'
            remaining = max(0.,self.origin[2]+.105-self.command[2])
            desired = min(.015, np.sqrt(2*.1*remaining))
            self.velocity += np.clip(desired-self.velocity,-.1/60,.1/60)
            requested = self.command.copy()
            requested[2] += min(remaining,self.velocity/60)
            error = requested-measured
            self.command = measured+error*min(1.,.003/max(np.linalg.norm(error),1e-12))
            if measured[2]-self.origin[2] >= .100 and remaining < .0005:
                self.phase = 'endpoint_verify'
                self.verify_frames += 1
            else:
                self.verify_frames = 0
        self.last_command = self.command.copy()
        return {'ee_pos':self.command.copy(),'grip_targets':dict(zip(FINGERS,finger))},self.verify_frames >= 61


def summarize(rows, origin, target, target_quat, failure, native_failed):
    if not rows:
        return {'necessary_endpoint_pass':False,'reason':failure or 'no_extension_samples','fullcycle_certified':False}
    position = np.asarray([r['retaining_clip_pos'] for r in rows])
    quaternion = np.asarray([r['retaining_clip_quat'] for r in rows])
    ee = np.asarray([r['ee_pos'] for r in rows])
    phase = np.asarray([r['phase_before'] for r in rows])
    last = rows[-61:]
    enough = len(last)==61 and all(r['phase_before']=='endpoint_verify' for r in last)
    tail = position[-61:]
    linear = np.diff(tail,axis=0)*60
    rotations = Rotation.from_quat(quaternion[-61:])
    angular = (rotations[:-1].inv()*rotations[1:]).as_rotvec()*60 if len(tail)>1 else np.zeros((0,3))
    pos_error = np.linalg.norm(tail-np.asarray(target),axis=1)
    angles = (Rotation.from_quat(target_quat).inv()*rotations).magnitude()
    speed = float(np.sqrt(np.mean(np.sum(linear**2,axis=1)))) if len(linear) else None
    angular_speed = float(np.sqrt(np.mean(np.sum(angular**2,axis=1)))) if len(angular) else None
    p2p = float(np.linalg.norm(np.ptp(tail,axis=0)))
    withdrawal = float(ee[-1,2]-origin[2])
    stable = bool(enough and pos_error.max()<=.002 and angles.max()<=np.pi/180 and speed<=.002 and angular_speed<=.02 and p2p<=.0005)
    return {'necessary_endpoint_pass':bool(not native_failed and failure is None and withdrawal>=.100 and stable),
            'reason':failure,'sample_count':len(rows),'withdrawal_vertical_m':withdrawal,
            'verify_1s_complete':enough,'position_peak_m':float(pos_error.max()),'angle_peak_rad':float(angles.max()),
            'speed_rms_m_s':speed,'angular_speed_rms_rad_s':angular_speed,'position_p2p_m':p2p,
            'pose_stability_pass':stable,'native_controller_failed':native_failed,
            'gripper_clearance_certified':False,'contact_separation_certified':False,
            'fullcycle_certified':False,'note':'Finite-difference pose stability diagnostic. Contact clearance, compensated wrench and full-history hard safety not certified.'}
