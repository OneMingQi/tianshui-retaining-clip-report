#!/usr/bin/env python3
"""上帝视角数据生成器：并行跑 N 个环境，从随机初始态出发采集"成功下降插入"数据。

客户端（这里）完全知道目标位置，所以策略是：
  1) z 不变、水平回到初始 xy（消除随机偏移）；
  2) 匀速下降到底（可选侧向力柔顺）；
  3) 落座后松爪；下降中遇到异常力则松爪并记为失败。

每回合保存一个 npz：
  joint_pos / ee_pos / ee_cmd / opening / obj_pos / obj_delta / force / torque / phase
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import shutil
import sys
import time

import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tianshui_insert_pack.controllers import ReplayInsertController, ScriptedInsertController
from tianshui_insert_pack.trajectories import load_descend_path
from tianshui_insert_pack.world import _resolve as _resolve_spec_path
from tianshui_insert_pack.world import TianshuiWorld


class DataGenerator:
    @staticmethod
    def _resid(x, w=11):
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < w + 2:
            return 0.0, 0.0
        sm = np.convolve(x, np.ones(w) / w, mode="same")
        core = (x - sm)[w:-w]
        return float(np.sqrt(np.mean(core ** 2))), float(np.abs(core).max())

    def _quality(self, rec):
        """下降/接触段的抖动与合力（用于数据质量门，单位 mm / N）。"""
        ph = np.asarray(rec["phase"], dtype=str)
        m = np.isin(ph, ["descend", "cart_descend"])
        if m.sum() < 30 or len(rec["ee_pos"]) == 0:
            return {"jitter_rms": 0.0, "jitter_peak": 0.0, "force_max": 0.0,
                    "force_med_max": 0.0, "n": int(m.sum())}
        ee = np.asarray(rec["ee_pos"], dtype=float)[m] * 1000.0
        cmd = np.asarray(rec["ee_cmd"], dtype=float)[m] * 1000.0
        f = np.asarray(rec["force"], dtype=float)[m]
        # 用"跟随误差"量抖动：主动的下压/试探位移不该算抖动，只有末端跟不住指令的那种晃才是
        trk = ee - cmd
        rx, px = self._resid(trk[:, 0])
        ry, py = self._resid(trk[:, 1])
        fn = np.linalg.norm(f, axis=1)
        # 5 帧滑动中值：量"持续压紧力"，把单帧接触冲击（0.7~10 帧的尖峰）剔掉
        k = min(5, len(fn))
        f_med = np.convolve(fn, np.ones(k) / k, mode="valid") if k > 1 else fn
        return {"jitter_rms": max(rx, ry), "jitter_peak": max(px, py),
                "force_max": float(fn.max()),
                "force_med_max": float(f_med.max()) if len(f_med) else float(fn.max()),
                "n": int(m.sum())}

    def __init__(self, n_envs=8, types=None, out_dir="data", randomize=True, amp=0.005,
                 controller_factory=None, max_steps=800, seed=0, enable_camera=False,
                 headless=True, lat_gain=0.0, descend_step=0.00075, fail_force=150.0,
                 lat_limit=0.003, grasp_hold=-99.0, policy="scripted", force_gain=0.0,
                 save_video=False, video_fps=10, video_every=6,
                 camera_name="left_gripper_camera",
                 solver_pos=0, solver_vel=0,
                 cart_linear_damping=1.5, cart_angular_damping=2.0,
                 cart_sleep_threshold=1e-3, carrier_friction=-1.0,
                 max_jitter_peak=2.0, max_jitter_rms=0.5, max_contact_force=250.0,
                 keep_explore=True, probe_prob=0.0, xy_model="", max_peak_force=500.0,
                 entry_amp=0.0, config_path="", amp_z=None, xy_noise=0.0,
                 par_workers=1, profile=False, gpu_dynamics=False):
                 
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.randomize = bool(randomize)
        self.amp = float(amp)
        self.xy_noise = float(xy_noise or 0.0)
        self.amp_z = None if amp_z in (None, "") else float(amp_z)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.lat_gain = float(lat_gain)
        self.descend_step = float(descend_step)
        self.fail_force = float(fail_force)
        self.lat_limit = float(lat_limit)
        self.grasp_hold = float(grasp_hold)
        self.policy = str(policy)
        self.force_gain = float(force_gain)
        self.save_video = bool(save_video)
        self.video_fps = int(video_fps)
        self.video_every = max(1, int(video_every))
        self.camera_name = str(camera_name)
        self.cart_linear_damping = float(cart_linear_damping)
        self.cart_angular_damping = float(cart_angular_damping)
        self._damping_applied = False
        self.cart_sleep_threshold = float(cart_sleep_threshold)
        self.carrier_friction = float(carrier_friction)
        self.max_jitter_peak = float(max_jitter_peak)
        self.max_jitter_rms = float(max_jitter_rms)
        self.max_contact_force = float(max_contact_force)
        self.keep_explore = bool(keep_explore)
        self.max_peak_force = float(max_peak_force)
        # 笛卡尔段起点的横向扰动幅度（m）。关节回放会抹掉 reset 的偏移，这里重新施加
        self.entry_amp = float(entry_amp)
        self.probe_prob = float(probe_prob)
        self.xy_model = str(xy_model or "")
        self.xy_model_scale = 1.0
        self.xy_model_mode = "add"
        self._carrier_friction_base = float(carrier_friction) if carrier_friction >= 0 else 0.0
        self._carrier_friction_managed = False
        self._video_tmp_dir = os.path.join(self.out_dir, "_videos_tmp")
        # 每个物理 step 之前"逐环境"的 observe/act/记录；>1 时用线程池并行
        self.par_workers = max(1, int(par_workers))
        self.profile = bool(profile)
        self.gpu_dynamics = bool(gpu_dynamics)
        self.prof = {}

        self.world = TianshuiWorld(config_path=(config_path or None),
                                   headless=headless,
                                   enable_camera=bool(enable_camera or self.save_video),
                                   gpu_dynamics=bool(gpu_dynamics))
        if solver_pos > 0 or solver_vel > 0:
            ok = self.world.set_solver_iterations(position=int(solver_pos),
                                                  velocity=int(solver_vel))
            print(f"[gen] solver 迭代 position={solver_pos} velocity={solver_vel} ok={ok}",
                  flush=True)
        types = types or ["lab1", "lab2", "lab3"]
        for i in range(int(n_envs)):
            self.world.add_env(types[i % len(types)])
        # 阻尼必须在首次 reset（构建 physics view）之前写入，否则会重建 sim view、
        # 把 articulation 的 _physics_view 句柄弄失效（实测整轮崩在 step 里）
        if self.cart_linear_damping > 0 or self.cart_angular_damping > 0:
            n = sum(1 for e in self.world.envs.values()
                    if e.set_retaining_clip_damping(self.cart_linear_damping,
                                               self.cart_angular_damping,
                                               self.cart_sleep_threshold))
            print(f"[gen] 弹夹阻尼 linear={self.cart_linear_damping} "
                  f"angular={self.cart_angular_damping} 生效 {n}/{len(self.world.envs)}",
                  flush=True)
            self._damping_applied = True
        if self.carrier_friction >= 0:
            n = sum(1 for e in self.world.envs.values()
                    if e.set_carrier_friction(self.carrier_friction))
            print(f"[gen] 中转车摩擦 mu={self.carrier_friction} 生效 {n}/{len(self.world.envs)}",
                  flush=True)
            self._carrier_friction_managed = True
        self.controller_factory = controller_factory
        self.controllers = {}

    def _ensure_controllers(self):
        """控制器要在 env 绑定（world.reset）之后建，否则拿不到标准关节姿态。"""
        if self.controllers:
            return
        self.controllers = {eid: self._make_controller(self.controller_factory, e)
                            for eid, e in self.world.envs.items()}

    def _make_controller(self, factory, env):
        if factory is None:
            pol = env.spec.get("policy", self.policy)
            if pol == "replay":
                path = load_descend_path(_resolve_spec_path(env.spec["traj"]),
                                         env.dof_names,
                                         env.home_joint_positions())
                return ReplayInsertController(
                    path, env.dof_names, force_gain=self.force_gain,
                    home_frames=int(env.spec.get("home_frames", 25)),
                    path_speed=float(env.spec.get("path_speed", 1.0)),
                    release_mode=str(env.spec.get("release_mode", "sync")),
                    release_open=float(env.spec.get("release_open", 0.0)),
                    release_frames=int(env.spec.get("release_frames", 60)),
                    release_guide_limit=float(env.spec.get("release_guide_limit", 0.001)),
                    release_guide_gain=float(env.spec.get("release_guide_gain", 0.0)),
                    release_guide_alpha=float(env.spec.get("release_guide_alpha", 0.25)),
                    release_lift=float(env.spec.get("release_lift", 0.008)),
                    release_exit_lift=float(env.spec.get("release_exit_lift", 0.0)),
                    release_exit_frames=int(env.spec.get("release_exit_frames", 60)),
                    release_retract=float(env.spec.get("release_retract", 0.0)),
                    release_lift_delay=float(env.spec.get("release_lift_delay", 0.0)),
                    jam_force=float(env.spec.get("jam_force", 120.0)),
                    shift_limit=float(env.spec.get("shift_limit", 0.0)),
                    shift_every=int(env.spec.get("shift_every", 10)),
                    track_tol=float(env.spec.get("track_tol", 0.03)),
                    release_max_z_err=float(env.spec.get("release_max_z_err", 0.005)),
                    search_after=int(env.spec.get("search_after", 40)),
                    search_radius=float(env.spec.get("search_radius", 0.0025)),
                    search_mode=str(env.spec.get("search_mode", "random")),
                    spiral_dtheta=float(env.spec.get("spiral_dtheta", 0.12)),
                    spiral_grow=float(env.spec.get("spiral_grow", 1.5e-4)),
                    grad_gain=float(env.spec.get("grad_gain", 2.0e-5)),
                    grad_step_max=float(env.spec.get("grad_step_max", 4.0e-4)),
                    grad_force_min=float(env.spec.get("grad_force_min", 4.0)),
                    xy_noise=float(env.spec.get("xy_noise", self.xy_noise)),
                    grad_sign=float(env.spec.get("grad_sign", -1.0)),
                    release_force_gain=float(env.spec.get("release_force_gain", 0.0)),
                    release_force_limit=float(env.spec.get("release_force_limit", 0.004)),
                    release_force_min=float(env.spec.get("release_force_min", 3.0)),
                    release_force_sign=float(env.spec.get("release_force_sign", 1.0)),
                    release_force_dir=env.spec.get("release_force_dir", (0.0, 0.0)),
                    release_force_thr=float(env.spec.get("release_force_thr", 0.0)),
                    release_gate_force=float(env.spec.get("release_gate_force", 0.0)),
                    release_gate_extra=env.spec.get("release_gate_extra", (0.0, 0.0)),
                    release_pre_frames=int(env.spec.get("release_pre_frames", 0)),
                    release_pre_frac=float(env.spec.get("release_pre_frac", 0.15)),
                    release_pre_lift=float(env.spec.get("release_pre_lift", 0.0)),
                    release_pre_lift_frames=int(env.spec.get("release_pre_lift_frames", 30)),
                    probe_prob=float(env.spec.get("probe_prob", self.probe_prob)),
                    probe_step=tuple(env.spec.get("probe_step", (8e-5, 3e-4))),
                    probe_hold=tuple(env.spec.get("probe_hold", (12, 30))),
                    probe_grow=float(env.spec.get("probe_grow", 1.4)),
                    probe_step_max=float(env.spec.get("probe_step_max", 0.0015)),
                    explore_limit=float(env.spec.get("explore_limit", 0.008)),
                    probe_bias=env.spec.get("probe_bias", (0.0, 0.0)),
                    probe_bias_prob=float(env.spec.get("probe_bias_prob", 0.5)),
                    probe_unload_force=float(env.spec.get("probe_unload_force", 0.0)),
                    probe_unload_lift=float(env.spec.get("probe_unload_lift", 0.0015)),
                    unload_lift=float(env.spec.get("unload_lift", 0.0)),
                    entry_xy_bias=env.spec.get("entry_xy_bias", (0.0, 0.0)),
                    cart_xy_bias=env.spec.get("cart_xy_bias", (0.0, 0.0)),
                    entry_z=float(env.spec.get("entry_z", 0.025)),
                    entry_force=float(env.spec.get("entry_force", 90.0)),
                    entry_dither=float(env.spec.get("entry_dither", 0.0003)),
                    entry_slow=float(env.spec.get("entry_slow", 0.35)),
                    retry_lift=float(env.spec.get("retry_lift", 0.004)),
                    retry_max=int(env.spec.get("retry_max", 8)),
                    retry_slow=float(env.spec.get("retry_slow", 0.0)),
                    retry_slow_clear=float(env.spec.get("retry_slow_clear", 0.005)),
                    cart_step=float(env.spec.get("cart_step", 0.00075)),
                    entry_force_ctrl=bool(env.spec.get("entry_force_ctrl", True)),
                    entry_kf=float(env.spec.get("entry_kf", 4e-6)),
                    entry_creep=float(env.spec.get("entry_creep", 0.0)),
                    xy_model=(env.spec.get("xy_model") or self.xy_model),
                    xy_model_scale=float(env.spec.get("xy_model_scale",
                                                      self.xy_model_scale)),
                    xy_model_ema=float(env.spec.get("xy_model_ema",
                                                    getattr(self, "xy_model_ema", 1.0))),
                    xy_model_mode=str(env.spec.get("xy_model_mode",
                                                   self.xy_model_mode)),
                    cart_switch_force=float(env.spec.get("cart_switch_force", 110.0)),
                    cart_switch_min_frames=int(env.spec.get("cart_switch_min_frames", 40)),
                    z_goal_extra=float(env.spec.get("z_goal_extra", 0.010)),
                    cart_xy_free_boost=float(env.spec.get("cart_xy_free_boost", 1.0)),
                    cart_xy_free_force=float(env.spec.get("cart_xy_free_force", 20.0)),
                    dither_gate=float(env.spec.get("dither_gate", 0.0)),
                    dither_gate_scale=float(env.spec.get("dither_gate_scale", 0.2)),
                    entry_offset_frames=int(env.spec.get("entry_offset_frames", 0)),
                    entry_offset_lift=float(env.spec.get("entry_offset_lift", 0.003)),
                    seat_shift=env.spec.get("seat_shift", (0.0, 0.0)),
                    seat_shift_frames=int(env.spec.get("seat_shift_frames", 20)),
                    readjust=bool(env.spec.get("readjust", False)),
                    readjust_max=int(env.spec.get("readjust_max", 1)),
                    readjust_min_err=float(env.spec.get("readjust_min_err", 5e-4)),
                    seat_z_tol=float(env.spec.get("seat_z_tol", 0.0012)),
                    seat_xy_tol=float(env.spec.get("seat_xy_tol", 0.0)),
                    mouth_xy_tol=float(env.spec.get("mouth_xy_tol", 0.0)),
                    mouth_z=float(env.spec.get("mouth_z", 0.02)),
                    lat_hold_tol=float(env.spec.get("lat_hold_tol", 0.0)),
                    lat_hold_z=float(env.spec.get("lat_hold_z", 0.03)),
                    lat_hold_force=float(env.spec.get("lat_hold_force", 25.0)),
                    lat_hold_step=float(env.spec.get("lat_hold_step", 0.0002)),
                    seat_hold=int(env.spec.get("seat_hold", 8)),
                    rim_regrip=int(env.spec.get("rim_regrip", 0)),
                    rim_regrip_frames=int(env.spec.get("rim_regrip_frames", 90)))
            return ScriptedInsertController(descend_step=self.descend_step,
                                            lat_gain=self.lat_gain,
                                            fail_force=self.fail_force,
                                            lat_limit=self.lat_limit,
                                            grasp_hold=self.grasp_hold)
        if isinstance(factory, type):
            try:
                return factory(descend_step=self.descend_step, lat_gain=self.lat_gain)
            except TypeError:
                return factory()
        return factory

    # ------------------------------------------------------------------
    def _advance_env(self, eid, env, rec_row):
        """单环境一个仿真步的"观测 → 策略 → 记录"。

        par_workers>1 时由线程池并发调用（各环境互不共享数据，只有只读的
        USD/PhysX 查询）。返回 (action, finished)。
        """
        t0 = time.perf_counter() if self.profile else 0.0
        obs = env.observe()
        if self.profile:
            t1 = time.perf_counter()
            self.prof["observe"] = self.prof.get("observe", 0.0) + (t1 - t0)
        act, fin = self.controllers[eid].act(env, obs)
        if self.profile:
            t2 = time.perf_counter()
            self.prof["act"] = self.prof.get("act", 0.0) + (t2 - t1)
            _t_book = t2
        # "发送的末端位姿命令"：EE 动作直接用；关节动作经 FK 换算成末端位置
        if act and act.get("ee_pos") is not None:
            ee_cmd = np.asarray(act["ee_pos"], dtype=float)
        elif act and act.get("joint_positions") is not None:
            ee_cmd = env.fk_ee_base(act["joint_positions"])
        else:
            ee_cmd = np.asarray(obs["ee_pos"], dtype=float)
        if act and act.get("opening") is not None:
            opening_cmd = float(act["opening"])
        elif act and act.get("grip_targets"):
            g = act["grip_targets"]
            opening_cmd = float(g.get("left_gripper_zjb_Joint", obs["opening"]))
        elif act and act.get("joint_positions") is not None and \
                "left_gripper_zjb_Joint" in env.dof_names:
            opening_cmd = float(act["joint_positions"][
                env.dof_names.index("left_gripper_zjb_Joint")])
        else:
            opening_cmd = float(obs["opening"])
        rec_row["joint_pos"].append(obs["joint_positions"])
        rec_row["joint_vel"].append(obs["joint_velocities"])
        rec_row["ee_pos"].append(obs["ee_pos"])
        rec_row["ee_quat"].append(obs["ee_quat"])
        rec_row["ee_cmd"].append(ee_cmd)
        rec_row["expert_dcmd"].append(np.asarray(
            getattr(self.controllers[eid], "_expert_dxy", np.zeros(2)), dtype=float))
        rec_row["opening"].append(obs["opening"])
        rec_row["opening_cmd"].append(opening_cmd)
        rec_row["obj_pos"].append(obs["retaining_clip_pos"])
        rec_row["obj_delta"].append(obs["retaining_clip_delta"])
        rec_row["force"].append(obs["force"])
        rec_row["torque"].append(obs["torque"])
        rec_row["phase"].append(str(getattr(self.controllers[eid], "phase", "")))
        if self.profile:
            self.prof["record"] = self.prof.get("record", 0.0) + (time.perf_counter() - _t_book)
        return act, fin

    # ------------------------------------------------------------------
    def collect_round(self, round_idx=0):
        """跑一轮（所有 env 同时）：返回 {env_id: episode_dict}。"""
        deltas = self.world.reset(randomize=self.randomize, amp=self.amp, amp_z=self.amp_z,
                                  seed=int(self.rng.integers(1 << 30)))
        # 中转车摩擦是 env 级材质属性，上一轮松手时改成过 0.6，这里复位回插入阶段的值
        if self._carrier_friction_managed:
            for e in self.world.envs.values():
                e.set_carrier_friction(self._carrier_friction_base)
        self._ensure_controllers()
        if self.save_video:
            os.makedirs(self._video_tmp_dir, exist_ok=True)
            # 相机/Replicator 需要几帧渲染才有数据
            for _ in range(6):
                self.world.step({}, render=True)
        rec = {eid: {"joint_pos": [], "joint_vel": [], "ee_pos": [], "ee_quat": [],
                     "ee_cmd": [], "opening": [], "opening_cmd": [],
                     "expert_dcmd": [],
                     "obj_pos": [], "obj_delta": [], "force": [], "torque": [],
                     "phase": []}
               for eid in self.world.envs}
        offsets = {}
        for eid, c in self.controllers.items():
            c.search_seed = int(self.rng.integers(1 << 30))
            if self.entry_amp > 0.0 and getattr(c, "entry_offset_frames", 0) > 0:
                c.entry_offset = self.rng.uniform(-self.entry_amp, self.entry_amp, size=2)
            else:
                c.entry_offset = np.zeros(2)
            offsets[eid] = np.asarray(c.entry_offset, dtype=float).copy()
            c.reset(None)
        pool = (ThreadPoolExecutor(max_workers=min(self.par_workers, len(self.world.envs)))
                if self.par_workers > 1 else None)
        done = {eid: False for eid in self.world.envs}
        writers = {}
        vid_paths = {}
        vid_closed = {}
        prof = self.prof
        for step in range(self.max_steps):
            actions = {}
            finished_now = set()
            active = [(eid, env) for eid, env in self.world.envs.items() if not done[eid]]
            if pool is not None and len(active) > 1:
                futs = [pool.submit(self._advance_env, eid, env, rec[eid]) for eid, env in active]
                results = [f.result() for f in futs]
                active = [active[i] for i in range(len(active))]
            else:
                results = [self._advance_env(eid, env, rec[eid]) for eid, env in active]
            for (eid, _env), (act, fin) in zip(active, results):
                if act is not None:
                    actions[eid] = act
                if fin:
                    done[eid] = True
                    finished_now.add(eid)
            do_render = self.save_video and (step % self.video_every == 0)
            _t0 = time.perf_counter()
            self.world.step(actions, render=do_render, profile=bool(prof))
            if prof:
                prof["world_step"] = prof.get("world_step", 0.0) + (time.perf_counter() - _t0)
                prof["steps"] = prof.get("steps", 0) + 1
                prof["env_steps"] = prof.get("env_steps", 0) + len(active)
            if do_render:
                for eid, env in self.world.envs.items():
                    if vid_closed.get(eid):
                        continue
                    img = env.camera_rgb(self.camera_name)
                    if img is None:
                        if eid in finished_now:
                            vid_closed[eid] = True
                        continue
                    frame = np.ascontiguousarray(np.asarray(img)[:, :, :3][:, :, ::-1])
                    if eid not in writers:
                        h, w = frame.shape[:2]
                        vp = os.path.join(self._video_tmp_dir,
                                          f"r{round_idx:03d}_env{eid}.mp4")
                        wr = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"avc1"),
                                             self.video_fps, (w, h))
                        if not wr.isOpened():
                            print(f"[video][warn] avc1 writer 打开失败 {vp}", flush=True)
                            vid_closed[eid] = True
                            continue
                        writers[eid] = wr
                        vid_paths[eid] = vp
                    writers[eid].write(frame)
                    if eid in finished_now:
                        writers[eid].release()
                        vid_closed[eid] = True
            if all(done.values()):
                break
        self._last_records = rec
        if pool is not None:
            pool.shutdown(wait=True)
        self.last_prof = dict(self.prof)
        rp = getattr(self.world, "last_reset_prof", None)
        if rp:
            print("[prof]   reset: " + " ".join(f"{k}={v:.2f}s" for k, v in rp.items()),
                  flush=True)
        if self.profile and self.prof.get("steps"):
            n = float(self.prof["steps"])
            es = float(self.prof.get("env_steps", 0))
            tot = sum(self.prof.get(k, 0.0) for k in ("observe", "act", "record", "world_step"))
            print(f"[prof] workers={self.par_workers} steps={int(n)} envs={es / max(n, 1):.1f}"
                  f" per-step={tot / n * 1e3:.1f}ms"
                  f" | observe={self.prof.get('observe', 0) / n * 1e3:.1f}"
                  f" act={self.prof.get('act', 0) / n * 1e3:.1f}"
                  f" record={self.prof.get('record', 0) / n * 1e3:.1f}"
                  f" world_step={self.prof.get('world_step', 0) / n * 1e3:.1f}"
                  f" | throughput={es / max(tot, 1e-9):.1f} env-frames/s", flush=True)
            wp = getattr(self.world, "last_step_prof", None)
            if wp and wp.get("n"):
                k = float(wp["n"])
                print(f"[prof]   world.step 内部: apply={wp.get('apply', 0) / k * 1e3:.1f}ms"
                      f" physics={wp.get('physics', 0) / k * 1e3:.1f}ms"
                      f" kinematics={wp.get('kinematics', 0) / k * 1e3:.1f}ms", flush=True)
                self.world.last_step_prof = {}
            self.prof = {}
        for eid, wr in writers.items():
            if not vid_closed.get(eid):
                wr.release()
                vid_closed[eid] = True
        eps = {}
        for eid, env in self.world.envs.items():
            st = env.status()
            ctrl = self.controllers[eid]
            eps[eid] = {
                "type": env.type_key,
                "status": st,
                "failed": bool(getattr(ctrl, "failed", False)),
                "reason": getattr(ctrl, "reason", ""),
                "offset": np.asarray(deltas.get(eid, np.zeros(3)), dtype=float),
                "entry_offset": np.asarray(offsets.get(eid, np.zeros(2)), dtype=float),
                "obj_target": env.retaining_clip_target(),
                "final_delta": env.retaining_clip_delta_to_target(),
                # insert_ok = 纯插入判定（弹夹到位、未判失败）；success = insert_ok 且过力门质检
                "insert_ok": bool(st == "success_insert"
                                  and not getattr(ctrl, "failed", False)),
                "success": bool(st == "success_insert" and not getattr(ctrl, "failed", False)),
                "handover": getattr(ctrl, "handover", None),
                "video": vid_paths.get(eid),
                "quality": self._quality(rec[eid]),
                "explore": bool(getattr(ctrl, "_search_used", False)),
                "data": {k: (np.asarray(v) if len(v) else np.zeros((0,)))
                         for k, v in rec[eid].items()},
            }
            q = eps[eid]["quality"]
            # 抖动只记录不卡（慢速爬行时量到的是伺服跟随滞后，不是抖动）；
            # 硬门 = 合力上限 + 松手高度(release_max_z_err) + 末态 success_insert
            # 力门用"持续力"（5 帧中值的峰值），单帧冲击单列一个更宽的上限：
            # 实测接触冲击 300~500N 只持续 2~10 帧，用它判失败会把 9 成样本误杀
            qc_ok = (q["force_med_max"] <= self.max_contact_force
                     and q["force_max"] <= self.max_peak_force)
            eps[eid]["qc_ok"] = bool(qc_ok)
            if not qc_ok:
                eps[eid]["success"] = False
                if not eps[eid].get("reason"):
                    eps[eid]["reason"] = (
                        f"qc jitter_peak={q['jitter_peak']:.2f}mm "
                        f"rms={q['jitter_rms']:.2f}mm "
                        f"force={q['force_med_max']:.0f}N peak={q['force_max']:.0f}N")
        return eps

    def run(self, n_episodes, max_rounds=200, only_success=True, keep_failed=True):
        index_path = os.path.join(self.out_dir, "index.jsonl")
        fail_dir = os.path.join(self.out_dir, "failed")
        explore_dir = os.path.join(self.out_dir, "explore")
        explore_fail_dir = os.path.join(self.out_dir, "explore_failed")
        if keep_failed:
            os.makedirs(fail_dir, exist_ok=True)
        if self.keep_explore:
            os.makedirs(explore_dir, exist_ok=True)
            os.makedirs(explore_fail_dir, exist_ok=True)
        index_fail = os.path.join(fail_dir, "index.jsonl") if keep_failed else None
        saved = 0
        n_fail = 0
        rnd = 0
        t0 = time.time()
        with open(index_path, "a", encoding="utf-8") as idx, \
                (open(index_fail, "a", encoding="utf-8") if index_fail else _NullCtx()) as idx_f:
            while saved < n_episodes and rnd < max_rounds:
                eps = self.collect_round(rnd + 1)
                rnd += 1
                for eid, ep in eps.items():
                    if saved >= n_episodes:
                        break
                    if only_success and not ep["success"]:
                        n_fail += 1
                        if keep_failed:
                            fname = f"fail_{n_fail:05d}_env{eid}_{ep['type']}.npz"
                            np.savez_compressed(os.path.join(fail_dir, fname),
                                                **ep["data"], offset=ep["offset"],
                                                entry_offset=ep["entry_offset"],
                                                obj_target=ep["obj_target"])
                            vname = None
                            tmp = ep.get("video")
                            if self.save_video and tmp and os.path.isfile(tmp):
                                vname = fname[:-4] + ".mp4"
                                os.replace(tmp, os.path.join(fail_dir, vname))
                            idx_f.write(json.dumps({"file": fname, "type": ep["type"],
                                                    "status": ep["status"],
                                                    "reason": ep["reason"],
                                                    "video": vname,
                                                    "quality": ep.get("quality", {}),
                                                    "final_delta": ep["final_delta"].tolist(),
                                                    "offset": ep["offset"].tolist(),
                                                    "entry_offset": ep["entry_offset"].tolist()},
                                                   ensure_ascii=False) + "\n")
                            idx_f.flush()
                            if self.keep_explore and ep.get("explore"):
                                dst = os.path.join(explore_fail_dir, fname)
                                shutil.copy2(os.path.join(fail_dir, fname), dst)
                                if vname:
                                    shutil.copy2(os.path.join(fail_dir, vname),
                                                 os.path.join(explore_fail_dir, vname))
                                with open(os.path.join(explore_fail_dir, "index.jsonl"), "a",
                                          encoding="utf-8") as ex:
                                    ex.write(json.dumps({"file": fname, "type": ep["type"],
                                                         "status": ep["status"],
                                                         "reason": ep["reason"], "video": vname,
                                                         "quality": ep.get("quality", {}),
                                                         "final_delta": ep["final_delta"].tolist()},
                                                        ensure_ascii=False) + "\n")
                        else:
                            tmp = ep.get("video")
                            if tmp and os.path.isfile(tmp):
                                os.remove(tmp)
                        continue
                    name = f"ep_{saved:05d}_env{eid}_{ep['type']}.npz"
                    path = os.path.join(self.out_dir, name)
                    np.savez_compressed(path, **ep["data"], offset=ep["offset"],
                                        entry_offset=ep["entry_offset"],
                                        obj_target=ep["obj_target"])
                    vname = None
                    tmp = ep.get("video")
                    if self.save_video and tmp and os.path.isfile(tmp):
                        vname = name[:-4] + ".mp4"
                        os.replace(tmp, os.path.join(self.out_dir, vname))
                    idx.write(json.dumps({"file": name, "type": ep["type"], "status": ep["status"],
                                          "success": ep["success"], "failed": ep["failed"],
                                          "reason": ep["reason"],
                                          "video": vname,
                                          "explore": bool(ep.get("explore", False)),
                                          "quality": ep.get("quality", {}),
                                          "offset": ep["offset"].tolist(),
                                          "entry_offset": ep["entry_offset"].tolist(),
                                          "final_delta_mm": (ep["final_delta"] * 1000).round(3).tolist(),
                                          "handover": ep.get("handover"),
                                          "steps": int(ep["data"]["joint_pos"].shape[0])},
                                         ensure_ascii=False) + "\n")
                    idx.flush()
                    if self.keep_explore and ep.get("explore"):
                        dst = os.path.join(explore_dir, name)
                        shutil.copy2(path, dst)
                        if vname:
                            shutil.copy2(os.path.join(self.out_dir, vname),
                                         os.path.join(explore_dir, vname))
                        with open(os.path.join(explore_dir, "index.jsonl"), "a",
                                  encoding="utf-8") as ex:
                            ex.write(json.dumps({"file": name, "type": ep["type"],
                                                 "status": ep["status"], "success": True,
                                                 "reason": ep["reason"], "video": vname,
                                                 "quality": ep.get("quality", {}),
                                                 "final_delta": ep["final_delta"].tolist(),
                                                 "offset": ep["offset"].tolist(),
                                                 "entry_offset": ep["entry_offset"].tolist()},
                                                ensure_ascii=False) + "\n")
                    saved += 1
                ok = sum(1 for e in eps.values() if e["success"])
                print(f"[gen] 第 {rnd} 轮: {ok}/{len(eps)} 成功, 成功累计 {saved} 条, "
                      f"失败 {n_fail} 条, {time.time() - t0:.1f}s", flush=True)
        print(f"[gen] DONE {saved} 条 -> {self.out_dir}", flush=True)
        return saved


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="",
                    help="覆盖 configs/envs.json 的配置文件路径（A/B 参数用）")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="data")
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--amp", type=float, default=0.005,
                    help="抓取点 xy 随机偏置幅度（m）")
    ap.add_argument("--xy-noise", type=float, default=0.0,
                    help="专家采集时的低频横向试探幅度（m），0=关")
    ap.add_argument("--amp-z", type=float, default=None,
                    help="抓取点 z 随机偏置幅度（m），缺省=同 --amp")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--lat-gain", type=float, default=0.0)
    ap.add_argument("--descend-step", type=float, default=0.00075)
    ap.add_argument("--fail-force", type=float, default=150.0)
    ap.add_argument("--lat-limit", type=float, default=0.003)
    ap.add_argument("--grasp-hold", type=float, default=-99.0,
                    help="下降段夹爪闭合指令；-99=沿当前，0.025=夹到最紧")
    ap.add_argument("--policy", default="replay", choices=["replay", "scripted"],
                    help="replay=关节空间倒放抬升轨迹（稳）；scripted=力/xy 闭环")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--types", default="lab1,lab2,lab3",
                    help="逗号分隔的环境类型，如 lab2,lab3")
    ap.add_argument("--force-gain", type=float, default=0.0,
                    help="笛卡尔闭环的力柔顺增益 C：xy -= C*(F-F0)")
    ap.add_argument("--max-rounds", type=int, default=200,
                    help="最多跑多少轮；每轮并行采 envs 条")
    ap.add_argument("--only-success", type=int, default=1,
                    help="1=只把成功样本写进 out（失败进 failed/）")
    ap.add_argument("--keep-failed", type=int, default=1,
                    help="1=失败样本另存 failed/，便于分析失败原因")
    ap.add_argument("--save-video", type=int, default=0,
                    help="1=保存 left_gripper_camera 的 avc1 mp4（会强制开启相机与渲染）")
    ap.add_argument("--video-fps", type=int, default=10)
    ap.add_argument("--camera-name", default="left_gripper_camera")
    ap.add_argument("--solver-pos", type=int, default=0,
                    help="physicsScene 最小位置迭代次数（0=用 USD 默认）")
    ap.add_argument("--solver-vel", type=int, default=0)
    ap.add_argument("--cart-damping", type=float, nargs=2, default=[1.5, 2.0],
                    metavar=("LINEAR", "ANGULAR"),
                    help="弹夹刚体线性/角阻尼，传 0 0 关闭")
    ap.add_argument("--cart-sleep-threshold", type=float, default=1e-3,
                    help="弹夹刚体 sleep/stabilization 阈值，越大越早静止（0=用默认）")
    ap.add_argument("--carrier-friction", type=float, default=0.15,
                    help="中转车摩擦系数（默认 0.15；<0 = 保持 USD 里的 0.0）")
    ap.add_argument("--max-jitter-peak", type=float, default=2.0,
                    help="下降/接触段末端残差峰值上限(mm)，超过判失败")
    ap.add_argument("--max-jitter-rms", type=float, default=0.5,
                    help="下降/接触段末端残差 RMS 上限(mm)")
    ap.add_argument("--xy-model", default="",
                    help="学到的 xy 修正策略 .pt（空=用脚本修正）")
    ap.add_argument("--xy-model-scale", type=float, default=1.0,
                    help="模型输出增益：<1 更保守，=0 等价于不用模型")
    ap.add_argument("--xy-model-mode", default="add", choices=["add", "replace"],
                    help="add=脚本修正+模型增量（默认）；replace=模型增量替代脚本修正")
    ap.add_argument("--entry-amp", type=float, default=0.0,
                    help="笛卡尔段起点的横向扰动幅度(m)，0=关闭")
    ap.add_argument("--probe-prob", type=float, default=0.0,
                    help="每回合做随机试探的比例(0~1)，试探样本会另存 explore/")
    ap.add_argument("--max-contact-force", type=float, default=250.0,
                    help="下降/接触段持续合力上限(N)，用 5 帧中值")
    ap.add_argument("--max-peak-force", type=float, default=500.0,
                    help="下降/接触段单帧合力硬上限(N)，只拦真正的撞击")
    ap.add_argument("--par-workers", type=int, default=1,
                    help="逐环境 observe/act 的线程数（1=串行）")
    ap.add_argument("--gpu-dynamics", type=int, default=0,
                    help="1=打开 PhysX GPU 动力学（换后端，接触数值会变）")
    ap.add_argument("--profile", type=int, default=0,
                    help="1=累计 observe/act/record/world_step 耗时并打印")
    ap.add_argument("--ik-batch", type=int, default=1,
                    help="1=批量 IK（默认，把多种子全解合成一次迭代）；0=逐 env 旧路径")
    ap.add_argument("--video-every", type=int, default=6,
                    help="每隔多少个仿真步渲染/写一帧")
    a = ap.parse_args()
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    gen = DataGenerator(n_envs=a.envs, types=types, out_dir=a.out, randomize=not a.no_randomize,
                        amp=a.amp, max_steps=a.max_steps, seed=a.seed,
                        headless=bool(a.headless), lat_gain=a.lat_gain,
                        descend_step=a.descend_step, fail_force=a.fail_force,
                        lat_limit=a.lat_limit, grasp_hold=a.grasp_hold, policy=a.policy,
                        force_gain=a.force_gain, save_video=bool(a.save_video),
                        video_fps=a.video_fps, video_every=a.video_every,
                        camera_name=a.camera_name,
                        solver_pos=a.solver_pos, solver_vel=a.solver_vel,
                        cart_linear_damping=a.cart_damping[0],
                        cart_angular_damping=a.cart_damping[1],
                        cart_sleep_threshold=a.cart_sleep_threshold,
                        carrier_friction=a.carrier_friction,
                        max_jitter_peak=a.max_jitter_peak,
                        max_jitter_rms=a.max_jitter_rms,
                        max_contact_force=a.max_contact_force,
                        probe_prob=a.probe_prob, xy_model=a.xy_model,
                        max_peak_force=a.max_peak_force, entry_amp=a.entry_amp,
                        config_path=a.config, amp_z=a.amp_z,
                        xy_noise=a.xy_noise, par_workers=a.par_workers,
                        profile=bool(a.profile), gpu_dynamics=bool(a.gpu_dynamics))
    gen.xy_model_scale = float(a.xy_model_scale)
    gen.xy_model_mode = str(a.xy_model_mode)
    gen.world.batch_ik_enabled = bool(a.ik_batch)
    gen.run(a.episodes, max_rounds=a.max_rounds,
            only_success=bool(a.only_success), keep_failed=bool(a.keep_failed))


if __name__ == "__main__":
    main()
