"""控制器（策略）接口 + 本任务的传统控制实现。

后续换成深度学习模型时，只要实现同样的 act(env, obs) -> (action, done) 即可。
动作统一是 {"ee_pos": [x,y,z] (base 系), "opening": v}，末端姿态固定 (0,0,-90) 度。
"""

import os

import numpy as np
from collections import deque


class BaseController:
    name = "base"

    def reset(self, env):
        pass

    def _load_xy_model(self, path):
        """加载监督/离线RL 学到的 xy 修正策略（state_dict + 归一化）。"""
        try:
            import torch
        except Exception:
            print("[xy-model] torch 不可用，回退脚本修正", flush=True)
            return
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(path, map_location="cpu")
        sd = ck["state_dict"]
        if any(k.startswith("net.") for k in sd):
            sd = {k[4:]: v for k, v in sd.items()}
        # 按 state_dict 的形状还原网络（训练侧隐藏层宽度可能不同）
        ws = [sd[k] for k in sorted(sd) if k.endswith("weight")]
        layers = []
        for i, w in enumerate(ws):
            layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
            if i < len(ws) - 1:
                layers.append(torch.nn.SiLU())
        net = torch.nn.Sequential(*layers)
        net.load_state_dict(sd)
        net.eval()
        self.xy_model = {"net": net, "s_mu": np.asarray(ck["s_mu"], dtype=float),
                         "s_sd": np.asarray(ck["s_sd"], dtype=float),
                         "a_mu": np.asarray(ck["a_mu"], dtype=float),
                         "a_sd": np.asarray(ck["a_sd"], dtype=float),
                         "feats": str(ck.get("feats", "full")),
                         "a_bias": (np.asarray(ck["a_bias"], dtype=float)
                                    if ck.get("a_bias") is not None else None),
                         "window": int(ck.get("window", 4) or 4)}
        print(f"[xy-model] 已加载 {path} (features={self.xy_model['feats']})", flush=True)

    def _xy_model_action(self, obs, err):
        """用模型给出每帧 xy 指令增量（mm）。"""
        if not self.xy_model:
            return None
        import torch
        ee = np.asarray(obs["ee_pos"], dtype=float)
        obj = np.asarray(obs["retaining_clip_pos"], dtype=float)
        F = np.asarray(obs.get("force", np.zeros(3)), dtype=float)
        T = np.asarray(obs.get("torque", np.zeros(3)), dtype=float)
        e = err * 1000.0
        rel = (obj - ee) * 1000.0
        de = e[:2] - (self._e_prev if self._e_prev is not None else e[:2])
        self._e_prev = e[:2].copy()
        x = np.concatenate([F, T, e, rel, de]).astype(np.float32)[None, :]
        feats = self.xy_model.get("feats", "full")
        if feats in ("ft", "ftw"):
            ft = np.concatenate([F, T]).astype(np.float32)
            if feats == "ftw":
                # 六维力时间窗：仍然只有腕部 F/T，用最近 window 帧（不足用首帧补齐）
                k = int(self.xy_model.get("window", 4) or 4)
                if getattr(self, "_ft_hist", None) is None:
                    self._ft_hist = []
                self._ft_hist.append(ft)
                if len(self._ft_hist) > k:
                    self._ft_hist = self._ft_hist[-k:]
                hist = [self._ft_hist[0]] * (k - len(self._ft_hist)) + self._ft_hist
                x = np.concatenate(hist).astype(np.float32)[None, :]
            else:
                # 纯力觉模型：只有六维力，不需要弹夹真值位姿
                x = ft[None, :]
        xn = (x - self.xy_model["s_mu"]) / self.xy_model["s_sd"]
        with torch.no_grad():
            a = self.xy_model["net"](torch.tensor(xn, dtype=torch.float32)).numpy()[0]
        a = np.ravel(a * self.xy_model["a_sd"] + self.xy_model["a_mu"]).astype(float)
        bias = self.xy_model.get("a_bias")
        if bias is not None:
            a = a - np.asarray(bias, dtype=float).ravel()
        ema = float(getattr(self, "xy_model_ema", 1.0))
        if 0.0 < ema < 1.0:
            prev = getattr(self, "_xy_ema", None)
            a = a if prev is None else ema * a + (1.0 - ema) * prev
            self._xy_ema = np.asarray(a, dtype=float).copy()
        return a * self.xy_model_scale

    def act(self, env, obs):
        raise NotImplementedError


class ReplayInsertController(BaseController):
    """上帝视角插入策略：关节空间回位 → 把"抬升段"轨迹倒放（=下降插入）→ 松爪。

    抬升轨迹（z24_labN_traj_final.csv 的 up/up_hold）已经在实机上验证过，倒放它
    等价于"从最高点沿原路放回槽位"，完全避开 IK 工作空间死角/肘关节限位问题。
    随机偏移由第一段关节空间回位消掉，所以倒放路径仍然对得上。
    """

    name = "replay_insert"

    def __init__(self, joint_path, dof_names, home_frames=25, release_open=0.0,
                 release_frames=60, hold_frames=10, settle_frames=120,
                 unload_frames=40, unload_hold=10,
                 jam_force=120.0, jam_hold=6, trim_frames=90, trim_limit=0.004,
                 trim_tol=0.0006, track_tol=0.03, unload_lift=0.0,
                 shift_limit=0.0, shift_every=10, grip_ramp_frames=20,
                 cart_switch_frames=60, cart_switch_force=110.0,
                 cart_switch_min_frames=40,
                 cart_step=0.00075, cart_xy_step=0.00012,
                 cart_timeout=600, jam_lift=0.0,
                 release_max_z_err=0.005,
                 release_lift=0.008,
                 search_after=40, search_radius=0.0025, search_period=90,
                 spiral_dtheta=0.12, spiral_grow=1.5e-4,
                 grad_gain=2.0e-5, grad_step_max=4.0e-4, grad_force_min=4.0,
                 xy_noise=0.0, xy_noise_period=15, grad_sign=-1.0,
                 release_force_gain=0.0, release_force_limit=0.004,
                 release_force_min=3.0, release_force_sign=1.0,
                 release_force_dir=(0.0, 0.0), release_force_thr=0.0,
                 release_gate_force=0.0, release_gate_extra=(0.0, 0.0),
                 release_pre_frames=0, release_pre_frac=0.15, release_pre_lift=0.0,
                 release_pre_lift_frames=30,
                 release_exit_lift=0.0, release_exit_frames=60, release_retract=0.0,
                 search_stall_z=0.008, search_mode="random",
                 probe_step=(8e-5, 3e-4), probe_hold=(12, 30),
                 probe_prob=0.0,
                 probe_grow=1.4, probe_step_max=0.0015, explore_limit=0.008,
                 probe_bias=(0.0, 0.0), probe_bias_prob=0.5, probe_unload_force=0.0,
                 probe_unload_lift=0.0015,
                 entry_xy_bias=(0.0, 0.0), entry_xy_bias_fade=0.01,
                 search_radius_lo=0.004, search_radius_hi=0.008,
                 entry_z=0.025, entry_slow=0.35, entry_dither=0.0003,
                 entry_dither_period=24, entry_force=90.0, entry_probe_step=0.0002,
                 dither_gate=0.0, dither_gate_scale=0.2,
                 entry_probe_hold=(10, 22),
                 retry_lift=0.004, retry_frames=25, retry_max=8,
                 retry_slow=0.0, retry_slow_clear=0.005,
                 entry_force_ctrl=True, entry_kf=4e-6, entry_filt=0.3,
                 entry_creep=0.0,
                 xy_model=None, xy_model_scale=1.0, xy_model_ema=1.0,
                 xy_model_mode="add",
                 cart_xy_bias=(0.0, 0.0),
                 probe_near=0.02,
                 cart_xy_gain=0.5, cart_xy_deadband=0.0002, cart_err_alpha=0.25,
                 cart_xy_free_boost=1.0, cart_xy_free_force=20.0,
                 entry_offset_frames=0, entry_offset_lift=0.003,
                 seat_shift=(0.0, 0.0), seat_shift_frames=20,
                 grip_settle_frames=20, grip_settle_hold=20, force_gain=0.0,
                 hold_through_home=True, z_goal_extra=0.010,
                 path_speed=1.0, release_guide_limit=0.001, release_mode="sync",
                 release_guide_gain=0.0, release_guide_alpha=0.25,
                 release_lift_delay=0.0,
                 post_open_hold=20, retreat_frames=30, retreat_dist=0.0, retreat_hold=70,
                 readjust=False, readjust_max=1, readjust_min_err=5e-4,
                 seat_z_tol=1.2e-3, seat_xy_tol=0.0, seat_hold=8,
                 mouth_xy_tol=0.0, mouth_z=0.02,
                 lat_hold_tol=0.0, lat_hold_z=0.03, lat_hold_force=25.0,
                 lat_hold_step=0.0002,
                 rim_regrip=0, rim_regrip_frames=90):
        self.track_tol = float(track_tol)
        self.unload_lift = float(unload_lift)
        # joint_path: (N, len(dof_names)) 从最高点 -> 抓取点的关节序列
        self.joint_path = np.asarray(joint_path, dtype=float)
        self.dof_names = list(dof_names)
        self.home_frames = int(home_frames)
        self.release_open = float(release_open)
        self.release_frames = int(release_frames)
        self.hold_frames = int(hold_frames)
        self.settle_frames = int(settle_frames)
        self.unload_frames = max(1, int(unload_frames))
        self.unload_hold = max(0, int(unload_hold))
        self.jam_force = float(jam_force)
        self.jam_hold = int(jam_hold)
        self.trim_frames = int(trim_frames)
        self.trim_limit = float(trim_limit)
        self.trim_tol = float(trim_tol)
        self.shift_limit = float(shift_limit)
        self.shift_every = max(1, int(shift_every))
        self.grip_ramp_frames = max(1, int(grip_ramp_frames))
        self.cart_switch_frames = max(1, int(cart_switch_frames))
        self.cart_switch_force = float(cart_switch_force)
        self.cart_switch_min_frames = max(0, int(cart_switch_min_frames))
        self.cart_timeout = int(cart_timeout)
        self.jam_lift = float(jam_lift)
        self.release_max_z_err = float(release_max_z_err)
        self.release_lift = float(release_lift)
        # 抬离延迟：0 = 开爪全程同步抬（旧行为，指头会沿弹夹侧面往上蹭，
        # 实测把弹夹带高 6~10mm）；0.6 = 前 60% 只在原位开爪，后 40% 才抬。
        self.release_lift_delay = min(max(float(release_lift_delay), 0.0), 0.95)
        self.search_after = int(search_after)
        self.search_radius = float(search_radius)
        self.search_radius_lo = float(search_radius_lo)
        self.search_radius_hi = float(search_radius_hi)
        self.entry_z = float(entry_z)
        self.entry_slow = float(entry_slow)
        self.entry_dither = float(entry_dither)
        self.entry_dither_period = max(5, int(entry_dither_period))
        # 已经对准（|err_xy| < dither_gate）时把抖动缩小：抖动幅度和槽口容差同量级，
        # 对准后还在抖等于自己把自己推出容差窗口（实测进孔时刻 |err| 恰在 0.7~1mm）
        self.dither_gate = float(dither_gate)
        self.dither_gate_scale = float(dither_gate_scale)
        self.entry_force = float(entry_force)
        self.entry_probe_step = float(entry_probe_step)
        self.entry_probe_hold = tuple(entry_probe_hold)
        self.retry_lift = float(retry_lift)
        self.retry_frames = max(3, int(retry_frames))
        self.retry_max = int(retry_max)
        # 抬起后慢速重新下压：实测"抬 7mm 再原速砸下去"会重复卡在同一深度，
        # 抬完先慢速爬过原来的卡点（zer 比卡点小 retry_slow_clear）再恢复原速
        self.retry_slow = float(retry_slow)
        self.retry_slow_clear = float(retry_slow_clear)
        self.entry_force_ctrl = bool(entry_force_ctrl)
        self.entry_kf = float(entry_kf)
        self.entry_filt = float(entry_filt)
        self.entry_creep = float(entry_creep)
        self.xy_model = None
        self.xy_model_scale = float(xy_model_scale)
        self.xy_model_ema = float(xy_model_ema)   # <1 时对模型增量做 EMA 平滑
        self._xy_ema = None
        self._ft_hist = None
        # add = 脚本修正 + 模型增量；replace = 用模型增量替代脚本修正
        self.xy_model_mode = str(xy_model_mode)
        if xy_model:
            self._load_xy_model(str(xy_model))
        self.search_period = max(10, int(search_period))
        self.search_stall_z = float(search_stall_z)
        self.search_mode = str(search_mode)
        self.spiral_dtheta = float(spiral_dtheta)
        self.spiral_grow = float(spiral_grow)
        self.grad_gain = float(grad_gain)
        self.grad_step_max = float(grad_step_max)
        self.grad_force_min = float(grad_force_min)
        self.grad_sign = float(grad_sign)
        # 开爪时的力觉补偿：按腕部 F_xy 反推弹夹被带偏的方向（真机可用）
        self.release_force_gain = float(release_force_gain)
        self.release_force_limit = float(release_force_limit)
        self.release_force_min = float(release_force_min)
        self.release_force_sign = float(release_force_sign)
        # 门控补偿：只有开爪时力超过 thr 才补，沿标定的拖拽方向（每任务一个常量）
        self.release_force_dir = np.asarray(release_force_dir, dtype=float)
        self.release_force_thr = float(release_force_thr)
        # 交接瞬间力门控：|F| > gate_force 说明这次开爪会明显拖拽，整段多补 extra
        self.release_gate_force = float(release_gate_force)
        self.release_gate_extra = np.asarray(release_gate_extra, dtype=float)
        # 分段松手：先开一小段解除挤压（避免手指卡在槽口里带着弹夹走），抬出槽口后再全开
        self.release_pre_frames = max(0, int(release_pre_frames))
        self.release_pre_frac = float(release_pre_frac)
        self.release_pre_lift = float(release_pre_lift)
        self.release_pre_lift_frames = max(1, int(release_pre_lift_frames))
        # 松手后抬离：爪全张开后再把末端抬高，让指尖彻底离开弹夹/槽口，
        # 否则收尾阶段指尖刮到弹夹会把落点甩偏几毫米
        self.release_exit_lift = float(release_exit_lift)
        self.release_exit_frames = max(1, int(release_exit_frames))
        # 松手只需"退让到刚好脱开"：release_open 是绝对值（指尖要走 ~35mm，
        # 一路摩擦把弹夹拖走 4~5mm）。>0 时改成从当前指尖位置各自外撤这么多。
        self.release_retract = float(release_retract)
        self._gate_bias = np.zeros(2)
        self._rel_f = np.zeros(2)
        # 专家数据采集时叠加的低频横向试探（DAgger 式状态覆盖），0=关
        self.xy_noise = float(xy_noise)
        self.xy_noise_period = max(1, int(xy_noise_period))
        self._xy_noise_off = np.zeros(2)
        self._ft_hist = None
        self._expert_dxy = np.zeros(2)
        self._rel_f = np.zeros(2)
        self._gate_bias = np.zeros(2)
        self._spiral_theta = 0.0
        self._search_center = None
        self._xy_noise_off = np.zeros(2)
        self.probe_step = tuple(probe_step)
        self.probe_hold = tuple(probe_hold)
        self.probe_near = float(probe_near)
        self.probe_prob = float(probe_prob)
        # 试探步长逐圈放大 + 方向偏置：卡在槽沿时先小范围试，试不到就扩大搜索半径
        self.probe_grow = float(probe_grow)
        self.probe_step_max = float(probe_step_max)
        self.explore_limit = float(explore_limit)
        self.probe_bias = np.asarray(probe_bias, dtype=float).reshape(2)
        self.probe_bias_prob = float(probe_bias_prob)
        # 试探前先把正压力卸到 probe_unload_force 以下再横移，否则夹爪是"压着磨"
        # 弹夹会在爪内打滑（实测滑 12~117mm），0 = 沿用 jam_force
        self.probe_unload_force = float(probe_unload_force)
        # 试探前卸力能抬多高：只抬 1mm 脱不开楔住的弹夹，横移会把弹夹拖出去、力冲到 1000N+
        self.probe_unload_lift = float(probe_unload_lift)
        # 入口 xy 系统偏差补偿：实测接触瞬间弹夹恒偏 +2.2mm(x)，把归中目标整体挪 -x，
        # 但离座位 entry_xy_bias_fade 以内线性收回到 0，避免落座后仍被横向顶着
        self.entry_xy_bias = np.asarray(entry_xy_bias, dtype=float).reshape(2)
        self.entry_xy_bias_fade = float(entry_xy_bias_fade)
        self.cart_xy_bias = np.asarray(cart_xy_bias, dtype=float).reshape(2)
        self.cart_step = float(cart_step)
        self.cart_xy_step = float(cart_xy_step)
        self.cart_xy_gain = float(cart_xy_gain)
        self.cart_xy_deadband = float(cart_xy_deadband)
        # 自由空间里可以走快一点：初始偏置越大，接触瞬间的残余误差越大（实测 5mm→2.2mm，
        # 6mm→2.4~3.4mm），而在接触后再加速会激出左右摆动，所以只在低力时放大步长
        self.cart_xy_free_boost = float(cart_xy_free_boost)
        self.cart_xy_free_force = float(cart_xy_free_force)
        # 笛卡尔段开始时的"横向扰动"：关节轨迹回放会把 reset 的随机偏移完全抹掉
        # （实测 corr(offset, contact_err)≈-0.15），所以偏移必须在这里重新施加，
        # 先抬起 entry_offset_lift 再横移，落成"偏离→纠正→插入"的真实样本
        self.entry_offset = np.zeros(2)
        self.entry_offset_frames = int(entry_offset_frames)
        self.entry_offset_lift = float(entry_offset_lift)
        # 落座后夹爪张开时弹夹会被横推 2~3mm（实测同向）。补偿必须在"还夹着"的时候
        # 做：unload 阶段指头已经松到预紧了，那时再移动末端弹夹根本不跟。
        self.seat_shift = np.asarray(seat_shift, dtype=float).reshape(2)
        self.seat_shift_frames = max(4, int(seat_shift_frames))
        self.cart_err_alpha = float(cart_err_alpha)
        self.force_gain = float(force_gain)
        self.path_speed = max(0.05, float(path_speed))
        self.release_guide_limit = float(release_guide_limit)
        # >0 = 比例引导（末端偏移 ∝ 当前弹夹误差），0 = 老的积分引导。
        # 积分版在松手瞬间的弹跳里会把偏移攒到上限，弹夹回中后仍被顶出去 3mm+。
        self.release_guide_gain = float(release_guide_gain)
        self.release_guide_alpha = float(release_guide_alpha)
        self._err_xy_f = None
        self.release_mode = str(release_mode)
        self.post_open_hold = max(0, int(post_open_hold))
        self.retreat_frames = max(1, int(retreat_frames))
        self.retreat_dist = float(retreat_dist)
        self.retreat_hold = max(0, int(retreat_hold))
        self.readjust = bool(readjust)
        # 重夹重放只在误差大到值得冒险时才做：≤这个值说明弹夹基本已经落好，
        # 再闭一次爪反而会把它推开 1~3mm（实测）。
        self.readjust_min_err = float(readjust_min_err)
        # 阶段拆分：下降段只要"弹夹 z 到位"就交出控制权，松手由固定原语完成，
        # 模型只在 cart_descend 里负责末端 xy 位姿，不参与松手。
        self.seat_z_tol = float(seat_z_tol)
        # 落座还要看横向：lab2 实测过弹夹落进"隔壁插槽"（dy=+23mm）而 z 刚好到位的回合，
        # 只看 z 会在槽沿上就松手。>0 时要求 |xy 误差| 也小于该值才算落座。
        self.seat_xy_tol = float(seat_xy_tol)
        # 槽口护栏：弹夹还在槽口上方(mouth_z)且横向偏太多时，先不许继续下降，
        # 让 xy 纠偏把人拉到正确槽位上方再下去（lab2 会掉进隔壁槽：间距只有 23.5mm）
        self.mouth_xy_tol = float(mouth_xy_tol)
        self.mouth_z = float(mouth_z)
        # 横向护栏 + 力梯度蹭：离槽还有 lat_hold_z 以上、横向偏差 > lat_hold_tol 时，
        # 不许继续下降；一旦蹭到槽沿(>lat_hold_force)就沿 -F_xy 蹭，把弹夹往槽口方向挪。
        # （纯 F/T 在悬空 20mm 时没有任何信息，必须先制造接触才能判断方向）
        self.lat_hold_tol = float(lat_hold_tol)
        self.lat_hold_z = float(lat_hold_z)
        self.lat_hold_force = float(lat_hold_force)
        self.lat_hold_step = float(lat_hold_step)
        self.seat_hold = max(1, int(seat_hold))
        self.handover = None      # 交给松手原语时的触发信息（写进 index.jsonl）
        # 松手后重夹重放单次只有约一半概率落到 <0.5mm，另一半会甩到 ~8mm。
        # 多给几次机会：每次都是"按当前实测误差重抓→平移→松手"。
        self.readjust_max = max(1, int(readjust_max))
        # 卡在槽沿时"重新夹一次"：卡死由弹夹相对夹爪的姿态决定，抬升重试改变不了它，
        # 只有重夹才能换一个相对姿态（lab2 的 6 次纯参数调整全部无效）
        self.rim_regrip = int(rim_regrip)
        self.rim_regrip_frames = max(20, int(rim_regrip_frames))
        self.grip_settle_frames = max(1, int(grip_settle_frames))
        # return_home 原来把夹爪指令退回"初始张开值"，等于空中松爪一次、下降时再夹紧；
        # 实测弹夹会随机滑 3~6mm，且与夹持力/速度无关。默认改成全程保持强夹持。
        self.hold_through_home = bool(hold_through_home)
        # _z_goal 是用标称 grasp_rel 算的；实际夹持姿态变了以后它会高 3~5mm，
        # 导致"命令到底、弹夹却还悬在座上方"（实测 lab2 停在 −3~−5mm）。放开这个下限，
        # 由实测 z 误差/力控决定何时停。
        self.z_goal_extra = float(z_goal_extra)
        self.grip_settle_hold = max(0, int(grip_settle_hold))
        self.reset(None)

    @staticmethod
    def _hold_grip(env, cmd, grip=None):
        """把夹爪关节替换成指定目标（默认下降段的强夹持）。"""
        cmd = np.asarray(cmd, dtype=float).copy()
        for n, v in (grip or env.grip_hold_positions()).items():
            if n in env.dof_names:
                cmd[env.dof_names.index(n)] = float(v)
        return cmd

    def _descend_grip(self, env):
        """下降前 grip_ramp_frames 帧把夹爪从复位实测值缓升到强夹持，避免一帧夹飞弹夹。"""
        g1 = env.grip_hold_positions()
        if self.hold_through_home:
            # 回位段已经保持强夹持，这里再"从张开值缓升"会二次松爪
            return dict(g1)
        g0 = env.grip_initial_positions()
        s = min(1.0, self.i / float(self.grip_ramp_frames))
        return {n: g0.get(n, v) + (v - g0.get(n, v)) * s for n, v in g1.items()}

    def _guide_xy(self, err_xy):
        """松手引导的 xy 偏移。

        release_guide_gain>0：比例 + EMA，偏移跟着"当前误差"走，误差回中就自动收回；
        =0：老的积分版（误差攒到上限后弹夹回中仍会被顶偏）。
        """
        lim = self.release_guide_limit
        if lim <= 0.0:
            self._rel_xy = np.zeros(2)
            return self._rel_xy
        if self.release_guide_gain > 0.0:
            e = np.clip(np.asarray(err_xy, dtype=float)[:2], -0.006, 0.006)
            if self._err_xy_f is None:
                self._err_xy_f = e.copy()
            else:
                a = min(max(self.release_guide_alpha, 0.02), 1.0)
                self._err_xy_f = (1.0 - a) * self._err_xy_f + a * e
            self._rel_xy = np.clip(self.release_guide_gain * self._err_xy_f, -lim, lim)
            return self._rel_xy
        if self._rel_xy is None:
            self._rel_xy = np.zeros(2)
        self._rel_xy = np.clip(self._rel_xy + np.clip(err_xy, -3e-4, 3e-4),
                               -lim, lim)
        return self._rel_xy

    def reset(self, env):
        self._path = np.asarray(self.joint_path, dtype=float).copy()
        self.phase = "grip_settle"
        self.k = 0
        self.i = 0
        self._path_u = 0.0
        self.done = False
        self.failed = False
        self.reason = ""
        self._q_start = None
        self._hold = 0
        self._jam = 0
        self._trim = np.zeros(3)
        self._hold_cmd = None
        self._ee_seat = None
        self._shift_total = np.zeros(2)
        self._grip_unload0 = None
        self._grip_release0 = None
        self._z_cmd = None
        self._z_goal = None
        self._cart_hold = 0
        self._cart_frames = 0
        self._progress_low_z = None
        self._progress_age_frames = 0
        self._z_lift = 0.0
        self._stall = 0
        self._zref = None
        self._zh = deque(maxlen=60)
        self._search_k = 0
        self._spiral_theta = 0.0
        self._search_center = None
        self._z_search_top = None
        self._search_used = False
        self._probe_age = 0
        self._probe_dir = np.zeros(2)
        self._probe_step = 1e-4
        self._probe_cycle = 0
        self._entry_k = 0
        self._entry_probing = False
        self._entry_probe_age = 0
        self._entry_dir = np.zeros(2)
        self._retry_left = 0.0
        self._retry_count = 0
        self._slow_active = False
        self._zer_stall = None
        self._f_filt = None
        self._z_entry_top = None
        self._e_prev = None
        self._rng = np.random.default_rng(int(getattr(self, "search_seed", 0)))
        self._entry_phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self._bounded_dither_offset = np.zeros(2)
        self._last_projected_ee = None
        if self.search_mode == "random":
            # 每次搜索的半径也随机一点，避免所有回合都在同一个圈上绕
            self.search_radius = float(self._rng.uniform(self.search_radius_lo,
                                                         self.search_radius_hi))
        self._explore_episode = bool(self._rng.random() < self.probe_prob)
        self._rel_xy = None
        self._err_xy_f = None
        self._f0 = None
        self._err_f = np.zeros(2)
        self._readjusted = False
        self._offset_left = 0
        self._seat_shift_done = False
        self._regrip_count = 0
        self._regrip_from = None

    def _enter_cart_descend(self, env, obs):
        """切到笛卡尔闭环：保持当前 z，按弹夹实测 xy 归中后继续递降。"""
        self.phase = "cart_descend"
        self._z_cmd = float(obs["ee_pos"][2])
        self._z_goal = float(env.retaining_clip_target()[2] - env.grasp_rel[2]
                             + env.release_height) - self.z_goal_extra
        self._xy_goal = np.asarray(obs["ee_pos"], dtype=float)[:2].copy()
        self._xy_nominal = self._xy_goal.copy()
        self._f0 = np.asarray(obs.get("force", np.zeros(3)), dtype=float).copy()
        self._err_f = np.zeros(2)
        self._cart_hold = 0
        self._cart_frames = 0
        self._progress_low_z = None
        self._progress_age_frames = 0
        # 横向扰动：抬起一点再横移，跳过关节回放抹掉随机偏移的问题
        self._offset_left = 0
        if self.entry_offset_frames > 0 and float(np.linalg.norm(self.entry_offset)) > 1e-9:
            self._xy_goal = self._xy_goal + self.entry_offset
            self._z_cmd += self.entry_offset_lift
            self._offset_left = int(self.entry_offset_frames)
        return {"ee_pos": np.asarray(obs["ee_pos"], dtype=float),
                "grip_targets": env.grip_hold_positions()}, False

    def act(self, env, obs):
        action, finished = self._act_unprojected(env, obs)
        measured = np.asarray(obs["ee_pos"], dtype=float)
        if action is not None and action.get("ee_pos") is not None:
            requested = np.asarray(action["ee_pos"], dtype=float)
            previous = measured if self._last_projected_ee is None else self._last_projected_ee
            delta = requested - previous
            target = previous + delta * min(1.0, 0.00035 / max(float(np.linalg.norm(delta)), 1e-12))
            error = target - measured
            target = measured + error * min(1.0, 0.003 / max(float(np.linalg.norm(error)), 1e-12))
            action = dict(action)
            action["ee_pos"] = target
            self._last_projected_ee = target.copy()
        else:
            self._last_projected_ee = measured.copy()
        return action, finished

    def _act_unprojected(self, env, obs):
        if self.done:
            return None, True
        home = env.home_joint_positions()
        if self.phase == "grip_settle":
            # 先原地把弹夹夹紧并稳住，再开始回位/下降；否则带着松爪走会滑 20~30mm。
            env.capture_grip_hold()
            self.k += 1
            s = min(1.0, self.k / float(self.grip_settle_frames))
            g0 = env.grip_initial_positions()
            g1 = env.grip_hold_positions()
            grip = {n: g0.get(n, v) + (v - g0.get(n, v)) * s for n, v in g1.items()}
            cmd = self._hold_grip(env, obs["joint_positions"], grip)
            if self.k >= self.grip_settle_frames + self.grip_settle_hold:
                self.phase = "return_home"
                self.k = 0
            return {"joint_positions": cmd}, False
        if self.phase == "return_home":
            if self._q_start is None:
                self._q_start = np.asarray(obs["joint_positions"], dtype=float).copy()
            self.k += 1
            s = min(1.0, self.k / float(max(1, self.home_frames)))
            cmd = self._q_start + (home - self._q_start) * (0.5 - 0.5 * np.cos(np.pi * s))
            if s >= 1.0 and self.k > self.home_frames + 3:
                self.phase = "align"
                self.k = 0
            g_home = (env.grip_hold_positions() if self.hold_through_home
                      else env.grip_initial_positions())
            return {"joint_positions": self._hold_grip(env, cmd, g_home)}, False

        if self.phase == "align":
            # 最高点手臂贴着肘限位、没法横移，先直接进入下降，等降下几厘米再平移路径
            self.phase = "descend"
            self.i = 0
            self._shifted = False
            return {"joint_positions": self._hold_grip(env, obs["joint_positions"],
                                                       env.grip_initial_positions())}, False

        if self.phase == "descend":
            f_norm = float(np.linalg.norm(np.asarray(obs.get("force", np.zeros(3)), dtype=float)))
            # 离开顶部限位后改用笛卡尔闭环：倒放路径在负载下垂时会累积 20~30mm 误差，
            # 继续开环走会卡在槽沿。这里按弹夹实测 xy 归中，z 绝对递降。
            # 提前切换：关节回放段一旦压到槽沿，软肘/腕在 100N 以上的反力下会起 3~7Hz
            # 极限环（末端左右晃 1~6mm），所以一检测到接触力就交给笛卡尔闭环收敛。
            switched = False
            if (self._path_u >= self.cart_switch_frames or self._path_u >= len(self._path)) and \
                    (self._path_u >= len(self._path)
                     or env.limit_margin(env._read_model_q()) > 0.05):
                switched = True
            elif (self.cart_switch_force > 0.0 and self._path_u >= self.cart_switch_min_frames
                  and f_norm > self.cart_switch_force):
                switched = True
            if switched:
                return self._enter_cart_descend(env, obs)
            # 下降 40 帧后（手臂离开肘限位死区）每 15 帧按弹夹实测 xy 偏差修正剩余路径，
            # 只在路径前 70% 做，避免已经在槽里时横向硬拽；每次最多 0.5mm，累计有上限。
            i0 = int(min(self._path_u, len(self._path) - 1))
            i1 = min(i0 + 1, len(self._path) - 1)
            frac = float(self._path_u - i0)
            if (self.shift_limit > 0.0 and i0 > 40 and i0 % self.shift_every == 0
                    and i0 < int(0.7 * len(self._path))
                    and float(np.linalg.norm(self._shift_total)) < self.shift_limit):
                obj = np.asarray(obs["retaining_clip_pos"], dtype=float)
                tgt = env.retaining_clip_target()
                err = np.array([tgt[0] - obj[0], tgt[1] - obj[1], 0.0])
                err[:2] = np.clip(err[:2], -2e-3, 2e-3)
                if (1e-4 < float(np.linalg.norm(err[:2]))
                        and float(np.linalg.norm(self._shift_total + err[:2])) <= self.shift_limit):
                    ee = np.asarray(obs["ee_pos"], dtype=float)
                    q_new, pe = env.solve_ee(ee + err)
                    if pe < 1.5e-3 and env.limit_margin(np.asarray(q_new, dtype=float)) > 0.02:
                        q_dof = env._model_q_to_dof(np.asarray(q_new, dtype=float))
                        applied = False
                        for name in env.world.ik.controlled_scalar_joint_names:
                            if name in env.dof_names:
                                k = env.dof_names.index(name)
                                d = float(np.clip(q_dof[k] - self._path[i0, k], -0.02, 0.02))
                                if abs(d) > 1e-6:
                                    self._path[i0:, k] += d
                                    applied = True
                        if applied:
                            self._shift_total += err[:2]
            base_cmd = (1.0 - frac) * self._path[i0] + frac * self._path[i1]
            cmd = self._hold_grip(env, base_cmd, self._descend_grip(env))
            # 一旦弹夹已经到座位（<1.5mm），立刻停止下压，进入松手
            if float(np.linalg.norm(np.asarray(obs["retaining_clip_pos"], dtype=float)
                                    - env.retaining_clip_target())) < 1.5e-3:
                self.phase = "unload"
                self.k = 0
                self._hold_cmd = self._hold_grip(env, obs["joint_positions"])
                self._ee_seat = np.asarray(obs["ee_pos"], dtype=float).copy()
                return {"joint_positions": self._hold_cmd}, False
            if f_norm > self.jam_force:
                # 顶住了：原地不动等力释放，别硬往下压
                self._jam += 1
                if self._jam > self.jam_hold:
                    # 顶太久：先按侧向力反方向轻推一下卸力
                    fxy = np.asarray(obs.get("force", np.zeros(3)), dtype=float)[:2]
                    n = float(np.linalg.norm(fxy))
                    self._jam = 0
                    if n > 5.0:
                        ee = np.asarray(obs["ee_pos"], dtype=float)
                        ee[:2] -= fxy / n * 5e-4
                        return {"ee_pos": ee, "grip_targets": env.grip_hold_positions()}, False
                    self._path_u += self.path_speed
            else:
                self._jam = 0
                # 伺服同步：关节还没跟上就不要往下走，否则反力会把手臂顶死
                q = np.asarray(obs["joint_positions"], dtype=float)
                track = float(np.max(np.abs(q - cmd))) if q.shape == cmd.shape else 0.0
                if track < self.track_tol:
                    self._path_u += self.path_speed
            self.i = int(min(self._path_u, len(self._path) - 1))
            return {"joint_positions": cmd}, False

        if self.phase == "cart_descend":
            self._cart_frames += 1
            # Measured wrist progress controls the internal stall timeout.
            # Total external attempt budget remains exactly 45 seconds.
            measured_z = float(obs["ee_pos"][2])
            if self._progress_low_z is None or measured_z < self._progress_low_z - 0.002:
                self._progress_low_z = measured_z
                self._progress_age_frames = 0
            else:
                self._progress_age_frames += 1

            cart = np.asarray(obs["retaining_clip_pos"], dtype=float)
            tgt = env.retaining_clip_target()
            tgt = np.asarray(tgt, dtype=float).copy()
            tgt[:2] += self.cart_xy_bias
            err = tgt - cart
            f_norm = float(np.linalg.norm(np.asarray(obs.get("force", np.zeros(3)), dtype=float)))
            ee = np.asarray(obs["ee_pos"], dtype=float)
            zer = abs(float(err[2]))
            entry = self.entry_z > 0.0 and zer < self.entry_z
            if self._offset_left > 0:
                self._offset_left -= 1
                cmd = np.array([self._xy_goal[0], self._xy_goal[1], self._z_cmd])
                return {"ee_pos": cmd, "grip_targets": env.grip_hold_positions()}, False
            # Bounded retaining clip search is an output offset, not an integrator.
            self._bounded_dither_offset = np.zeros(2)
            # 抬起-重对中-再插：卡在槽沿时先抬 retry_lift 让弹夹脱离接触，
            # 抬的过程中按弹夹误差继续归中（此时无摩擦阻力），再压下第二次试探。
            if self._retry_left > 0.0:
                step = min(self.retry_lift / float(self.retry_frames), self._retry_left)
                self._z_cmd += step
                self._retry_left -= step
                self._z_lift = max(self._z_lift, self.retry_lift - self._retry_left)
            # z：只在没有明显反力时下降；卡住就停 z，先把 xy 归中
            #     接触力越大下压越慢，避免硬顶槽沿时末端被反力激出 3~7Hz 摆动
            if entry and self.entry_force_ctrl:
                # 导纳式入口：把接触力维持在 entry_force 附近——
                # 力大就回抬、力小就下压，横向同时做抖动/试探。避免 400N 以上硬顶把弹夹顶滑。
                if self._f_filt is None:
                    self._f_filt = float(f_norm)
                self._f_filt = (1.0 - self.entry_filt) * self._f_filt + self.entry_filt * f_norm
                if self._z_entry_top is None:
                    self._z_entry_top = float(self._z_cmd) + 0.006
                e_f = float(self._f_filt) - self.entry_force
                # 下压偏置 entry_creep：即使力到了设定值也以恒速慢慢压（力上限爬行），
                # 避免"力一超就悬停"；力大时向上修正自然压过这个偏置。
                dz = float(np.clip(self.entry_kf * e_f - self.entry_creep,
                                   -self.cart_step, self.cart_step * 0.6))
                if self._slow_active and dz < 0.0:
                    dz *= self.retry_slow
                self._z_cmd = float(np.clip(self._z_cmd + dz, self._z_goal, self._z_entry_top))
            elif f_norm > self.jam_force * 2.0:
                # 硬顶（>2×jam_force）：小幅回抬卸力，最多 jam_lift，避免把弹夹从槽沿硬挤出去
                if self._z_lift < self.jam_lift:
                    step = min(self.cart_step * 0.5, self.jam_lift - self._z_lift)
                    self._z_cmd += step
                    self._z_lift += step
            elif f_norm < self.jam_force:
                dz = self.cart_step * max(0.1, 1.0 - f_norm / max(1.0, self.jam_force))
                if self._retry_left > 0.0:
                    dz = 0.0
                if entry:
                    # 接近座位：慢压；一旦反力超 entry_force 就停住转试探
                    dz *= self.entry_slow
                    if f_norm > self.entry_force:
                        dz = 0.0
                self._z_cmd = max(self._z_goal, self._z_cmd - dz)
            if (self.mouth_xy_tol > 0.0 and zer > self.mouth_z
                    and float(np.linalg.norm(np.asarray(err[:2], dtype=float)))
                    > self.mouth_xy_tol):
                # 还没到槽口就先别下去，等 xy 归中
                self._z_cmd = max(float(self._z_cmd), float(ee[2]))
            if self.lat_hold_tol > 0.0 and zer > self.lat_hold_z:
                e_xy = np.asarray(err[:2], dtype=float)
                if float(np.linalg.norm(e_xy)) > self.lat_hold_tol:
                    # 横向差太多：停在当前高度，别掉进隔壁槽
                    self._z_cmd = max(float(self._z_cmd), float(ee[2]))
                    # 只有蹭到槽沿才有方向信息：沿 -F_xy 蹭（减力方向 = 往槽口里挪）
                    f3 = np.asarray(obs.get("force", np.zeros(3)), dtype=float)
                    fn2 = float(np.linalg.norm(f3[:2]))
                    if fn2 > self.lat_hold_force:
                        self._xy_goal = self._xy_goal - self.lat_hold_step * f3[:2] / fn2
                        self._search_used = True
            if (self._slow_active and self._zer_stall is not None
                    and zer <= self._zer_stall - self.retry_slow_clear):
                self._slow_active = False
            # 误差积分到"命令末端位置"上：手臂下垂时命令继续外推，直到弹夹归中；
            # 不能写成 ee + dx（实测不动时命令也不动，会一直卡在 20mm 外）。
            # 误差 EMA + 死区 + 小步长：直接按单帧误差积分会在槽沿接触时形成左右极限环
            err_xy = np.asarray(err[:2], dtype=float)
            if float(np.linalg.norm(self.entry_xy_bias)) > 1e-9:
                fade = float(np.clip((zer - self.entry_xy_bias_fade) / 0.04, 0.0, 1.0))
                err_xy = err_xy + self.entry_xy_bias * fade
            self._err_f = (1.0 - self.cart_err_alpha) * self._err_f + \
                self.cart_err_alpha * err_xy
            # DAgger：无论当前由谁驱动 xy，都按脚本口径记录"专家在这一状态的修正量"
            # （mm/帧）。deploy 时这个量不可得，只用于给策略状态打标签。
            _xy_step_exp = self.cart_xy_step
            if (self.cart_xy_free_boost > 1.0
                    and f_norm < self.cart_xy_free_force):
                _xy_step_exp *= self.cart_xy_free_boost
            self._expert_dxy = np.clip(self.cart_xy_gain * self._err_f,
                                       -_xy_step_exp, _xy_step_exp) * 1000.0
            if float(np.linalg.norm(self._err_f)) <= self.cart_xy_deadband:
                self._expert_dxy = np.zeros(2)
            model_replace = (self.xy_model is not None
                             and self.xy_model_mode == "replace")
            if model_replace:
                # replace：模型增量替代脚本的误差积分（模型是"行为克隆"时 ≈ 脚本本身）
                d_xy = self._xy_model_action(obs, np.asarray(err, dtype=float))
                if d_xy is not None:
                    self._xy_goal = self._xy_goal + np.clip(
                        d_xy / 1000.0, -self.cart_xy_step, self.cart_xy_step)
            else:
                if float(np.linalg.norm(self._err_f)) > self.cart_xy_deadband:
                    xy_step = self.cart_xy_step
                    if (self.cart_xy_free_boost > 1.0
                            and f_norm < self.cart_xy_free_force):
                        xy_step *= self.cart_xy_free_boost
                    self._xy_goal = self._xy_goal + np.clip(
                        self.cart_xy_gain * self._err_f, -xy_step, xy_step)
                if self.xy_model is not None:
                    # add：脚本修正 + 模型增量（默认；scale=1 时相当于双倍修正）
                    d_xy = self._xy_model_action(obs, np.asarray(err, dtype=float))
                    if d_xy is not None:
                        self._xy_goal = self._xy_goal + np.clip(
                            d_xy / 1000.0, -self.cart_xy_step, self.cart_xy_step)
            if self.force_gain > 0.0 and self._f0 is not None:
                fres = np.asarray(obs.get("force", np.zeros(3)), dtype=float) - self._f0
                if float(np.linalg.norm(fres[:2])) > 5.0:
                    self._xy_goal = self._xy_goal - self.force_gain * fres[:2]
            if self.xy_noise > 0.0 and self._cart_frames % self.xy_noise_period == 0:
                # 低频随机试探：每 period 帧换一个方向，幅度 ±xy_noise
                ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
                r = float(self._rng.uniform(0.3, 1.0)) * self.xy_noise
                self._xy_noise_off = r * np.array([np.cos(ang), np.sin(ang)])
            self._xy_goal = np.clip(self._xy_goal, self._xy_nominal - 0.03,
                                    self._xy_nominal + 0.03)
            if entry and self.search_mode == "random":
                # 近座位柔顺入口：小幅圆周抖动 + 反力大时随机试探（试探-插入循环）
                self._entry_k += 1
                ph = 2.0 * np.pi * self._entry_k / float(self.entry_dither_period) + self._entry_phase
                amp = self.entry_dither
                if self.dither_gate > 0.0 and \
                        float(np.linalg.norm(err[:2])) < self.dither_gate:
                    amp *= self.dither_gate_scale
                self._bounded_dither_offset = amp * np.array([np.cos(ph), np.sin(ph)])
                if f_norm > self.entry_force:
                    self._search_used = True
                    if self._entry_probe_age <= 0:
                        if self._rng.random() < 0.5 and f_norm > 5.0:
                            fxy = np.asarray(obs.get("force", np.zeros(3)), dtype=float)[:2]
                            n2 = float(np.linalg.norm(fxy))
                            d = -fxy / n2 if n2 > 1e-6 else None
                        else:
                            d = None
                        if d is None:
                            a = self._rng.uniform(0.0, 2.0 * np.pi)
                            d = np.array([np.cos(a), np.sin(a)])
                        self._entry_dir = d
                        self._entry_probe_age = int(self._rng.integers(
                            self.entry_probe_hold[0], self.entry_probe_hold[1] + 1))
                    self._entry_probe_age -= 1
                    self._xy_goal = self._xy_goal + self._entry_dir * self.entry_probe_step
                    if self.probe_unload_force > 0.0:
                        if self._z_search_top is None:
                            self._z_search_top = float(self._z_cmd) + self.probe_unload_lift
                        if f_norm > self.probe_unload_force:
                            self._z_cmd = min(self._z_cmd + 2.5e-4, self._z_search_top)
                lim = max(self.search_radius, self.explore_limit)
                self._xy_goal = np.clip(self._xy_goal, self._xy_nominal - lim,
                                        self._xy_nominal + lim)
            # 卡在槽沿上（力大、z 不降）：绕圈找孔。经典 peg-in-hole 搜索，
            # 只在"明显高于座位 + 有反力"时启用，落座前 8mm 内不再搜索，避免把已进孔的弹夹拽出来。
            # 卡死判定用"z 多久没有新进展"，不看力：贴槽沿时力就在阈值附近抖
            zer = abs(float(err[2]))
            self._zh.append(zer)
            if len(self._zh) >= 60:
                arr = np.fromiter(self._zh, dtype=float, count=len(self._zh))
                old = float(np.median(arr[:30]))
                new = float(np.median(arr[30:]))
                if new < old - 0.0015:
                    self._stall = 0
                    self._probe_cycle = 0
                else:
                    self._stall += 1
            if zer <= self.search_stall_z:
                self._stall = 0
            near_seat = 0.0 < zer < self.probe_near
            if self._stall >= self.search_after or (near_seat and self._explore_episode
                                                    and self.search_mode == "random"):
                if (self.rim_regrip > 0 and self._regrip_count < self.rim_regrip
                        and zer > 0.05):
                    self._regrip_count += 1
                    self._search_used = True
                    self.phase = "rim_regrip"
                    self.k = 0
                    self._regrip_from = np.asarray(obs["ee_pos"], dtype=float).copy()
                    self._stall = 0
                    self._zh.clear()
                    return {"ee_pos": self._regrip_from,
                            "grip_targets": env.grip_hold_positions()}, False
                # 优先用"抬起-再压"的次数控制：超过 retry_max 才退化成平面随机试探
                if self._retry_left <= 0.0 and self._retry_count < self.retry_max:
                    self._retry_count += 1
                    self._retry_left = self.retry_lift
                    if self.retry_slow > 0.0:
                        self._slow_active = True
                        self._zer_stall = zer
                    self._stall = 0
                    self._zh.clear()
                    self._search_used = True
                    cmd = np.array([self._xy_goal[0], self._xy_goal[1], self._z_cmd])
                    return {"ee_pos": cmd, "grip_targets": env.grip_hold_positions()}, False
                if self._z_search_top is None:
                    self._z_search_top = float(self._z_cmd) + self.probe_unload_lift
                    self._probe_age = 0
                self._search_used = True
                self._search_k += 1
                if self._search_k == 1:
                    # 搜索中心 = 卡住瞬间的目标点（不用弹夹真值，真机同样可得）
                    self._search_center = self._xy_goal.copy()
                    self._spiral_theta = 0.0
                if self.search_mode == "random":
                    # 随机试探：每隔 12~30 帧换一个随机方向/步长；一半概率沿"减力方向"，
                    # 一半纯随机，保证覆盖既有信息量又能走出局部卡死。
                    if self._probe_age <= 0:
                        self._probe_cycle += 1
                        if self._rng.random() < 0.5 and f_norm > 5.0:
                            fxy = np.asarray(obs.get("force", np.zeros(3)), dtype=float)[:2]
                            n2 = float(np.linalg.norm(fxy))
                            d = -fxy / n2 if n2 > 1e-6 else None
                        else:
                            d = None
                        if d is None:
                            a = self._rng.uniform(0.0, 2.0 * np.pi)
                            d = np.array([np.cos(a), np.sin(a)])
                        if (float(np.linalg.norm(self.probe_bias)) > 1e-9
                                and self._rng.random() < self.probe_bias_prob):
                            # 偏置方向：失败样本集中在某个轴向时，优先往那个方向找
                            a = float(np.arctan2(self.probe_bias[1], self.probe_bias[0]))
                            a += float(self._rng.uniform(-0.9, 0.9))
                            d = np.array([np.cos(a), np.sin(a)])
                        self._probe_dir = d
                        base = float(self._rng.uniform(*self.probe_step))
                        step = base * self.probe_grow ** max(0, self._probe_cycle - 1)
                        self._probe_step = min(self.probe_step_max, step) \
                            if self.probe_step_max > 0 else step
                        self._probe_age = int(self._rng.integers(self.probe_hold[0],
                                                                 self.probe_hold[1] + 1))
                    self._probe_age -= 1
                    self._xy_goal = self._xy_goal + self._probe_dir * self._probe_step
                elif self.search_mode == "spiral":
                    # 确定性阿基米德螺旋：半径随转角单调增长，第 n 圈覆盖 n×2π×grow。
                    # 轨迹完全由帧数决定（可复现、可学习），不用随机数。
                    self._spiral_theta += self.spiral_dtheta
                    r_max = max(self.search_radius, self.explore_limit)
                    r = min(r_max, self.spiral_grow * self._spiral_theta)
                    c = self._search_center if self._search_center is not None \
                        else self._xy_nominal
                    self._xy_goal = c + r * np.array(
                        [np.cos(self._spiral_theta), np.sin(self._spiral_theta)])
                elif self.search_mode == "grad":
                    # 力梯度搜索：沿 −F_xy（远离接触挤压）方向按力大小步进。
                    # 力小（已经脱开）就原地等 z 继续压；力大才推。
                    fxy = np.asarray(obs.get("force", np.zeros(3)), dtype=float)[:2]
                    n2 = float(np.linalg.norm(fxy))
                    c = self._search_center if self._search_center is not None \
                        else self._xy_nominal
                    if n2 >= self.grad_force_min:
                        d = self.grad_sign * fxy / n2
                        step = min(self.grad_step_max,
                                   max(5e-5, self.grad_gain * n2))
                        self._xy_goal = self._xy_goal + d * step
                    else:
                        # 力很小：说明这一步让开了，回到中心再下压（轻微的螺旋回中）
                        self._spiral_theta += self.spiral_dtheta
                        r = min(0.001, self.spiral_grow * self._spiral_theta)
                        self._xy_goal = c + r * np.array(
                            [np.cos(self._spiral_theta), np.sin(self._spiral_theta)])
                else:
                    ang = 2.0 * np.pi * (self._search_k % self.search_period) / self.search_period
                    grow = min(1.0, 0.5 + 0.5 * self._search_k / float(self.search_period * 3))
                    self._xy_goal = self._xy_nominal + self.search_radius * grow * np.array(
                        [np.cos(ang), np.sin(ang)])
                # 找孔时把 z 稍微退一点卸力，不然会一直以 200N 压在槽沿上磨
                unload_f = self.probe_unload_force if self.probe_unload_force > 0.0 \
                    else self.jam_force
                if f_norm > unload_f:
                    self._z_cmd = min(self._z_cmd + 2.5e-4, self._z_search_top)
                # 试探必须限制在座位附近一个小范围内，否则随机游走会越走越远
                lim = max(self.search_radius, self.explore_limit)
                self._xy_goal = np.clip(self._xy_goal, self._xy_nominal - lim,
                                        self._xy_nominal + lim)
                # 护栏：弹夹已经被探出槽口（xy 偏差超过 explore_limit）就停试探，免得越推越远
                if float(np.linalg.norm(err[:2])) > self.explore_limit:
                    self._explore_episode = False
                    self._search_used = True
            cmd = np.array([self._xy_goal[0] + self._xy_noise_off[0] + self._bounded_dither_offset[0],
                            self._xy_goal[1] + self._xy_noise_off[1] + self._bounded_dither_offset[1], self._z_cmd])
            if entry and self.search_mode == "random":
                limit = max(self.search_radius, self.explore_limit)
                cmd[:2] = np.clip(cmd[:2], self._xy_nominal-limit, self._xy_nominal+limit)
            # 落座判定只看 z：xy 残余 2mm 时若仍要求 |xy|<1mm，就会一直压着不松手，
            # 把力从 ~60N 顶到 300~400N（实测）。xy 留给松手阶段处理。
            seated = abs(err[2]) < self.seat_z_tol
            if seated and self.seat_xy_tol > 0.0:
                seated = float(np.linalg.norm(err[:2])) < self.seat_xy_tol
            if seated:
                self._cart_hold += 1
                cmd[2] = float(ee[2])       # 到位后不再下压
                if self._cart_hold >= self.seat_hold:
                    self.handover = {"trigger": "seat",
                                     "z_err_mm": float(err[2] * 1000.0),
                                     "xy_err_mm": float(np.linalg.norm(err[:2]) * 1000.0),
                                     "frames": int(self._cart_frames)}
                    if (not self._seat_shift_done
                            and float(np.linalg.norm(self.seat_shift)) > 1e-9):
                        self._seat_shift_done = True
                        self.phase = "seat_shift"
                        self.k = 0
                        self._shift_from = np.asarray(ee, dtype=float).copy()
                        return {"ee_pos": self._shift_from,
                                "grip_targets": env.grip_hold_positions()}, False
                    self.phase = "unload"
                    self.k = 0
                    self._ee_seat = ee.copy()
                    self._hold_cmd = self._hold_grip(env, obs["joint_positions"])
                    return {"ee_pos": cmd, "grip_targets": env.grip_hold_positions()}, False
            if self._z_cmd <= self._z_goal + 1e-5 and abs(err[2]) > 2e-3:
                # z 到底但弹夹没跟上：被顶住了，保持 xy 归中并原地等
                cmd[2] = float(ee[2])
            if (not seated and self._cart_frames >= 240
                    and abs(err[2]) < 2.0e-3):
                # 兜底：xy 收不到 1mm 也别无限等，z 到位就进入卸力松手
                self.handover = {"trigger": "seat_z_only",
                                 "z_err_mm": float(err[2] * 1000.0),
                                 "xy_err_mm": float(np.linalg.norm(err[:2]) * 1000.0),
                                 "frames": int(self._cart_frames)}
                if (not self._seat_shift_done
                        and float(np.linalg.norm(self.seat_shift)) > 1e-9):
                    self._seat_shift_done = True
                    self.phase = "seat_shift"
                    self.k = 0
                    self._shift_from = np.asarray(ee, dtype=float).copy()
                    return {"ee_pos": self._shift_from,
                            "grip_targets": env.grip_hold_positions()}, False
                self.phase = "unload"
                self.k = 0
                self._ee_seat = ee.copy()
                self._hold_cmd = self._hold_grip(env, obs["joint_positions"])
                return {"ee_pos": cmd, "grip_targets": env.grip_hold_positions()}, False
            if (not seated and self.cart_timeout > 0
                    and self._cart_frames >= self.cart_timeout
                    and (self._progress_age_frames >= 180 or self._cart_frames >= 1800)
                    and abs(err[2]) > 3.0e-3):
                # 卡在槽沿上方死活不下去：别耗满 max_steps，直接松手让状态判定收尾。
                # 注意不要置 failed：超时后弹夹可能在松手时自己落座，硬判失败会误杀好样本
                self.reason = f"cart_timeout z_err={err[2] * 1000:.0f}mm"
                self.handover = {"trigger": "timeout",
                                 "z_err_mm": float(err[2] * 1000.0),
                                 "xy_err_mm": float(np.linalg.norm(err[:2]) * 1000.0),
                                 "frames": int(self._cart_frames)}
                self.phase = "unload"
                self.k = 0
                self._ee_seat = ee.copy()
                self._hold_cmd = self._hold_grip(env, obs["joint_positions"])
                return {"ee_pos": cmd, "grip_targets": env.grip_hold_positions()}, False
            return {"ee_pos": cmd, "grip_targets": env.grip_hold_positions()}, False

        if self.phase == "trim":
            # 落座前把弹夹 xy 误差收一收（每次最多 0.1mm，最多 trim_limit）
            self.k += 1
            obj = np.asarray(obs["retaining_clip_pos"], dtype=float)
            tgt = env.retaining_clip_target()
            err = tgt - obj
            if float(np.linalg.norm(err)) < self.trim_tol:
                self.phase = "unload"        # 已经到位：卸预载后松手
                self.k = 0
                self._ee_seat = np.asarray(obs["ee_pos"], dtype=float).copy()
                return {"ee_pos": np.asarray(obs["ee_pos"], dtype=float),
                        "grip_targets": env.grip_hold_positions()}, False
            err[2] = 0.0
            self._trim = self._trim + np.clip(err - self._trim, -1.5e-4, 1.5e-4)
            self._trim = np.clip(self._trim, -self.trim_limit, self.trim_limit)
            ee = np.asarray(obs["ee_pos"], dtype=float)
            ee_tgt = ee + self._trim
            if self.k >= self.trim_frames:
                self.phase = "unload"
                self._ee_seat = np.asarray(obs["ee_pos"], dtype=float).copy()
                self.k = 0
            return {"ee_pos": ee_tgt, "grip_targets": env.grip_hold_positions()}, False

        if self.phase == "rim_regrip":
            self.k += 1
            n = self.rim_regrip_frames
            third = max(6, n // 3)
            p0 = self._regrip_from
            g_hold = env.grip_hold_positions()
            g_open = {nn: (-self.release_open if nn.endswith("zjb_Joint")
                           else self.release_open) for nn in g_hold}
            cmd = p0.copy()
            if self.k <= third:                      # 张开 + 抬 5mm
                sfrac = self.k / float(third)
                grip = {nn: g_hold[nn] + (g_open[nn] - g_hold[nn]) * sfrac for nn in g_hold}
                cmd[2] = p0[2] + 0.005 * sfrac
            elif self.k <= 2 * third:                # 落下 10mm，让指头包住弹夹更深
                sfrac = (self.k - third) / float(third)
                grip = dict(g_open)
                cmd[2] = p0[2] + 0.005 - 0.010 * sfrac
            else:                                    # 重新夹紧
                sfrac = (self.k - 2 * third) / float(max(1, n - 2 * third))
                grip = {nn: g_open[nn] + (g_hold[nn] - g_open[nn]) * min(1.0, sfrac)
                        for nn in g_hold}
                cmd[2] = p0[2] - 0.005
            if self.k >= n:
                self.phase = "cart_descend"
                self.k = 0
                self._z_cmd = float(cmd[2])
                self._z_search_top = None
                self._f_filt = None
                return {"ee_pos": cmd, "grip_targets": grip}, False
            return {"ee_pos": cmd, "grip_targets": grip}, False

        if self.phase == "seat_shift":
            # 还夹着弹夹时把末端平移 seat_shift（补偿开爪时的横推），再进入卸力松手
            self.k += 1
            sfrac = min(1.0, self.k / float(self.seat_shift_frames))
            cmd = self._shift_from.copy()
            cmd[:2] += self.seat_shift * sfrac
            if self.k >= self.seat_shift_frames + 6:
                self.phase = "unload"
                self.k = 0
                self._ee_seat = cmd.copy()
                self._hold_cmd = self._hold_grip(env, obs["joint_positions"])
            return {"ee_pos": cmd, "grip_targets": env.grip_hold_positions()}, False

        if self.phase == "unload":
            # 第一段：把夹爪从"全闭"退到 USD 预紧，缓慢卸掉过盈力；此时仍夹着弹夹。
            if self.k == 0 and self.release_max_z_err > 0.0:
                # 质量门：松手瞬间弹夹还悬在座位上方（卡在槽沿被超时放弃）→ 直接判失败。
                # 否则会出现"从 2~5cm 高处掉进槽里"的假成功，视频里就是松手后剧烈弹跳。
                ez = float(np.asarray(obs["retaining_clip_pos"], dtype=float)[2]
                           - env.retaining_clip_target()[2])
                if abs(ez) > self.release_max_z_err:
                    self.failed = True
                    self.reason = f"release_high z={ez * 1000:.0f}mm"
            self.k += 1
            if self._grip_unload0 is None:
                self._grip_unload0 = env.grip_command()
            target = env.grip_release_positions()
            s = min(1.0, self.k / float(self.unload_frames))
            grip = {n: a + (target[n] - a) * s for n, a in self._grip_unload0.items() if n in target}
            ee_tgt = self._ee_seat.copy()
            ee_tgt[2] += self.unload_lift * min(1.0, self.k / float(self.unload_frames))
            ee_tgt += np.asarray(env.release_bias, dtype=float) * s
            # 卸力阶段就开始按弹夹 xy 误差引导（弹跳从退回预紧时就已经开始）
            err_xy = (np.asarray(env.retaining_clip_target(), dtype=float)[:2]
                      - np.asarray(obs["retaining_clip_pos"], dtype=float)[:2])
            self._rel_xy = self._guide_xy(err_xy)
            ee_tgt[:2] += self._rel_xy + self._gate_bias
            if self.k >= self.unload_frames + self.unload_hold:
                self.phase = "release"
                self.k = 0
                self._grip_release0 = dict(target)
                self._hold_cmd = np.asarray(obs["joint_positions"], dtype=float).copy()
                env.detach_retaining_clip()
            return {"ee_pos": ee_tgt, "grip_targets": grip}, False

        if self.phase == "readjust":
            # 松手后落点差 0.5~8mm：重新夹住 → 平移到目标 → 再松开。
            # 单次只约一半概率落到 <0.5mm，所以按 readjust_max 重复，每次都用
            # 当前实测误差重新瞄准（失败的那次会把弹夹甩到 ~8mm，下一轮再拉回来）。
            CY = 170
            self.k += 1
            cyc = (self.k - 1) // CY
            k = (self.k - 1) % CY + 1
            if k == 1:
                err_now = np.asarray(env.retaining_clip_target(), dtype=float) - \
                    np.asarray(obs["retaining_clip_pos"], dtype=float)
                err_now[2] = 0.0
                self._readjust_err = np.clip(err_now, -0.012, 0.012)
                self._readjust_ee = np.asarray(obs["ee_pos"], dtype=float).copy()
                # 关键：松手后末端已抬 release_lift(10~12mm)，在这个高度闭爪只会抓到
                # 弹夹上沿并把它带起来（实测失败样本 fin=(±5,0,−10.5)）。
                # 重夹必须回到落座时的夹持高度。
                if self._ee_seat is not None:
                    self._readjust_ee[2] = float(self._ee_seat[2])
            g_hold = env.grip_hold_positions()
            g_open = {n: (-self.release_open if n.endswith("zjb_Joint") else self.release_open)
                      for n in g_hold}
            ee = self._readjust_ee.copy()
            if k <= 30:
                s = k / 30.0
                grip = {n: g_open[n] + (g_hold[n] - g_open[n]) * s for n in g_hold}
            elif k <= 50:
                grip = dict(g_hold)
            elif k <= 80:
                s = (k - 50) / 30.0
                grip = dict(g_hold)
                ee = self._readjust_ee + self._readjust_err * s
            elif k <= 110:
                s = (k - 80) / 30.0
                grip = {n: g_hold[n] + (g_open[n] - g_hold[n]) * s for n in g_hold}
                ee = self._readjust_ee + self._readjust_err
            else:
                grip = dict(g_open)
                ee = self._readjust_ee + self._readjust_err
            done = (k >= CY) and (cyc + 1 >= self.readjust_max)
            if done:
                self.done = True
            return {"ee_pos": ee, "grip_targets": grip}, done

        # release
        self.k += 1
        if self.k == 1 and self.release_gate_force > 0.0:
            fn0 = float(np.linalg.norm(
                np.asarray(obs.get("force", np.zeros(3)), dtype=float)[:2]))
            if fn0 > self.release_gate_force:
                self._gate_bias = self.release_gate_extra.copy()
        s = min(1.0, self.k / float(max(1, self.release_frames)))
        g0 = self._grip_release0 or env.grip_hold_positions()
        grip = {}
        # 分段松手：前 release_pre_frames 帧只张开 release_pre_frac 的行程（解除挤压）
        npf = self.release_pre_frames
        if npf > 0 and self.k <= npf:
            prog = self.release_pre_frac * (self.k / float(npf))
        elif npf > 0:
            rest = max(1, self.release_frames - npf)
            prog = self.release_pre_frac + (1.0 - self.release_pre_frac) * \
                min(1.0, (self.k - npf) / float(rest))
        else:
            prog = min(1.0, self.k / float(max(1, self.release_frames)))
        if self.release_mode in ("sequential", "sequential_z"):
            # 先松 yjb，再松 zjb；两指同时回缩会把弹夹挤偏
            half = max(1, self.release_frames // 2)
            s1 = min(1.0, max(0.0, prog * self.release_frames) / float(half))
            s2 = min(1.0, max(0.0, prog * self.release_frames - half) / float(half))
            for n, a in g0.items():
                first = (n.endswith("yjb_Joint") if self.release_mode == "sequential"
                         else n.endswith("zjb_Joint"))
                g_end = -self.release_open if n.endswith("zjb_Joint") else self.release_open
                if self.release_retract > 0.0:
                    sg = 1.0 if n.endswith("zjb_Joint") else -1.0
                    g_end = a - sg * self.release_retract
                grip[n] = a + (g_end - a) * (s1 if first else s2)
        else:
            for n in ("left_gripper_zjb_Joint", "left_gripper_yjb_Joint"):
                if n not in g0:
                    continue
                g_end = -self.release_open if n.endswith("zjb_Joint") else self.release_open
                if self.release_retract > 0.0:
                    sg = 1.0 if n.endswith("zjb_Joint") else -1.0
                    g_end = g0[n] - sg * self.release_retract
                grip[n] = g0[n] + (g_end - g0[n]) * s
        # 松手阶段保持"落座时"的关节姿态，只开爪；否则会继续下压把弹夹挤走
        if getattr(self, "_hold_cmd", None) is None:
            self._hold_cmd = np.asarray(obs["joint_positions"], dtype=float).copy()
        cmd = self._hold_cmd.copy()
        for name, v in grip.items():
            if name in env.dof_names:
                cmd[env.dof_names.index(name)] = v
        # 引导释放：开爪过程中按弹夹 xy 误差微调末端（指尖仍有接触时能把弹夹按回中）。
        # 张开完成后必须停止引导、原地保持，否则最后一次修正会把弹夹带偏。
        if self._ee_seat is not None and self.k <= self.release_frames:
            err = np.asarray(env.retaining_clip_target(), dtype=float)[:2] - \
                np.asarray(obs["retaining_clip_pos"], dtype=float)[:2]
            self._rel_xy = self._guide_xy(err)
            ee_tgt = np.asarray(self._ee_seat, dtype=float).copy()
            ee_tgt[:2] += self._rel_xy
            # 边开爪边抬离：指头不再沿弹夹侧面持续蹭，避免把弹夹从槽里挤出去/弹起
            s_rel = self.k / float(max(1, self.release_frames))
            d = self.release_lift_delay
            s_lift = 0.0 if s_rel <= d else min(1.0, (s_rel - d) / max(1e-6, 1.0 - d))
            ee_tgt[2] += self.release_lift * s_lift
            if self.release_pre_lift > 0.0:
                ee_tgt[2] += self.release_pre_lift * min(
                    1.0, self.k / float(self.release_pre_lift_frames))
            if self.release_force_gain > 0.0:
                f3 = np.asarray(obs.get("force", np.zeros(3)), dtype=float)
                fn = float(np.linalg.norm(f3[:2]))
                if (self.release_force_thr > 0.0
                        and float(np.linalg.norm(self.release_force_dir)) > 1e-9):
                    # 门控：力大说明指头在拖着弹夹走，按超出量沿标定方向补一段（不积分）
                    over = max(0.0, fn - self.release_force_thr)
                    extra = min(self.release_force_limit,
                                self.release_force_gain * over)
                    if extra > 0.0:
                        d = self.release_force_dir / np.linalg.norm(self.release_force_dir)
                        self._rel_f = np.clip(d * extra,
                                              -self.release_force_limit,
                                              self.release_force_limit)
                elif fn >= self.release_force_min:
                    self._rel_f = np.clip(
                        self._rel_f + self.release_force_sign * self.release_force_gain
                        * f3[:2], -self.release_force_limit, self.release_force_limit)
                ee_tgt[:2] += self._rel_f
            ee_tgt += np.asarray(env.release_bias, dtype=float)
            return {"ee_pos": ee_tgt, "grip_targets": grip}, False
        finished = self.k >= self.release_frames + self.settle_frames + self.hold_frames
        if (self.readjust and finished and not self._readjusted
                and getattr(self, "_readjust_count", 0) < self.readjust_max):
            err_now = np.asarray(env.retaining_clip_target(), dtype=float) - \
                np.asarray(obs["retaining_clip_pos"], dtype=float)
            if self.readjust_min_err < float(np.linalg.norm(err_now[:2])) < 8e-3:
                self._readjust_count = getattr(self, "_readjust_count", 0) + 1
                self._readjusted = True
                self.phase = "readjust"
                self.k = 0
                self._readjust_err = np.asarray(err_now, dtype=float)
                self._readjust_ee = np.asarray(self._ee_seat, dtype=float).copy()
                return {"ee_pos": self._readjust_ee,
                        "grip_targets": dict(grip)}, False
        if finished:
            self.done = True
        if self._ee_seat is not None:
            hold = np.asarray(self._ee_seat, dtype=float).copy()
            hold[:2] += self._rel_xy if self._rel_xy is not None else 0.0
            hold[2] += self.release_lift
            if self.release_exit_lift > 0.0:
                hold[2] += self.release_exit_lift * min(
                    1.0, max(0, self.k - self.release_frames)
                    / float(self.release_exit_frames))
            hold += np.asarray(env.release_bias, dtype=float)
            return {"ee_pos": hold,
                    "grip_targets": grip}, finished
        # 不做"松手后立刻判失败"：最后一个动作执行完状态才会更新，
        # 由数据生成器按回合结束时的 env.status() 统一判定。
        return {"joint_positions": cmd}, finished

class ScriptedInsertController(BaseController):
    """上帝视角脚本策略：先水平回到初始 xy（z 不变），再匀速下降到底，最后松爪。

    下降过程：
      * 侧向力超过 lat_force_thresh 时，按 lat_gain * F_xy 做一点横向柔顺；
      * 合力超过 fail_force 判定异常，立刻松爪并标记失败。
    """

    name = "scripted_insert"

    def __init__(self, descend_step=0.00075, xy_speed=0.0015, lat_gain=0.0,
                 lat_force_thresh=12.0, lat_limit=0.003, fail_force=150.0,
                 home_frames=25,
                 release_open=0.0, release_frames=20, hold_frames=6,
                 settle_frames=60, grasp_hold=-99.0):
        self.descend_step = float(descend_step)
        self.xy_speed = float(xy_speed)
        self.lat_gain = float(lat_gain)
        self.home_frames = int(home_frames)
        self.lat_force_thresh = float(lat_force_thresh)
        self.lat_limit = float(lat_limit)
        self.fail_force = float(fail_force)
        self.release_open = float(release_open)
        self.release_frames = int(release_frames)
        self.hold_frames = int(hold_frames)
        self.settle_frames = int(settle_frames)
        self.grasp_hold = float(grasp_hold)
        self.reset(None)

    def reset(self, env):
        self.phase = "grip_settle"
        self.lat = np.zeros(3)
        self.k = 0
        self.done = False
        self.failed = False
        self.reason = ""
        self._q_start = None
        self._base_xy = None
        self._z_goal = None
        self._descend_k = 0

    @staticmethod
    def _nominal_xy(env):
        return env.ee_init_rel[:2]

    def _grip_action(self, env, ramp=1.0):
        """夹持动作：显式给 grasp_hold 时用它；否则从复位值缓升到强夹持。"""
        if self.grasp_hold > -90:
            return {"opening": self.grasp_hold}
        g0 = env.grip_initial_positions()
        g1 = env.grip_hold_positions()
        s = min(1.0, float(ramp))
        return {"grip_targets": {n: g0.get(n, v) + (v - g0.get(n, v)) * s
                                 for n, v in g1.items()}}

    def act(self, env, obs):
        if self.done:
            return None, True
        ee = np.asarray(obs["ee_pos"], dtype=float)
        force = np.asarray(obs.get("force", np.zeros(3)), dtype=float)
        f_norm = float(np.linalg.norm(force))
        if self.phase == "grip_settle":
            env.capture_grip_hold()
            self.k += 1
            s = min(1.0, self.k / 20.0)
            g0 = env.grip_initial_positions()
            g1 = env.grip_hold_positions()
            cmd = np.asarray(obs["joint_positions"], dtype=float).copy()
            for n in g1:
                if n in env.dof_names:
                    cmd[env.dof_names.index(n)] = g0.get(n, g1[n]) + (g1[n] - g0.get(n, g1[n])) * s
            if self.k >= 40:
                self.phase = "return_home"
                self.k = 0
            return {"joint_positions": cmd}, False
        if self.phase == "return_home":
            # 关节空间回位：直接回到记录的标准抓取姿态（弹夹被爪子带着一起回去），
            # 比"按末端 xy 回位"稳，也不会把手臂拧到关节限位死角
            home = env.home_joint_positions()
            if self._q_start is None:
                self._q_start = np.asarray(obs["joint_positions"], dtype=float)
            self.k += 1
            s = min(1.0, self.k / float(max(1, self.home_frames)))
            cmd = self._q_start + (home - self._q_start) * (0.5 - 0.5 * np.cos(np.pi * s))
            cmd = np.asarray(cmd, dtype=float)
            # 回位阶段沿复位实测值；下降段再缓升到强夹持
            if self.grasp_hold > -90:
                for n in ("left_gripper_zjb_Joint", "left_gripper_yjb_Joint"):
                    if n in env.dof_names:
                        cmd[env.dof_names.index(n)] = self.grasp_hold if n.endswith("zjb_Joint") \
                            else -self.grasp_hold
            else:
                for n, v in env.grip_initial_positions().items():
                    if n in env.dof_names:
                        cmd[env.dof_names.index(n)] = float(v)
            if s >= 1.0 and self.k > self.home_frames + 3:
                self.phase = "descend"
                self._base_xy = ee[:2].copy()
                # 弹夹在末端下方（grasp_rel.z 为负），目标末端 z = 目标弹夹 z - grasp_rel.z
                self._z_goal = float(env.retaining_clip_target()[2] - env.grasp_rel[2])
                self._z_cmd = float(ee[2])
                self.k = 0
            return {"joint_positions": cmd}, False
        if self.phase == "descend":
            self._descend_k += 1
            if f_norm > self.fail_force:
                self.failed = True
                self.reason = f"force {f_norm:.0f}N"
                self.phase = "release"
                self.k = 0
                return {"ee_pos": ee, "opening": 0.0}, False
            if self.lat_gain > 0.0 and float(np.linalg.norm(force[:2])) > self.lat_force_thresh:
                self.lat[:2] = np.clip(self.lat[:2] + self.lat_gain * force[:2],
                                       -self.lat_limit, self.lat_limit)
            # 绝对递降（不能用"实测 z - step"：手臂跟随有滞后时目标会停滞不动）
            self._z_cmd = max(self._z_goal, self._z_cmd - self.descend_step)
            z = self._z_cmd
            # 下降过程中按弹夹 xy 误差做小幅跟随（每帧最多 0.2mm，避免横向猛拽卡住）
            obj_xy = np.asarray(obs["retaining_clip_pos"], dtype=float)[:2]
            want = np.clip(env.retaining_clip_target()[:2] - obj_xy, -self.lat_limit, self.lat_limit)
            self.lat[:2] = self.lat[:2] + np.clip(want - self.lat[:2], -2e-4, 2e-4)
            err_xy = self.lat[:2]
            tgt = np.array([self._base_xy[0] + err_xy[0] + self.lat[0],
                            self._base_xy[1] + err_xy[1] + self.lat[1], z])
            if obs["status"] == "success_insert":
                self.phase = "release"
                self.k = 0
            elif self._z_cmd <= self._z_goal + 1e-9:
                self.k += 1
                if self.k > self.settle_frames:
                    self.phase = "release"
                    self.k = 0
            return {"ee_pos": tgt, **self._grip_action(env, self._descend_k / 20.0)}, False
        if self.phase == "release":
            self.k += 1
            if self.k == 1:
                env.detach_retaining_clip()
            s = min(1.0, self.k / float(self.release_frames))
            grip = float(obs["opening"]) * (1.0 - s) + self.release_open * s
            if self.k >= self.release_frames + self.hold_frames:
                self.done = True
            return {"ee_pos": ee, "opening": grip}, self.done
        self.done = True
        return None, True
