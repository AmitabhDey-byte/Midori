"""Physics-informed behavior cloning for the Midori quadruped.

The model is a temporal command-conditioned policy with auxiliary one-step
dynamics prediction. It applies a vertical Newton/contact residual, so this is
PINN-style physics-informed learning, not a full articulated-body PINN.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

JOINTS = [
    'front_left_hip_abduction', 'front_left_hip_pitch', 'front_left_knee_pitch',
    'front_right_hip_abduction', 'front_right_hip_pitch', 'front_right_knee_pitch',
    'rear_left_hip_abduction', 'rear_left_hip_pitch', 'rear_left_knee_pitch',
    'rear_right_hip_abduction', 'rear_right_hip_pitch', 'rear_right_knee_pitch',
]
LEGS = ['front_left', 'front_right', 'rear_left', 'rear_right']
COMMANDS = ['stop', 'forward', 'backward', 'left', 'right']
TASK_TO_COMMAND = {
    'walk_forward': 'forward', 'walk_backward': 'backward',
    'turn_left': 'left', 'turn_right': 'right',
}
ANGLE_LIMITS_DEG = np.array([[-40.107, 40.107], [-86.0, 60.16], [-91.673, 91.673]] * 4, dtype=np.float32)
FEATURE_COUNT = 54
VEL_COUNT = 18
DT = 0.02
GRAVITY = 9.81


def quat_inverse_rotate(q, v):
    w, x, y, z = q
    u = np.array([-x, -y, -z], dtype=np.float32)
    t = 2.0 * np.cross(u, v)
    return np.asarray(v + w * t + np.cross(u, t), dtype=np.float32)


def quaternion_roll_pitch(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w*y - z*x))))
    return roll, pitch


def is_teacher_label(row, metadata=None):
    """Resolve the target source without confusing it with the executed controller."""
    metadata = metadata or {}
    source = (row.get('target_label_source') or row.get('target_source')
              or metadata.get('target_label_source') or metadata.get('action_label_source'))
    if source:
        source = str(source).strip().lower().replace('-', '_').replace(' ', '_')
        return source in {
            'teacher', 'expert', 'scripted_teacher', 'scripted_teacher_targets',
            'teacher_targets_logged_for_policy_visited_states',
        }


    applied = str(row.get('applied_controller') or '').strip().lower()
    if applied:
        return applied in {'teacher', 'scripted_teacher'}
    return True


def read_episode(csv_path: Path, metadata=None):
    with csv_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r['step_index']))
    xs, ys, vs, contacts, zvel, target_valid = [], [], [], [], [], []
    for r in rows:
        command = str(r['movement_mode']).lower()
        if command not in COMMANDS:
            continue
        q = np.array([float(r[f'root_quat_{a}']) for a in 'wxyz'], dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-8)
        roll, pitch = quaternion_roll_pitch(q)
        actual = np.array([float(r[f'actual_{j}_deg']) for j in JOINTS], dtype=np.float32)
        joint_vel = np.array([float(r[f'actual_{j}_vel_degps']) for j in JOINTS], dtype=np.float32)
        actual_norm = 2 * (actual - ANGLE_LIMITS_DEG[:, 0]) / (ANGLE_LIMITS_DEG[:, 1] - ANGLE_LIMITS_DEG[:, 0]) - 1
        root = np.array([float(r[f'root_{a}_m']) for a in 'xyz'], dtype=np.float32)
        feet = []
        for leg in LEGS:
            foot = np.array([float(r[f'foot_{leg}_{a}_m']) for a in 'xyz'], dtype=np.float32)
            feet.extend(quat_inverse_rotate(q, foot - root))
        command_onehot = np.eye(len(COMMANDS), dtype=np.float32)[COMMANDS.index(command)]
        body = np.array([float(r['root_z_m']), roll, pitch,
                         float(r['root_vx_mps']), float(r['root_vy_mps']), float(r['root_vz_mps']),
                         float(r['root_wx_radps']), float(r['root_wy_radps']), float(r['root_wz_radps'])], dtype=np.float32)
        contact = np.array([float(r[f'foot_{leg}_contact']) for leg in LEGS], dtype=np.float32)
        xs.append(np.concatenate([command_onehot, actual_norm, joint_vel / 360.0, body,
                                  np.asarray(feet, dtype=np.float32), contact]))
        target = np.array([float(r[f'target_{j}_deg']) for j in JOINTS], dtype=np.float32)
        ys.append(np.clip(2 * (target - ANGLE_LIMITS_DEG[:, 0]) /
                          (ANGLE_LIMITS_DEG[:, 1] - ANGLE_LIMITS_DEG[:, 0]) - 1, -1, 1))
        target_valid.append(is_teacher_label(r, metadata))

        vs.append(np.concatenate([
            np.array([float(r[f'root_v{a}_mps']) for a in 'xyz'], dtype=np.float32),
            np.array([float(r[f'root_w{a}_radps']) for a in 'xyz'], dtype=np.float32),
            np.deg2rad(joint_vel),
        ]))
        contacts.append(contact)
        zvel.append(float(r['root_vz_mps']))
    x, y, v = map(lambda a: np.asarray(a, dtype=np.float32), (xs, ys, vs))
    return {'x': x, 'y': y, 'target_label_valid': np.asarray(target_valid, dtype=bool),
            'v': v, 'contact': np.asarray(contacts, dtype=np.float32),
            'zvel': np.asarray(zvel, dtype=np.float32)}


def split_episodes(episodes, seed=41):
    rng = random.Random(seed)
    by_task = {c: [] for c in COMMANDS}
    for ep in episodes:
        task = TASK_TO_COMMAND.get(ep['task'], ep['task'])
        if task in by_task:
            by_task[task].append(ep)
    train, val, test = [], [], []
    for task_eps in by_task.values():
        rng.shuffle(task_eps)
        n = len(task_eps)
        nt = max(1, round(n * .15)) if n >= 5 else max(0, round(n * .15))
        nv = max(1, round(n * .15)) if n >= 5 else max(0, round(n * .15))
        if n - nt - nv < 1 and n:
            nt, nv = 1, 0
        train.extend(task_eps[:n-nt-nv])
        val.extend(task_eps[n-nt-nv:n-nt])
        test.extend(task_eps[n-nt:])
    return train, val, test


class TemporalPhysicsPolicy(nn.Module):
    """4-layer, width-512 transformer policy plus next-velocity/contact head."""
    def __init__(self, input_size=FEATURE_COUNT, context=64, width=512, heads=8,
                 layers=4, ff=1344, action_size=12):
        super().__init__()
        self.context = context
        self.input = nn.Linear(input_size, width)
        self.position = nn.Parameter(torch.zeros(1, context, width))
        block = nn.TransformerEncoderLayer(width, heads, ff, dropout=.10,
                                           activation='gelu', batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.action_head = nn.Sequential(nn.Linear(width, 512), nn.GELU(),
                                         nn.Linear(512, action_size), nn.Tanh())

        self.dynamics_head = nn.Sequential(nn.Linear(width + action_size, 512), nn.GELU(),
                                           nn.Linear(512, VEL_COUNT + 4))

    def forward(self, x):
        h = self.input(x) + self.position[:, -x.shape[1]:]
        h = self.norm(self.encoder(h))
        actions = self.action_head(h)
        action = actions[:, -1]
        dynamics = self.dynamics_head(torch.cat([h[:, -1], action], dim=-1))
        action_delta = actions[:, -1] - actions[:, -2]
        return action, dynamics[:, :VEL_COUNT], torch.nn.functional.softplus(dynamics[:, VEL_COUNT:]), action_delta


class WindowDataset(Dataset):
    def __init__(self, episodes, mean, std, context):
        self.context, self.mean, self.std = context, mean, std
        self.episodes = []
        self.indices = []
        for ep_i, ep in enumerate(episodes):
            x = ((ep['x'] - mean) / std).astype(np.float32)
            self.episodes.append(x)
            target_valid = ep.get('target_label_valid', np.ones(len(x), dtype=bool))
            for t in range(len(x) - 1):
                previous = max(t - 1, 0)
                if target_valid[t] and target_valid[previous]:
                    self.indices.append((ep_i, t))

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        ep_i, t = self.indices[i]
        ep = self.episodes[ep_i]
        start = max(0, t - self.context + 1)
        seq = ep[start:t+1]
        if len(seq) < self.context:
            seq = np.concatenate([np.repeat(seq[:1], self.context-len(seq), axis=0), seq])
        e = self.episodes_data[ep_i] if hasattr(self, 'episodes_data') else None

        return (torch.from_numpy(seq), torch.from_numpy(self.targets[ep_i][t]),
                torch.from_numpy(self.targets[ep_i][max(t-1, 0)]),
                torch.from_numpy(self.velocities[ep_i][t+1]),
                torch.from_numpy(self.velocities[ep_i][t]),
                torch.from_numpy(self.contacts[ep_i][t]),
                torch.tensor(self.zvels[ep_i][t], dtype=torch.float32))


def attach_labels(dataset, episodes):
    dataset.targets = [e['y'] for e in episodes]
    dataset.velocities = [e['v'] for e in episodes]
    dataset.contacts = [e['contact'] for e in episodes]
    dataset.zvels = [e['zvel'] for e in episodes]
    dataset.episodes_data = episodes
    return dataset


def physics_loss(vnext_pred, force_pred, vnow, contact, znow, body_mass):


    supported_force = (force_pred * contact).sum(dim=1)
    az_pred = (vnext_pred[:, 2] - vnow[:, 2]) / DT
    residual = (az_pred - (supported_force / body_mass - GRAVITY)) / GRAVITY
    return torch.mean(residual.square())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('midori-output'))
    ap.add_argument('--mass-kg', type=float, default=0.828,
                    help='total simulated robot mass; replace with MuJoCo model body_mass.sum()')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()
    seed = 41
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device('cuda:0' if gpu_count else 'cpu')
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    parallel_backend = 'torch.nn.DataParallel' if gpu_count > 1 else 'single_device'
    if gpu_count:
        torch.cuda.set_device(0)
    print(f'Device: {device}; visible CUDA GPUs: {gpu_count}; backend: {parallel_backend}')
    for gpu_index, gpu_name in enumerate(gpu_names):
        print(f'  cuda:{gpu_index}: {gpu_name}')
    if gpu_count == 1:
        print('Warning: only one GPU is visible. Select the Kaggle T4 x2 accelerator to use both GPUs.')
    episodes = []
    audit = {'folders': 0, 'accepted_success': 0, 'accepted_failed_or_other': 0,
             'missing_csv': 0, 'wrong_gait': 0, 'teacher_label_rows': 0,
             'ignored_non_teacher_label_rows': 0, 'tasks': {}}
    metadata_files = set(args.episodes.rglob('episode.json'))
    metadata_files.update(args.episodes.rglob('*_episode.json'))
    for meta_path in sorted(metadata_files):
        audit['folders'] += 1
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('gait_controller_version') != 'stance_transition_sequential_crawl_v3':
            audit['wrong_gait'] += 1; continue
        if meta_path.name.endswith('_episode.json'):
            episode_prefix = meta_path.name[:-len('_episode.json')]
            csv_path = meta_path.with_name(f'{episode_prefix}_state_action.csv')
        else:
            csv_path = meta_path.parent / 'state_action.csv'
        if not csv_path.exists():
            audit['missing_csv'] += 1; continue
        outcome = meta.get('outcome', meta.get('status', 'unknown'))
        if outcome != 'success':
            audit['accepted_failed_or_other'] += 1
            continue


        task = TASK_TO_COMMAND.get(meta.get('task', ''), meta.get('task', ''))
        if task not in COMMANDS:
            continue
        d = read_episode(csv_path, meta)
        d.update({'id': meta.get('episode_id', meta_path.parent.name), 'task': task})
        if len(d['x']) < 2:
            continue
        teacher_rows = int(d['target_label_valid'].sum())
        audit['teacher_label_rows'] += teacher_rows
        audit['ignored_non_teacher_label_rows'] += len(d['x']) - teacher_rows
        if teacher_rows < 2:
            continue
        episodes.append(d)
        audit['accepted_success'] += 1
        audit['tasks'][task] = audit['tasks'].get(task, 0) + 1
    if len(episodes) < 12:
        raise RuntimeError(f'Only {len(episodes)} usable success episodes found. Audit: {audit}')
    train_eps, val_eps, test_eps = split_episodes(episodes, seed)
    x_train = np.concatenate([e['x'] for e in train_eps])
    mean = x_train.mean(0).astype(np.float32)
    std = np.maximum(x_train.std(0), 1e-4).astype(np.float32)
    def make_loader(eps, shuffle):
        d = attach_labels(WindowDataset(eps, mean, std, 64), eps)
        if shuffle:

            command_counts = np.zeros(len(COMMANDS), dtype=np.int64)
            for ep_i, t in d.indices:
                command = int(np.argmax(eps[ep_i]['x'][t, :len(COMMANDS)]))
                command_counts[command] += 1
            weights = []
            for ep_i, t in d.indices:
                command = int(np.argmax(eps[ep_i]['x'][t, :len(COMMANDS)]))
                weights.append(1.0 / max(1, int(command_counts[command])))
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(d, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        return DataLoader(d, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_loader, val_loader, test_loader = (make_loader(train_eps, True),
                                              make_loader(val_eps, False),
                                              make_loader(test_eps, False))
    if not len(train_loader.dataset) or not len(val_loader.dataset) or not len(test_loader.dataset):
        raise RuntimeError('Teacher-tag filtering left an empty train, validation, or test window set.')
    model = TemporalPhysicsPolicy().to(device)
    if gpu_count > 1:


        model = nn.DataParallel(model, device_ids=list(range(gpu_count)), output_device=0)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Usable successful episodes: {len(episodes)}; train/val/test: '
          f'{len(train_eps)}/{len(val_eps)}/{len(test_eps)}')
    print('Task counts:', audit['tasks'], '| audit:', audit)
    print(f'Parameters: {params:,} ({params/1e6:.3f}M); body mass: {args.mass_kg:.3f} kg')
    print(f'Global batch size: {args.batch_size}; GPUs: {gpu_count}; backend: {parallel_backend}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    best, stale, history = float('inf'), 0, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_losses = []
        for xb, yb, yprev, vnext, vnow, contact, znow in train_loader:
            xb, yb, yprev = xb.to(device), yb.to(device), yprev.to(device)
            vnext, vnow = vnext.to(device), vnow.to(device)
            contact, znow = contact.to(device), znow.to(device)
            optimizer.zero_grad(set_to_none=True)

            xb = xb.clone()
            xb[:, :, 5:50] += 0.01 * torch.randn_like(xb[:, :, 5:50])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                action, pred_v, pred_force, action_delta = model(xb)
                loss_bc = nn.functional.smooth_l1_loss(action, yb)
                loss_dyn = nn.functional.smooth_l1_loss(pred_v, vnext, beta=0.5)
                loss_phys = physics_loss(pred_v, pred_force, vnow, contact, znow, args.mass_kg)

                loss_smooth = nn.functional.smooth_l1_loss(action_delta, yb - yprev)
                loss = loss_bc + .20 * loss_dyn + .10 * loss_phys + .005 * loss_smooth
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); epoch_losses.append(float(loss.detach()))
        model.eval(); vals=[]
        with torch.no_grad():
            for xb, yb, yprev, vnext, vnow, contact, znow in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vnext, vnow, contact = vnext.to(device), vnow.to(device), contact.to(device)
                action, pred_v, pred_force, action_delta = model(xb)
                vals.append(float((nn.functional.smooth_l1_loss(action, yb) +
                                   .20*nn.functional.smooth_l1_loss(pred_v, vnext) +
                                   .10*physics_loss(pred_v, pred_force, vnow, contact, znow.to(device), args.mass_kg)).item()))
        val = float(np.mean(vals)); tr = float(np.mean(epoch_losses))
        history.append({'epoch': epoch, 'train_loss': tr, 'val_loss': val})
        print(f'epoch {epoch:03d} train {tr:.5f} val {val:.5f} elapsed {time.time()-started:.0f}s')
        if val < best:
            best, stale = val, 0
            policy_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({'state_dict': policy_model.state_dict(), 'input_mean': mean, 'input_std': std,
                        'angle_limits_deg': ANGLE_LIMITS_DEG, 'joints': JOINTS, 'commands': COMMANDS,
                        'context_steps': 64, 'sample_rate_hz': 50, 'gait_version': 'stance_transition_sequential_crawl_v3',
                        'sample_stride': 1, 'parameter_count': params,
                        'model_config': {'width':512,'heads':8,'layers':4,'ff':1344},
                        'body_mass_kg': args.mass_kg, 'physics': 'vertical_newton_contact_residual',
                        'training_parallel_backend': parallel_backend,
                        'training_gpu_count': gpu_count,
                        'feature_count': FEATURE_COUNT}, output/'best_policy_10m.pt')
        else:
            stale += 1
            if stale >= 10:
                print('Early stopping.'); break
    best_checkpoint = torch.load(output/'best_policy_10m.pt', map_location=device, weights_only=False)
    policy_model = model.module if isinstance(model, nn.DataParallel) else model
    policy_model.load_state_dict(best_checkpoint['state_dict'])
    model.eval(); maes=[]
    with torch.no_grad():
        for xb, yb, *_ in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred, _, _, _ = model(xb)
            scale = torch.as_tensor((ANGLE_LIMITS_DEG[:,1]-ANGLE_LIMITS_DEG[:,0])/2, device=device)
            maes.extend(((pred-yb).abs()*scale).mean(dim=1).cpu().tolist())
    summary = {'dataset_audit': audit, 'episodes_total_success': len(episodes),
               'episodes_train':len(train_eps),'episodes_validation':len(val_eps),'episodes_test':len(test_eps),
               'parameter_count':params,'body_mass_kg':args.mass_kg,'physics_residual':'m*az = sum(contact Fz) - m*g',
               'device':str(device),'gpu_count':gpu_count,'gpu_names':gpu_names,
               'parallel_backend':parallel_backend,'global_batch_size':args.batch_size,
               'test_mean_joint_mae_deg':float(np.mean(maes)),'best_validation_loss':best,'history':history,
               'training_seconds':time.time()-started}
    (output/'training_summary_10m.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved', output/'best_policy_10m.pt', 'and training_summary_10m.json')


if __name__ == '__main__':
    main()
