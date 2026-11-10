"""Measure held-out joint-target error separately for each movement command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from physics_policy_10m import (
    ANGLE_LIMITS_DEG, COMMANDS, JOINTS, TASK_TO_COMMAND, TemporalPhysicsPolicy,
    read_episode, split_episodes,
)


class CommandTestWindows(Dataset):
    def __init__(self, episodes, mean, std, context):
        self.inputs, self.targets, self.commands = [], [], []
        for episode in episodes:
            x = ((episode['x'] - mean) / std).astype(np.float32)
            y = episode['y'].astype(np.float32)
            target_valid = episode.get('target_label_valid', np.ones(len(x), dtype=bool))
            for t in range(len(x) - 1):
                if not target_valid[t] or not target_valid[max(t - 1, 0)]:
                    continue
                start = max(0, t - context + 1)
                seq = x[start:t+1]
                if len(seq) < context:
                    seq = np.concatenate([np.repeat(seq[:1], context-len(seq), axis=0), seq])
                self.inputs.append(seq)
                self.targets.append(y[t])
                self.commands.append(int(np.argmax(episode['x'][t, :len(COMMANDS)])))

    def __len__(self): return len(self.targets)

    def __getitem__(self, index):
        return (torch.from_numpy(self.inputs[index]), torch.from_numpy(self.targets[index]),
                self.commands[index])


def load_episodes(root: Path):
    metadata = set(root.rglob('episode.json'))
    metadata.update(root.rglob('*_episode.json'))
    episodes = []
    audit = {'metadata_files': len(metadata), 'successful_episodes': 0,
             'ignored_non_success': 0, 'missing_csv': 0, 'wrong_gait': 0,
             'teacher_label_rows': 0, 'ignored_non_teacher_label_rows': 0,
             'task_counts': {name: 0 for name in COMMANDS[1:]}}
    for path in sorted(metadata):
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('gait_controller_version') != 'stance_transition_sequential_crawl_v3':
            audit['wrong_gait'] += 1
            continue
        if path.name.endswith('_episode.json'):
            prefix = path.name[:-len('_episode.json')]
            csv_path = path.with_name(prefix + '_state_action.csv')
        else:
            csv_path = path.parent / 'state_action.csv'
        if not csv_path.exists():
            audit['missing_csv'] += 1
            continue
        if meta.get('outcome', meta.get('status')) != 'success':
            audit['ignored_non_success'] += 1
            continue
        task = TASK_TO_COMMAND.get(meta.get('task', ''), meta.get('task', ''))
        if task not in COMMANDS:
            continue
        episode = read_episode(csv_path, meta)
        if len(episode['x']) < 2:
            continue
        teacher_rows = int(episode['target_label_valid'].sum())
        audit['teacher_label_rows'] += teacher_rows
        audit['ignored_non_teacher_label_rows'] += len(episode['x']) - teacher_rows
        if teacher_rows < 2:
            continue
        episode.update({'id': meta.get('episode_id', path.parent.name), 'task': task})
        episodes.append(episode)
        audit['successful_episodes'] += 1
        audit['task_counts'][task] += 1
    return episodes, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('training/best_policy_10m.pt'))
    parser.add_argument('--output', type=Path, default=Path('training/per_command_test_metrics.json'))
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['model_config']
    model = TemporalPhysicsPolicy(input_size=checkpoint['feature_count'],
                                  context=checkpoint['context_steps'],
                                  width=config['width'], heads=config['heads'],
                                  layers=config['layers'], ff=config['ff'],
                                  action_size=len(checkpoint['joints']))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    episodes, audit = load_episodes(args.episodes)
    train, validation, test = split_episodes(episodes, seed=41)
    dataset = CommandTestWindows(test, checkpoint['input_mean'], checkpoint['input_std'],
                                 checkpoint['context_steps'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    errors = {command: [] for command in COMMANDS}
    scales = torch.as_tensor((ANGLE_LIMITS_DEG[:, 1] - ANGLE_LIMITS_DEG[:, 0]) / 2,
                             dtype=torch.float32, device=device)
    with torch.inference_mode():
        offset = 0
        for xb, yb, command_ids in loader:
            xb, yb = xb.to(device), yb.to(device)
            result = model(xb)
            predictions = result[0]
            mae = ((predictions - yb).abs() * scales).cpu().numpy()
            for row, command_id in zip(mae, command_ids.tolist()):
                errors[COMMANDS[command_id]].append(row)
            offset += len(mae)
    report = {
        'checkpoint': str(args.checkpoint.resolve()),
        'dataset_audit': audit,
        'episode_split': {'train': len(train), 'validation': len(validation), 'test': len(test)},
        'test_windows': offset,
        'metrics': {},
        'interpretation': 'Per-command held-out action-target MAE; it measures imitation, not closed-loop stability.',
    }
    for command, rows in errors.items():
        if rows:
            values = np.asarray(rows, dtype=np.float64)
            report['metrics'][command] = {
                'windows': int(len(values)),
                'mean_joint_mae_deg': float(values.mean()),
                'per_joint_mae_deg': {joint: float(v) for joint, v in zip(JOINTS, values.mean(axis=0))},
            }
        else:
            report['metrics'][command] = {'windows': 0, 'mean_joint_mae_deg': None,
                                          'per_joint_mae_deg': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'metrics'}, indent=2))
    for command, metric in report['metrics'].items():
        print(f"{command:8s} windows={metric['windows']:6d} "
              f"target_MAE={metric['mean_joint_mae_deg'] if metric['mean_joint_mae_deg'] is not None else float('nan'):.3f} deg")
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
