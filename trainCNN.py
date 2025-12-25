# operator
# EveryMeis
# Time: 20230710
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

# New imports
from scipy.optimize import minimize
import copy
import optuna

import dataprepare as dp
from model_CNN import LSM_cnn, LSM_cnn_Enhanced
from utils import drawAUC_TwoClass, drawAUC_MultiClass, draw_acc, draw_loss, draw_auc_curve, compute_auc


class PatchDataset(Dataset):
    def __init__(self, padded_features, labels, width, window_size, indices):
        self.padded_features = padded_features
        self.labels = labels
        self.width = width
        self.window_size = window_size
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        flat_idx = int(self.indices[idx])
        row = flat_idx // self.width
        col = flat_idx % self.width
        row_start = row
        col_start = col
        patch = self.padded_features[row_start:row_start + self.window_size,
                                     col_start:col_start + self.window_size, :]
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        label = int(self.labels[row, col])
        return patch_tensor, label


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN Processes on data")
    parser.add_argument("--feature_path",default='C:/Users/Lenovo/Desktop/envirodis/ex/feature', type=str)
    parser.add_argument("--label_path", default='C:/Users/Lenovo/Desktop/envirodis/ex/hazard/hazard.tif', type=str)
    parser.add_argument("--window_size", default=15, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--val_ratio", default=0.3, type=float)
    parser.add_argument("--log_interval", default=0, type=int,
                        help="每隔多少个训练 batch 输出一次进度（0 表示仅在每个 epoch 结束时输出）")
    parser.add_argument("--feature_files", nargs="+",
                        default=[
                            "aspect.tif", "slope.tif", "landuse.tif", "diceng.tif", "curv.tif",
                            "gzx.tif", "water.tif", "road.tif", "dem.tif", "qfd.tif"
                        ],
                        help="需要参与训练的特征文件列表，相对于 feature_path，可自定义顺序。")
    parser.add_argument("--sample_seed", default=42, type=int,
                        help="抽样随机种子，便于复现实验")
    parser.add_argument("--plot_roc", action="store_true",
                        help="训练结束后计算并绘制最终 ROC 曲线（默认关闭以加速训练）")

    # New parameters for regularization and Anderson
    parser.add_argument("--use_regularization", action="store_true",
                        help="启用正则化 (Dropout, BatchNorm, L2 decay)")
    parser.add_argument("--weight_decay", default=0.0005, type=float,
                        help="L2 权重衰减系数 (仅当 use_regularization 启用时有效)")
    parser.add_argument("--use_anderson", action="store_true",
                        help="启用 Anderson 加速")
    parser.add_argument("--anderson_k", default=4, type=int,
                        help="每多少 epoch 应用一次 Anderson 加速")
    parser.add_argument("--anderson_m", default=4, type=int,
                        help="Anderson 加速的历史长度")
    parser.add_argument("--use_bayes_opt", action="store_true",
                        help="启用贝叶斯优化以搜索超参数")
    parser.add_argument("--bayes_trials", default=20, type=int,
                        help="贝叶斯优化的试验次数")
    parser.add_argument("--bayes_epochs", default=20, type=int,
                        help="贝叶斯优化每次试验的训练 epoch 数")
    parser.add_argument("--bayes_max_train", default=5000, type=int,
                        help="贝叶斯优化时每次试验使用的最大训练样本数（0 表示使用全部）")
    parser.add_argument("--use_se_layer", action="store_true",
                        help="启用通道注意力 SE-Layer 以提升弱特征权重")
    parser.add_argument("--neg_pos_ratio", default=3.0, type=float,
                        help="训练集中非滑坡:滑坡样本的目标比例（默认 3:1，<=0 表示不调整）")

    args = parser.parse_args()
    return args


def _build_loader(padded_features, labels, width, window_size, indices,
                  batch_size, shuffle=True, max_samples=0, seed=0):
    selected = indices
    if max_samples and max_samples > 0 and len(indices) > max_samples:
        rng_local = np.random.default_rng(seed)
        selected = np.sort(rng_local.choice(indices, size=max_samples, replace=False))
    dataset = PatchDataset(padded_features, labels, width, window_size, selected)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _balance_class_ratio(indices, label_map, width, target_ratio=5.0, seed=0):
    """
    下采样多数类（非滑坡）以维持 target_ratio 的负:正样本比例。
    约定 label>0 视为滑坡正样本，label==0 视为非滑坡样本。
    """
    if target_ratio is None or target_ratio <= 0:
        return np.asarray(indices, dtype=np.int64), 0, 0

    flat_indices = np.asarray(indices, dtype=np.int64)
    if flat_indices.size == 0:
        return flat_indices, 0, 0

    rows = flat_indices // width
    cols = flat_indices % width
    labels = label_map[rows, cols]

    pos_mask = labels > 0
    pos_indices = flat_indices[pos_mask]
    neg_indices = flat_indices[~pos_mask]

    if pos_indices.size == 0 or neg_indices.size == 0:
        return flat_indices, pos_indices.size, neg_indices.size

    max_neg = int(np.ceil(pos_indices.size * target_ratio))
    rng = np.random.default_rng(seed)
    if neg_indices.size > max_neg:
        neg_indices = rng.choice(neg_indices, size=max_neg, replace=False)

    balanced = np.concatenate([pos_indices, neg_indices])
    rng.shuffle(balanced)
    return balanced.astype(np.int64), pos_indices.size, neg_indices.size


def _sample_validation_indices(indices, desired_size, seed=0):
    """
    随机采样验证集索引以匹配目标大小，保持与训练集 7:3 的样本占比。
    """
    indices = np.asarray(indices, dtype=np.int64)
    if desired_size <= 0 or desired_size >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(indices, size=desired_size, replace=False))
    return selected


def _count_pos_neg(indices, label_map, width):
    if indices is None or len(indices) == 0:
        return 0, 0
    flat = np.asarray(indices, dtype=np.int64)
    rows = flat // width
    cols = flat % width
    labels = label_map[rows, cols]
    pos = int(np.count_nonzero(labels > 0))
    neg = int(flat.size - pos)
    return pos, neg


def _build_validation_subset(indices, label_map, width, desired_size, seed=0, min_neg_multiplier=2.0):
    """
    保留全部正样本，并按一定比例随机抽取负样本，避免验证集缺少灾害点。
    """
    flat = np.asarray(indices, dtype=np.int64)
    if desired_size <= 0 or desired_size >= flat.size:
        return flat
    rows = flat // width
    cols = flat % width
    labels = label_map[rows, cols]
    pos_mask = labels > 0
    pos_indices = flat[pos_mask]
    neg_indices = flat[~pos_mask]

    if pos_indices.size == 0:
        return _sample_validation_indices(flat, desired_size, seed)

    target_size = max(desired_size, int(np.ceil(pos_indices.size * (1.0 + min_neg_multiplier))))
    target_size = min(target_size, flat.size)
    rng = np.random.default_rng(seed)
    selected = pos_indices.copy()
    remaining = target_size - selected.size
    if remaining > 0 and neg_indices.size > 0:
        if neg_indices.size <= remaining:
            neg_pick = neg_indices
        else:
            neg_pick = rng.choice(neg_indices, size=remaining, replace=False)
        selected = np.concatenate([selected, neg_pick])
    rng.shuffle(selected)
    return np.sort(selected)


def main():
    args = parse_args()

    # Prepare result dir and logging early so logs are available during reading data
    result_dir = Path("Result")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / f"train_cnn_{args.epochs}e_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("运行日志将同步输出到控制台与文件: %s", log_file)
    logging.info("=" * 20 + " 运行参数 " + "=" * 20)
    for key, val in vars(args).items():
        logging.info("%s: %s", key, val)
    logging.info("=" * 54)

    logging.info("加载特征: %s", args.feature_path)
    if args.feature_files:
        logging.info("特征文件列表: %s", ", ".join(args.feature_files))
    height, width, n_feature, data = dp.get_feature_data(args.feature_path, args.window_size, args.feature_files)
    logging.info("加载标签: %s", args.label_path)
    label = dp.get_label_data(args.label_path, args.window_size)
    unique_labels, label_counts = np.unique(label, return_counts=True)
    logging.info("检测到的标签唯一值: %s", unique_labels)
    logging.info("整体标签分布: %s", dict(zip(unique_labels.astype(int).tolist(),
                                        label_counts.astype(int).tolist())))
    num_classes = int(label.max()) + 1
    logging.info("检测到的类别数: %d", num_classes)
    train_indices, _, val_indices, _ = dp.get_CNN_data(
        data, label, args.window_size, val_ratio=args.val_ratio)
    # -------------------------------------------------------------
    # 过滤掉 label==15 的无效区域像元（不属于研究区）
    # -------------------------------------------------------------
    INVALID_LABEL = 15

    def _filter_invalid(indices, label_map, width, invalid_label=INVALID_LABEL):
        """移除标签为 invalid_label 的所有像元"""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            return indices
        rows = indices // width
        cols = indices % width
        mask = label_map[rows, cols] != invalid_label
        return indices[mask]

    before_train = len(train_indices)
    before_val = len(val_indices)

    train_indices = _filter_invalid(train_indices, label, width)
    val_indices = _filter_invalid(val_indices, label, width)

    logging.info(
        "已过滤标签=%d 的无效样本: 训练集 %d → %d, 验证集 %d → %d",
        INVALID_LABEL, before_train, len(train_indices),
        before_val, len(val_indices)
    )

    logging.info("全量训练样本: %d, 全量验证样本: %d", len(train_indices), len(val_indices))

    balanced_train_indices, pos_count, neg_count = _balance_class_ratio(
        train_indices, label, width, target_ratio=args.neg_pos_ratio, seed=args.sample_seed)
    if args.neg_pos_ratio > 0 and len(train_indices) > 0:
        actual_ratio = (neg_count / pos_count) if pos_count > 0 else float('inf')
        logging.info(
            "训练集按 %.1f:1 负:正目标采样 -> 正样本: %d, 非滑坡样本: %d, 实际负:正=%.2f:1, 采样后总数: %d (原总数: %d)",
            args.neg_pos_ratio, pos_count, neg_count, actual_ratio, len(balanced_train_indices), len(train_indices)
        )
        train_indices = balanced_train_indices
    else:
        logging.info("未对训练集执行负:正比例采样（neg_pos_ratio<=0 或训练样本不足）")
    train_pos, train_neg = _count_pos_neg(train_indices, label, width)
    logging.info("训练集样本统计 -> pos: %d, neg: %d, neg:pos=%.2f:1",
                 train_pos, train_neg, (train_neg / train_pos) if train_pos > 0 else float('inf'))

    if 0 < args.val_ratio < 1 and len(train_indices) > 0:
        target_val_size = int(max(1, round(len(train_indices) * args.val_ratio / (1 - args.val_ratio))))
        subset = _build_validation_subset(
            val_indices,
            label,
            width,
            min(target_val_size, len(val_indices)),
            seed=args.sample_seed + 2025,
            min_neg_multiplier=2.0
        )
        logging.info(
            "验证集抽样(保留全部正样本): 目标训练:验证 = %.1f:%.1f -> 训练样本 %d, 验证样本 %d (原验证样本 %d)",
            1 - args.val_ratio, args.val_ratio, len(train_indices), len(subset), len(val_indices)
        )
        val_indices = subset
    else:
        logging.info("未对验证集执行抽样（val_ratio<=0, >=1 或训练样本为空）")
    val_pos, val_neg = _count_pos_neg(val_indices, label, width)
    logging.info("验证集样本统计 -> pos: %d, neg: %d, neg:pos=%.2f:1",
                 val_pos, val_neg, (val_neg / val_pos) if val_pos > 0 else float('inf'))
    pad = args.window_size // 2
    padded_features = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_features = np.nan_to_num(padded_features, nan=0.0, posinf=0.0, neginf=0.0)

    train_dataset = PatchDataset(padded_features, label, width, args.window_size, train_indices)
    val_dataset = PatchDataset(padded_features, label, width, args.window_size, val_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    logging.info("训练样本: %d, 验证样本: %d, 输入通道: %d", len(train_dataset), len(val_dataset), n_feature)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("使用设备: %s", device)

    # Bayes opt (Optuna)
    if args.use_bayes_opt:
        logging.info("启用贝叶斯优化，试验次数: %d", args.bayes_trials)

        def objective(trial):
            trial_lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
            trial_weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            trial_anderson_k = trial.suggest_int("anderson_k", 3, 6)
            trial_anderson_m = trial.suggest_int("anderson_m", 2, 6)
            trial_use_regularization = trial.suggest_categorical("use_regularization", [True, False])

            model_trial = LSM_cnn(n_feature, num_classes=num_classes,
                                  use_regularization=trial_use_regularization).to(device)
            optimizer_trial = optim.SGD(model_trial.parameters(),
                                        lr=trial_lr,
                                        weight_decay=trial_weight_decay if trial_use_regularization else 0.0)
            criterion_trial = nn.CrossEntropyLoss().to(device)

            train_loader_trial = _build_loader(
                padded_features, label, width, args.window_size, train_indices,
                args.batch_size, shuffle=True, max_samples=args.bayes_max_train,
                seed=args.sample_seed + trial.number)
            val_loader_trial = _build_loader(
                padded_features, label, width, args.window_size, val_indices,
                args.batch_size, shuffle=False,
                max_samples=min(len(val_indices), max(1, args.bayes_max_train // 4)),
                seed=args.sample_seed + trial.number + 2024)

            epochs_trial = min(args.bayes_epochs, args.epochs)
            best_val_acc_trial = 0.0

            for _ in range(epochs_trial):
                model_trial.train()
                for images, target in train_loader_trial:
                    images, target = images.to(device), target.to(device)
                    optimizer_trial.zero_grad()
                    outputs = model_trial(images)
                    loss = criterion_trial(outputs, target)
                    loss.backward()
                    optimizer_trial.step()

                # Validation 353
                model_trial.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, target in val_loader_trial:
                        images, target = images.to(device), target.to(device)
                        outputs = model_trial(images)
                        preds = torch.argmax(outputs, dim=1)

                        val_correct += preds.eq(target).sum().item()
                        val_total += target.size(0)
                if val_total > 0:
                    epoch_val_acc = val_correct / val_total
                    best_val_acc_trial = max(best_val_acc_trial, epoch_val_acc)

            torch.cuda.empty_cache()
            trial.set_user_attr("anderson_k", trial_anderson_k)
            trial.set_user_attr("anderson_m", trial_anderson_m)
            return best_val_acc_trial

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.bayes_trials)
        best_params = study.best_params
        logging.info("贝叶斯优化完成，最佳验证准确率: %.4f", study.best_value if study.best_value else 0.0)
        logging.info("最佳参数: %s", best_params)

        args.lr = best_params["lr"]
        args.weight_decay = best_params["weight_decay"]
        args.use_regularization = best_params["use_regularization"]
        args.use_anderson = True
        args.anderson_k = int(study.best_trial.user_attrs.get("anderson_k", args.anderson_k))
        args.anderson_m = int(study.best_trial.user_attrs.get("anderson_m", args.anderson_m))

        bayes_summary = {
            "best_params": best_params,
            "best_val_acc": study.best_value,
            "anderson_k": args.anderson_k,
            "anderson_m": args.anderson_m,
            "trials": args.bayes_trials,
            "epochs_per_trial": min(args.bayes_epochs, args.epochs),
        }
        summary_path = result_dir / "bayes_opt_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(bayes_summary, f, ensure_ascii=False, indent=2)
        logging.info("贝叶斯优化总结已保存至 %s", summary_path)

    # Build model
    if args.use_se_layer or args.use_regularization:
        model_core = LSM_cnn_Enhanced(
            n_feature,
            num_classes=num_classes,
            use_magdrop=args.use_regularization,
            p_base=0.5,
            use_se_layer=args.use_se_layer,
            use_batch_norm=True,
            weight_decay_l2=args.weight_decay,
        )
    else:
        model_core = LSM_cnn(n_feature, num_classes=num_classes, use_regularization=False)
    model = model_core.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # Add weight_decay if use_regularization
    optimizer_kwargs = {}
    if args.use_regularization:
        optimizer_kwargs['weight_decay'] = args.weight_decay
    optimizer = optim.SGD(model.parameters(), lr=args.lr, **optimizer_kwargs)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    max_acc = 0
    record = {"train": {"acc": [], "loss": [], "auc": [], "miou": []},
              "val": {"acc": [], "loss": [], "auc": [], "miou": []}}
    num_train_steps = len(train_loader)

    # Anderson acceleration variables
    residuals = []
    params_history = []

    # 混淆矩阵与 mIoU 计算
    def _update_conf(conf, preds, target, num_classes):
        preds = preds.view(-1).cpu()
        target = target.view(-1).cpu()
        mask = (target >= 0) & (target < num_classes)
        preds = preds[mask]
        target = target[mask]
        idx = target * num_classes + preds
        bins = torch.bincount(idx, minlength=num_classes * num_classes)
        conf += bins.view(num_classes, num_classes)

    def _compute_miou(conf):
        union = conf.sum(0) + conf.sum(1) - conf.diag()
        iou = conf.diag().float() / torch.clamp(union.float(), min=1.0)
        return iou.mean().item()

    for epoch in range(args.epochs):
        logging.info("Epoch %03d/%03d 开始", epoch + 1, args.epochs)
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        train_outputs_list = []
        train_labels_list = []
        val_outputs_list = []
        val_labels_list = []
        conf_train = torch.zeros(num_classes, num_classes, dtype=torch.long)
        conf_val = torch.zeros(num_classes, num_classes, dtype=torch.long)

        model.train()
        for batch_idx, (images, target) in enumerate(train_loader, 1):
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()


            preds = torch.argmax(outputs, dim=1)
            _update_conf(conf_train, preds, target, num_classes)

            train_acc += preds.eq(target).sum().item()
            train_loss += loss.item() * images.size(0)
            train_outputs_list.append(outputs.detach().cpu().numpy())
            train_labels_list.append(target.detach().cpu().numpy())
            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                logging.info(
                    "Epoch %03d/%03d Step %04d/%04d | train_loss=%.4f",
                    epoch + 1, args.epochs, batch_idx, num_train_steps,
                    loss.item()
                )

        # Avoid division by zero if train_dataset is empty
        if len(train_dataset) > 0:
            epoch_train_loss = train_loss / len(train_dataset)
            epoch_train_acc = train_acc / len(train_dataset)
        else:
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
        miou_train = _compute_miou(conf_train)

        record["train"]["loss"].append(epoch_train_loss)
        record["train"]["acc"].append(epoch_train_acc)
        record["train"]["miou"].append(miou_train)

        model.eval()

        with torch.no_grad():
            for images, target in val_loader:
                images, target = images.to(device), target.to(device)
                outputs = model(images)
                loss = criterion(outputs, target)
                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                _update_conf(conf_val, preds, target, num_classes)

                val_acc += preds.eq(target).sum().item()
                val_outputs_list.append(outputs.detach().cpu().numpy())
                val_labels_list.append(target.detach().cpu().numpy())

        if len(val_dataset) > 0:
            epoch_val_loss = val_loss / len(val_dataset)
            epoch_val_acc = val_acc / len(val_dataset)
        else:
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0
        miou_val = _compute_miou(conf_val)
        record["val"]["loss"].append(epoch_val_loss)
        record["val"]["acc"].append(epoch_val_acc)
        record["val"]["miou"].append(miou_val)

        logging.info(
            '[%03d/%03d] Train Acc: %.4f Loss: %.4f mIoU: %.4f | Val Acc: %.4f Loss: %.4f mIoU: %.4f',
            epoch + 1, args.epochs, epoch_train_acc, epoch_train_loss, miou_train, epoch_val_acc, epoch_val_loss, miou_val
        )

        train_array = np.concatenate(train_outputs_list, axis=0) if train_outputs_list else np.empty((0, num_classes))
        train_labels_array = np.concatenate(train_labels_list, axis=0) if train_labels_list else np.empty((0,))
        val_array = np.concatenate(val_outputs_list, axis=0) if val_outputs_list else np.empty((0, num_classes))
        val_labels_array = np.concatenate(val_labels_list, axis=0) if val_labels_list else np.empty((0,))

        # 计算每个 epoch 的 AUC（函数应能处理空输入）
        train_auc = compute_auc(train_labels_array, train_array, num_classes)
        val_auc = compute_auc(val_labels_array, val_array, num_classes)
        record["train"]["auc"].append(train_auc)
        record["val"]["auc"].append(val_auc)
        logging.info(
            '[%03d/%03d] Train AUC: %.4f | Val AUC: %.4f',
            epoch + 1, args.epochs, train_auc, val_auc
        )

        if epoch_val_acc > max_acc and len(val_array) > 0:
            max_acc = epoch_val_acc
            logging.info("验证精度提升至 %.4f，保存最优模型", epoch_val_acc)
            torch.save(model.state_dict(), result_dir / 'best.pth')

        # step lr scheduler once per epoch
        scheduler.step()

        # Anderson acceleration
        if args.use_anderson and (epoch + 1) % args.anderson_k == 0:
            # Compute residual on validation set
            model.eval()
            residual = 0.0
            with torch.no_grad():
                for images, target in val_loader:
                    images, target = images.to(device), target.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, target)
                    residual += loss.item()

            residuals.append(residual)

            # Save model parameters at this step
            current_params = {name: param.data.clone() for name, param in model.named_parameters()}
            params_history.append(current_params)

            # If we have enough history, apply Anderson Acceleration
            if len(residuals) >= args.anderson_m:
                # use last m residuals/history
                recent_residuals = residuals[-args.anderson_m:]
                recent_params = params_history[-args.anderson_m:]

                def objective(w):
                    # squared weighted sum of recent residuals
                    return float(sum(w[i] * recent_residuals[i] for i in range(len(w))) ** 2)

                constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                bounds = [(0, 1) for _ in range(len(recent_residuals))]
                initial_w = np.ones(len(recent_residuals)) / len(recent_residuals)

                result = minimize(objective, initial_w, bounds=bounds, constraints=constraints)

                if result.success:
                    optimal_w = result.x

                    # Update parameters with weighted combination
                    for name, param in model.named_parameters():
                        param.data.zero_()
                        for i, w in enumerate(optimal_w):
                            param.data.add_(w * recent_params[i][name])

                    logging.info("Anderson acceleration applied with weights: %s", np.array2string(optimal_w, precision=4))

                # Keep only the latest m history
                residuals = residuals[-args.anderson_m:]
                params_history = params_history[-args.anderson_m:]

    # End of training loop

    # 训练结束后绘图与保存
    try:
        draw_acc(record["train"]["acc"], record["val"]["acc"])
        draw_loss(record["train"]["loss"], record["val"]["loss"])
        draw_auc_curve(record["train"]["auc"], record["val"]["auc"])
        logging.info("训练曲线已保存到 Result (如函数内部指定路径)")
    except Exception as e:
        logging.warning("绘制训练曲线时出错: %s", str(e))

    # 保存记录与最新模型
    try:
        record_path = result_dir / 'record.json'
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        torch.save(model.state_dict(), result_dir / 'latest.pth')
        logging.info("训练完成，最新模型已保存到 %s", result_dir / 'latest.pth')
    except Exception as e:
        logging.warning("保存 record 或 model 时出错: %s", str(e))

    # 训练完成后绘制最终 ROC（如启用）
    if args.plot_roc:
        logging.info("开始绘制最终 ROC 曲线（优先使用最佳模型 Result/best.pth）")

        best_model_path = result_dir / 'best.pth'
        try:
            if best_model_path.exists():
                state_dict = torch.load(best_model_path, map_location=device)
                model.load_state_dict(state_dict)
                logging.info("已加载最佳模型进行评估: %s", best_model_path)
        except Exception as e:
            logging.warning("加载最佳模型失败，将使用最新模型进行评估：%s", str(e))

        model.eval()

        def _collect_outputs_and_labels(loader):
            outs, labs = [], []
            with torch.no_grad():
                for images, target in loader:
                    images, target = images.to(device), target.to(device)
                    outputs = model(images)
                    outs.append(outputs.detach().cpu().numpy())
                    labs.append(target.detach().cpu().numpy())

            out_arr = np.concatenate(outs, axis=0) if outs else np.empty((0, num_classes))
            lab_arr = np.concatenate(labs, axis=0) if labs else np.empty((0,))
            return out_arr, lab_arr

        val_array, val_labels_array = _collect_outputs_and_labels(val_loader)
        train_array, train_labels_array = _collect_outputs_and_labels(train_loader)

        # ------ 验证集 ROC 绘制 ------
        if val_array.size > 0:
            unique_val_classes = len(np.unique(val_labels_array))

            if num_classes == 2 and unique_val_classes == 2 and val_array.shape[1] >= 2:
                try:
                    drawAUC_TwoClass(val_labels_array, val_array[:, 1], str(result_dir / 'H5_val_AUC.png'))
                except Exception as e:
                    logging.warning("绘制二分类验证集 ROC 出错: %s", str(e))
            else:
                logging.info(
                    "多分类任务（模型输出 %d 类，验证集包含 %d 类），绘制 micro-average ROC 曲线",
                    num_classes, unique_val_classes
                )
                try:
                    drawAUC_MultiClass(val_labels_array, val_array, str(result_dir / 'H5_val_AUC.png'))
                except Exception as e:
                    logging.warning("绘制多分类验证集 ROC 出错: %s", str(e))
        else:
            logging.warning("验证集为空或未能产生输出，跳过验证集 ROC 绘制")

        # ------ 训练集 ROC 绘制 ------
        if train_array.size > 0:
            unique_train_classes = len(np.unique(train_labels_array))

            if num_classes == 2 and unique_train_classes == 2 and train_array.shape[1] >= 2:
                try:
                    drawAUC_TwoClass(train_labels_array, train_array[:, 1], str(result_dir / 'H5_train_AUC.png'))
                except Exception as e:
                    logging.warning("绘制二分类训练集 ROC 出错: %s", str(e))
            else:
                try:
                    drawAUC_MultiClass(train_labels_array, train_array, str(result_dir / 'H5_train_AUC.png'))
                except Exception as e:
                    logging.warning("绘制多分类训练集 ROC 出错: %s", str(e))
        else:
            logging.warning("训练集为空或未能产生输出，跳过训练集 ROC 绘制")


if __name__ == "__main__":
    main()
