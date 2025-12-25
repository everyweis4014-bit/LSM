# operator
# EveryMeis
# Time: 20230710
# geneLSM_with_shap.py
import argparse
import math
import os
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from osgeo import gdal
from tqdm import tqdm

import dataprepare as dp
from model import LSM_cnn
from model_CNN import LSM_cnn_Enhanced

# --- 新增：shap 可能会非常慢，请确保已安装 shap ---
try:
    import shap
except Exception as e:
    shap = None
    print("Warning: shap not available. To enable SHAP feature importance, run `pip install shap`.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default="C:/Users/Lenovo/Desktop/envirodis/ex/feature")
    parser.add_argument("--label_path", type=str, default="C:/Users/Lenovo/Desktop/envirodis/ex/hazard/hazard.tif")
    parser.add_argument("--window_size", default=15, type=int)
    parser.add_argument("--slide_window", default=4096, type=int)
    parser.add_argument("--model_path", default="D:/pythonProject1/Result/latest.pth")
    parser.add_argument("--output_path", default="hazard_predict.tif", type=str)
    parser.add_argument("--smooth", action="store_true", help="是否对输出概率图做核卷积平滑（参考 RF 推理脚本）")
    parser.add_argument("--smooth_kernel", default=5, type=int, help="平滑核大小（奇数，推荐 3/5/7）")
    parser.add_argument("--smooth_iterations", default=2, type=int, help="平滑迭代次数，默认 1-2 即可")
    parser.add_argument("--quantile_calibration", action="store_true", help="是否对概率做分位数校准（保持固定的风险比例结构）")
    parser.add_argument("--temperature", type=float, default=None, help="可选的温度缩放参数 (>0)，用于对概率进行校准，None 表示不启用")
    parser.add_argument("--use_se_layer", action="store_true", help="如果训练时启用了 SE-Layer，推理时也需要指定此参数")
    parser.add_argument("--use_regularization", action="store_true", help="如果训练时启用了正则化，推理时也需要指定此参数")
    parser.add_argument("--dangerous_weights", nargs="+", type=float, default=None, help="多分类时各危险等级的权重，长度需等于危险类别数，默认按等级递增权重")
    parser.add_argument("--disable_auto_postprocess", action="store_true", help="关闭自动温度缩放/分位数校准/平滑处理")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma 校正系数，<1 拉高高值，>1 压制高值")

    # 1️⃣ 去极值百分位拉伸
    parser.add_argument(
        "--percentile_stretch",
        nargs=2,
        type=float,
        default=None,
        help="按百分位拉伸，例如 2 98 表示用 [P2,P98] 重映射到 [0,1]"
    )

    # 2️⃣ 高斯平滑抑噪
    parser.add_argument(
        "--gaussian_smooth",
        action="store_true",
        help="是否启用 Gaussian smoothing (比卷积核平滑更自然)"
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=1.0,
        help="高斯平滑 sigma（推荐 0.8~1.5）"
    )

    # 3️⃣ Sigmoid 拉伸（提升中高值）
    parser.add_argument(
        "--sigmoid_gain",
        type=float,
        default=None,
        help="Sigmoid 增益系数，>1 拉高中高值，推荐 5~15"
    )
    parser.add_argument("--kernel", type=int, default=51,
                        help="kernel size for local quantile calibration (LQC)")

    # SHAP相关参数
    parser.add_argument("--shap_enable", action="store_true", help="是否在推理前计算 SHAP 特征贡献度（基于补丁通道均值 + KernelExplainer）")
    parser.add_argument("--shap_background_size", type=int, default=50, help="SHAP 背景样本数量（默认50，越大越稳但越慢）")
    parser.add_argument("--shap_explain_size", type=int, default=200, help="SHAP 解释样本数量（默认200，会从所有窗口随机抽样）")
    parser.add_argument("--shap_out", type=str, default="feature_shap.csv", help="SHAP 输出 CSV 文件路径")
    return parser.parse_args()


def save_prob_to_tif(reference_tif, output_path, prob_array):
    ds = gdal.Open(reference_tif)
    width = ds.RasterXSize
    height = ds.RasterYSize

    prob_array = prob_array.reshape(height, width)  # reshape 回影像大小

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(prob_array)
    out_ds.FlushCache()
    out_ds = None


def _smooth_probability_map(prob_2d: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    if kernel_size < 3 or kernel_size % 2 == 0:
        kernel_size = 3
    if iterations <= 0:
        return prob_2d

    h, w = prob_2d.shape
    pad = kernel_size // 2
    result = prob_2d.astype(np.float32)
    for _ in range(iterations):
        padded = np.pad(result, pad_width=pad, mode="edge")
        acc = np.zeros_like(result, dtype=np.float32)
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                acc += padded[dy:dy + h, dx:dx + w]
        result = acc / float(kernel_size * kernel_size)
    return np.clip(result, 0.0, 1.0)


def _temperature_scaling(prob: np.ndarray, temperature: float) -> np.ndarray:
    if temperature is None or temperature <= 0:
        return prob
    eps = 1e-7
    prob = np.clip(prob, eps, 1.0 - eps)
    logits = np.log(prob) - np.log(1.0 - prob)
    scaled_logits = logits / temperature
    return 1.0 / (1.0 + np.exp(-scaled_logits))


def _quantile_calibration_local(prob_2d: np.ndarray, kernel: int = 51) -> np.ndarray:
    """
    局部分位数校准（Local Quantile Calibration, LQC）
    保持空间连续性的概率增强方法
    """
    height, width = prob_2d.shape
    prob_2d = prob_2d.astype(np.float32)

    pad = kernel // 2
    padded = np.pad(prob_2d, pad, mode='reflect')

    out = np.zeros_like(prob_2d, dtype=np.float32)

    # 遍历每个像素（可后续加速）
    for i in range(height):
        for j in range(width):
            win = padded[i:i+kernel, j:j+kernel].reshape(-1)
            # 计算局部分位数（像素在邻域的排名）
            rank = (win < prob_2d[i, j]).mean()
            out[i, j] = rank

    # 分位数映射成最终的增强效果
    q_breaks = np.array([0.00, 0.25, 0.55, 0.70, 1.00], dtype=np.float32)
    v_breaks = np.array([0.00, 0.10, 0.40, 0.75, 1.00], dtype=np.float32)
    calibrated_flat = np.interp(out.reshape(-1), q_breaks, v_breaks)
    calibrated_2d = calibrated_flat.reshape(height, width)
    return calibrated_2d  # 返回 2D





def compute_shap_channel_importance(
        model,
        device,
        features,
        window_size,
        slide_window,        # <- 这里接收 slide_window
        background_size=50,
        explain_size=200,
        num_classes=6,
        dangerous_weights=None,
        seed=42):

    import shap
    rng = np.random.RandomState(seed)

    print("Generating per-window channel means for SHAP sampling...")

    # ============= ✔ 正确兼容 generate_windows 的写法 ============
    all_means = []
    batches = dp.generate_windows(features, window_size, slide_window)  # <- 位置参数，不要关键字
    # ==============================================================

    for batch in batches:
        # batch shape: (B, win, win, C)
        means = batch.mean(axis=(1, 2))  # -> (B, C)
        all_means.append(means)

    if len(all_means) == 0:
        raise RuntimeError("generate_windows 没有生成任何样本，请检查 window_size/block_size")

    all_means = np.vstack(all_means)  # (N_windows, C)
    n_windows, n_feature = all_means.shape
    print(f"Total windows = {n_windows}, feature channels = {n_feature}")

    # 采样 background 和 explain
    bg_idx = rng.choice(n_windows, size=min(background_size, n_windows), replace=False)
    ex_idx = rng.choice(n_windows, size=min(explain_size, n_windows), replace=False)

    background = all_means[bg_idx]
    to_explain = all_means[ex_idx]

    # ========== 预测函数：把通道均值还原为窗口送入 CNN ==========
    def predict_from_channel_means(X_np):
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        m = X_np.shape[0]

        window = np.ones((window_size, window_size), dtype=np.float32)
        batch_windows = np.empty((m, window_size, window_size, n_feature), dtype=np.float32)

        for i in range(n_feature):
            batch_windows[:, :, :, i] = X_np[:, i].reshape(m, 1, 1) * window

        tensor = torch.from_numpy(batch_windows).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs_all = torch.softmax(logits, dim=1).cpu().numpy()

        if num_classes == 2:
            return probs_all[:, 1]
        else:
            dangerous_probs = probs_all[:, 1:]
            if dangerous_probs.shape[1] == 0:
                return np.zeros(m, dtype=np.float32)
            return np.dot(dangerous_probs, dangerous_weights)

    # ========== SHAP ==========
    print("Initializing KernelExplainer...")
    explainer = shap.KernelExplainer(predict_from_channel_means, background)

    print("Computing SHAP values (slow step)...")
    shap_values = explainer.shap_values(to_explain, nsamples="auto")

    # 兼容 SHAP 不同返回格式
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values)[0]  # binary task
    else:
        shap_arr = shap_values

    mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)  # (C,)

    return mean_abs_shap, background, to_explain


def main():
    args = parse_args()
    print("Loading features...")
    height, width, n_feature, features = dp.get_feature_data(args.feature_path, args.window_size)

    # ----------------------------
    # 加载 label
    # ----------------------------
    label_ds = gdal.Open(args.label_path)
    label = label_ds.ReadAsArray().astype(np.int32)
    label_ds = None

    # label shape: (H, W) → 展平成一维
    labels = label.reshape(-1)

    # ----------------------------
    # 过滤掉 15（无效像元）和 0（研究区背景）
    # ----------------------------
    mask_valid = (labels != 15)
    filtered_labels = labels[mask_valid]

    print(f"过滤掉 标签=15 的像元数量: {(labels == 15).sum()}")

    print(f"有效标签数量（用于权重计算）: {filtered_labels.shape[0]}")

    # ----------------------------
    # 现在 filtered_labels 只包含 {1,2,3}
    # ----------------------------
    filtered_labels = torch.tensor(filtered_labels, dtype=torch.long)

    # 找出现类别
    unique_classes = torch.unique(filtered_labels)
    print("实际出现类别 =", unique_classes.tolist())

    # ----------------------------------
    # ★ 手动设置 class weights ★
    # 顺序必须与 unique_classes 对应！
    # 如果 unique_classes = [1, 2, 3]：
    # 那就写成 [w1, w2, w3]
    # ----------------------------------

    manual_weights = torch.tensor([25, 6, 1], dtype=torch.float32)

    # 归一化（可选）
    manual_weights = manual_weights / manual_weights.sum()

    class_weights = manual_weights
    print("最终标签权重（手动） =", class_weights.tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model_path, map_location=device)
    
    has_enhanced_keys = any("conv1.weight" in k or "se1.fc.0.weight" in k for k in state_dict.keys())
    has_original_keys = any("net.0.weight" in k or "classifier.0.weight" in k for k in state_dict.keys())
    
    num_classes = 0
    if "fc2.weight" in state_dict:
        num_classes = state_dict["fc2.weight"].shape[0]
    elif "classifier.3.weight" in state_dict:
        num_classes = state_dict["classifier.3.weight"].shape[0]
    elif "net.classifier.3.weight" in state_dict:
        num_classes = state_dict["net.classifier.3.weight"].shape[0]
    
    if num_classes == 0:
        raise ValueError("无法从 state_dict 中确定类别数，请检查模型文件")
    
    dangerous_weights = None
    if num_classes > 2:
        if args.dangerous_weights:
            weights = np.asarray(args.dangerous_weights, dtype=np.float32)
            if weights.size != num_classes - 1:
                raise ValueError(f"dangerous_weights 长度应为 {num_classes - 1}，当前为 {weights.size}")
            dangerous_weights = weights / (weights.sum() + 1e-8)
        else:
            auto_weights = np.linspace(1.0, num_classes - 1, num_classes - 1, dtype=np.float32)
            dangerous_weights = auto_weights / (auto_weights.sum() + 1e-8)
        print(f"Dangerous class weights: {np.round(dangerous_weights, 4).tolist()}")

    if args.use_se_layer or args.use_regularization or has_enhanced_keys:
        print("检测到 Enhanced 模型结构，使用 LSM_cnn_Enhanced...")
        model = LSM_cnn_Enhanced(
            n_feature,
            num_classes=num_classes,
            use_magdrop=args.use_regularization,
            p_base=0.5,
            use_se_layer=args.use_se_layer or has_enhanced_keys,
            use_batch_norm=True,
            weight_decay_l2=0.0001,
        )
    else:
        print("使用标准 LSM_cnn 模型...")
        model = LSM_cnn(n_feature, num_classes=num_classes)
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"警告: 严格加载失败，尝试部分加载...")
        model.load_state_dict(state_dict, strict=False)
        print("部分加载成功，某些层可能使用随机初始化")
    
    model.to(device)
    model.eval()



    # ----------------------------
    # 在推理前计算 SHAP（可选）
    # ----------------------------
    if args.shap_enable:
        print("SHAP 计算已启用，开始计算通道级别特征贡献度...")
        try:
            mean_abs_shap, bg, to_explain = compute_shap_channel_importance(
                model=model,
                device=device,
                features=features,
                slide_window=args.slide_window,
                window_size=args.window_size,
                background_size=args.shap_background_size,
                explain_size=args.shap_explain_size,
                num_classes=num_classes,
                dangerous_weights=dangerous_weights,
            )
            # 保存 CSV
            idxs = np.arange(n_feature)
            out_csv = args.shap_out
            header = "feature_index,mean_abs_shap\n"
            with open(out_csv, "w") as f:
                f.write(header)
                for i, val in enumerate(mean_abs_shap):
                    f.write(f"{i},{float(val)}\n")
            print(f"SHAP 特征贡献度已保存到: {out_csv}")

            # ================================
            # ② 特征按贡献度排序（从高到低）
            # ================================
            ranking = np.argsort(-mean_abs_shap)

            print("\n===============================")
            print(" 特征贡献度排序（从高到低）")
            print("===============================")
            for rank_idx, feat_idx in enumerate(ranking):
                imp = mean_abs_shap[feat_idx]
                print(f"Rank {rank_idx + 1:02d} | Feature {feat_idx} | Importance = {imp:.6f}")

            # ================================
            # ③ 保存排序后的 CSV
            # ================================
            out_rank_csv = out_csv.replace(".csv", "_ranking.csv")
            with open(out_rank_csv, "w") as f:
                f.write("rank,feature_index,importance\n")
                for r, feat_idx in enumerate(ranking):
                    f.write(f"{r + 1},{feat_idx},{float(mean_abs_shap[feat_idx])}\n")

            print(f"\n✔ 排序后的 SHAP 贡献度已保存到: {out_rank_csv}")
            print("================================\n")

        except Exception as e:
            print("计算 SHAP 时发生错误（继续推理），错误信息：", e)

    # ----------------------------
    # 推理主流程（和你原来的基本一致）
    # ----------------------------
    total_pixels = height * width
    all_probs = np.empty(total_pixels, dtype=np.float32)
    offset = 0

    batches = dp.generate_windows(features, args.window_size, args.slide_window)
    total_steps = math.ceil(total_pixels / args.slide_window)

    print("Predicting...")
    sample_batch_for_debug = None  # 保存第一个batch用于调试
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, total=total_steps)):
            # 保存第一个batch用于调试
            if batch_idx == 0:
                sample_batch_for_debug = batch.copy()

            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            logits = model(tensor)
            probs_all = torch.softmax(logits, dim=1).cpu().numpy()

            if num_classes == 2:
                probs = probs_all[:, 1]
            else:
                dangerous_probs = probs_all[:, 1:]
                if dangerous_probs.shape[1] == 0:
                    probs = np.zeros(dangerous_probs.shape[0], dtype=np.float32)
                else:
                    probs = np.dot(dangerous_probs, dangerous_weights)

            next_offset = offset + probs.shape[0]
            all_probs[offset:next_offset] = probs
            offset = next_offset

        if offset != total_pixels:
            raise RuntimeError(f"预测像元数量 ({offset}) 与影像像元总数 ({total_pixels}) 不一致")

    # ========== 添加调试信息 ==========
    print(f"\n=== 模型输出调试 ===")
    print(f"原始易发性概率统计: min={all_probs.min():.6f}, max={all_probs.max():.6f}, "
          f"mean={all_probs.mean():.6f}, std={all_probs.std():.6f}")
    print(f"概率分布: <0.01: {(all_probs < 0.01).sum()}, 0.01-0.1: {((all_probs >= 0.01) & (all_probs < 0.1)).sum()}, "
          f">=0.1: {(all_probs >= 0.1).sum()}")

    # 使用保存的batch查看原始logits
    if sample_batch_for_debug is not None:
        model.eval()
        sample_tensor = torch.from_numpy(sample_batch_for_debug[:10]).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            sample_logits = model(sample_tensor).cpu().numpy()
            sample_probs = torch.softmax(torch.from_numpy(sample_logits), dim=1).numpy()

        print(f"\n样本Logits统计 (前10个样本):")
        print(f"  min={sample_logits.min():.4f}, max={sample_logits.max():.4f}, mean={sample_logits.mean():.4f}")
        print(f"  示例logits (前3个样本):\n{sample_logits[:3]}")
        print(f"\n样本Softmax概率 (前10个样本):")
        print(f"  类别概率范围: {sample_probs.min(axis=0)}, {sample_probs.max(axis=0)}")
        print(f"  示例概率 (前3个样本):\n{sample_probs[:3]}")
        if num_classes > 2:
            print(f"  类别0(安全)概率: mean={sample_probs[:, 0].mean():.4f}, max={sample_probs[:, 0].max():.4f}")
            print(f"  危险类概率和: mean={(sample_probs[:, 1:].sum(axis=1)).mean():.4f}")
    else:
        print("警告: 无法获取样本batch进行调试")
    print("=" * 40 + "\n")
    # ==================================

    auto_postprocess = not args.disable_auto_postprocess

    temperature_to_apply = args.temperature
    if temperature_to_apply is None and auto_postprocess:
        temperature_to_apply = 0.6 if num_classes > 2 else 0.8
        print(f"Auto temperature scaling (temperature={temperature_to_apply})...")
    if temperature_to_apply is not None:
        print(f"Applying temperature scaling (temperature={temperature_to_apply})...")
        all_probs  = _temperature_scaling(all_probs, temperature_to_apply)

    # 分位数校准会破坏空间连续性（全局排序），默认禁用
    # 如需使用，请显式指定 --quantile_calibration
    quantile_requested = args.quantile_calibration
    # 分位数校准（增强对比度，不破坏空间连续性）
    if args.quantile_calibration:
        print(f"Applying Local Quantile Calibration (kernel={args.kernel}) ...")
        prob_2d = all_probs.reshape(height, width)
        calibrated = _quantile_calibration_local(prob_2d, kernel=args.kernel)
        all_probs = calibrated  # 已经是 1D

    smooth_requested = args.smooth
    if not smooth_requested and auto_postprocess:
        smooth_requested = True
        auto_kernel = int(max(3, args.smooth_kernel))
        auto_iter = max(1, args.smooth_iterations)
        print(f"Auto smoothing (kernel={auto_kernel}, iterations={auto_iter})...")
        prob_2d = all_probs.reshape(height, width)
        prob_2d = _smooth_probability_map(prob_2d, kernel_size=auto_kernel, iterations=auto_iter)
        all_probs = prob_2d.reshape(-1)
    elif smooth_requested:
        print(f"Applying spatial smoothing (kernel={args.smooth_kernel}, iterations={args.smooth_iterations})...")
        prob_2d = all_probs.reshape(height, width)
        prob_2d = _smooth_probability_map(prob_2d, kernel_size=int(args.smooth_kernel),
                                          iterations=int(args.smooth_iterations))
        all_probs = prob_2d.reshape(-1)

    # 如果概率值范围太小，进行线性拉伸
    prob_min = float(all_probs.min())
    prob_max = float(all_probs.max())
    prob_range = prob_max - prob_min
    
    print(f"拉伸前统计: min={prob_min:.6f}, max={prob_max:.6f}, range={prob_range:.6f}")
    
    if prob_range < 0.01 or prob_max < 0.1:  # 如果范围太小或最大值太小
        print(f"警告: 概率值范围过小，进行线性拉伸到 [0, 1]...")
        if prob_range > 1e-8:
            all_probs = (all_probs - prob_min) / prob_range
        else:
            print("警告: 所有概率值几乎相同，模型可能没有学到有效特征！")
            all_probs = np.full_like(all_probs, 0.5)
        print(f"拉伸后: min={all_probs.min():.6f}, max={all_probs.max():.6f}")

    print("Saving result tif...")
    # 删除概率反转！模型输出的是"易发性概率"，不需要反转
    # all_probs = 1.0 - all_probs  # 删除这行！
    
    # 处理标签=15的无效区域
    prob_2d = all_probs.reshape(height, width)
    label_ds = gdal.Open(args.label_path)
    if label_ds:
        label_array = label_ds.ReadAsArray()
        prob_2d[label_array == 15] = 0.0
        label_ds = None
    all_probs = prob_2d.reshape(-1)
    
    print(f"最终统计: min={all_probs.min():.6f}, max={all_probs.max():.6f}, "
          f"mean={all_probs.mean():.6f}")
    # ====================== 概率增强增强模块（直接粘贴即可） ======================



    print("\n=== 应用概率增强 ===")
    # 1) 去极值百分位拉伸
   # p_low, p_high = 1, 99
   # p1 = np.percentile(all_probs, p_low)
   # p99 = np.percentile(all_probs, p_high)
   # all_probs = (all_probs - p1) / (p99 - p1 + 1e-12)
   # all_probs = np.clip(all_probs, 0.0, 1.0)
   # print(f"Percentile stretch: p{p_low}={p1:.6f}, p{p_high}={p99:.6f}")

    # 2) 高斯平滑抑噪
    prob_2d = all_probs.reshape(height, width)
    prob_s = gaussian_filter(prob_2d, sigma=1.0)
    prob_2d = 0.6 * prob_2d + 0.4 * prob_s
    all_probs = np.clip(prob_2d.reshape(-1), 0.0, 1.0)
    print("Applied Gaussian smoothing (sigma=1.0)")

    # 3) Sigmoid 拉伸（可选）
    # def sigmoid_stretch(prob, k=12.0, mid=0.03):
    #     return 1.0 / (1.0 + np.exp(-k * (prob - mid)))
    # all_probs = sigmoid_stretch(all_probs)

    # 4) 自动 gamma 强化
    #def find_gamma_for_target_mean(prob, target_mean=0.08, low=0.01, high=2.0, iters=40):
       # prob = np.clip(prob, 0.0, 1.0)
       # lo, hi = low, high
       # for _ in range(iters):
           # mid = (lo + hi) / 2.0
           # cur_mean = np.mean(prob ** mid)
           # if cur_mean < target_mean:
               # hi = mid
           # else:
               # lo = mid
       # return (lo + hi) / 2.0
   # target_mean = 0.12
   # gamma_found  = find_gamma_for_target_mean(all_probs, target_mean)
    #all_probs = np.clip(all_probs ** gamma_found, 0.0, 1.0)
   # print(f"Applied auto-gamma, target_mean={target_mean}, gamma={gamma_found:.4f}")
    print("增强后统计: min={:.6f}, max={:.6f}, mean={:.6f}".format(
        all_probs.min(), all_probs.max(), all_probs.mean()
    ))

    save_prob_to_tif(args.label_path, args.output_path, all_probs)
    print(f"Done! Output: {args.output_path}")
    print("输出说明：栅格值范围 [0, 1]，值越大表示易发性越高（越危险）")


if __name__ == "__main__":
    main()
