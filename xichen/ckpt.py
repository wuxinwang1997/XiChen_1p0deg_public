import torch
import os


# 交付的 *.ckpt 内嵌训练期的 OmegaConf(Hydra) 配置对象；torch>=2.6 起默认 weights_only=True
# 会拒绝反序列化这些非张量对象,导致 `Weights only load failed`。这里统一全量反序列化
# (weights_only=False)——仅用于加载官方/自训练的受信 checkpoint。
_LOAD_KWARGS = {"map_location": "cpu", "weights_only": False}


def _resolve_ckpt_path(ckpt_dir, model_name):
    """把「目录」或「扁平 .ckpt 文件」解析为实际 checkpoint 路径。

    支持两种布局：
    1. 训练日志目录（旧）：``ckpt_dir/model_name/runs/checkpoints/best.ckpt``
    2. 扁平 .ckpt 文件（DCU 服务器）：``ckpt_dir`` 本身就是一个 ``*.ckpt`` 文件，
       此时忽略 ``model_name`` 直接加载。

    两种布局都搜索不到时抛 FileNotFoundError 并列出候选路径，避免错误深埋在
    torch ``open()`` 内部（便于定位集群上 ckpt 路径笔误 / 未解包）。

    Args:
        ckpt_dir: 目录路径或 ``*.ckpt`` 文件路径。
        model_name: 旧布局中 run 名（如 ``finetune_xichen_state_forecast_ar15_20260616``）。

    Returns:
        实际 checkpoint 文件路径。

    Raises:
        FileNotFoundError: 两种布局均未找到 checkpoint。
    """
    if os.path.isfile(ckpt_dir):
        return ckpt_dir
    run_ckpt = os.path.join(ckpt_dir, model_name, "runs", "checkpoints", "best.ckpt")
    if os.path.isfile(run_ckpt):
        return run_ckpt
    raise FileNotFoundError(
        f"Checkpoint not found: ckpt_dir={ckpt_dir!r} is neither an existing "
        f"'*.ckpt' file nor a run directory holding\n  {run_ckpt}\n"
        "Expect one of:\n"
        f"  flat file    : ckpt_dir points directly at a '*.ckpt' file\n"
        "  run directory: <ckpt_dir>/<model_name>/runs/checkpoints/best.ckpt\n"
        "Resolve the path (or unpack/rename the archive) so one of the two exists."
    )


def _strip_module_prefix(state_dict):
    """DDP 保存的 ckpt 带 ``module.`` 前缀 → 剥掉后匹配。

    返回 (剥皮后的 dict, 是否剥过)。
    """
    if isinstance(state_dict, dict) and state_dict:
        first_key = next(iter(state_dict))
        if first_key.startswith("module."):
            return {k[len("module."):]: v for k, v in state_dict.items()}, True
    return state_dict, False


def load_state_dict_strict(model, state_dict, ignore_suffix=("_max_logit",)):
    """strict 加载权重,但忽略特定常量 buffer(如 ``_max_logit``)。

    ``_max_logit``(SwinAttention 的 logit 截断上限,恒 ln(100))是非学习常量 buffer,
    历史 ckpt 是否存在不影响前向(cascade 用缺省值也能正常出 z-500)。因此不作为权重
    参与 strict 校验 —— 权重(Linear/LayerNorm/Embedding 的 weight/bias 等可学习参数和
    buffer)与形状仍严格要求全匹配,缺键/多余键/尺寸不匹配一律抛错,避免静默让该层
    保持随机初始化(否则会出现「训练好、推理差」)。

    Args:
        model: 待加载的 ``torch.nn.Module``。
        state_dict: 要加载的权重字典(已经剥好 ``module.`` 前缀)。
        ignore_suffix: 不作为权重参与校验的 key 后缀集合,默认 ``("_max_logit",)``。

    Returns:
        加载后的 ``model``。

    Raises:
        RuntimeError: 除忽略集外存在 missing / unexpected key,或尺寸不匹配。
    """
    result = model.load_state_dict(state_dict, strict=False)

    def _ignored(k):
        return any(k.endswith(s) for s in ignore_suffix)

    missing = [k for k in result.missing_keys if not _ignored(k)]
    unexpected = [k for k in result.unexpected_keys if not _ignored(k)]
    if missing or unexpected:
        raise RuntimeError(
            f"{type(model).__name__} strict weights mismatch "
            f"(ignoring non-learn buffers {ignore_suffix}):\n"
            f"  missing={missing[:12]}\n  unexpected={unexpected[:12]}"
        )
    return model


def _extract_state_dict(checkpoint, model):
    """从 torch.load 结果中取模型权重并加载,兼容多种保存格式。

    训练端/扁平 ckpt 可能用以下任一 key:
      - ``model_state_dict``(BaseTrainer / ObsOperatorTrainer 保存格式)
      - ``state_dict``(PyTorch Lightning 或裸 ``model.state_dict()`` 包装)
      - 直接就是 state_dict 本身(整文件即权重字典)
    优先按 ``model_state_dict`` → ``state_dict`` → 整个 dict 兜底。

    推理必须 ``load_state_dict(strict=True)``:任何缺键/多余键/尺寸不匹配都显式抛错,
    避免静默让该层保持随机初始化(否则会出现「训练好、推理差」)。多格式解析(model_state_dict /
    state_dict / module. 前缀)由上方完成,这里不再向下兼容缺键。
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # 兜底:整文件就是权重字典(可能是错误的打包格式,下方会校验命中数)
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise KeyError(
            f"Checkpoint contains no state dict: got {type(state_dict).__name__}. "
            "Expected 'model_state_dict' / 'state_dict' keys or a bare state dict."
        )

    # DDP module. 前缀容错
    state_dict, _ = _strip_module_prefix(state_dict)

    # 权重 strict：任何可学习参数缺键/多余键/尺寸不匹配都显式抛错（_max_logit 等常量
    # buffer 由 load_state_dict_strict 忽略）。避免静默让该层保持随机初始化，否则会
    # 出现「训练好、推理差」。多格式解析（model_state_dict / state_dict / module. 前缀）
    # 的职责已由上方完成，这里不再向下兼容缺键。
    load_state_dict_strict(model, state_dict)
    return model


def load_forecast_ckpt(ckpt_dir, model_name, forecast_model):
    checkpoint = torch.load(
        _resolve_ckpt_path(ckpt_dir, model_name),
        **_LOAD_KWARGS,
    )
    return _extract_state_dict(checkpoint, forecast_model)


def load_obsop_ckpt(ckpt_dir, model_name, obsop_model):
    checkpoint = torch.load(
        _resolve_ckpt_path(ckpt_dir, model_name),
        **_LOAD_KWARGS,
    )
    return _extract_state_dict(checkpoint, obsop_model)
