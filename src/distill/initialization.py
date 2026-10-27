"""删层学生的权重选择；随机模式只保留 blocks 的原生初始化。"""


def student_state_dict(teacher_state, initial_state, keep_layers, block_prefix, initialization):
    if initialization not in ("teacher", "random"):
        raise ValueError("model.block_initialization 只支持 teacher 或 random")
    # 随机模式保留完整的新模型状态，再覆盖非 block 参数；不调用通用 reset_parameters，
    # 避免破坏模型原生的 LayerNorm 常数初始化、LangFlow AdaLN 零门控等规则。
    weights = dict(initial_state) if initialization == "random" else {}
    weights.update({key: value for key, value in teacher_state.items()
                    if not key.startswith(block_prefix)})
    if initialization == "teacher":
        for student_index, teacher_index in enumerate(keep_layers):
            prefix = f"{block_prefix}{teacher_index}."
            weights.update({f"{block_prefix}{student_index}." + key[len(prefix):]: value
                            for key, value in teacher_state.items() if key.startswith(prefix)})
    return weights
