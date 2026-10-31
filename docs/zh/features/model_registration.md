# 内置模型自动注册

首次导入 `fsdp_turbo.models` 或其子模块时，Python 会先执行
`models/__init__.py` 中的模型类导入。
这些模型类通过 `@register_model(...)` 注册到 `fsdp_turbo.models.utils` 的类注册表。

因此可以直接导入查询函数，不需要先手动导入模型实现：

```python
from fsdp_turbo.models.utils import get_model_by_name

model_class = get_model_by_name("qwen4_exp")
```

当前内置模型名称与类的对应关系为：

| 模型名称 | 模型类 |
| --- | --- |
| `kimi_k3` | `KimiK3ForConditionalGeneration` |
| `qwen4_exp` | `Qwen4ExpForConditionalGeneration` |
| `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` |

查询返回实际模型类；未知名称返回 `None`，名称按原字符串精确匹配。
普通的重复导入使用 Python 的模块缓存，不会再次执行模型类的注册装饰器。

导入查询函数时，父包 `fsdp_turbo.models` 会先完成初始化，使三个内置模型类都已注册。
`models.utils` 自身只提供注册和查询函数，不反向导入模型实现，避免循环导入。
仅导入 `fsdp_turbo` 或不经过 `models` 的其他子模块，不会触发这组模型导入。

模型实现自身需要的 Transformers、PyTorch、算子与设备检测等依赖会随模型导入执行。
运行环境必须满足所有内置模型的导入依赖；若任一模型导入失败，异常会直接向上传播，
`fsdp_turbo.models` 初始化不会被当作成功完成。

该类注册表独立于 Transformers 的 `AutoConfig` / `AutoModel` 注册表。
例如，通过 Transformers 加载 Kimi 时，仍需按示例调用 `register_kimi_k3()`。
