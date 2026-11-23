# Kimi训推一致性

此实验分支将原checkpoint Python代码中的训练实现接入仓库源码。
verl在Kimi模型且KIMI_ALIGNMENT_RECURRENT_KDA=1时直接选择
fsdp_turbo.models.kimi.kimi_k31.dsa.KimiK31ForConditionalGeneration，
其余模型沿用原AutoModel入口。

Kimi K3.1专家类直接实现ep_forward_mc2，复用公共MC2调度与算子接口，
保持原实验中的计算顺序和共享KV反向语义。训练递归实现在
fsdp_turbo/ops/npu/kimi_kda_alignment.py，开关读取集中在utils/env.py。

该分支需要与verl、vLLM和vLLM-Ascend的同名分支配套使用。
迁移前已获得4层模型的短测零差结果。当前源码布局及模型加载入口变化后，
已完成语法、导入闭包、计算AST等价性和加载分支检查，尚待重新进行NPU验证。
反向有效更新、完整BI与PD验收仍按各自证据记录。
