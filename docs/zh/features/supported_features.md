# 特性列表

本表汇总FSDPTurbo支持的训练特性，并按特性大类归类，详细说明见各特性文档。

<table>
  <thead>
    <tr>
      <th>特性大类</th>
      <th>特性名称</th>
      <th>说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">并行特性</td>
      <td><a href="./fully_shard_parallel.md">全分片数据并行</a></td>
      <td>基于PyTorch FSDP2，参数/梯度/优化器状态全分片，支持混合精度与预取</td>
    </tr>
    <tr>
      <td><a href="./tensor_parallel.md">张量并行</a></td>
      <td>列并行、行并行、序列并行</td>
    </tr>
    <tr>
      <td><a href="./expert_parallel.md">专家并行</a></td>
      <td>MoE专家并行，支持eager/fused/mc2/domino dispatcher与EFSDP</td>
    </tr>
    <tr>
      <td><a href="./context_parallel.md">上下文并行</a></td>
      <td>Ulysses上下文并行，支持长序列训练</td>
    </tr>
    <tr>
      <td><a href="./vocab_parallel.md">词表并行</a></td>
      <td>超大词表嵌入表按词表维切分（VP/V-FSDP），All-to-All查表，常驻设备且可训练</td>
    </tr>
    <tr>
      <td rowspan="7">内存优化特性</td>
      <td><a href="./recompute.md">激活重计算</a></td>
      <td>选择性激活重计算，降低激活显存占用</td>
    </tr>
    <tr>
      <td><a href="./chunk_batch.md">微批次切分</a></td>
      <td>沿batch维切分模块forward，降低单层显存峰值</td>
    </tr>
    <tr>
      <td><a href="./chunk_loss.md">分块损失计算</a></td>
      <td>损失计算分块处理，适用于大词表模型</td>
    </tr>
    <tr>
      <td><a href="./swap_activation.md">激活值交换</a></td>
      <td>激活值异步卸载到CPU内存，YAML配置开启</td>
    </tr>
    <tr>
      <td><a href="./swap_optimizer.md">优化器状态交换</a></td>
      <td>优化器状态卸载到CPU（<code>optimizer_type: AdamWSwap</code>）</td>
    </tr>
    <tr>
      <td><a href="./fully_shard_parallel.md">CPU卸载</a></td>
      <td>FSDP2参数/梯度CPU卸载</td>
    </tr>
    <tr>
      <td><a href="./checkpoint_resume.md">断点续训</a></td>
      <td>权重与优化器状态保存/恢复</td>
    </tr>
    <tr>
      <td>量化特性</td>
      <td><a href="./quantization.md">量化训练</a></td>
      <td>MXFP8量化（E4M3/E5M2/HIF8格式与分块量化），FSDP2低精度AllGather通信</td>
    </tr>
    <tr>
      <td rowspan="2">其他</td>
      <td><a href="./memory_snapshot.md">显存快照</a></td>
      <td>采集分配器历史与快照用于显存分析，YAML配置开启，支持OOM自动落档</td>
    </tr>
    <tr>
      <td><a href="./environment_variable.md">环境变量</a></td>
      <td>集群配置与NPU性能调优环境变量</td>
    </tr>
  </tbody>
</table>
