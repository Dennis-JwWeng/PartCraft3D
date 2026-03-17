# Prompt Style Comparison: Default vs Action

## 问题

Phase 1 enricher 生成的 edit prompt 过于详细（平均 14.7 词），导致：
1. VLM 图像编辑时修改超出 part 边界（提到整体对象名称触发全局编辑）
2. 跨 part 描述导致一个 prompt 影响多个 part
3. 形状修改导致相邻 part 位移

与 3DEditVerse 数据集对比：
- 3DEditVerse Alpaca: 平均 **3.8 词**（"Grow a tail", "Add a lid"）
- 3DEditVerse Flux:   平均 **8.4 词**（"Replace the teeth with a smooth edge"）
- 我们 (default):     平均 **14.7 词**

## 两种 Prompt 风格

### `--prompt-style default`（原版）

```
"Change the camera's body material from black textured leather
 to a smooth, polished chrome finish."
```

特点：
- 详细描述 part 当前外观 + 目标外观
- 引用整体对象名称（"camera's body"）
- 可能包含形状修改
- 每个 part 1 个 modification

### `--prompt-style action`（3DEditVerse 风格）

```
"Change material to chrome"
```

特点：
- ≤8 词，动作导向
- 不引用整体对象，只描述 part 本身的变化
- 核心 part 只做材质/颜色/纹理修改（不改形状）
- 严格限制在 part 空间边界内
- 每个 part 2 个 modification（增加多样性）

## 使用方式

```bash
# 原版详细 prompt
python scripts/run_enrich.py --config configs/partobjaverse.yaml

# 3DEditVerse 风格短 prompt（输出到单独文件）
python scripts/run_enrich.py --config configs/partobjaverse.yaml \
    --prompt-style action

# 测试 5 个对象
python scripts/run_enrich.py --config configs/partobjaverse.yaml \
    --prompt-style action --limit 5
```

action 风格输出到 `semantic_labels_action.jsonl`，不覆盖原版。

## 预期效果

| | default | action |
|---|---|---|
| VLM 编辑越界 | 常见 | 极少 |
| Part mask 匹配 | 经常不匹配 | 高度匹配 |
| 编辑幅度 | 大（可能破坏性） | 小（保守、可控） |
| 多样性 | 1 mod/part | 2 mod/part |
