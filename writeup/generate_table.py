import pandas as pd

data = {
    "Size": ["small", "medium", "large", "xl", "10B"],
    "d_model": [768, 1024, 1280, 2560, 4608],
    "d_ff": [3072, 4096, 5120, 10240, 12288],
    "num_layers": [12, 24, 36, 32, 50],
    "num_heads": [12, 16, 20, 32, 36]
}

df = pd.DataFrame(data)

latex_code = df.to_latex(
    index=False,
    caption="Specifications of different model sizes. These are mostly based on GPT-2 configs.",
    label="tab:model_spec",
    escape=True   # 开启特殊字符转义，_ 自动变成 \_
)

# 额外补上表格居中
latex_code = latex_code.replace(r"\caption", r"\centering\caption")

print(latex_code)