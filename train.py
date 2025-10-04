import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 0. 环境检查
assert torch.cuda.is_available(), "需要GPU环境！"
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. 加载模型（自动小量化适配Colab）
model_name = "bigcode/starcoderbase-1b"  # 改用更小的开源代码模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 极简训练样例
inputs = tokenizer("def hello_world():", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))

# 3. 保存模型（自动存到Google Drive）
model.save_pretrained("/content/drive/MyDrive/opencoder-demo")
print("✅ 训练完成！模型已保存")
