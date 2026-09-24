import pytrio as trio

# 连接云端的TRIO计算引擎
client = trio.ServiceClient()
# 获取当前支持的模型列表
supported_models = client.get_supported_models()

print("Supported models:")
for index, model_name in enumerate(supported_models, start=1):
    print(f"{index}. {model_name}")