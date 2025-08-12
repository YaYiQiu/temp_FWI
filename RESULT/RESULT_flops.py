from thop import profile, clever_format

from model.InversionNet import *
from model.FCNVMB import *
from model.IAEDN import *
from model.DDNet70 import *
from torchstat import stat
from torchsummary import summary
from ptflops import get_model_complexity_info
# 定义网络模型
model = IAEDN_UU()  #InversionNet  IAEDN_WTUU   DDNet70Model
# 创建一个输入张量作为模型的输入
# wtuu 模型的 FLOPs: 3.129G
# 模型的参数数量: 9.858M
# 模型的 FLOPs: 2.150G
# 模型的参数数量: 7.632M
def print_thop(model):
    input = torch.randn(1, 5, 1000, 70)  # 替换为你自己的输入尺寸
    # 使用 THOP 计算模型的 FLOPs 和参数数量
    flops, params = profile(model, inputs=(input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"模型的 FLOPs: {flops}")
    print(f"模型的参数数量: {params}")

def print_ptflops(model):
    flops, params = get_model_complexity_info(model, (5, 1000, 70), as_strings=True, print_per_layer_stat=True)
    print(flops)
    print(params)

print_thop(model)
#print_ptflops(model)