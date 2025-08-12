import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 创建子图并显示图像
fig, axes = plt.subplots(12, 2, figsize=(2, 12))  # 创建12行8列的子图

for i in range(12):
    for j in range(2):
        img = mpimg.imread("./feature_map/FlatFaultA(ABA-FWI)/" + str(43) + '_' +str(j) + '_' +str(i) + '.png')  #CurveFaultA  FlatFaultA
        axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')  # 关闭坐标轴

plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0.05, hspace=0.01)
fig.savefig("./feature_map/FlatFaultA_43(ABA-Net).png")
plt.show()