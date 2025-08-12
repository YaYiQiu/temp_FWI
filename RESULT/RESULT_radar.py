# coding=utf-8

import numpy as np

import matplotlib
import matplotlib.pylab as plt

# 与ABA-FWI的消融
dataset_dict = {
    'FlatVelA': {
                 "ABA-Net": [47.518123,	0.991902, 	0.000039, 	0.003492, 	0.855830, 	0.000617, 	0.000107, 	0.004705],
                 "ABA-Loss": [47.04763,	0.985962, 	0.000079, 	0.003558, 	0.855304, 	0.000990, 	0.000089, 	0.004466],
                 "ABA-FWI": [49.104591,	0.990921, 	0.000027, 	0.002977, 	0.855797, 	0.000310, 	0.000083, 	0.003755]},
    'FlatFaultA': {
                   "ABA-Net": [34.538292,	0.908250, 	0.000436, 	0.011472, 	0.836372, 	0.010498, 	0.001811, 	0.022333],
                   "ABA-Loss": [34.335894,	0.955402, 	0.000489, 	0.009629, 	0.841845, 	0.011779, 	0.002216, 	0.021605],
                   "ABA-FWI": [34.97732,	0.964317, 	0.000423, 	0.008871, 	0.843387, 	0.009887, 	0.001923, 	0.019691]},
    'CurveVelA': {
                  "ABA-Net": [24.102809,0.830786, 0.004767, 0.032166, 0.861971, 0.033955, 0.012552, 0.054597],
                  "ABA-Loss": [23.770575,0.825447, 0.005149, 0.033022, 0.861241, 0.034201, 0.013459, 0.056177],
                  "ABA-FWI": [24.812243,0.849809, 0.004073, 0.031965, 0.864743, 0.032258, 0.010308, 0.053147]},
    'CurveFaultA': {
                    "ABA-Net": [29.347719, 0.902384, 0.001389, 0.017483, 0.825597, 0.022047, 0.004984, 0.039679],
                    "ABA-Loss": [29.079473, 0.898061, 0.001564, 0.016249, 0.830261, 0.018497, 0.005897, 0.038724],
                    "ABA-FWI": [29.879054, 0.911255, 0.001297, 0.014707, 0.831338, 0.015587, 0.004976, 0.035452]},
}

for index in range(8):
    typ = index  # MSE:0 | MAE:1 | UIQ: 2 | LPIPS: 3
    typ_name = ""

    if typ == 0:
        typ_name = "PSNR"
        limset = (20, 50)
    elif typ == 1:
        typ_name = "SSIM"
        limset = (0.82, 1)
    elif typ == 2:
        typ_name = "MSE"
        limset = (0, 0.0050)  # 依据自己的指标值确定
    elif typ == 3:
        typ_name = "MAE"
        limset = (0, 0.035)
    elif typ == 4:
        typ_name = "UIQ"
        limset = (0.8, 0.87)
    elif typ == 5:
        typ_name = "LPIPS"
        limset = (0, 0.07)
    elif typ == 6:
        typ_name = "BMSE"
        limset = (0, 0.015)  # 依据自己的指标值确定
    else:
        typ_name = "BMAE"
        limset = (0, 0.06)  # 依据自己的指标值确定
    savename = "Ablation-{}".format(typ_name)

    legends_names = ["ABA-Net", "ABA-Loss", "ABA-FWI"]

    results = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

    for i, dataset in enumerate(dataset_dict):
        for j, strategy in enumerate(legends_names):
            results[j].update({dataset: dataset_dict[dataset][strategy][typ]})

    data_length = len(results[0])

    # 将极坐标根据数据长度进行等分

    angles = np.linspace(np.pi / 4, 2.25 * np.pi, data_length, endpoint=False)

    labels = [str(key) for key in results[0].keys()]

    score = [[v for v in result.values()] for result in results]

    # 使雷达图数据封闭

    score_a = np.concatenate((score[0], [score[0][0]]))

    score_b = np.concatenate((score[1], [score[1][0]]))

    score_c = np.concatenate((score[2], [score[2][0]]))

    angles = np.concatenate((angles, [angles[0]]))

    labels = np.concatenate((labels, [labels[0]]))

    # 设置图形的大小

    fig = plt.figure(figsize=(4, 4), dpi=200)

    # 新建一个子图

    ax = plt.subplot(111, polar=True)

    # 绘制雷达图

    ax.plot(angles, score_a, color='g')

    ax.plot(angles, score_b, color='b')

    ax.plot(angles, score_c, color='c')

    # 设置雷达图中每一项的标签显示

    ax.set_thetagrids(angles * 180 / np.pi, labels)

    # 设置雷达图的0度起始位置

    ax.set_theta_zero_location('N')

    # 设置雷达图的坐标刻度范围

    ax.set_rlim(limset)

    # 设置雷达图的坐标值显示角度，相对于起始角度的偏移量

    ax.set_rlabel_position(300)

    # ax.set_title(Title_str)
    plt.legend(legends_names, bbox_to_anchor=(0.38, 0))

    plt.subplots_adjust(left=0, bottom=0.195, right=1, top=0.88,
                        wspace=0.225, hspace=0)
    plt.show()
    fig.savefig(str(savename) + '.png')
