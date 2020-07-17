def plot_change(model1, model2):
    diff = torch.nn.MSELoss()
    norm_w, norm_wx, norm_b, norm_bx, conv_w, conv_wx = [], [], [], [], [], []
    i = 0
    for (name, parameter1), parameter2 in  zip(model1.named_parameters(), model2.parameters()):
        name = name.split(".")
        if name[-2][:4] == "conv":
            if name[-1] == "weight":
                conv_w.append(diff(parameter1, parameter2).item())
                conv_wx.append(i)
        elif name[-2][:4] == "norm":
            if name[-1] == "weight":
                norm_w.append(diff(parameter1, parameter2).item())
                norm_wx.append(i)
            elif name[-1] == "bias":
                norm_b.append(diff(parameter1, parameter2).item())
                norm_bx.append(i)
        i = i + 1

    plt.bar(norm_wx, norm_w, label="BN gamma")
    plt.bar(norm_bx, norm_b, label="BN beta")
    plt.legend()
    plt.show()

    plt.bar(conv_wx, conv_w, label="CONV weights")
    plt.legend()
    plt.show()
