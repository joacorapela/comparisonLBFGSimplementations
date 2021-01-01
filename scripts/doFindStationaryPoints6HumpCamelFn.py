
import sys
import pdb
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def main(argv):
    ylim = [-1, 1]
    x1 = np.arange(start=-2, stop=2, step=.05)
    x2 = np.arange(start=-1, stop=1, step=.05)
    # x1 = np.arange(start=1, stop=1.75, step=.0001)
    # x2 = np.arange(start=0, stop=0.75, step=.0001)
    # x1 = np.arange(start=0, stop=0.1, step=.0001)
    # x2_0 = np.arange(start=-0.8, stop=-0.6, step=.0001)
    # x2_1 = np.arange(start=0.5, stop=0.6, step=.0001)
    # x2_2 = np.arange(start=-0.9, stop=-0.7, step=.0001)
    allData = np.zeros((len(x1)*len(x2), 3))

    # scx2 = lambda x1: -((-4.2*x1+4*x1**3/3)*x1**2+(4-2.1*x1**2+x1**4/3)*2*x1) # stationary candidate x2
    # scx1 = lambda x1: x1+8*scx2(x1)**3+(-4+4*scx2(x1)**2)*2*scx2(x1) # stationary candidate x1
    # scx2ForSX1 = lambda x1, x2: x1+8*x2**3+(-4+4*x2**2)*2*x2 # stationary candidate x2 for stationary x1
    sixHumpCamel = lambda x1, x2: (4-2.1*x1**2+x1**4/3)*x1**2+x1*x2+(-4+4*x2**2)*x2**2

    i = 0;
    for i1 in range(len(x1)):
        for i2 in range(len(x2)):
            allData[i, 0] = x1[i1]
            allData[i, 1] = x2[i2]
            allData[i, 2] = sixHumpCamel(x1=x1[i1], x2=x2[i2])
            i += 1

    fig = go.Figure(data=[go.Mesh3d(x=allData[:,0], y=allData[:,1], z=allData[:,2], opacity=0.50)])
    # fig = go.Figure(data=[go.Contour(x=allData[:,0], y=allData[:,1], z=allData[:,2], opacity=0.50)])
    fig.show()

    # X1, X2 = np.meshgrid(x1, x2)
    # Z = sixHumpCamel(x1=X1, x2=X2)
    # plt.contour(X1, X2, Z, levels=100)
    # plt.colorbar()
    # plt.show()
    # pdb.set_trace()

    # yscx1 = scx1(x1)
    # fig = go.Figure(data=go.Scatter(x=x1, y=yscx1, mode="lines+markers"))
    # fig.update_layout(yaxis_range=ylim)
    # fig.show()

    # stationaryX1 = np.array([0.0898, 1.6071, 1.7036])

    # yscx2_0 = scx2ForSX1(x1=stationaryX1[0], x2=x2_0)
    # fig = go.Figure(data=go.Scatter(x=x2_0, y=yscx2_0, mode="lines+markers"))
    # fig.update_layout(yaxis_range=ylim)
    # fig.show()

    # yscx2_1 = scx2ForSX1(x1=stationaryX1[1], x2=x2_1)
    # fig = go.Figure(data=go.Scatter(x=x2_1, y=yscx2_1, mode="lines+markers"))
    # fig.update_layout(yaxis_range=ylim)
    # fig.show()

    # yscx2_2 = scx2ForSX1(x1=stationaryX1[2], x2=x2_2)
    # fig = go.Figure(data=go.Scatter(x=x2_2, y=yscx2_2, mode="lines+markers"))
    # fig.update_layout(yaxis_range=ylim)
    # fig.show()

    # yscx2_1 = scx2ForSX1(x1=stationaryX1[1], x2=x2)
    # fig = go.Figure(data=go.Scatter(x=x2, y=yscx2_1, mode="lines+markers"))
    # fig.update_layout(yaxis_range=ylim)
    # fig.show()

    # yscx2_2 = scx2ForSX1(x1=stationaryX1[2], x2=x2)
    # fig = go.Figure(data=go.Scatter(x=x2, y=yscx2_2, mode="lines+markers"))
    # fig.update_layout(yaxis_range=ylim)
    # fig.show()

    # stationaryPoints = np.array([[-0.0898, -0.7126], [-0.0898, 0.0], [-0.0898, +0.7126], [-0.0000, -0.7126], [-0.0000, 0.0], [-0.0000, +0.7126], [+0.0898, -0.7126], [+0.0898, 0.0], [+0.0898, +0.7126]])
    # for i in range(stationaryPoints.shape[0]):
    #     x1 = stationaryPoints[i,0]
    #     x2 = stationaryPoints[i,1]
    #     y = sixHumpCamel(x1=x1, x2=x2)
    #     print("x1={:f}, x2={:f}, y={:f}".format(x1, x2, y))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
