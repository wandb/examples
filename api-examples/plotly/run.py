import plotly.graph_objs as go
import matplotlib.pyplot as plt
import wandb
wandb.init()
contour = go.Figure(
                data=go.Contour(
                    z=[[10, 10.625, 12.5, 15.625, 20],
                       [5.625, 6.25, 8.125, 11.25, 15.625],
                       [2.5, 3.125, 5., 8.125, 12.5],
                       [0.625, 1.25, 3.125, 6.25, 10.625],
                       [0, 0.625, 2.5, 5.625, 10]]),
                layout=go.Layout(
                    title=go.layout.Title(text="A Bar Chart")))
scatter = go.Figure(
        data=go.Scatter(x=[0, 1, 2]),
        layout=go.Layout(
            title=go.layout.Title(text="A Bar Chart")))

plt.plot([1, 2, 3, 4])
plt.ylabel('some interesting numbers')

wandb.log({'contour': contour, 'scatter': scatter, 'mpl': plt})
