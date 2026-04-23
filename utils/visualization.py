import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_learning_curve(rewards: list, smoothing_window: int = 10, title="Learning Curve"):
    """
    Plots a smoothed learning curve using Plotly.
    """
    if len(rewards) == 0:
        return go.Figure()

    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - smoothing_window + 1)
        smoothed.append(np.mean(rewards[start:i+1]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rewards, mode='lines', name='Raw Reward', opacity=0.3))
    fig.add_trace(go.Scatter(y=smoothed, mode='lines', name='Smoothed Reward'))

    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Reward",
        template="plotly_white"
    )
    return fig

def render_maze_cli(maze_grid: np.ndarray, agent_pos: tuple, target_pos: tuple):
    """
    Renders the maze as text in the CLI.
    """
    h, w = maze_grid.shape
    output = []
    for r in range(h):
        row_str = ""
        for c in range(w):
            if (r, c) == agent_pos:
                row_str += "A"
            elif (r, c) == target_pos:
                row_str += "T"
            elif maze_grid[r, c] == 1: # WALL
                row_str += "█"
            else:
                row_str += " "
        output.append(row_str)
    return "\n".join(output)
