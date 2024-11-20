import plotly.express as px
import pandas as pd
import re
import copy
import plotly.graph_objects as go
import plotly.io as pio


def render_graph_terminal_popup(box_map):
    fig = render_graph(box_map=trans_info_box(box_map))

    pio.renderers.default = "browser"
    fig.show()
    print("Graph displayed in a browser pop-up window.")


def render_map_terminal_popup(box_map, action_list):
    box_map, action_list = trans_data(box_map, action_list)
    render_map(box_map, action_list)
    print("Graph displayed in a browser pop-up window.")


# ------------------- preparation function ------------
# setup_function
def position_int(position):
    return tuple([float(pos) for pos in position.split("_")])


def split_info(info_list):
    return [info.split("_") for info in info_list]


# ---------------- Transforming Box_map ---------------
def trans_info_box(pg_state):
    boxes = {}
    for position, info in pg_state.items():
        boxes[position_int(position)] = split_info(info)
    return boxes


# ---------------- Transforming action ------------
def transform_action(action):
    trans_action = {}
    for cent, act in action.items():
        pattern = r"Agent\[(\d+\.\d+), (\d+\.\d+)\]"
        match = re.search(pattern, cent)
        cent = (float(match.group(1)), float(match.group(2)))

        trans_action[cent] = parse_action(act)
    return trans_action


def parse_action(action):
    """
    Parse an action string into its components.

    Args:
        action (str): The action string to parse (e.g., 'move(box_red, target_blue)' or 'move(box_red, square[1.5, 1.5])').

    Returns:
        tuple: A tuple containing the moving_item and the target (both as strings).
    """
    # Use regex to extract the command and details
    match = re.match(r"(\w+)\((.*)\)", action)

    if not match:
        raise ValueError(f"Invalid action format: {action}")

    details = [detail.strip() for detail in match.group(2).split(",")]

    # Parse the moving item and target
    moving_item = details[0]
    if len(details) == 2:
        target = details[1]
    else:
        target = details[1] + ", " + details[2]
        target = tuple(
            map(float, re.findall(r"\[([\d\.\-, ]+)\]", target)[0].split(","))
        )

    return moving_item, target


# ------------------------------ Preparation for visualizing the data
def trans_data(box_map, action_list):
    """
    action_list: Each element in the list represents all actions at that time
    box_map: It is the initial state of boxes
    """
    return trans_info_box(box_map), [transform_action(action) for action in action_list]


# ------------ transform box_map to a data Frame ----------------
def map_df(states):

    # Prepare data for plotting
    data = []
    offset_factor = 0.15  # Offset factor for spreading items within each 1x1 box
    max_offset = 0.45  # Ensure all items stay within the 1x1 box

    for center, items in states.items():
        cx, cy = center  # Extract center coordinates

        # Separate targets and boxes
        targets = [item for item in items if item[0] == "target"]
        boxes = [item for item in items if item[0] == "box"]

        # Distribute targets at the top within the box
        for i, item in enumerate(targets):
            item_type, color = item
            x_offset = (i - len(targets) / 2) * offset_factor
            x_offset = max(
                -max_offset, min(x_offset, max_offset)
            )  # Clamp within box bounds
            y_offset = 0.25  # Fixed offset to place targets above the center
            data.append(
                {
                    "x": cx + x_offset,
                    "y": cy + y_offset,
                    "type": "target",
                    "color": color,
                }
            )

        # Distribute boxes at the bottom within the box
        for i, item in enumerate(boxes):
            item_type, color = item
            x_offset = (i - len(boxes) / 2) * offset_factor
            x_offset = max(
                -max_offset, min(x_offset, max_offset)
            )  # Clamp within box bounds
            y_offset = -0.25  # Fixed offset to place boxes below the center
            data.append(
                {
                    "x": cx + x_offset,
                    "y": cy + y_offset,
                    "type": "box",
                    "color": color,
                }
            )

    df = pd.DataFrame(data)
    return df


# ------------------- apply action to map ---------------


def apply_action(box_map, actions):
    """
    Apply a set of actions to the given box_map.

    Args:
        box_map (dict): A dictionary where keys are positions (tuples) and values are lists of items.
        actions (dict): A dictionary where keys are positions (tuples) and values are actions.

    Returns:
        tuple: Updated box_map, the moving_item, and the target.
    """
    updated_box_map = copy.deepcopy(box_map)

    for position, action in actions.items():
        center = position  # The position is already a tuple

        # Use the parse_action helper function
        moving_item, target = action

        # Process the action
        if "target" in target:  # Remove both moving item and target
            target_item = next(
                (
                    item
                    for item in updated_box_map.get(center, [])
                    if f"{item[0]}_{item[1]}" == target
                ),
                None,
            )
            if target_item:
                updated_box_map[center].remove(target_item)

            moving_item_obj = next(
                (
                    item
                    for item in updated_box_map.get(center, [])
                    if f"{item[0]}_{item[1]}" == moving_item
                ),
                None,
            )
            if moving_item_obj:
                updated_box_map[center].remove(moving_item_obj)
        else:  # Move the item to a new square
            # Parse the target square coordinates
            target_coords = target  # tuple(map(float, re.findall(r"\[([\d\.\-, ]+)\]", target)[0].split(",")))

            # Find and remove only one instance of the moving item
            moving_item_obj = next(
                (
                    item
                    for item in updated_box_map.get(center, [])
                    if f"{item[0]}_{item[1]}" == moving_item
                ),
                None,
            )
            if moving_item_obj:
                updated_box_map[center].remove(moving_item_obj)

                # Add the moving item to the target location
                updated_box_map.setdefault(target_coords, []).append(moving_item_obj)

    return updated_box_map, moving_item, target


# ------------- plot single map ---------------


def render_map(initial_map, action_list):
    """
    Visualize the process of moving boxes step-by-step based on an initial map and a list of actions.

    Args:
        initial_map (dict): Initial state of the box_map.
        action_list (list[dict]): A list of actions, where each action is a dictionary of {position: action_string}.
    """
    # Generate sequential states
    current_state = copy.deepcopy(initial_map)
    state_list = [current_state]
    for action in action_list:
        current_state, moving_item, target = apply_action(current_state, action)
        state_list.append(current_state)

    iteration = 0
    for state in state_list:

        df = map_df(state)
        hover_text = df.apply(
            lambda row: f"Type: {row['type']}<br>Color: {row['color']}<br>X: {row['x']}<br>Y: {row['y']}",
            axis=1,
        )

        # Initialize the figure
        pio.renderers.default = "browser"
        fig = go.Figure()

        # Add scatter trace
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker=dict(
                    symbol=[
                        "diamond-open" if t == "target" else "square"
                        for t in df["type"]
                    ],  # Use square symbols
                    size=20,  # Set the size of the squares
                    color=df["color"],  # Use the 'color' column for marker colors
                ),
                text=hover_text,  # Display 'type' on hover
                hoverinfo="text",
            )
        )

        # Add rectangles for visualizing the "boxes"
        for center in current_state.keys():
            cx, cy = center
            fig.add_shape(
                type="rect",
                x0=cx - 0.5,
                y0=cy - 0.5,
                x1=cx + 0.5,
                y1=cy + 0.5,
                line=dict(color="black", width=2),
            )

        # Update layout to match the desired visual appearance
        fig.update_layout(
            title=dict(
                text=(
                    f"Box Item Plot with state {iteration}"
                    if iteration < 1
                    else f"Box Item Plot with state {iteration}<br>action {action}"
                ),  # Use the formatted title
                x=0.5,  # Center the title horizontally
                xanchor="center",
                yanchor="top",
                font=dict(size=10),  # Adjust font size if needed
            ),
            xaxis=dict(
                range=[0, 2], showgrid=False, zeroline=False, title="X Coordinate"
            ),
            yaxis=dict(
                range=[0, 2], showgrid=False, zeroline=False, title="Y Coordinate"
            ),
            plot_bgcolor="white",
            width=600,
            height=600,
        )

        # Show the plot
        fig.show()
        iteration += 1

    return df, fig


def render_graph(box_map):
    df = map_df(box_map)
    df["hover_text"] = df.apply(
        lambda row: f"""
                                    Type: {row['type']}<br>
                                     Color: {row['color']}<br>
                                     X: {row['x']}<br>
                                     Y: {row['y']}
                                    """,
        axis=1,
    )

    # Initialize the figure
    fig = go.Figure()

    # Add scatter trace
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                symbol=[
                    "diamond-open" if t == "target" else "square" for t in df["type"]
                ],  # Use square symbols
                size=20,  # Set the size of the squares
                color=df["color"],  # Use the 'color' column for marker colors
            ),
            text=df["hover_text"],  # Display 'type' on hover
            hoverinfo="text",
        )
    )

    # Add rectangles for visualizing the "boxes"
    for center in box_map.keys():
        cx, cy = center
        fig.add_shape(
            type="rect",
            x0=cx - 0.5,
            y0=cy - 0.5,
            x1=cx + 0.5,
            y1=cy + 0.5,
            line=dict(color="black", width=2),
        )

    # Update layout to match the desired visual appearance
    fig.update_layout(
        title=dict(
            text=f"Box Item Plot",  # Use the formatted title
            x=0.5,  # Center the title horizontally
            xanchor="center",
            yanchor="top",
            font=dict(size=18),  # Adjust font size if needed
        ),
        xaxis=dict(range=[0, 2], showgrid=False, zeroline=False, title="X Coordinate"),
        yaxis=dict(range=[0, 2], showgrid=False, zeroline=False, title="Y Coordinate"),
        plot_bgcolor="white",
        width=600,
        height=600,
    )

    # Show the plot
    return fig


# ---------------------------- Animation ------------------------------------------#


# import copy
# import plotly.graph_objects as go
# import pandas as pd
# import numpy as np

# def render_map_animate(initial_map, action_list):
#     """
#     Visualize the process of moving boxes step-by-step based on an initial map and a list of actions.

#     Args:
#         initial_map (dict): Initial state of the box_map.
#         action_list (list[dict]): A list of actions, where each action is a dictionary of {position: action_string}.
#     """
#     # Generate sequential states
#     current_state = copy.deepcopy(initial_map)
#     state_list = [current_state]

#     for action in action_list:
#         current_state, _, _ = apply_action(current_state, action)
#         state_list.append(current_state)

#     # Initialize the figure
#     fig = go.Figure()

#     # Generate animation frames
#     frames = []
#     for i, (state, actions) in enumerate(zip(state_list[:-1], action_list)):
#         # Extract the current and next states
#         df = map_df(state)
#         next_state = state_list[i + 1]
#         next_df = map_df(next_state)

#         # Create a frame for each action
#         for pos, (box, target) in actions.items():
#             # Extract initial and target positions
#             box_row = df[df['type'] == box]
#             if isinstance(target, tuple):
#                 target_x, target_y = target  # Coordinates provided directly
#             else:
#                 # Look up the target's coordinates in the DataFrame
#                 target_row = df.query(f"color == '{target.split('_')[1]}' and type == 'target'")
#                 if not target_row.empty:
#                     target_x, target_y = target_row[['x', 'y']].iloc[0]
#                 else:
#                     # Log the issue and skip this action
#                     raise ValueError(f"Warning: Target '{target}' not found in the current state.")
#             print(box_row['x'].iloc[0])
#             print(box_row['y'].iloc[0])

#             print(target_x, target_y)
#             # Interpolate movement
#             n_steps = 10  # Number of animation steps
#             x_interp = np.linspace(box_row['x'].iloc[0], target_x, n_steps)
#             y_interp = np.linspace(box_row['y'].iloc[0], target_y, n_steps)

#             for step in range(n_steps):
#                 # Modify the DataFrame for the current animation step
#                 temp_df = df.copy()
#                 temp_df.loc[temp_df['type'] == box, 'x'] = x_interp[step]
#                 temp_df.loc[temp_df['type'] == box, 'y'] = y_interp[step]

#                 # Create hover text
#                 hover_text = temp_df.apply(
#                     lambda row: f"Type: {row['type']}<br>Color: {row['color']}<br>X: {row['x']}<br>Y: {row['y']}",
#                     axis=1
#                 )

#                 # Add a frame for the current step
#                 frames.append(
#                     go.Frame(
#                         data=[
#                             go.Scatter(
#                                 x=temp_df['x'],
#                                 y=temp_df['y'],
#                                 mode='markers',
#                                 marker=dict(
#                                     symbol=['square-open' if t == 'target' else 'square' for t in temp_df['type']],
#                                     size=20,
#                                     color=temp_df['color']
#                                 ),
#                                 text=hover_text,
#                                 hoverinfo='text'
#                             )
#                         ]
#                     )
#                 )
#             # Mark disappearance for actions like `target_blue`
#             if target == f"target_{box.split('_')[1]}":  # If box reaches its target
#                 temp_df = temp_df[temp_df['type'] != box]
#                 hover_text = temp_df.apply(
#                     lambda row: f"Type: {row['type']}<br>Color: {row['color']}<br>X: {row['x']}<br>Y: {row['y']}",
#                     axis=1
#                 )
#                 frames.append(
#                     go.Frame(
#                         data=[
#                             go.Scatter(
#                                 x=temp_df['x'],
#                                 y=temp_df['y'],
#                                 mode='markers',
#                                 marker=dict(
#                                     symbol=['square-open' if t == 'target' else 'square' for t in temp_df['type']],
#                                     size=20,
#                                     color=temp_df['color']
#                                 ),
#                                 text=hover_text,
#                                 hoverinfo='text'
#                             )
#                         ]
#                     )
#                 )

#     # Add the initial state as the first frame
#     initial_df = map_df(state_list[0])
#     hover_text = initial_df.apply(
#         lambda row: f"Type: {row['type']}<br>Color: {row['color']}<br>X: {row['x']}<br>Y: {row['y']}",
#         axis=1
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=initial_df['x'],
#             y=initial_df['y'],
#             mode='markers',
#             marker=dict(
#                 symbol=['square-open' if t == 'target' else 'square' for t in initial_df['type']],
#                 size=20,
#                 color=initial_df['color']
#             ),
#             text=hover_text,
#             hoverinfo='text'
#         )
#     )

#     # Add the frames to the figure
#     fig.frames = frames

#     # Add animation controls
#     fig.update_layout(
#         updatemenus=[
#             {
#                 "buttons": [
#                     {
#                         "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
#                         "label": "Play",
#                         "method": "animate"
#                     },
#                     {
#                         "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
#                         "label": "Pause",
#                         "method": "animate"
#                     }
#                 ],
#                 "direction": "left",
#                 "pad": {"r": 10, "t": 87},
#                 "showactive": False,
#                 "type": "buttons",
#                 "x": 0.1,
#                 "xanchor": "right",
#                 "y": 0,
#                 "yanchor": "top"
#             }
#         ]
#     )

#     # Update layout for the plot
#     fig.update_layout(
#         title="Animated Box Movement",
#         xaxis=dict(range=[0, 2], title="X Coordinate"),
#         yaxis=dict(range=[0, 2], title="Y Coordinate"),
#         plot_bgcolor='white',
#         width=600,
#         height=600
#     )

#     fig.show()
