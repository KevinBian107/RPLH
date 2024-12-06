import plotly.express as px
import pandas as pd
import re
import copy
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import plotly.express as px

pio.renderers.default = "plotly_mimetype"
from PIL import Image

import sys
from pathlib import Path

main_path = Path(__file__).resolve().parent.parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))

# from rplh.h_vanilla.memory import *
import os
import json
import numpy as np


import shutil
import random


def render_animate_terminal_popup(box_map, action_list):
    box_map, action_list = trans_data(box_map, action_list)
    fig = render_animate(box_map, action_list[0])

    pio.renderers.default = "browser"
    fig.show()

    print("Graph displayed in a browser pop-up window.")


def render_graph_terminal_popup(box_map, pg_row_num, pg_column_num):
    fig = render_graph(
        box_map=trans_info_box(box_map),
        pg_row_num=pg_row_num,
        pg_column_num=pg_column_num,
    )

    pio.renderers.default = "browser"
    fig.show()
    print("Graph displayed in a browser pop-up window.")


def render_animate_terminal_popup(box_map, action_list):
    box_map, action_list = trans_data(box_map, action_list)
    render_animate(box_map, action_list[0])
    print("Graph displayed in a browser pop-up window.")


# ------------------- preparation function ------------
# setup_function
def position_int(position):
    return tuple([float(pos) for pos in position.split("_")])


def split_info(info_list):
    return [info.split("_") for i, info in enumerate(info_list)]


# ---------------- Transforming Box_map ---------------
def trans_info_box(pg_state):
    boxes = {}
    count = 0
    for position, info in pg_state.items():
        boxes[position_int(position)] = split_info(info)
        for lst in boxes[position_int(position)]:
            lst.append(count)
            count += 1
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
    if sum([len(state) for state in states.values()]) == 0:
        return pd.DataFrame(
            {
                "x": [None],
                "y": [None],
                "center": [None],
                "type": [None],
                "color": [None],
                "id": [None],
            }
        )

    # Prepare data for plotting
    data = []
    offset_factor = 0.15  # Offset factor for spreading items within each 1x1 box
    max_offset = 0.45  # Ensure all items stay within the 1x1 box

    for center, items in states.items():
        cx, cy = center  # Extract center coordinates

        # Separate targets and boxes
        targets = [item for item in items if item[0] == "target"]
        boxes = [item for item in items if item[0] == "box"]
        count = 1

        # Distribute targets at the top within the box
        for i, item in enumerate(targets):
            item_type, color, id = item
            position_factor = min(5, len(targets))
            if count <= 5:
                x_offset = (i - int(position_factor / 2)) * offset_factor
                y_offset = 0.35  # Fixed offset to place targets above the center
            else:
                x_offset = (i - 5 - int(position_factor / 2)) * offset_factor
                y_offset = 0.15  # Fixed offset to place targets above the center
            data.append(
                {
                    "x": cx + x_offset,
                    "y": cy + y_offset,
                    "center": (cx, cy),
                    "type": item_type,
                    "color": color,
                    "id": id,
                }
            )
            count += 1

        count = 1
        # Distribute boxes at the bottom within the box
        for i, item in enumerate(boxes):
            item_type, color, id = item
            position_factor = min(5, len(boxes))
            if count <= 5:
                x_offset = (i - int(position_factor / 2)) * offset_factor
                y_offset = -0.35  # Fixed offset to place boxes below the center
            else:
                x_offset = (i - 5 - int(position_factor / 2)) * offset_factor
                y_offset = -0.15  # Fixed offset to place boxes below the center
            data.append(
                {
                    "x": cx + x_offset,
                    "y": cy + y_offset,
                    "center": (cx, cy),
                    "type": item_type,
                    "color": color,
                    "id": id,
                }
            )
            count += 1

    df = pd.DataFrame(data)
    return df


# ------------------- apply action to map ---------------


def apply_action(
    pg_dict_input,  # {(0.5, 0.5): [['box', 'blue', 0], ['target', 'blue', 1], ['target', 'red', 2]]}
    original_response_dict,  # {(0.5, 0.5): ('box_blue', 'target_blue'), (1.5, 1.5): ('box_red', (1.5, 0.5))}
):
    """
    Updates the environment state based on the actions in the response.

    Args:
        pg_dict_input (dict[str, list[str]]): Current state of the playground.
        original_response_dict (dict): Actions to be executed.

    """
    system_error_feedback = ""
    pg_dict_original = copy.deepcopy(pg_dict_input)
    transformed_dict = {}
    for key, value in original_response_dict.items():
        coordinates = key
        # match the item and location in the value
        item, location = value
        if type(location) == tuple:
            location = location
        transformed_dict[coordinates] = [item, location]

    remove_item = []

    for key, value in transformed_dict.items():
        # print(f"Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
        condition_1 = False
        condition_2 = False
        for i, element in enumerate(pg_dict_original[key]):
            if element[0] + "_" + element[1] == value[0]:
                condition_1 = True
                location_1 = element
            if element[0] + "_" + element[1] == value[1]:
                condition_2 = True
                location_2 = element

        if (
            condition_1
            and type(value[1]) == tuple
            and (
                (
                    np.abs(key[0] - value[1][0]) == 0
                    and np.abs(key[1] - value[1][1]) == 1
                )
                or (
                    np.abs(key[0] - value[1][0]) == 1
                    and np.abs(key[1] - value[1][1]) == 0
                )
            )
        ):
            pg_dict_original[key].remove(location_1)
            pg_dict_original[(value[1][0], value[1][1])].append(location_1)
        elif (
            condition_1
            and type(value[1]) == str
            and condition_2
            and value[0][:4] == "box_"
            and value[1][:7] == "target_"
            and value[0][4:] == value[1][7:]
        ):
            pg_dict_original[key].remove(location_1)
            pg_dict_original[key].remove(location_2)
            remove_item.append((location_1, location_2))
        else:
            # print(f"Error, Iteration Num: {iteration_num}, Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
            system_error_feedback += f"Your assigned task for {key[0]}_{key[1]} is not in the doable action list; "

    return system_error_feedback, remove_item, pg_dict_original


# ------------- plot single map ---------------


def render_graph(box_map):
    df = map_df(box_map)
    df["hover_text"] = df.apply(
        lambda row: f"Type: {row['type']}<br>Color: {row['color']}", axis=1
    )

    # Initialize the Plotly figure


def render_graph(box_map, pg_row_num, pg_column_num):
    fig = go.Figure()
    robot = Image.open("demos/robot.png")

    # Add rectangles and robot images for the "boxes"
    for center in box_map.keys():
        cx, cy = center
        # Add a rectangle representing the box
        fig.add_shape(
            type="rect",
            x0=cx - 0.5,
            y0=cy - 0.5,
            x1=cx + 0.5,
            y1=cy + 0.5,
            line=dict(color="black", width=2),
        )
        # Add a robot image in the center of the box
        fig.add_layout_image(
            dict(
                source=robot,  # Path or URL to the robot image
                x=cx,
                y=cy,
                xref="x",
                yref="y",
                sizex=0.3,  # Robot image size
                sizey=0.3,
                xanchor="center",
                yanchor="middle",
                layer="above",  # Layer position
            )
        )

    # Update the layout for better visual appearance
    fig.update_layout(
        title=dict(
            text="Moving Box to The Right Target",
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=18),
        ),
        xaxis=dict(
            range=[0, pg_row_num], showgrid=False, zeroline=False, title="X Coordinate"
        ),
        yaxis=dict(
            range=[0, pg_column_num],
            showgrid=False,
            zeroline=False,
            title="Y Coordinate",
        ),
        plot_bgcolor="white",
        width=600,
        height=600,
    )

    if sum([len(i) for i in box_map.values()]) == 0:
        return fig

    df = map_df(box_map)
    df["hover_text"] = df.apply(
        lambda row: f"Type: {row['type']}<br>Color: {row['color']}", axis=1
    )

    # Add scatter trace for the points
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                symbol=[
                    "diamond-open" if t == "target" else "square" for t in df["type"]
                ],
                size=20,
                color=df["color"],  # Marker colors from 'color' column
            ),
            text=df["hover_text"],  # Hover information
            hoverinfo="text",
        )
    )

    return fig


def construct_plotting_df(box_map, actions, num_frames):
    interpolated_frames = []
    box_map = copy.deepcopy(box_map)
    df_init = map_df(box_map)
    frame_counter = 0  # Keep track of the global frame number

    # Loop through each action
    for action in actions:
        # Apply the action
        _, remove_item, box_map_after = apply_action(box_map, action)
        df_after = map_df(box_map_after)

        # Generate interpolated frames between df_init and df_after
        for frame in range(num_frames + 1):  # Include final frame
            alpha = frame / num_frames
            interpolated_frame = df_init.copy()
            # Add metadata for the current frame
            interpolated_frame["frame"] = frame_counter

            interpolated_frame["size"] = 100  # Fixed size for uniform markers
            for id_num in interpolated_frame["id"].unique():
                # Select the rows corresponding to the current ID

                init_row = df_init.query(f"id == {id_num}")
                after_row = df_after.query(f"id == {id_num}")
                if not after_row.empty:
                    # Interpolate x and y values for the current ID
                    x_after = after_row["x"].values[0]
                    y_after = after_row["y"].values[0]

                else:
                    for pair in remove_item:
                        if (
                            id_num in pair[0]
                            or id_num in pair[1]
                            or df_after.shape[0] == 0
                        ):
                            x_after = df_init.query(f"id == {pair[1][2]}")[
                                "center"
                            ].values[0][0]
                            y_after = df_init.query(f"id == {pair[1][2]}")[
                                "center"
                            ].values[0][1]

                # Perform interpolation
                interpolated_frame.loc[interpolated_frame["id"] == id_num, "x"] = (
                    1 - alpha
                ) * init_row["x"].values[0] + alpha * x_after
                interpolated_frame.loc[interpolated_frame["id"] == id_num, "y"] = (
                    1 - alpha
                ) * init_row["y"].values[0] + alpha * y_after
                interpolated_frame["size"] = (1 - alpha) * 100

            interpolated_frame = interpolated_frame.drop_duplicates(subset="id")

            # Append the current frame to the list
            interpolated_frames.append(interpolated_frame)
            frame_counter += 1  # Increment global frame counter

        # Update df_init and box_map for the next action
        df_init = df_after
        box_map = box_map_after

        if pd.isna(df_after.iloc[0]["x"]):
            break

    # Combine all frames into one DataFrame
    df_combined = pd.concat(interpolated_frames, ignore_index=True)
    df_combined["hover_info"] = (
        df_combined["color"]
        + " | "
        + df_combined["type"]
        + " | "
        + df_combined["id"].apply(str)
    )
    return df_combined


def render_animate(box_map, actions, num_frames=2):
    """
    Create an animated scatter plot showing smooth transitions between points for a sequence of actions.

    Args:
        box_map (dict): Initial box map.
        actions (list): List of actions to apply.
        num_frames (int): Number of frames for smooth transition between each action.

    Returns:
        plotly.express.scatter object with animation.
    """
    # Initialize variables
    df_combined = construct_plotting_df(box_map, actions, num_frames)
    # Create animation using plotly express
    fig = px.scatter(
        df_combined,
        x="x",
        y="y",
        animation_frame="frame",
        animation_group="id",
        color="color",
        symbol="type",  # Map symbol column to scatter shapes
        hover_name="hover_info",  # Use the new combined column for hover info
        size="size",  # Ensure the size column is applied
        range_x=[0, 2],
        range_y=[0, 2],
    )

    robot = Image.open("robot.png")

    # Add rectangles and robot images for the "boxes"
    for center in box_map.keys():
        cx, cy = center
        # Add a rectangle representing the box
        fig.add_shape(
            type="rect",
            x0=cx - 0.5,
            y0=cy - 0.5,
            x1=cx + 0.5,
            y1=cy + 0.5,
            line=dict(color="black", width=2),
        )
        # Add a robot image in the center of the box
        fig.add_layout_image(
            dict(
                source=robot,  # Path or URL to the robot image
                x=cx,
                y=cy,
                xref="x",
                yref="y",
                sizex=0.3,  # Robot image size
                sizey=0.3,
                xanchor="center",
                yanchor="middle",
                layer="above",  # Layer position
            )
        )

    # Update the layout for better visual appearance
    fig.update_layout(
        showlegend=False,
        title=dict(
            text="Moving Boxes Through a Sequence of Actions",
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=18),
        ),
        xaxis=dict(range=[0, 2], showgrid=False, zeroline=False, title="X Coordinate"),
        yaxis=dict(range=[0, 2], showgrid=False, zeroline=False, title="Y Coordinate"),
        plot_bgcolor="white",
        width=600,
        height=600,
    )
    return fig, df_combined
