# import sys
# from pathlib import Path

# main_path = Path(__file__).resolve().parent.parent.parent
# if str(main_path) not in sys.path:
#     sys.path.append(str(main_path))
    
# from rplh.rendering.render_state import *
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.io as pio
# import os
# import json

# def trans_info_box(pg_state):
#     boxes = {}
#     for position, items in pg_state.items():
#         if items:  # Ensure items are not empty
#             position_tuple = tuple(map(float, position.split("_")))
#             boxes[position_tuple] = [item.split("_") for item in items]
#     return boxes

# def map_df(states):
#     if not states:  # Handle empty states
#         return pd.DataFrame(columns=["x", "y", "type", "color"])

#     data = []
#     y_offset_target = 0.15  # Offset for targets
#     y_offset_box = -0.15  # Offset for boxes
#     x_offset_factor = 0.2  # Horizontal offset factor for spreading items
#     max_x_offset = 0.4  # Clamp the maximum horizontal offset

#     for center, items in states.items():
#         cx, cy = center  # Extract center coordinates

#         # Separate items by type
#         targets = [item for item in items if item[0] == "target"]
#         boxes = [item for item in items if item[0] == "box"]

#         # Distribute targets horizontally
#         for i, item in enumerate(targets):
#             item_type, color = item
#             x_offset = (i - len(targets) / 2) * x_offset_factor
#             x_offset = max(-max_x_offset, min(x_offset, max_x_offset))  # Clamp offset
#             data.append(
#                 {
#                     "x": cx + x_offset,  # Apply x offset
#                     "y": cy + y_offset_target,  # Fixed y offset for targets
#                     "type": item_type,
#                     "color": color,
#                 }
#             )

#         # Distribute boxes horizontally
#         for i, item in enumerate(boxes):
#             item_type, color = item
#             x_offset = (i - len(boxes) / 2) * x_offset_factor
#             x_offset = max(-max_x_offset, min(x_offset, max_x_offset))  # Clamp offset
#             data.append(
#                 {
#                     "x": cx + x_offset,  # Apply x offset
#                     "y": cy + y_offset_box,  # Fixed y offset for boxes
#                     "type": item_type,
#                     "color": color,
#                 }
#             )

#     return pd.DataFrame(data)


# def render_maps_as_video(folder_path, renderer="notebook"):
#     """
#     Render multiple box_map states from JSON files as an animation in Jupyter Notebook or browser.

#     Args:
#         folder_path (str): Path to the folder containing JSON files with box_map data.
#         renderer (str): Plotly renderer to use ("notebook", "browser", etc.).
#     """
#     # Set the renderer
#     pio.renderers.default = renderer

#     # List all JSON files in the folder
#     file_names = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])

#     # Initialize the figure
#     fig = go.Figure()

#     # Collect frames for animation
#     frames = []
#     initial_transformed_box_map = None  # Store the first frame for fixed targets

#     for i, file_name in enumerate(file_names):
#         file_path = os.path.join(folder_path, file_name)
#         print(f"Processing file: {file_name}")

#         # Load the file content
#         with open(file_path, "r") as f:
#             box_map = json.load(f)

#         # Transform the box_map into the required format
#         transformed_box_map = trans_info_box(box_map)

#         # Save the first frame for fixed targets
#         if i == 0:
#             initial_transformed_box_map = transformed_box_map

#         # Separate fixed targets from moving boxes
#         targets = {k: [item for item in v if item[0] == "target"] for k, v in initial_transformed_box_map.items()}
#         moving_boxes = {k: [item for item in v if item[0] == "box"] for k, v in transformed_box_map.items()}

#         # Convert targets and boxes to DataFrames
#         targets_df = map_df(targets)
#         boxes_df = map_df(moving_boxes)

#         # Create hover text
#         hover_text_boxes = boxes_df.apply(
#             lambda row: f"Type: {row['type']}<br>Color: {row['color']}<br>X: {row['x']}<br>Y: {row['y']}",
#             axis=1,
#         )
#         hover_text_targets = targets_df.apply(
#             lambda row: f"Type: {row['type']}<br>Color: {row['color']}<br>X: {row['x']}<br>Y: {row['y']}",
#             axis=1,
#         )

#         # Add a frame for this state
#         frames.append(
#             go.Frame(
#                 data=[
#                     # Moving boxes
#                     go.Scatter(
#                         x=boxes_df["x"],
#                         y=boxes_df["y"],
#                         mode="markers",
#                         marker=dict(
#                             symbol="square",
#                             size=20,
#                             color=boxes_df["color"],
#                         ),
#                         text=hover_text_boxes,
#                         hoverinfo="text",
#                     ),
#                     # Fixed targets
#                     go.Scatter(
#                         x=targets_df["x"],
#                         y=targets_df["y"],
#                         mode="markers",
#                         marker=dict(
#                             symbol="diamond-open",
#                             size=20,
#                             color=targets_df["color"],
#                         ),
#                         text=hover_text_targets,
#                         hoverinfo="text",
#                     ),
#                 ],
#                 name=f"frame_{i}",
#             )
#         )

#     # Add the initial data (first frame)
#     if frames:
#         initial_df_targets = map_df({k: [item for item in v if item[0] == "target"] for k, v in initial_transformed_box_map.items()})
#         initial_df_boxes = map_df({k: [item for item in v if item[0] == "box"] for k, v in transformed_box_map.items()})

#         hover_text_boxes = initial_df_boxes.apply(
#             lambda row: f"Type: {row['type']}<br>Color: {row['color']}<br>X: {row['x']}<br>Y: {row['y']}",
#             axis=1,
#         )
#         hover_text_targets = initial_df_targets.apply(
#             lambda row: f"Type: {row['type']}<br>Color: {row['color']}<br>X: {row['x']}<br>Y: {row['y']}",
#             axis=1,
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=initial_df_boxes["x"],
#                 y=initial_df_boxes["y"],
#                 mode="markers",
#                 marker=dict(
#                     symbol="square",
#                     size=20,
#                     color=initial_df_boxes["color"],
#                 ),
#                 text=hover_text_boxes,
#                 hoverinfo="text",
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=initial_df_targets["x"],
#                 y=initial_df_targets["y"],
#                 mode="markers",
#                 marker=dict(
#                     symbol="diamond-open",
#                     size=20,
#                     color=initial_df_targets["color"],
#                 ),
#                 text=hover_text_targets,
#                 hoverinfo="text",
#             )
#         )

#     # Add frames to the figure
#     fig.frames = frames

#     # Add animation controls
#     fig.update_layout(
#     updatemenus=[
#         {
#             "buttons": [
#                 {
#                     "args": [None, {"frame": {"duration": 2000, "redraw": True}}],
#                     "label": "Play",
#                     "method": "animate",
#                 },
#                 {
#                     "args": [[None], {"frame": {"duration": 0, "redraw": True}}],
#                     "label": "Pause",
#                     "method": "animate",
#                 },
#             ],
#             "direction": "left",
#             "pad": {"r": 10, "t": 87},
#             "showactive": False,
#             "type": "buttons",
#             "x": 0.1,
#             "xanchor": "right",
#             "y": 0,
#             "yanchor": "top",
#         }
#     ],
#     title="Animated Box Movement",
#     xaxis=dict(
#         range=[0, 2],
#         title="X Coordinate",
#         tick0=0,
#         dtick=1,
#         showgrid=True,  # Enable gridlines
#         gridcolor="black",  # Gridline color
#         gridwidth=1,  # Gridline width
#     ),
#     yaxis=dict(
#         range=[0, 2],
#         title="Y Coordinate",
#         tick0=0,
#         dtick=1,
#         showgrid=True,  # Enable gridlines
#         gridcolor="black",  # Gridline color
#         gridwidth=1,  # Gridline width
#     ),
#     plot_bgcolor="white",
#     width=600,
#     height=600,
#     )
    
#     # Render the animation
#     fig.show()
