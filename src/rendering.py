import os
import math
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_3D_AVAILABLE = True
except ImportError:
    MATPLOTLIB_3D_AVAILABLE = False


# Load the .glb file and rotate the x-axis by 90 degrees, to get a front facing, upright pottery
def render_glb_matplotlib(glb_path, output_size=(512, 512)):
    try:
        fig = plt.figure(figsize=(output_size[0] / 100, output_size[1] / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        mesh = trimesh.load(glb_path)
        if hasattr(mesh, 'geometry'):
            mesh = list(mesh.geometry.values())[0]

        rotation_matrix = trimesh.transformations.rotation_matrix(angle=np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0])
        mesh.apply_transform(rotation_matrix)
        vertices, faces = mesh.vertices, mesh.faces

        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, alpha=0.9, cmap='copper', linewidth=0, antialiased=True)
        ax.view_init(elev=10, azim=0)

        max_range = np.array([vertices[:, i].max() - vertices[:, i].min() for i in range(3)]).max() / 2.0
        mid = np.mean(vertices, axis=0)

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_axis_off()
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return buf
    
    except Exception as e:
        print(f"Error with matplotlib rendering {glb_path}: {e}")
        return np.ones((*output_size[::-1], 3), dtype=np.uint8) * 180


# Renders a front view of a .glb file and returns it as a numpy array.
def render_glb_front_view(glb_path, output_size=(512, 512)):
    if MATPLOTLIB_3D_AVAILABLE:
        return render_glb_matplotlib(glb_path, output_size)

    # Return blank if rendering fails
    return np.ones((*output_size[::-1], 3), dtype=np.uint8) * 200


# Creates a collage of pottery models for a given cluster.
def create_cluster_collage(pottery_ids, pottery_dir, cluster_id, output_dir, image_size=(256, 256), collage_columns=None):
    num_items = len(pottery_ids)
    if collage_columns is None:
        collage_columns = min(5, int(math.ceil(math.sqrt(num_items))))

    collage_rows = int(math.ceil(num_items / collage_columns))
    collage = Image.new('RGB', (collage_columns * image_size[0], collage_rows * image_size[1]), color=(240, 240, 240))
    rendered_count = 0

    for idx, pottery_id in enumerate(tqdm(pottery_ids, desc=f"Cluster {cluster_id}")):
        row, col = idx // collage_columns, idx % collage_columns
        glb_files = [f for f in os.listdir(pottery_dir) if f.startswith(pottery_id) and f.endswith('.glb')]

        if glb_files:
            try:
                rendered_image = render_glb_front_view(os.path.join(pottery_dir, glb_files[0]), image_size)
                pil_image = Image.fromarray(rendered_image)
                rendered_count += 1
            except Exception as e:
                print(f"Failed to render {pottery_id}: {e}")

            draw = ImageDraw.Draw(pil_image)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), pottery_id, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_x, text_y = 5, image_size[1] - text_height - 10
            draw.rectangle([text_x, text_y - 2, text_x + text_width + 4, text_y + text_height + 2], fill=(255, 255, 255, 180))
            draw.text((text_x + 2, text_y), pottery_id, fill=(0, 0, 0), font=font)

        collage.paste(pil_image, (col * image_size[0], row * image_size[1]))

    collage_path = os.path.join(output_dir, f"cluster_{cluster_id}.png")
    collage.save(collage_path, "PNG")
    return collage_path
