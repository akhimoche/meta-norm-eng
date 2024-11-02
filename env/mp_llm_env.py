import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional


class LLMPrepObject:
    """A wrapper for melting pot environments to allow for LLM compatibility.
    Outputs a dictionary of states from an image. currently tested for global obs in
        - `commons_harvest__open`
    """
    def __init__(self, label_folder: Path) -> None:
        self.label_folder = label_folder
        self.load_sprite_labels(label_folder)
        #self.knn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.sprites)

    def load_sprite_labels(self, label_folder: Path):
        sprites = []
        sprite_labels = []
        for filename in os.listdir(label_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image = np.array(Image.open(os.path.join(label_folder, filename)))
                sprites.append(image.flatten())
                sprite_labels.append(filename.split('.')[0])
        self.sprites = np.array(sprites)
        self.sprite_labels = np.array(sprite_labels)

    def exact_pixel_match(self, patch):
        patch_vector = patch.flatten()
        for ref_img, label in zip(self.sprites, self.sprite_labels):
            if np.array_equal(patch_vector, ref_img):
                return label

        return None

    def image_to_state(self, image: np.ndarray):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        width, height = image.size
        sprite_size = 8  # sprite size is always 8 for melting pot
        self.width = width
        self.height = height
        self.sprite_size = sprite_size

        patch_number = 0
        states = {'global': {}}
        patch_coords = []

        timestamp = int(time.time())
        date_time = datetime.fromtimestamp(timestamp)
        timestamp = date_time.strftime("%m%d_%H%M%S")  # Get the current timestamp

        for y in range(0, height, sprite_size):
            for x in range(0, width, sprite_size):
                box = (x, y, x + sprite_size, y + sprite_size)
                patch = np.array(image.crop(box))
                label = self.exact_pixel_match(patch)
                if label is None:
                    # save patch in unlabeled_patches folder
                    patch_img = Image.fromarray(patch)
                    # Get the directory one level up
                    sprite_labels_folder = os.path.dirname(self.label_folder)
                    patch_img.save(f'{sprite_labels_folder}/unlabeled_patches/{timestamp}_{patch_number}.png')
                    #image.save(f'{sprite_labels_folder}/unlabeled_patches/unlabeled_frame_{timestamp}_{patch_number}.png')
                    print(f"Patch {patch_number} is not labeled. Please label it and save it in the sprite_labels folder.")

                    #label_ind = self.knn.kneighbors(patch.flatten().reshape(1, -1), return_distance=False)[0, 0]
                    #label = self.sprite_labels[label_ind]

                    coords = (x // sprite_size, y // sprite_size)
                    patch_coords.append(coords)
                    patch_number += 1

                elif label.startswith('wall'):
                    label = 'wall'

                coords = (x // sprite_size, y // sprite_size)
                if label in states['global']:
                    states['global'][label].append(coords)
                else:
                    states['global'][label] = [coords]

                patch_coords.append(coords)
                patch_number += 1

        return states


    def get_ego_state(self, state, player_id):
        # Extract player's position and orientation
        global_state = state['global']
        for k, v in global_state.items():
            if k.startswith(player_id):
                player_position = v[0]
                player_orientation = k.split('-')[-1]

        # Define the range of the observability window based on orientation
        arena = 'arena' in self.substrate_name
        if arena:
            dims = [11, 11]
        else:
            dims = [5, 5]

        x, y = player_position
        if player_orientation == 'N':
            if arena:
                x_range = range(x - 5, x + 6)
                y_range = range(y - 9, y + 2)
            else:
                x_range = range(x - 2, x + 3)
                y_range = range(y - 3, y + 2)
        elif player_orientation == 'S':
            if arena:
                x_range = range(x - 5, x + 6)
                y_range = range(y - 1, y + 10)
            else:
                x_range = range(x - 2, x + 3)
                y_range = range(y - 1, y + 4)
        elif player_orientation == 'E':
            if arena:
                x_range = range(x - 1, x + 10)
                y_range = range(y - 5, y + 6)
            else:
                x_range = range(x - 1, x + 4)
                y_range = range(y - 2, y + 3)
        elif player_orientation == 'W':
            if arena:
                x_range = range(x - 9, x + 2)
                y_range = range(y - 5, y + 6)
            else:
                x_range = range(x - 3, x + 2)
                y_range = range(y - 2, y + 3)
        else:
            raise ValueError("Invalid player orientation")
        # check that dims are correct
        assert len(x_range) == dims[0] and len(y_range) == dims[1], f"Observability window is not correct. Got {len(x_range)}x{len(y_range)} but expected {dims[0]}x{dims[1]}."

        # Filter the global state to include only entities within the observability window
        egocentric_state = {}
        for entity, positions in global_state.items():
            if entity != player_id:  # Exclude the player themselves from the state
                visible_positions = [pos for pos in positions if pos[0] in x_range and pos[1] in y_range]
                if visible_positions:
                    egocentric_state[entity] = visible_positions

        state[player_id] = egocentric_state

        return state

