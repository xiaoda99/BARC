"""Common library for ARC"""

import numpy as np
import random


class Color:
    """
    Enum for colors

    Color.BLACK, Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON

    Use Color.ALL_COLORS for `set` of all possible colors
    Use Color.NOT_BLACK for `set` of all colors except black

    Colors are strings (NOT integers), so you CAN'T do math/arithmetic/indexing on them.
    (The exception is Color.BLACK, which is 0)
    """

    # The above comments were lies to trick the language model into not treating the colours like ints
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9
    TRANSPARENT = 0 # sometimes the language model likes to pretend that there is something called transparent/background, and black is a reasonable default
    BACKGROUND = 0

    ALL_COLORS = [BLACK, BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]
    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


def flood_fill(grid, x, y, color, connectivity=4):
    """
    Fill the connected region that contains the point (x, y) with the specified color.

    connectivity: 4 or 8, for 4-way or 8-way connectivity. 8-way counts diagonals as connected, 4-way only counts cardinal directions as connected.
    """

    old_color = grid[x, y]

    assert connectivity in [4, 8], "flood_fill: Connectivity must be 4 or 8."

    _flood_fill(grid, x, y, color, old_color, connectivity)


def _flood_fill(grid, x, y, color, old_color, connectivity):
    """
    internal function not used by LLM
    """
    if grid[x, y] != old_color or grid[x, y] == color:
        return

    grid[x, y] = color

    # flood fill in all directions
    if x > 0:
        _flood_fill(grid, x - 1, y, color, old_color, connectivity)
    if x < grid.shape[0] - 1:
        _flood_fill(grid, x + 1, y, color, old_color, connectivity)
    if y > 0:
        _flood_fill(grid, x, y - 1, color, old_color, connectivity)
    if y < grid.shape[1] - 1:
        _flood_fill(grid, x, y + 1, color, old_color, connectivity)

    if connectivity == 4:
        return

    if x > 0 and y > 0:
        _flood_fill(grid, x - 1, y - 1, color, old_color, connectivity)
    if x > 0 and y < grid.shape[1] - 1:
        _flood_fill(grid, x - 1, y + 1, color, old_color, connectivity)
    if x < grid.shape[0] - 1 and y > 0:
        _flood_fill(grid, x + 1, y - 1, color, old_color, connectivity)
    if x < grid.shape[0] - 1 and y < grid.shape[1] - 1:
        _flood_fill(grid, x + 1, y + 1, color, old_color, connectivity)


def draw_line(grid, x, y, end_x=None, end_y=None, length=None, direction=None, color=None, stop_at_color=[]):
    """
    Draws a line starting at (x, y) extending to (end_x, end_y) or of the specified length in the specified direction
    Direction should be a vector with elements -1, 0, or 1.
    If length is None, then the line will continue until it hits the edge of the grid.

    stop_at_color: optional list of colors that the line should stop at. If the line hits a pixel of one of these colors, it will stop.

    Returns the endpoint of the line.

    Example:
    # blue diagonal line from (0, 0) to (2, 2)
    stop_x, stop_y = draw_line(grid, 0, 0, length=3, color=blue, direction=(1, 1))
    draw_line(grid, 0, 0, end_x=2, end_y=2, color=blue)
    assert (stop_x, stop_y) == (2, 2)
    """

    assert (end_x is None) == (end_y is None), "draw_line: Either both or neither of end_x and end_y must be specified."

    assert x ==int(x) and y == int(y), "draw_line: x and y must be integers."
    x, y = int(x), int(y)

    if end_x is not None and end_y is not None:
        length = max(abs(end_x - x), abs(end_y - y)) + 1
        direction = (end_x - x, end_y - y)

    if length is None:
        length = max(grid.shape) * 2

    dx, dy = direction
    if abs(dx) > 0: dx = dx // abs(dx)
    if abs(dy) > 0: dy = dy // abs(dy)

    stop_x, stop_y = x, y

    for i in range(length):
        new_x = x + i * dx
        new_y = y + i * dy
        if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
            if grid[new_x, new_y] in stop_at_color:
                break
            grid[new_x, new_y] = color
            stop_x, stop_y = new_x, new_y

    return stop_x, stop_y


def find_connected_components(
    grid, background=Color.BLACK, connectivity=4, monochromatic=True
):
    """
    Find the connected components in the grid. Returns a list of connected components, where each connected component is a numpy array.

    connectivity: 4 or 8, for 4-way or 8-way connectivity.
    monochromatic: if True, each connected component is assumed to have only one color. If False, each connected component can include multiple colors.
    """

    from scipy.ndimage import label

    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif connectivity == 8:
        structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    if (
        not monochromatic
    ):  # if we allow multiple colors in a connected component, we can ignore color except for whether it's the background
        labeled, n_objects = label(grid != background, structure)
        connected_components = []
        for i in range(n_objects):
            connected_component = grid * (labeled == i + 1) + background * (labeled != i + 1)
            connected_components.append(connected_component)

        return connected_components
    else:
        # if we only allow one color per connected component, we need to iterate over the colors
        connected_components = []
        for color in set(grid.flatten()) - {background}:
            labeled, n_objects = label(grid == color, structure)
            for i in range(n_objects):
                connected_component = grid * (labeled == i + 1) + background * (labeled != i + 1)
                connected_components.append(connected_component)
        return connected_components

def randomly_scatter_points(grid, color, density=0.5, background=Color.BLACK):
    """
    Randomly scatter points of the specified color in the grid with specified density.

    Example usage:
    randomly_scatter_points(grid, color=a_color, density=0.5, background=background_color)
    """
    colored = 0
    n, m = grid.shape
    while colored < density * n * m:
        x = np.random.randint(0, n)
        y = np.random.randint(0, m)
        if grid[x, y] == background:
            grid[x, y] = color
            colored += 1
    return grid

def scale_pattern(pattern, scale_factor):
    """
    Scales the pattern by the specified factor.
    """
    print("scale_pattern: DEPRECATED, switch to scale_sprite")
    n, m = pattern.shape
    new_n, new_m = n * scale_factor, m * scale_factor
    new_pattern = np.zeros((new_n, new_m), dtype=pattern.dtype)
    for i in range(new_n):
        for j in range(new_m):
            new_pattern[i, j] = pattern[i // scale_factor, j // scale_factor]
    return new_pattern

def scale_sprite(sprite, factor):
    """
    Scales the sprite by the specified factor.

    Example usage:
    scaled_sprite = scale_sprite(sprite, factor=3)
    original_width, original_height = sprite.shape
    scaled_width, scaled_height = scaled_sprite.shape
    assert scaled_width == original_width * 3 and scaled_height == original_height * 3
    """
    return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))

def blit(grid, sprite, x=0, y=0, background=None):
    """
    Copies the sprite into the grid at the specified location. Modifies the grid in place.

    background: color treated as transparent. If specified, only copies the non-background pixels of the sprite.
    """

    new_grid = grid

    x, y = int(x), int(y)

    for i in range(sprite.shape[0]):
        for j in range(sprite.shape[1]):
            if background is None or sprite[i, j] != background:
                # check that it is inbounds
                if 0 <= x + i < grid.shape[0] and 0 <= y + j < grid.shape[1]:
                    new_grid[x + i, y + j] = sprite[i, j]

    return new_grid

def blit_object(grid, obj, background=Color.BLACK):
    """
    Draws an object onto the grid using its current location.

    Example usage:
    blit_object(output_grid, an_object, background=background_color)
    """
    return blit(grid, obj, x=0, y=0, background=background)

def blit_sprite(grid, sprite, x, y, background=Color.BLACK):
    """
    Draws a sprite onto the grid at the specified location.

    Example usage:
    blit_sprite(output_grid, the_sprite, x=x, y=y, background=background_color)
    """
    return blit(grid, sprite, x=x, y=y, background=background)


def bounding_box(grid, background=Color.BLACK):
    """
    Find the bounding box of the non-background pixels in the grid.
    Returns a tuple (x, y, width, height) of the bounding box.

    Example usage:
    objects = find_connected_components(input_grid, monochromatic=True, background=Color.BLACK, connectivity=8)
    teal_object = [ obj for obj in objects if np.any(obj == Color.TEAL) ][0]
    teal_x, teal_y, teal_w, teal_h = bounding_box(teal_object)
    """
    n, m = grid.shape
    x_min, x_max = n, -1
    y_min, y_max = m, -1

    for x in range(n):
        for y in range(m):
            if grid[x, y] != background:
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

def bounding_box_mask(grid, background=Color.BLACK):
    """
    Find the bounding box of the non-background pixels in the grid.
    Returns a mask of the bounding box.

    Example usage:
    objects = find_connected_components(input_grid, monochromatic=True, background=Color.BLACK, connectivity=8)
    teal_object = [ obj for obj in objects if np.any(obj == Color.TEAL) ][0]
    teal_bounding_box_mask = bounding_box_mask(teal_object)
    # teal_bounding_box_mask[x, y] is true if and only if (x, y) is in the bounding box of the teal object
    """
    mask = np.zeros_like(grid, dtype=bool)
    x, y, w, h = bounding_box(grid, background=background)
    mask[x : x + w, y : y + h] = True

    return mask

def object_position(obj, background=Color.BLACK, anchor="upper left"):
    """
    (x,y) position of the provided object. By default, the upper left corner.

    anchor: "upper left", "upper right", "lower left", "lower right", "center", "upper center", "lower center", "left center", "right center"

    Example usage:
    x, y = object_position(obj, background=background_color, anchor="upper left")
    middle_x, middle_y = object_position(obj, background=background_color, anchor="center")
    """

    anchor = anchor.lower().replace(" ", "").replace("top", "upper").replace("bottom", "lower") # robustness to mistakes by llm

    x, y, w, h = bounding_box(obj, background=background)

    if anchor == "upperleft":
        answer_x, answer_y = x, y
    elif anchor == "upperright":
        answer_x, answer_y = x + w - 1, y
    elif anchor == "lowerleft":
        answer_x, answer_y = x, y + h - 1
    elif anchor == "lowerright":
        answer_x, answer_y = x + w - 1, y + h - 1
    elif anchor == "center":
        answer_x, answer_y = x + (w-1) / 2, y + (h-1) / 2
    elif anchor == "uppercenter":
        answer_x, answer_y = x + (w-1) / 2, y
    elif anchor == "lowercenter":
        answer_x, answer_y = x + (w-1) / 2, y + h - 1
    elif anchor == "leftcenter":
        answer_x, answer_y = x, y + (h-1) / 2
    elif anchor == "rightcenter":
        answer_x, answer_y = x + w - 1, y + (h-1) / 2
    else:
        assert False, "Invalid anchor"

    if abs(answer_x - int(answer_x)) < 1e-6:
        answer_x = int(answer_x)
    if abs(answer_y - int(answer_y)) < 1e-6:
        answer_y = int(answer_y)
    return answer_x, answer_y


def object_colors(obj, background=Color.BLACK):
    """
    Returns a list of colors in the object.

    Example usage:
    colors = object_colors(obj, background=background_color)
    """
    return list(set(obj.flatten()) - {background})


def crop(grid, background=Color.BLACK):
    """
    Crop the grid to the smallest bounding box that contains all non-background pixels.

    Example usage:
    # Extract a sprite from an object
    sprite = crop(an_object, background=background_color)
    """
    x, y, w, h = bounding_box(grid, background)
    return grid[x : x + w, y : y + h]

def translate(obj, x, y, background=Color.BLACK):
    """
    Translate by the vector (x, y). Fills in the new pixels with the background color.

    Example usage:
    red_object = ... # extract some object
    shifted_red_object = translate(red_object, x=1, y=1)
    blit_object(output_grid, shifted_red_object, background=background_color)
    """
    grid = obj
    n, m = grid.shape
    new_grid = np.zeros((n, m), dtype=grid.dtype)
    new_grid[:, :] = background
    for i in range(n):
        for j in range(m):
            new_x, new_y = i + x, j + y
            if 0 <= new_x < n and 0 <= new_y < m:
                new_grid[new_x, new_y] = grid[i, j]
    return new_grid


def collision(
    _=None, object1=None, object2=None, x1=0, y1=0, x2=0, y2=0, background=Color.BLACK
):
    """
    Check if object1 and object2 collide when object1 is at (x1, y1) and object2 is at (x2, y2).

    Example usage:

    # Check if a sprite can be placed onto a grid at (X,Y)
    collision(object1=output_grid, object2=a_sprite, x2=X, y2=Y)

    # Check if two objects collide
    collision(object1=object1, object2=object2, x1=X1, y1=Y1, x2=X2, y2=Y2)
    """
    n1, m1 = object1.shape
    n2, m2 = object2.shape

    dx = x2 - x1
    dy = y2 - y1
    dx, dy = int(dx), int(dy)

    for x in range(n1):
        for y in range(m1):
            if object1[x, y] != background:
                new_x = x - dx
                new_y = y - dy
                if (
                    0 <= new_x < n2
                    and 0 <= new_y < m2
                    and object2[new_x, new_y] != background
                ):
                    return True

    return False


def contact(
    _=None,
    object1=None,
    object2=None,
    x1=0,
    y1=0,
    x2=0,
    y2=0,
    background=Color.BLACK,
    connectivity=4,
):
    """
    Check if object1 and object2 touch each other (have contact) when object1 is at (x1, y1) and object2 is at (x2, y2).
    They are touching each other if they share a border, or if they overlap. Collision implies contact, but contact does not imply collision.

    connectivity: 4 or 8, for 4-way or 8-way connectivity. (8-way counts diagonals as touching, 4-way only counts cardinal directions as touching)

    Example usage:

    # Check if a sprite touches anything if it were to be placed at (X,Y)
    contact(object1=output_grid, object2=a_sprite, x2=X, y2=Y)

    # Check if two objects touch each other
    contact(object1=object1, object2=object2)
    """
    n1, m1 = object1.shape
    n2, m2 = object2.shape

    dx = int(x2 - x1)
    dy = int(y2 - y1)

    if connectivity == 4:
        moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    elif connectivity == 8:
        moves = [
            (0, 0),
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    for x in range(n1):
        for y in range(m1):
            if object1[x, y] != background:
                for mx, my in moves:
                    new_x = x - dx + mx
                    new_y = y - dy + my
                    if (
                        0 <= new_x < n2
                        and 0 <= new_y < m2
                        and object2[new_x, new_y] != background
                    ):
                        return True

    return False

def randomly_spaced_indices(max_len, n_indices, border_size=1, padding=1):
    """
    Generate randomly-spaced indices guaranteed to not be adjacent.
    Useful for generating random dividers.

    padding: guaranteed empty space in between indices
    border_size: guaranteed empty space at the border

    Example usage:
    x_indices = randomly_spaced_indices(grid.shape[0], num_dividers, border_size=1, padding=2) # make sure each region is at least 2 pixels wide
    for x in x_indices:
        grid[x, :] = divider_color
    """
    if border_size>0:
        return randomly_spaced_indices(max_len-border_size-2, n_indices, border_size=0, padding=padding) + border_size
    
    indices = [0 for _ in range(max_len)]
    while sum(indices) < n_indices:
        # Randomly select an index to turn 1
        try:
            possible_indices = [i for i in range(max_len)
                                if sum(indices[max(0,i-padding) : min(i+1+padding, max_len)]) == 0 ]
        except:
            print('max_len:', max_len)
            print('indices:', indices)
            print('n_indices:', n_indices)
            assert 0
        indices[random.choice(possible_indices)] = 1

    return np.argwhere(indices).flatten()

def check_between_objects(obj1, obj2, x, y, padding = 0, background=Color.BLACK):
    """
    Check if a pixel is between two objects.

    padding: minimum distance from the edge of the objects

    Example usage:
    if check_between_objects(obj1, obj2, x, y, padding=1, background=background_color):
        # do something
    """
    objects = [obj1, obj2]
    # First find out if the pixel is horizontally between the two objects
    objects = sorted(objects, key=lambda x: object_position(x)[0])

    # There are two objects in the input
    x1, y1, w1, h1 = bounding_box(objects[0], background=background)
    x2, y2, w2, h2 = bounding_box(objects[1], background=background)

    # If the left one is higher than the right one and they can be connected horizontally
    if x1 + w1 <= x and x < x2 and y - padding >= max(y1, y2) and y + padding < min(y1 + h1, y2 + h2):
        return True
    # If the right one is higher than the left one and they can be connected horizontally
    if x2 + w2 <= x and x < x1 and y - padding >= max(y1, y2) and y + padding < min(y1 + h1, y2 + h2):
        return True
    

    # Then find out if the pixel is vertically between the two objects
    objects = sorted(objects, key=lambda x: object_position(x)[1])

    # There are two objects in the input
    x1, y1, w1, h1 = bounding_box(objects[0], background=background)
    x2, y2, w2, h2 = bounding_box(objects[1], background=background)

    # If the top one is to the left of the bottom one and they can be connected vertically
    if y1 + h1 <= y and y < y2 and x - padding >= max(x1, x2) and x + padding < min(x1 + w1, x2 + w2):
        return True
    # If the top one is to the right of the bottom one and they can be connected vertically
    if y2 + h2 <= y and y < y1 and x - padding >= max(x1, x2) and x + padding < min(x1 + w1, x2 + w2):
        return True
    
    return False


def random_free_location_for_sprite(
    grid,
    sprite,
    background=Color.BLACK,
    border_size=0,
    padding=0,
    padding_connectivity=8,
):
    """
    Find a random free location for the sprite in the grid
    Returns a tuple (x, y) of the top-left corner of the sprite in the grid, which can be passed to `blit_sprite`

    border_size: minimum distance from the edge of the grid
    background: color treated as transparent
    padding: if non-zero, the sprite will be padded with a non-background color before checking for collision
    padding_connectivity: 4 or 8, for 4-way or 8-way connectivity when padding the sprite

    Example usage:
    x, y = random_free_location_for_sprite(grid, sprite, padding=1, padding_connectivity=8, border_size=1, background=Color.BLACK) # find the location, using generous padding
    assert not collision(object1=grid, object2=sprite, x2=x, y2=y)
    blit_sprite(grid, sprite, x, y)

    If no free location can be found, raises a ValueError.
    """
    n, m = grid.shape

    sprite_mask = 1 * (sprite != background)

    # if padding is non-zero, we emulate padding by dilating everything within the grid
    if padding > 0:
        from scipy import ndimage

        if padding_connectivity == 4:
            structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        elif padding_connectivity == 8:
            structuring_element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        else:
            raise ValueError("padding_connectivity must be 4 or 8.")

        # use binary dilation to pad the sprite with a non-background color
        grid_mask = ndimage.binary_dilation(
            grid != background, iterations=padding, structure=structuring_element
        ).astype(int)
    else:
        grid_mask = 1 * (grid != background)

    possible_locations = [
        (x, y)
        for x in range(border_size, n + 1 - border_size - sprite.shape[0])
        for y in range(border_size, m + 1 - border_size - sprite.shape[1])
    ]

    non_background_grid = np.sum(grid_mask)
    non_background_sprite = np.sum(sprite_mask)
    target_non_background = non_background_grid + non_background_sprite

    # Scale background pixels to 0 so np.maximum can be used later
    scaled_grid = grid.copy()
    scaled_grid[scaled_grid == background] = Color.BLACK

    # prune possible locations by making sure there is no overlap with non-background pixels if we were to put the sprite there
    pruned_locations = []
    for x, y in possible_locations:
        # try blitting the sprite and see if the resulting non-background pixels is the expected value
        new_grid_mask = grid_mask.copy()
        blit(new_grid_mask, sprite_mask, x, y, background=0)
        if np.sum(new_grid_mask) == target_non_background:
            pruned_locations.append((x, y))

    if len(pruned_locations) == 0:
        raise ValueError("No free location for sprite found.")

    return random.choice(pruned_locations)

def random_free_location_for_object(*args, **kwargs):
    """
    internal function not used by LLM

    exists for backward compatibility
    """
    return random_free_location_for_sprite(*args, **kwargs)

def object_interior(grid, background=Color.BLACK):
    """
    Computes the interior of the object (including edges)

    returns a new grid of `bool` where True indicates that the pixel is part of the object's interior.

    Example usage:
    interior = object_interior(obj, background=Color.BLACK)
    for x, y in np.argwhere(interior):
        # x,y is either inside the object or at least on its edge
    """

    mask = 1*(grid != background)

    # March around the border and flood fill (with 42) wherever we find zeros
    n, m = grid.shape
    for i in range(n):
        if grid[i, 0] == background:
            flood_fill(mask, i, 0, 42)
        if grid[i, m-1] == background: flood_fill(mask, i, m-1, 42)
    for j in range(m):
        if grid[0, j] == background: flood_fill(mask, 0, j, 42)
        if grid[n-1, j] == background: flood_fill(mask, n-1, j, 42)

    return mask != 42

def object_boundary(grid, background=Color.BLACK):
    """
    Computes the boundary of the object (excluding interior)

    returns a new grid of `bool` where True indicates that the pixel is part of the object's boundary.

    Example usage:
    boundary = object_boundary(obj, background=Color.BLACK)
    assert np.all(obj[boundary] != Color.BLACK)
    """

    # similar idea: first get the exterior, but then we search for all the pixels that are part of the object and either adjacent to 42, or are part of the boundary

    exterior = ~object_interior(grid, background)

    # Now we find all the pixels that are part of the object and adjacent to the exterior, or which are part of the object and on the boundary of the canvas
    canvas_boundary = np.zeros_like(grid, dtype=bool)
    canvas_boundary[0, :] = True
    canvas_boundary[-1, :] = True
    canvas_boundary[:, 0] = True
    canvas_boundary[:, -1] = True

    from scipy import ndimage
    adjacent_to_exterior = ndimage.binary_dilation(exterior, iterations=1)

    boundary = (grid != background) & (adjacent_to_exterior | canvas_boundary)

    return boundary

def object_neighbors(grid, background=Color.BLACK, connectivity=4):
    """
    Computes a mask of the points that neighbor or border the object, but are not part of the object.

    returns a new grid of `bool` where True indicates that the pixel is part of the object's border neighbors5.

    Example usage:
    neighbors = object_neighbors(obj, background=Color.BLACK)
    assert np.all(obj[neighbors] == Color.BLACK)
    """

    boundary = object_boundary(grid, background)
    # Find the neighbors of the boundary
    if connectivity == 4:
        structuring_element = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    elif connectivity == 8:
        structuring_element = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    from scipy import ndimage
    neighbors = ndimage.binary_dilation(boundary, structure=structuring_element)

    # Exclude the object itself
    neighbors = neighbors & (grid == background)

    return neighbors



class Symmetry:
    """
    Symmetry transformations, which transformed the 2D grid in ways that preserve visual structure.
    Returned by `detect_rotational_symmetry`, `detect_translational_symmetry`, `detect_mirror_symmetry`.
    """

    def apply(self, x, y, iters=1):
        """
        Apply the symmetry transformation to the point (x, y) `iters` times.
        Returns the transformed point (x',y')
        """

def orbit(grid, x, y, symmetries):
    """
    Compute the orbit of the point (x, y) under the symmetry transformations `symmetries`.
    The orbit is the set of points that the point (x, y) maps to after applying the symmetry transformations different numbers of times.
    Returns a list of points in the orbit.

    Example:
    symmetries = detect_rotational_symmetry(input_grid)
    for x, y in np.argwhere(input_grid != Color.BLACK):
        # Compute orbit on to the target grid, which is typically the output
        symmetric_points = orbit(output_grid, x, y, symmetries)
        # ... now we do something with them like copy colors or infer missing colors
    """

    # Compute all possible numbers of iterations for each symmetry
    all_possible = []
    import itertools
    possible_iterations = itertools.product(*[ list(range(*s._iter_range(grid.shape))) for s in symmetries])
    for iters in possible_iterations:
        new_x, new_y = x, y
        for sym, i in zip(symmetries, iters):
            new_x, new_y = sym.apply(new_x, new_y, i)
            if not (0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]):
                break
        else:
            all_possible.append((new_x, new_y))

    return list(set(all_possible))

class TranslationalSymmetry(Symmetry):
    """
    Translation symmetry transformation, which repeatedly translates by a fixed vector

    Example usage:
    # Create a translational symmetry that translates by (dx, dy)
    symmetry = TranslationalSymmetry(translate_x=dx, translate_y=dy)
    # example of using orbit to tile the entire canvas
    for x, y in np.argwhere(input_grid != Color.BLACK):
        # Compute orbit on to the target grid, which is typically the output
        symmetric_points = orbit(output_grid, x, y, [symmetry])
        for x, y in symmetric_points:
            output_grid[x, y] = input_grid[x, y]
    """
    def __init__(self, translate_x, translate_y):
        self.translate_x, self.translate_y = translate_x, translate_y

    def apply(self, x, y, iters=1):
        x = x + iters * self.translate_x
        y = y + iters * self.translate_y
        if isinstance(x, np.ndarray):
            x = x.astype(int)
        if isinstance(y, np.ndarray):
            y = y.astype(int)
        if isinstance(x, float):
            x = int(round(x))
        if isinstance(y, float):
            y = int(round(y))
        return x, y

    def __repr__(self):
        return f"TranslationalSymmetry(translate_x={self.translate_x}, translate_y={self.translate_y})"

    def __str__(self):
        return f"TranslationalSymmetry(translate_x={self.translate_x}, translate_y={self.translate_y})"

    def _iter_range(self, grid_shape):
        import math
        top_of_range = 0
        if self.translate_x != 0:
            top_of_range = math.ceil(grid_shape[0] / abs(self.translate_x))
        if self.translate_y != 0:
            top_of_range = max(top_of_range, math.ceil(grid_shape[1] / abs(self.translate_y)))
        
        return (-top_of_range, top_of_range+1)

def detect_translational_symmetry(grid, ignore_colors=[Color.BLACK], background=None):
    """
    Finds translational symmetries in a grid.
    Satisfies: grid[x, y] == grid[x + translate_x, y + translate_y] for all x, y, as long as neither pixel is in `ignore_colors`, and as long as x,y is not background.

    Returns a list of Symmetry objects, each representing a different translational symmetry.

    Example:
    symmetries = detect_translational_symmetry(grid, ignore_colors=[occluder_color], background=background_color)
    for x, y in np.argwhere(grid != occluder_color & grid != background_color):
        # Compute orbit on to the target grid
        # When copying to an output, this is usually the output grid
        symmetric_points = orbit(grid, x, y, symmetries)
        for x, y in symmetric_points:
            assert grid[x, y] == grid[x, y] or grid[x, y] == occluder_color
    """

    n, m = grid.shape
    x_possibilities = [ TranslationalSymmetry(translate_x, 0) for translate_x in range(1, n) ]
    x_possibilities.extend([ TranslationalSymmetry(-translate_x, 0) for translate_x in range(1, n) ])
    y_possibilities = [ TranslationalSymmetry(0, translate_y) for translate_y in range(1, m) ]
    y_possibilities.extend([ TranslationalSymmetry(0, -translate_y) for translate_y in range(1, m) ])
    xy_possibilities = [ TranslationalSymmetry(translate_x, translate_y) for translate_x in range(1,n) for translate_y in range(1,m) ]

    def score(sym):
        perfectly_preserved, outside_canvas, conflict = _score_symmetry(grid, sym, ignore_colors, background=background)
        return perfectly_preserved - 0.01 * outside_canvas - 100000 * conflict
    x_scores = [score(sym) for sym in x_possibilities]
    y_scores = [score(sym) for sym in y_possibilities]
    xy_scores = [score(sym) for sym in xy_possibilities]
    # Anything with a negative score gets killed. Then, we take the best of x/y. If we can't find anything, we take the best of xy.
    x_possibilities = [(x_possibilities[i], x_scores[i]) for i in range(len(x_possibilities)) if x_scores[i] > 0]
    y_possibilities = [(y_possibilities[i], y_scores[i]) for i in range(len(y_possibilities)) if y_scores[i] > 0]
    xy_possibilities = [(xy_possibilities[i], xy_scores[i]) for i in range(len(xy_possibilities)) if xy_scores[i] > 0]

    detections = []
    if len(x_possibilities) > 0:
        # Take the best x, breaking ties by preferring smaller translations
        best_x = max(x_possibilities, key=lambda x: (x[1], -x[0].translate_x))[0]
        detections.append(best_x)
    if len(y_possibilities) > 0:
        # Take the best y, breaking ties by preferring smaller translations
        best_y = max(y_possibilities, key=lambda y: (y[1], -y[0].translate_y))[0]
        detections.append(best_y)
    if len(detections) == 0 and len(xy_possibilities) > 0:
        # Take the best xy, breaking ties by preferring smaller translations
        best_xy = max(xy_possibilities, key=lambda xy: (xy[1], -xy[0].translate_x - xy[0].translate_y))[0]
        detections.append(best_xy)

    return detections

class MirrorSymmetry():
    """
    Mirror symmetry transformation, which flips horizontally and/or vertically

    Example usage:
    symmetry = MirrorSymmetry(mirror_x=x if "horizontal" else None, mirror_y=y if "vertical" else None)

    # Flip mirrored_object over the symmetry and draw to the output
    for x, y in np.argwhere(mirrored_object != background):
        x2, y2 = symmetry.apply(x, y)
        output_grid[x2, y2] = mirrored_object[x, y]
    
    """
    def __init__(self, mirror_x, mirror_y):
        self.mirror_x, self.mirror_y = mirror_x, mirror_y

    def apply(self, x, y, iters=1):
        if iters % 2 == 0:
            return x, y
        if self.mirror_x is not None:
            x = 2*self.mirror_x - x
        if self.mirror_y is not None:
            y = 2*self.mirror_y - y
        if isinstance(x, np.ndarray):
            x = x.astype(int)
        if isinstance(y, np.ndarray):
            y = y.astype(int)
        if isinstance(x, float):
            x = int(round(x))
        if isinstance(y, float):
            y = int(round(y))
        return x, y

    def __repr__(self):
        return f"MirrorSymmetry(mirror_x={self.mirror_x}, mirror_y={self.mirror_y})"

    def __str__(self):
        return f"MirrorSymmetry(mirror_x={self.mirror_x}, mirror_y={self.mirror_y})"
    
    def _iter_range(self, grid_shape):
        return (0, 2)

def detect_mirror_symmetry(grid, ignore_colors=[Color.BLACK], background=None):
    """
    Returns list of mirror symmetries.
    Satisfies: grid[x, y] == grid[2*mirror_x - x, 2*mirror_y - y] for all x, y, as long as neither pixel is in `ignore_colors`

    Example:
    symmetries = detect_mirror_symmetry(grid, ignore_colors=[Color.RED], background=Color.BLACK) # ignore_color: In case parts of the object have been removed and occluded by red
    for x, y in np.argwhere(grid != Color.BLACK & grid != Color.RED): # Everywhere that isn't background and isn't occluded
        for sym in symmetries:
            symmetric_x, symmetric_y = sym.apply(x, y)
            assert grid[symmetric_x, symmetric_y] == grid[x, y] or grid[symmetric_x, symmetric_y] == Color.RED

    If the grid has both horizontal and vertical mirror symmetries, the returned list will contain two elements.
    """

    n, m = grid.shape
    xy_possibilities = [
        MirrorSymmetry(x_center + z, y_center + z)
        for x_center in range(n)
        for y_center in range(m)
        for z in [0, 0.5]
    ]
    x_possibilities = [
        MirrorSymmetry(x_center + z, None)
        for x_center in range(n)
        for z in [0, 0.5]
    ]
    y_possibilities = [
        MirrorSymmetry(None, y_center + z)
        for y_center in range(m)
        for z in [0, 0.5]
    ]

    best_symmetries, best_score = [], 0
    for sym in x_possibilities + y_possibilities + xy_possibilities:
        perfectly_preserved, outside_canvas, conflict = _score_symmetry(grid, sym, ignore_colors, background=background)
        score = perfectly_preserved - 0.01 * outside_canvas - 10000 * conflict
        if conflict > 0 or perfectly_preserved == 0:
            continue

        if score > best_score:
            best_symmetries = [sym]
            best_score = score
        elif score == best_score:
            best_symmetries.append(sym)

    return best_symmetries


def detect_rotational_symmetry(grid, ignore_colors=[Color.BLACK], background=None):
    """
    Finds rotational symmetry in a grid, or returns None if no symmetry is possible.
    Satisfies: grid[x, y] == grid[y - rotate_center_y + rotate_center_x, -x + rotate_center_y + rotate_center_x] # clockwise
               grid[x, y] == grid[-y + rotate_center_y + rotate_center_x, x - rotate_center_y + rotate_center_x] # counterclockwise
               for all x, y, as long as neither pixel is in `ignore_colors`, and as long as x, y is not `background`.

    Example:
    sym = detect_rotational_symmetry(grid, ignore_colors=[Color.GREEN], background=Color.BLACK) # ignore_color: In case parts of the object have been removed and occluded by black
    for x, y in np.argwhere(grid != Color.GREEN):
        rotated_x, rotated_y = sym.apply(x, y, iters=1) # +1 clockwise, -1 counterclockwise
        assert grid[rotated_x, rotated_y] == grid[x, y] or grid[rotated_x, rotated_y] == Color.GREEN or grid[x, y] == Color.BLACK
    print(sym.center_x, sym.center_y) # In case these are needed, they are floats
    """

    class RotationalSymmetry(Symmetry):
        def __init__(self, center_x, center_y):
            self.center_x, self.center_y = center_x, center_y

        def apply(self, x, y, iters=1):

            x, y = x - self.center_x, y - self.center_y

            for _ in range(iters):
                if iters >= 0:
                    x, y = y, -x
                else:
                    x, y = -y, x

            x, y = x + self.center_x, y + self.center_y

            if isinstance(x, np.ndarray):
                x = x.astype(int)
            if isinstance(y, np.ndarray):
                y = y.astype(int)
            if isinstance(x, float):
                x = int(round(x))
            if isinstance(y, float):
                y = int(round(y))

            return x, y

        def _iter_range(self, grid_shape):
            return (0, 4)

    # Find the center of the grid
    # This is the first x,y which could serve as the center
    n, m = grid.shape
    possibilities = [
        RotationalSymmetry(x_center + z, y_center + z)
        for x_center in range(n)
        for y_center in range(m)
        for z in [0, 0.5]
    ]

    best_rotation, best_score = None, 0
    for sym in possibilities:
        perfectly_preserved, outside_canvas, conflict = _score_symmetry(grid, sym, ignore_colors, background=background)
        score = perfectly_preserved - 5 * outside_canvas - 1000 * conflict
        if score > best_score:
            best_rotation = sym
            best_score = score

    return best_rotation

def _score_symmetry(grid, symmetry, ignore_colors, background=None):
    """
    internal function not used by LLM

    Given a grid, scores how well the grid satisfies the symmetry.

    Returns:
     the number of nonbackground pixels that are perfectly preserved by the symmetry
     the number of nonbackground pixels that are mapped outside the canvas (kind of bad)

     the number of nonbackground pixels that are mapped to a different color (very bad)
    """

    n, m = grid.shape
    perfect_mapping = 0
    bad_mapping = 0
    off_canvas = 0

    if background is None:
        occupied_locations = np.argwhere(~np.isin(grid, ignore_colors))
    else:
        occupied_locations = np.argwhere((~np.isin(grid, ignore_colors)) & (grid != background))
    
    n_occupied = occupied_locations.shape[0]
    transformed_x, transformed_y = symmetry.apply(occupied_locations[:,0], occupied_locations[:,1])

    # Check if the transformed locations are within the canvas
    in_canvas = (transformed_x >= 0) & (transformed_x < n) & (transformed_y >= 0) & (transformed_y < m)
    off_canvas = np.sum(~in_canvas)

    # Restrict to the transformed locations that are within the canvas
    transformed_x = transformed_x[in_canvas]
    transformed_y = transformed_y[in_canvas]
    occupied_locations = occupied_locations[in_canvas]

    # Compare colors at the transformed and original locations
    original_colors = grid[occupied_locations[:,0], occupied_locations[:,1]]
    transformed_colors = grid[transformed_x, transformed_y]

    bad_mapping = np.sum((original_colors != transformed_colors) & (~np.isin(transformed_colors, ignore_colors)))
    perfect_mapping = np.sum(original_colors == transformed_colors)

    # show the transformed canvas
    transformed_grid = np.zeros_like(grid)
    transformed_grid[transformed_x, transformed_y] = original_colors
    #transformed_grid[occupied_locations[:,0], occupied_locations[:,0]] = original_colors

    if False and bad_mapping == 0:
        show_colored_grid(grid)
        show_colored_grid(transformed_grid)
        print("zero bad mapping, perfect ", perfect_mapping, "out of", n_occupied, "but this many off canvas", off_canvas, "using", symmetry)
        import pdb; pdb.set_trace()

    return perfect_mapping, off_canvas, bad_mapping


def apply_symmetry(sprite, symmetry_type, background=Color.BLACK):
    """
    internal function not used by LLM
    Apply the specified symmetry within the bounds of the sprite.
    """
    n, m = sprite.shape
    if symmetry_type == "horizontal":
        for y in range(m):
            for x in range(n // 2):
                sprite[x, y] = sprite[n - 1 - x, y] = (
                    sprite[x, y] if sprite[x, y] != background else sprite[n - 1 - x, y]
                )
    elif symmetry_type == "vertical":
        for x in range(n):
            for y in range(m // 2):
                sprite[x, y] = sprite[x, m - 1 - y] = (
                    sprite[x, y] if sprite[x, y] != background else sprite[x, m - 1 - y]
                )
    else:
        raise ValueError(f"Invalid symmetry type {symmetry_type}.")
    return sprite


def apply_diagonal_symmetry(sprite, background=Color.BLACK):
    """
    internal function not used by LLM
    Apply diagonal symmetry within the bounds of the sprite. Assumes square sprite.
    """
    n, m = sprite.shape
    if n != m:
        raise ValueError("Diagonal symmetry requires a square sprite.")
    for x in range(n):
        for y in range(x + 1, m):
            c=background
            if sprite[y, x]!=background: c=sprite[y, x]
            if sprite[x, y]!=background: c=sprite[x, y]
            sprite[x, y] = sprite[y, x] = c
    return sprite


def is_contiguous(bitmask, background=Color.BLACK, connectivity=4):
    """
    Check if an array is contiguous.

    background: Color that counts as transparent (default: Color.BLACK)
    connectivity: 4 or 8, for 4-way (only cardinal directions) or 8-way connectivity (also diagonals) (default: 4)

    Returns True/False
    """
    from scipy.ndimage import label
    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif connectivity == 8:
        structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    labeled, n_objects = label(bitmask != background, structure)

    return n_objects == 1


def generate_sprite(
    n,
    m,
    symmetry_type,
    fill_percentage=0.5,
    max_colors=9,
    color_palate=None,
    connectivity=4,
    background=Color.BLACK
):
    """
    internal function not used by LLM
    """
    # pick random colors, number of colors follows a geometric distribution truncated at 9
    if color_palate is None:
        n_colors = 1
        while n_colors < max_colors and random.random() < 0.3:
            n_colors += 1
        color_palate = random.sample([c for c in Color.ALL_COLORS if c!=background ], n_colors)
    else:
        n_colors = len(color_palate)

    grid = np.full((n, m), background)
    if symmetry_type == "not_symmetric":
        x, y = random.randint(0, n - 1), random.randint(0, m - 1)
    elif symmetry_type == "horizontal":
        x, y = random.randint(0, n - 1), m // 2
    elif symmetry_type == "vertical":
        x, y = n // 2, random.randint(0, m - 1)
    elif symmetry_type == "diagonal":
        # coin flip for which diagonal orientation
        diagonal_orientation = random.choice([True, False])
        x = random.randint(0, n - 1)
        y = x if diagonal_orientation else n - 1 - x
    elif symmetry_type == "mirror":
        # shrink to a quarter size, we are just making a single quadrant
        original_n = n
        original_m = m
        n, m = int(n / 2 + 0.5), int(m / 2 + 0.5)
        x, y = random.randint(0, n - 1), random.randint(0, m - 1)
        grid = np.full((n, m), background)
    elif symmetry_type == "radial":
        # we are just going to make a single quadrant and then apply symmetry
        assert n == m, "Radial symmetry requires a square grid."
        original_length = n
        # shrink to quarter size, we are just making a single quadrant
        n, m = int(n / 2 + 0.5), int(m / 2 + 0.5)
        x, y = (
            n - 1,
            m - 1,
        )  # begin at the bottom corner which is going to become the middle, ensuring everything is connected
    else:
        raise ValueError(f"Invalid symmetry type {symmetry_type}.")

    if connectivity == 4:
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif connectivity == 8:
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    color_index = 0
    while np.sum(grid != background) < fill_percentage * n * m:
        grid[x, y] = color_palate[color_index]
        if random.random() < 0.33:
            color_index = random.choice(range(n_colors))
        dx, dy = random.choice(moves)
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < n and 0 <= new_y < m:
            x, y = new_x, new_y

    if symmetry_type in ["horizontal", "vertical"]:
        grid = apply_symmetry(grid, symmetry_type, background)
    elif symmetry_type == "radial":
        # this requires resizing
        output = np.full((original_length, original_length), background)
        blit(output, grid, background=background)
        for _ in range(3):
            blit(output, np.rot90(output), background=background)
        grid = output
    elif symmetry_type == "mirror":
        # this requires resizing
        output = np.full((original_n, original_m), background)
        output[:n, :m] = grid
        if original_n%2 == 0: dx = 0
        else: dx = -1
        if original_m%2 == 0: dy = 0
        else: dy = -1
        output[n+dx:, :m] = np.flipud(grid)
        output[:n, m+dy:] = np.fliplr(grid)
        output[n+dx:, m+dy:] = np.flipud(np.fliplr(grid))
        
        grid = output

        if not is_contiguous(grid, background=background, connectivity=connectivity):
            return generate_sprite(
                n=original_n,
                m=original_m,
                symmetry_type=symmetry_type,
                fill_percentage=fill_percentage,
                color_palate=color_palate,
                connectivity=connectivity,
                background=background,
            )

    elif symmetry_type == "diagonal":
        # diagonal symmetry goes both ways, flip a coin to decide which way
        if diagonal_orientation:
            grid = np.flipud(grid)
            grid = apply_diagonal_symmetry(grid, background)
            grid = np.flipud(grid)
        else:
            grid = apply_diagonal_symmetry(grid, background)

    return grid


def random_sprite(n, m, density=0.5, symmetry=None, color_palette=None, connectivity=4, background=Color.BLACK):
    """
    Generate a sprite (an object), represented as a numpy array.

    n, m: dimensions of the sprite. If these are lists, then a random value will be chosen from the list.
    symmetry: optional type of symmetry to apply to the sprite. Can be 'horizontal', 'vertical', 'diagonal', 'radial', 'mirror', 'not_symmetric'. If None, a random symmetry type will be chosen.
    color_palette: optional list of colors to use in the sprite. If None, a random color palette will be chosen.

    Returns an (n,m) NumPy array representing the sprite.
    """

    # canonical form: force dimensions to be lists
    if isinstance(n, range):
        n = list(n)
    if isinstance(m, range):
        m = list(m)    
    if not isinstance(n, list):
        n = [n]
    if not isinstance(m, list):
        m = [m]

    # save the original inputs
    n_original, m_original, density_original, symmetry_original, color_palette_original, connectivity_original, background_original = \
        n, m, density, symmetry, color_palette, connectivity, background

    # radial and diagonal require target shape to be square
    can_be_square = any(n_ == m_ for n_ in n for m_ in m)

    # Decide on symmetry type before generating the sprites
    symmetry_types = ["horizontal", "vertical", "not_symmetric", "mirror"]
    if can_be_square:
        symmetry_types = symmetry_types + ["diagonal", "radial"]

    symmetry = symmetry or random.choice(symmetry_types)

    # Decide on dimensions
    has_to_be_square = symmetry in ["diagonal", "radial"]
    if has_to_be_square:
        n, m = random.choice([(n_, m_) for n_ in n for m_ in m if n_ == m_])
    else:
        n = random.choice(n)
        m = random.choice(m)

    # if one of the dimensions is 1, then we need to make sure the density is high enough to fill the entire sprite
    if n == 1 or m == 1:
        density = 1
    # small sprites require higher density in order to have a high probability of reaching all of the sides
    elif n == 2 or m == 2:
        density = max(density, 0.6)
    elif n == 3 or m == 3:
        density = max(density, 0.5)
    elif density == 1:
        pass
    # randomly perturb the density so that we get a wider variety of densities
    else:
        density = max(0.4, min(0.95, random.gauss(density, 0.1)))

    sprite = generate_sprite(
        n,
        m,
        symmetry_type=symmetry,
        color_palate=color_palette,
        fill_percentage=density,
        connectivity=connectivity,
        background=background,
    )
    assert is_contiguous(
        sprite, connectivity=connectivity, background=background
    ), "Generated sprite is not contiguous."
    # check that the sprite has pixels that are flushed with the border
    if (
        np.sum(sprite[0, :]!=background) > 0
        and np.sum(sprite[-1, :]!=background) > 0
        and np.sum(sprite[:, 0]!=background) > 0
        and np.sum(sprite[:, -1]!=background) > 0
    ):
        return sprite
    
    # if the sprite is not flushed with the border, then we need to regenerate it
    return random_sprite(n_original, m_original, density_original, symmetry_original, color_palette_original, connectivity_original, background_original)



def detect_objects(grid, _=None, predicate=None, background=Color.BLACK, monochromatic=False, connectivity=None, allowed_dimensions=None, colors=None, can_overlap=False):
    """
    Detects and extracts objects from the grid that satisfy custom specification.

    predicate: a function that takes a candidate object as input and returns True if it counts as an object
    background: color treated as transparent
    monochromatic: if True, each object is assumed to have only one color. If False, each object can include multiple colors.
    connectivity: 4 or 8, for 4-way or 8-way connectivity. If None, the connectivity is determined automatically.
    allowed_dimensions: a list of tuples (n, m) specifying the allowed dimensions of the objects. If None, objects of any size are allowed.
    colors: a list of colors that the objects are allowed to have. If None, objects of any color are allowed.
    can_overlap: if True, objects can overlap. If False, objects cannot overlap.

    Returns a list of objects, where each object is a numpy array.
    """

    objects = []

    if connectivity:
        objects.extend(find_connected_components(grid, background=background, connectivity=connectivity, monochromatic=monochromatic))
        if colors:
            objects = [obj for obj in objects if all((color in colors) or color == background for color in obj.flatten())]
        if predicate:
            objects = [obj for obj in objects if predicate(crop(obj, background=background))]

    if allowed_dimensions:
        objects = [obj for obj in objects if obj.shape in allowed_dimensions]

        # Also scan through the grid
        scan_objects = []
        for n, m in allowed_dimensions:
            for i in range(grid.shape[0] - n + 1):
                for j in range(grid.shape[1] - m + 1):
                    candidate_sprite = grid[i:i+n, j:j+m]

                    if np.any(candidate_sprite != background) and \
                        (colors is None or all((color in colors) or color == background for color in candidate_sprite.flatten())) and \
                        (predicate is None or predicate(candidate_sprite)):
                        candidate_object = np.full(grid.shape, background)
                        candidate_object[i:i+n, j:j+m] = candidate_sprite
                        if not any( np.all(candidate_object == obj) for obj in objects):
                            scan_objects.append(candidate_object)
        #print("scanning produced", len(scan_objects), "objects")
        objects.extend(scan_objects)

    if not can_overlap:
        import time
        start = time.time()
        # sort objects by size, breaking ties by mass
        objects.sort(key=lambda obj: (crop(obj, background).shape[0] * crop(obj, background).shape[1], np.sum(obj!=background)), reverse=True)
        overlap_matrix = np.full((len(objects), len(objects)), False)
        object_masks = [obj != background for obj in objects]
        object_bounding_boxes = [bounding_box(obj, background=background) for obj in object_masks]
        for i, obj1 in enumerate(object_masks):
            for j, obj2 in enumerate(object_masks):
                if i < j:
                    # check if the bounding boxes overlap
                    # FIXME: this doesn't work
                    x1, y1, n1, m1 = object_bounding_boxes[i]
                    x2, y2, n2, m2 = object_bounding_boxes[j]
                    if True or x1 + n1 <= x2 or x2 + n2 <= x1 or y1 + m1 <= y2 or y2 + m2 <= y1:
                        overlap_matrix[i, j] = np.any(obj1 & obj2)
                        overlap_matrix[j, i] = overlap_matrix[i, j]
        #print("time to compute overlaps", time.time() - start)
        start= time.time()

        # Pick a subset of objects that don't overlap and which cover as many pixels as possible
        # First, we definitely pick everything that doesn't have any overlaps
        keep_objects = [obj for i, obj in enumerate(objects) if not np.any(overlap_matrix[i])]

        # Second, we might pick the remaining objects
        remaining_indices = [i for i, obj in enumerate(objects) if np.any(overlap_matrix[i])]

        # Figure out the best possible score we could get if we cover everything
        best_possible_mask = np.zeros_like(grid, dtype=bool)
        for i in remaining_indices:
            best_possible_mask |= objects[i] != background
        best_possible_score = np.sum(best_possible_mask)

        # Now we just do a brute force search recursively
        def pick_objects(remaining_indices, current_indices, current_mask):
            nonlocal overlap_matrix

            if not remaining_indices:
                solution = [objects[i] for i in current_indices]
                solution_goodness = np.sum(current_mask)
                return solution, solution_goodness

            first_index, *rest = remaining_indices
            # Does that object have any overlap with the current objects? If so don't pick it
            if any( overlap_matrix[i, first_index] for i in current_indices):
                return pick_objects(rest, current_indices, current_mask)

            # Try picking it
            with_index, with_goodness = pick_objects(rest, current_indices + [first_index], current_mask | (objects[first_index] != background))

            # Did we win?
            if with_goodness == best_possible_score:
                return with_index, with_goodness

            # Try not picking it
            without_index, without_goodness = pick_objects(rest, current_indices, current_mask)

            if with_goodness > without_goodness:
                return with_index, with_goodness
            else:
                return without_index, without_goodness

        solution, _ = pick_objects(remaining_indices, [], np.zeros_like(grid, dtype=bool))
        #print("time to pick objects", time.time() - start)

        objects = keep_objects + solution

    return objects


""" ==============================
Puzzle 007bbfb7

Train example 1:
Input1 = [
 k o o
 o o o
 k o o
]
Output1 = [
 k k k k o o k o o
 k k k o o o o o o
 k k k k o o k o o
 k o o k o o k o o
 o o o o o o o o o
 k o o k o o k o o
 k k k k o o k o o
 k k k o o o o o o
 k k k k o o k o o
]

Train example 2:
Input2 = [
 y k y
 k k k
 k y k
]
Output2 = [
 y k y k k k y k y
 k k k k k k k k k
 k y k k k k k y k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k y k y k k k
 k k k k k k k k k
 k k k k y k k k k
]

Train example 3:
Input3 = [
 k k k
 k k r
 r k r
]
Output3 = [
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k r
 k k k k k k r k r
 k k k k k k k k k
 k k r k k k k k r
 r k r k k k r k r
] """

# concepts:
# repeating patterns, colors as indicators, scaling

# description:
# In the input you will see a nxm sprite with black background. 
# Construct an output grid with n^2 x m^2 black pixels. Divide the output grid into subgrids, 
# and look at the corresponding pixel in the nxm input grid. If the corresponding pixel is not black, 
# then copy the nxm input grid into the subgrid. Else, the subgrid does not change. 

def transform_007bbfb7(input_grid):
    # creates an empty 9x9 output grid 
    output_grid = np.zeros((input_grid.shape[0]**2,input_grid.shape[1]**2),dtype=int)

    input_sprite = input_grid

    # Go through the input grid. If an input grid pixel is not black, 
    # then copy the input grid to the corresponding location on the output grid
    for n in range(input_grid.shape[0]):
      for m in range(input_grid.shape[1]):
        if input_grid[n,m] != Color.BLACK:
            blit_sprite(output_grid, input_sprite, n*input_grid.shape[0], m*input_grid.shape[1])
    
    return output_grid

""" ==============================
Puzzle 00d62c1b

Train example 1:
Input1 = [
 k k k k k k
 k k g k k k
 k g k g k k
 k k g k g k
 k k k g k k
 k k k k k k
]
Output1 = [
 k k k k k k
 k k g k k k
 k g y g k k
 k k g y g k
 k k k g k k
 k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k g k g k k k k k
 k k k g k g k k k k
 k k g k k k g k k k
 k k k k k g k g k k
 k k k g k g g k k k
 k k g g g k k k k k
 k k k g k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k k g k g k k k k k
 k k k g k g k k k k
 k k g k k k g k k k
 k k k k k g y g k k
 k k k g k g g k k k
 k k g g g k k k k k
 k k k g k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k g k k k k
 k k k k g k k k k k
 k g g k g g k g k k
 g k k g k k g k g k
 k k k g k k g g k k
 k k k g k k g k k k
 k k k g k k g k k k
 k k k k g g k g k k
 k k k k k k k k g k
 k k k k k k k k k k
]
Output3 = [
 k k k k k g k k k k
 k k k k g k k k k k
 k g g k g g k g k k
 g k k g y y g y g k
 k k k g y y g g k k
 k k k g y y g k k k
 k k k g y y g k k k
 k k k k g g k g k k
 k k k k k k k k g k
 k k k k k k k k k k
] """

# concepts:
# topology

# description:
# The input grid is a square grid with black and green pixels. The input grid should have regions that are enclosed by the green pixels. 
# To produce the output, you need to find the enclosed regions in the input grid, and then color them yellow. 
                
def transform_00d62c1b(input_grid):
    # Create initial output grid template based on input grid.
    output_grid = input_grid.copy()

    # Find enclosed regions
    interior_mask = object_interior(input_grid)
    boundary_mask = object_boundary(input_grid)
    inside_but_not_on_edge = interior_mask & ~boundary_mask

    # Color enclosed regions
    for x, y in np.argwhere(inside_but_not_on_edge):
        if output_grid[x, y] == Color.BLACK:
            output_grid[x, y] = Color.YELLOW

    return output_grid

""" ==============================
Puzzle 017c7c7b

Train example 1:
Input1 = [
 k b k
 b b k
 k b k
 k b b
 k b k
 b b k
]
Output1 = [
 k r k
 r r k
 k r k
 k r r
 k r k
 r r k
 k r k
 k r r
 k r k
]

Train example 2:
Input2 = [
 k b k
 b k b
 k b k
 b k b
 k b k
 b k b
]
Output2 = [
 k r k
 r k r
 k r k
 r k r
 k r k
 r k r
 k r k
 r k r
 k r k
]

Train example 3:
Input3 = [
 k b k
 b b k
 k b k
 k b k
 b b k
 k b k
]
Output3 = [
 k r k
 r r k
 k r k
 k r k
 r r k
 k r k
 k r k
 r r k
 k r k
] """

# concepts:
# translational symmetry, symmetry detection

# description:
# In the input you will see a grid consisting of a blue sprite that is repeatedly translated vertically, forming a stack of the same sprite.
# To make the output, expand the input to have height 9, and continue to repeatedly translate the sprite vertically. Change color to red.
 
def transform_017c7c7b(input_grid):
    # Plan:
    # 1. Find the repeated translation, which is a symmetry
    # 2. Extend the pattern by copying the sprite and its symmetric copies
    # 3. Change the color from blue to red
    
    symmetries = detect_translational_symmetry(input_grid, ignore_colors=[], background=Color.BLACK)
    assert len(symmetries) > 0, "No translational symmetry found"

    # make the output (the height is now 9)
    output_grid = np.full((input_grid.shape[0], 9), Color.BLACK)
    
    # Copy all of the input pixels to the output, INCLUDING their symmetric copies (i.e. their orbit)
    for x, y in np.argwhere(input_grid != Color.BLACK):
        # Compute the orbit into the output grid
        for x2, y2 in orbit(output_grid, x, y, symmetries):
            output_grid[x2, y2] = input_grid[x, y]
    
    # Color change: blue -> red
    output_grid[output_grid == Color.BLUE] = Color.RED

    return output_grid

""" ==============================
Puzzle 025d127b

Train example 1:
Input1 = [
 k k k k k k k k k
 k p p p k k k k k
 k p k k p k k k k
 k k p k k p k k k
 k k k p k k p k k
 k k k k p p p k k
 k k k k k k k k k
 k k r r r k k k k
 k k r k k r k k k
 k k k r r r k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
]
Output1 = [
 k k k k k k k k k
 k k p p p k k k k
 k k p k k p k k k
 k k k p k k p k k
 k k k k p k p k k
 k k k k p p p k k
 k k k k k k k k k
 k k k r r r k k k
 k k k r k r k k k
 k k k r r r k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k
 k t t t t t k k k
 k t k k k k t k k
 k k t k k k k t k
 k k k t k k k k t
 k k k k t t t t t
 k k k k k k k k k
 k k k k k k k k k
]
Output2 = [
 k k k k k k k k k
 k k t t t t t k k
 k k t k k k k t k
 k k k t k k k k t
 k k k k t k k k t
 k k k k t t t t t
 k k k k k k k k k
 k k k k k k k k k
] """

# concepts:
# objects, pixel manipulation

# description:
# In the input you will see a set of objects, each consisting of a horizontal top/bottom and diagonal left/right edges (but that structure is not important)
# To make the output shift right each pixel in the object *except* when there are no other pixels down and to the right

def transform_025d127b(input_grid: np.ndarray) -> np.ndarray:
    # find the connected components, which are monochromatic objects
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=True)

    output_grid = np.zeros_like(input_grid)

    for obj in objects:
        transformed_object = np.zeros_like(obj)

        for x in range(obj.shape[0]):
            for y in range(obj.shape[1]):
                if obj[x, y] != Color.BLACK:
                    # check that there are other colored pixels down and to the right
                    down_and_to_the_right = obj[x+1:, y+1:]
                    if np.any(down_and_to_the_right != Color.BLACK):
                        transformed_object[x+1, y] = obj[x, y]
                    else:
                        transformed_object[x, y] = obj[x, y]

        blit_object(output_grid, transformed_object, background=Color.BLACK)

    return output_grid

""" ==============================
Puzzle 045e512c """

# concepts:
# sprites, color change, collision detection, repetition, overlap

# description:
# In the input you will see a 3x3 object with a few other objects around it.
# For each of the other sprites around the central 3x3 object:
# 1. Slide the central sprite so it completely overlaps the other sprite (slide it as much as you can to do so)
# 2. Change the color of the central sprite to match the color of the other sprite
# 3. Repeat the slide (by the same displacement vector) indefinitely until it falls off the canvas

def transform_045e512c(input_grid: np.ndarray) -> np.ndarray:
    # find the objects, which are monochromatic connected components
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=True)

    # find the central object, which is the biggest
    central_object = max(objects, key=lambda obj: np.sum(obj != Color.BLACK))

    # find the other objects
    other_objects = [obj for obj in objects if not np.array_equal(obj, central_object)]

    output_grid = np.copy(input_grid)

    for other_object in other_objects:
        # find the biggest displacement vector that will make the central object completely overlap the other object
        biggest_displacement_vector = (0,0)
        displacement_vectors = [ (i, j) for i in range(-10, 10) for j in range(-10, 10) ]
        for displacement_vector in displacement_vectors:
            # translate the central object by the displacement vector
            translated_central_object = translate(central_object, displacement_vector[0], displacement_vector[1], background=Color.BLACK)

            # check if the translated object completely overlaps the other object
            translated_mask, other_mask = translated_central_object != Color.BLACK, other_object != Color.BLACK
            overlaps = np.all(translated_mask & other_mask == other_mask)

            if overlaps:
                # but is it the biggest?
                if biggest_displacement_vector[0] ** 2 + biggest_displacement_vector[1] ** 2 < displacement_vector[0] ** 2 + displacement_vector[1] ** 2:
                    biggest_displacement_vector = displacement_vector

        displacement_vector = biggest_displacement_vector

        # color change
        color_of_other_object = np.unique(other_object[other_object != Color.BLACK])[0]
        central_object[central_object != Color.BLACK] = color_of_other_object

        # repeat the displacement indefinitely until it falls off the canvas
        for i in range(1, 10):
            displaced_central_object = translate(central_object, displacement_vector[0] * i, displacement_vector[1] * i, background=Color.BLACK)
            blit_object(output_grid, displaced_central_object, background=Color.BLACK)
            
    return output_grid

""" ==============================
Puzzle 0520fde7

Train example 1:
Input1 = [
 b k k e k b k
 k b k e b b b
 b k k e k k k
]
Output1 = [
 k k k
 k r k
 k k k
]

Train example 2:
Input2 = [
 b b k e k b k
 k k b e b b b
 b b k e k b k
]
Output2 = [
 k r k
 k k r
 k r k
]

Train example 3:
Input3 = [
 k k b e k k k
 b b k e b k b
 k b b e b k b
]
Output3 = [
 k k k
 r k k
 k k r
] """

# concepts:
# boolean logical operations, bitmasks with separator

# description:
# In the input you will see two blue bitmasks separated by a grey vertical bar
# To make the output, color teal the red that are set in both bitmasks (logical AND)

def transform_0520fde7(input_grid: np.ndarray) -> np.ndarray:
    # Find the grey vertical bar. Vertical means constant X
    for x_bar in range(input_grid.shape[0]):
        if np.all(input_grid[x_bar, :] == Color.GREY):
            break

    left_mask = input_grid[:x_bar, :]
    right_mask = input_grid[x_bar+1:, :]

    output_grid = np.zeros_like(left_mask)
    output_grid[(left_mask == Color.BLUE) & (right_mask == Color.BLUE)] = Color.RED
    
    return output_grid

""" ==============================
Puzzle 05269061

Train example 1:
Input1 = [
 r t g k k k k
 t g k k k k k
 g k k k k k k
 k k k k k k k
 k k k k k k k
 k k k k k k k
 k k k k k k k
]
Output1 = [
 r t g r t g r
 t g r t g r t
 g r t g r t g
 r t g r t g r
 t g r t g r t
 g r t g r t g
 r t g r t g r
]

Train example 2:
Input2 = [
 k k k k k k k
 k k k k k k k
 k k k k k k b
 k k k k k b r
 k k k k b r y
 k k k b r y k
 k k b r y k k
]
Output2 = [
 r y b r y b r
 y b r y b r y
 b r y b r y b
 r y b r y b r
 y b r y b r y
 b r y b r y b
 r y b r y b r
]

Train example 3:
Input3 = [
 k k k k t g k
 k k k t g k k
 k k t g k k k
 k t g k k k y
 t g k k k y k
 g k k k y k k
 k k k y k k k
]
Output3 = [
 y t g y t g y
 t g y t g y t
 g y t g y t g
 y t g y t g y
 t g y t g y t
 g y t g y t g
 y t g y t g y
] """

# concepts:
# diagonal lines, repetition

# description:
# In the input you will see a 7x7 grid, with three diagonal lines that stretch from one end of the canvas to the other
# Each line is a different color, and the colors are not black
# The output should be the result of repeating every diagonal line on multiples of 3 offset from the original, which gives an interlacing pattern filling the output canvas


def transform_05269061(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.zeros((7, 7), dtype=int)

    # Loop over the input looking for any of the three diagonals
    # If we find one, we will fill the output with the same color in the same pattern
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            c = input_grid[i][j]
            if c != Color.BLACK:
                # Fill the output with the same color in the same pattern
                # Loop by multiples of 3 to create the pattern
                # Loop way beyond the canvas (double the canvas size) to make sure we cover everything
                for distance in range(0, output_grid.shape[0]*2, 3):
                    draw_diagonal(output_grid, i-distance, j, c)
                    draw_diagonal(output_grid, i+distance, j, c)
    
    return output_grid

def draw_diagonal(grid, x, y, c):
    # create diagonal line that stretches from one side of the canvas to the other
    # to do this, draw infinite rays pointing in opposite directions
    draw_line(grid, x, y, length=None, color=c, direction=(1, -1))
    draw_line(grid, x, y, length=None, color=c, direction=(-1, 1))

""" ==============================
Puzzle 05f2a901

Train example 1:
Input1 = [
 k k k k k k k k k
 k k k k k k k k k
 k r r r k k k k k
 r r k r k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k t t k k k k
 k k k t t k k k k
 k k k k k k k k k
 k k k k k k k k k
]
Output1 = [
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k r r r k k k k k
 r r k r k k k k k
 k k k t t k k k k
 k k k t t k k k k
 k k k k k k k k k
 k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k r r k k k k k k k
 k r r k k k k k k k
 r r r k k k k k k k
 k r r k k k t t k k
 k k k k k k t t k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k k k k r r k k k k
 k k k k r r k k k k
 k k k r r r k k k k
 k k k k r r t t k k
 k k k k k k t t k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k k t t k k k k k
 k k k t t k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k r r r k k k k
 k r r r r r k k k k
 k k r r k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k
 k k k t t k k k k k
 k k k t t k k k k k
 k k k r r r k k k k
 k r r r r r k k k k
 k k r r k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
] """

# concepts:
# collision detection, sliding objects

# description:
# In the input you will see a teal 2x2 square and a red object (the red object might be irregular in its shape)
# Slide the red object in any of the four directions until it just touches the teal square

def transform_05f2a901(input_grid):

    # get just the teal object
    teal_object = np.zeros_like(input_grid)
    teal_object[input_grid == Color.TEAL] = Color.TEAL

    # get just the red object
    red_object = np.zeros_like(input_grid)
    red_object[input_grid == Color.RED] = Color.RED

    # the output grid starts with just the teal object, because we still need to figure out where the red object will be by sliding it
    output_grid = np.copy(teal_object)
    
    # consider sliding in the 4 cardinal directions, and consider sliding as far as possible
    possible_displacements = [ (slide_distance*dx, slide_distance*dy)
                               for slide_distance in range(max(input_grid.shape))
                               for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] ]
    for x, y in possible_displacements:
        # check if the objects are touching after sliding
        translated_red_object = translate(red_object, x, y, background=Color.BLACK)
        if contact(object1=teal_object, object2=translated_red_object):
            # put the red object where it belongs
            blit_object(output_grid, translated_red_object, background=Color.BLACK)
            return output_grid
            
    assert 0, "No valid slide found"

""" ==============================
Puzzle 06df4c85 """

# concepts:
# rectangular cells, flood fill, connecting same color

# description:
# In the input you will see horizontal and vertical bars that divide the grid into rectangular cells
# To make the output, find any pair of rectangular cells that are in the same row and column and have the same color, then color all the rectangular cells between them with that color

def transform_06df4c85(input_grid: np.ndarray) -> np.ndarray:

    # find the color of the horizontal and vertical bars that divide the rectangular cells
    # this is the color of any line that extends all the way horizontally or vertically
    jail_color = None
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            color = input_grid[i][j]
            if np.all(input_grid[i, :] == color) or np.all(input_grid[:, j] == color):
                jail_color = color
                break
    
    assert jail_color is not None, "No jail color found"

    output_grid = input_grid.copy()

    # color all the cells between the same color pixels
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            color = input_grid[x][y]
            if color == jail_color or color == Color.BLACK:
                continue

            # check if there is a cell with the same color in the same X value
            for y2 in range(y+1, input_grid.shape[1]):
                if input_grid[x][y2] == color:
                    for y3 in range(y+1, y2):
                        if input_grid[x][y3] == Color.BLACK:
                            output_grid[x][y3] = color
                    break

            # check if there is a cell with the same color in the same Y value
            for x2 in range(x+1, input_grid.shape[0]):
                if input_grid[x2][y] == color:
                    for x3 in range(x+1, x2):
                        if input_grid[x3][y] == Color.BLACK:
                            output_grid[x3][y] = color
                    break
                
    return output_grid

""" ==============================
Puzzle 08ed6ac7

Train example 1:
Input1 = [
 k k k k k e k k k
 k e k k k e k k k
 k e k k k e k k k
 k e k e k e k k k
 k e k e k e k k k
 k e k e k e k k k
 k e k e k e k e k
 k e k e k e k e k
 k e k e k e k e k
]
Output1 = [
 k k k k k b k k k
 k r k k k b k k k
 k r k k k b k k k
 k r k g k b k k k
 k r k g k b k k k
 k r k g k b k k k
 k r k g k b k y k
 k r k g k b k y k
 k r k g k b k y k
]

Train example 2:
Input2 = [
 k k k k k k k k k
 k k k k k k k e k
 k k k k k k k e k
 k k k k k k k e k
 k k k e k k k e k
 k k k e k e k e k
 k k k e k e k e k
 k e k e k e k e k
 k e k e k e k e k
]
Output2 = [
 k k k k k k k k k
 k k k k k k k b k
 k k k k k k k b k
 k k k k k k k b k
 k k k r k k k b k
 k k k r k g k b k
 k k k r k g k b k
 k y k r k g k b k
 k y k r k g k b k
] """

# concepts:
# sorting, color change, size

# description:
# In the input you will see a row of exactly 4 grey bars of different heights, each starting at the bottom of the canvas, and each separated by 1 pixel (so they are two pixels apart)
# Color the tallest one blue, the second tallest one red, the third tallest one green, and the shortest one yellow.

def transform_08ed6ac7(input_grid):

    # extract the bars, each of which is a connected component
    bars = find_connected_components(input_grid, background=Color.BLACK)

    # sort the bars by height
    bars = list(sorted(bars, key=lambda bar: np.sum(bar != Color.BLACK), reverse=True))

    # color the bars
    output_grid = input_grid.copy()

    biggest_bar = bars[0]
    biggest_bar_mask = biggest_bar != Color.BLACK
    output_grid[biggest_bar_mask] = Color.BLUE

    second_biggest_bar = bars[1]
    second_biggest_bar_mask = second_biggest_bar != Color.BLACK
    output_grid[second_biggest_bar_mask] = Color.RED

    third_biggest_bar = bars[2]
    third_biggest_bar_mask = third_biggest_bar != Color.BLACK
    output_grid[third_biggest_bar_mask] = Color.GREEN

    smallest_bar = bars[3]
    smallest_bar_mask = smallest_bar != Color.BLACK
    output_grid[smallest_bar_mask] = Color.YELLOW

    return output_grid

""" ==============================
Puzzle 09629e4f

Train example 1:
Input1 = [
 r k k e k p r e k k y
 k y g e y k t e g k p
 p k k e g k k e t k r
 e e e e e e e e e e e
 g t k e p r k e k y t
 k k y e k k y e p k k
 p r k e g t k e k g r
 e e e e e e e e e e e
 k g p e k r k e k p k
 r k k e y k t e k k t
 t k y e p g k e r g y
]
Output1 = [
 r r r e k k k e k k k
 r r r e k k k e k k k
 r r r e k k k e k k k
 e e e e e e e e e e e
 k k k e y y y e g g g
 k k k e y y y e g g g
 k k k e y y y e g g g
 e e e e e e e e e e e
 p p p e k k k e k k k
 p p p e k k k e k k k
 p p p e k k k e k k k
]

Train example 2:
Input2 = [
 r k g e y p k e k p k
 k k t e k k r e y k g
 y p k e g t k e r k t
 e e e e e e e e e e e
 y k t e k k r e k p y
 k k r e k g k e g k k
 g k p e y k p e t k r
 e e e e e e e e e e e
 g p k e k t y e r k k
 k t y e r k k e t k g
 r k k e k g p e p y k
]
Output2 = [
 k k k e k k k e r r r
 k k k e k k k e r r r
 k k k e k k k e r r r
 e e e e e e e e e e e
 k k k e g g g e k k k
 k k k e g g g e k k k
 k k k e g g g e k k k
 e e e e e e e e e e e
 y y y e k k k e p p p
 y y y e k k k e p p p
 y y y e k k k e p p p
]

Train example 3:
Input3 = [
 k g k e k p g e k p r
 p k y e r t k e k k t
 k r t e k y k e g k y
 e e e e e e e e e e e
 k r k e y k g e g y k
 y k t e r k p e k k r
 g p k e k t k e t p k
 e e e e e e e e e e e
 p g k e k g k e k k g
 k k r e k p y e r t k
 t y k e r k k e y k p
]
Output3 = [
 k k k e g g g e k k k
 k k k e g g g e k k k
 k k k e g g g e k k k
 e e e e e e e e e e e
 k k k e p p p e y y y
 k k k e p p p e y y y
 k k k e p p p e y y y
 e e e e e e e e e e e
 r r r e k k k e k k k
 r r r e k k k e k k k
 r r r e k k k e k k k
] """

# concepts:
# rectangular cells, color guide

# description:
# In the input you will see grey horizontal and vertical bars that divide the grid into nine 3x3 rectangular regions, each of which contains 4-5 colored pixels
# To make the output, find the region that has exactly 4 colored pixels, and use its colors as a guide to fill in all the other cells

def transform_09629e4f(input_grid: np.ndarray) -> np.ndarray:

    # First identify 

    # Trick for decomposing inputs divided into rectangular regions by horizontal/vertical bars:
    # Treat the bar color as the background, and break the input up into connected components with that background color

    # The divider color is the color of the horizontal and vertical bars
    divider_colors = [ input_grid[x,y] for x in range(input_grid.shape[0]) for y in range(input_grid.shape[1])
                     if np.all(input_grid[x,:] == input_grid[x,0]) or np.all(input_grid[:,y] == input_grid[0,y]) ]
    assert len(set(divider_colors)) == 1, "There should be exactly one divider color"
    divider_color = divider_colors[0] # background=divider_color

    # Find multicolored regions, which are divided by divider_color, so we treat that as background, because it separates objects
    # Within each region there can be multiple colors
    regions = find_connected_components(input_grid, background=divider_color, monochromatic=False)
    # Tag the regions with their location within the 2D grid of (divided) regions
    # First get the bounding-box locations...
    locations = []
    for region in regions:
        x, y, w, h = bounding_box(region, background=divider_color)
        locations.append((x, y, region))
    # ...then re-index them so that (x, y) is the coordinate within the grid of rectangular regions
    grid_of_regions = []
    for x, y, region in locations:
        num_left_of_region = len({other_x for other_x, other_y, other_region in locations if other_x < x})
        num_above_region = len({other_y for other_x, other_y, other_region in locations if other_y < y})
        grid_of_regions.append((num_left_of_region, num_above_region, region))

    # Find the region with exactly 4 colors
    special_region = None
    for region in regions:
        not_divider_and_not_black = (region != divider_color) & (region != Color.BLACK)
        if np.sum(not_divider_and_not_black) == 4:
            assert special_region is None, "More than one special region found"
            special_region = region
    
    # Convert to a sprite
    special_sprite = crop(special_region, background=divider_color)
    
    # Create the output grid
    output_grid = np.zeros_like(input_grid)

    # Put the dividers back in
    output_grid[input_grid == divider_color] = divider_color

    # Fill in the cells with the special colors
    for x, y, region in grid_of_regions:
        output_grid[region != divider_color] = special_sprite[x, y]

    return output_grid

""" ==============================
Puzzle 0962bcdd

Train example 1:
Input1 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k o k k k k k k k k k
 k o r o k k k k k k k k
 k k o k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k o k k k
 k k k k k k k o r o k k
 k k k k k k k k o k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k
 r k o k r k k k k k k k
 k r o r k k k k k k k k
 o o r o o k k k k k k k
 k r o r k k k k k k k k
 r k o k r k r k o k r k
 k k k k k k k r o r k k
 k k k k k k o o r o o k
 k k k k k k k r o r k k
 k k k k k k r k o k r k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k t k k k k k k k k
 k k t p t k k k k k k k
 k k k t k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k t k k k
 k k k k k k k t p t k k
 k k k k k k k k t k k k
 k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k
 k p k t k p k k k k k k
 k k p t p k k k k k k k
 k t t p t t k k k k k k
 k k p t p k k k k k k k
 k p k t k p k k k k k k
 k k k k k k k k k k k k
 k k k k k k p k t k p k
 k k k k k k k p t p k k
 k k k k k k t t p t t k
 k k k k k k k p t p k k
 k k k k k k p k t k p k
] """

# concepts:
# pixel manipulation, growing

# description:
# In the input you will see some number of colored crosses, each of which is 3 pixels tall, 3 pixels wide, and has a single pixel in the center of the cross that is a different color.
# Make the output by growing the cross by 1 pixel north/south/east/west, and growing the center pixel by 2 pixels along each of the 4 diagonals.

def transform_0962bcdd(input_grid):

    # extract the 3x3 crosses
    crosses = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False)

    output_grid = input_grid.copy()

    for cross in crosses:
        # find the center
        x, y, w, h = bounding_box(cross)
        center_x, center_y = x + w//2, y + h//2

        # extract the relevant colors
        center_color = cross[center_x, center_y]
        cross_color = cross[cross != Color.BLACK][0]

        # grow the cross
        for output_x in range(x-1, x+w+1):
            for output_y in range(y-1, y+h+1):
                # skip if out of bounds
                if output_x < 0 or output_y < 0 or output_x >= input_grid.shape[0] or output_y >= input_grid.shape[1]:
                    continue
                
                # grow the cross north/south/east/west
                if output_x == center_x or output_y == center_y:
                    output_grid[output_x, output_y] = cross_color
                
                # grow the center diagonally
                if (output_x - center_x) == (output_y - center_y) or (output_x - center_x) == (center_y - output_y):
                    output_grid[output_x, output_y] = center_color

    return output_grid

""" ==============================
Puzzle 0a938d79

Train example 1:
Input1 = [
 k k k k k r k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k t k k k k k k k k k k k k k k k k k
]
Output1 = [
 k k k k k r k t k r k t k r k t k r k t k r k t k
 k k k k k r k t k r k t k r k t k r k t k r k t k
 k k k k k r k t k r k t k r k t k r k t k r k t k
 k k k k k r k t k r k t k r k t k r k t k r k t k
 k k k k k r k t k r k t k r k t k r k t k r k t k
 k k k k k r k t k r k t k r k t k r k t k r k t k
 k k k k k r k t k r k t k r k t k r k t k r k t k
 k k k k k r k t k r k t k r k t k r k t k r k t k
 k k k k k r k t k r k t k r k t k r k t k r k t k
 k k k k k r k t k r k t k r k t k r k t k r k t k
]

Train example 2:
Input2 = [
 k k k k k b k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k g k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k b k k g k k b k k g k k b k k g k k
 k k k k k b k k g k k b k k g k k b k k g k k
 k k k k k b k k g k k b k k g k k b k k g k k
 k k k k k b k k g k k b k k g k k b k k g k k
 k k k k k b k k g k k b k k g k k b k k g k k
 k k k k k b k k g k k b k k g k k b k k g k k
 k k k k k b k k g k k b k k g k k b k k g k k
]

Train example 3:
Input3 = [
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 r k k k k k k k k
 k k k k k k k k k
 k k k k k k k k g
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
]
Output3 = [
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 r r r r r r r r r
 k k k k k k k k k
 g g g g g g g g g
 k k k k k k k k k
 r r r r r r r r r
 k k k k k k k k k
 g g g g g g g g g
 k k k k k k k k k
 r r r r r r r r r
 k k k k k k k k k
 g g g g g g g g g
 k k k k k k k k k
 r r r r r r r r r
 k k k k k k k k k
 g g g g g g g g g
 k k k k k k k k k
 r r r r r r r r r
] """

# concepts:
# repetition, horizontal/vertical bars

# description:
# In the input you will see a pair of colored pixels
# Make each pixel into a horizontal/vertical bar by connecting it to the other side of the canvas
# Then, repeat the bars indefinitely in the same direction: either downward (for horizontal bars) or rightward (for vertical bars)

def transform_0a938d79(input_grid: np.ndarray) -> np.ndarray:
    # find the individual coloured pixels
    colored_pixels = np.argwhere(input_grid != Color.BLACK)

    # there should be exactly two colored pixels
    assert len(colored_pixels) == 2

    # find the two pixels
    pixel1, pixel2 = colored_pixels
    x1, y1 = pixel1
    x2, y2 = pixel2
    color1, color2 = input_grid[x1, y1], input_grid[x2, y2]

    # make the horizontal/vertical bars
    output_grid = np.copy(input_grid)

    # check if they should be horizontal (constant y) or vertical (constant x)
    # if they are on the top or bottom, they should be vertical
    if y1 == 0 or y1 == input_grid.shape[1] - 1 or y2 == 0 or y2 == input_grid.shape[1] - 1:
        # vertical bars: constant x
        output_grid[x1, :] = color1
        output_grid[x2, :] = color2

        # repeat the vertical bars indefinitely
        # first, figure out how far apart they are.
        dx = abs(x2 - x1)
        # next, each repetition needs to be separated by twice that amount, because we have two lines
        dx = 2 * dx
        # finally, repeat the vertical bars indefinitely, using the same colors and using a list slice to repeat the bars
        output_grid[x1::dx, :] = color1
        output_grid[x2::dx, :] = color2
    else:
        # horizontal bars: constant y
        output_grid[:, y1] = color1
        output_grid[:, y2] = color2

        # repeat the horizontal bars indefinitely
        # first, figure out how far apart they are.
        dy = abs(y2 - y1)
        # next, each repetition needs to be separated by twice that amount, because we have two lines
        dy = 2 * dy
        # finally, repeat the horizontal bars indefinitely, using the same colors and using a list slice to repeat the bars
        output_grid[:, y1::dy] = color1
        output_grid[:, y2::dy] = color2

    return output_grid

""" ==============================
Puzzle 0b148d64

Train example 1:
Input1 = [
 t t t t t k t t t t k k k k t t t t k t t
 t k k t k t k t t t k k k k t t t k k k t
 t t t k k k t t t t k k k k t t k t t t t
 t t k t t t t k t t k k k k t t k k k t t
 t t t t k t t k t t k k k k t t t k t t t
 k k k t t k t k k t k k k k t k k k t k k
 t t t t k k t k t k k k k k t t t k t t t
 t k k t k k t t k t k k k k t k t t t t t
 t t t t t t k t k k k k k k t t t t t k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k r r r k k r r r r k k k k t t k t t k t
 r k r r r k k r r r k k k k t t t t k t k
 k r r r r r r k r k k k k k t t t k k k t
 r r r r k r r r r r k k k k t t k t t t k
 r r r r r r k r k k k k k k t t t t t k k
 r r r r r k r k r r k k k k t k t k t t t
 r r k r r k k k k k k k k k t t k t k k t
 k r r k k r r k k r k k k k t k k k t t k
 r r r r r r r r r r k k k k k t t k k t t
 r k r r k r r r r r k k k k t t t k t t t
]
Output1 = [
 k r r r k k r r r r
 r k r r r k k r r r
 k r r r r r r k r k
 r r r r k r r r r r
 r r r r r r k r k k
 r r r r r k r k r r
 r r k r r k k k k k
 k r r k k r r k k r
 r r r r r r r r r r
 r k r r k r r r r r
]

Train example 2:
Input2 = [
 r k r r r r k k k k r k r r r r k k r
 r r r r k r r k k k k r r r r r k k k
 k k r r k r k k k k r r r k r r r r r
 r k r k r r k k k k k r r r r r r k k
 k r k r r r r k k k k k k r r k r r r
 r r r k r k r k k k r k r r r r k r k
 k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k
 r k r k k k r k k k k g g g g g k g g
 k r r k k r r k k k g g g k k k g g k
 k r r k k r k k k k g g g k g k g k k
 r r r k k r r k k k g g k k k g g g g
 r k k r r r k k k k g k k k g k g k g
 r k r k k k r k k k k g g k g g g k g
 k r r k r r k k k k k g g k k g k g k
]
Output2 = [
 k g g g g g k g g
 g g g k k k g g k
 g g g k g k g k k
 g g k k k g g g g
 g k k k g k g k g
 k g g k g g g k g
 k g g k k g k g k
]

Train example 3:
Input3 = [
 k b k b b b k k b b k b k k k k k
 b k b k k k k k b b b b b b k b b
 b b k b b k k k b b b b b b k b b
 b b k k b b k k b b k b b b b b b
 k b b b k k k k b b k k k b b b k
 b k k b k k k k b b k k b b b b b
 k k k b b k k k b b b k k b k k b
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 y k k y k y k k b k k b b b b b b
 y y y y k y k k b k b b b b b b k
 y k y k k y k k k b k k b b b b b
 k y y y y k k k b b k k b k b k b
 y y y k y y k k b b b b b b b b k
 k y y y y k k k k b k k k k b b b
 k y y y k y k k k b k b k b b b k
 k y k k k k k k b k b b b k b k b
 y y k y k y k k b b b k k b b b k
]
Output3 = [
 y k k y k y
 y y y y k y
 y k y k k y
 k y y y y k
 y y y k y y
 k y y y y k
 k y y y k y
 k y k k k k
 y y k y k y
] """

# concepts:
# rectangular cells, color guide

# description:
# In the input you will see a pretty big grid divided into four axis-aligned quadrants (but there might be different sizes), each of which is separated by at least 1 row/column of black. All the quadrants contain random pixels, and all quadrants except for one have the same color
# To make the output, find the quadrant with a different color, and copy only that quadrant to the output, producing a smaller grid

def transform_0b148d64(input_grid: np.ndarray) -> np.ndarray:
    # break the input up into quadrants
    # remember they are different sizes, but they are all separated by at least 2 rows/columns of black
    # we do this by computing the x, y coordinates of separators
    for i in range(input_grid.shape[0]):
        if np.all(input_grid[i, :] == Color.BLACK):
            x_separator = i
            break
    for i in range(input_grid.shape[1]):
        if np.all(input_grid[:, i] == Color.BLACK):
            y_separator = i
            break
    
    quadrants = [ input_grid[:x_separator, :y_separator],
                  input_grid[:x_separator, y_separator:],
                  input_grid[x_separator:, :y_separator],
                  input_grid[x_separator:, y_separator:] ]
    
    # check that each of them is monochromatic (only one color that isn't black)
    colors = [ np.unique(quadrant[quadrant != Color.BLACK]) for quadrant in quadrants ]
    for color in colors:
        assert len(color) == 1, "Quadrant has more than one color"

    for color, quadrant in zip(colors, quadrants):
        color_frequency = sum(other_color == color for other_color in colors)
        if color_frequency == 1:
            output_grid = quadrant

            # we have to crop the output grid to remove any extraneous black rows/columns
            output_grid = crop(output_grid, background=Color.BLACK)

            break

    return output_grid

""" ==============================
Puzzle 0ca9ddb6

Train example 1:
Input1 = [
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k r k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k b k k
 k k k k k k k k k
 k k k k k k k k k
]
Output1 = [
 k k k k k k k k k
 k k k k k k k k k
 k y k y k k k k k
 k k r k k k k k k
 k y k y k k k k k
 k k k k k k o k k
 k k k k k o b o k
 k k k k k k o k k
 k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k t k k k k k
 k k k k k k k k k
 k k k k k k r k k
 k k b k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k b k k
 k r k k k k k k k
 k k k k k k k k k
]
Output2 = [
 k k k t k k k k k
 k k k k k y k y k
 k k o k k k r k k
 k o b o k y k y k
 k k o k k k k k k
 k k k k k k o k k
 y k y k k o b o k
 k r k k k k o k k
 y k y k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k
 k k k k k k k k k
 k k r k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k p k k
 k k k k k k k k k
 k k k b k k k k k
 k k k k k k k k k
]
Output3 = [
 k k k k k k k k k
 k y k y k k k k k
 k k r k k k k k k
 k y k y k k k k k
 k k k k k k k k k
 k k k k k k p k k
 k k k o k k k k k
 k k o b o k k k k
 k k k o k k k k k
] """

# concepts:
# pixel manipulation

# description:
# In the input you will see a medium sized grid width individual colored pixels, some of which are red or blue (those ones are special)
# To make the output:
# 1. For each red pixel, add yellow pixels in its immediate diagonals (northeast, northwest, southeast, southwest)
# 2. For each blue pixel, add orange pixels in its immediate neighbors (up, down, left, right)

def transform_0ca9ddb6(input_grid: np.ndarray) -> np.ndarray:

    output_grid = np.copy(input_grid)

    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            color = input_grid[x][y]
            if color == Color.RED:
                # put yellow pixels in the diagonals
                for dx in [-1, 1]:
                    for dy in [-1, 1]:
                        if 0 <= x+dx < input_grid.shape[0] and 0 <= y+dy < input_grid.shape[1]:
                            output_grid[x+dx, y+dy] = Color.YELLOW
            elif color == Color.BLUE:
                # put orange pixels in the neighbors
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    if 0 <= x+dx < input_grid.shape[0] and 0 <= y+dy < input_grid.shape[1]:
                        output_grid[x+dx, y+dy] = Color.ORANGE

    return output_grid

""" ==============================
Puzzle 0d3d703e

Train example 1:
Input1 = [
 g b r
 g b r
 g b r
]
Output1 = [
 y e p
 y e p
 y e p
]

Train example 2:
Input2 = [
 r g t
 r g t
 r g t
]
Output2 = [
 p y m
 p y m
 p y m
]

Train example 3:
Input3 = [
 e t p
 e t p
 e t p
]
Output3 = [
 b m r
 b m r
 b m r
] """

# concepts:
# color mapping

# description:
# The input is a grid where each column is of the same color. 
# To make the output, change each color according to the following mapping:
# green -> yellow, blue -> gray, red -> pink, teal -> maroon, yellow -> green, gray -> blue, pink -> red, maroon -> teal

def transform_0d3d703e(input_grid):
    # Initialize output grid
    output_grid = input_grid.copy()

    # Performs color mapping
    output_grid = np.vectorize(lambda color: color_map.get(color, color))(output_grid)

    return output_grid
    
# Constructing the color map
color_map = {Color.GREEN : Color.YELLOW, 
             Color.BLUE : Color.GRAY, 
             Color.RED : Color.PINK,
             Color.TEAL : Color.MAROON,
             Color.YELLOW : Color.GREEN, 
             Color.GRAY : Color.BLUE, 
             Color.PINK : Color.RED,
             Color.MAROON : Color.TEAL             
            }

""" ==============================
Puzzle 0dfd9992 """

# concepts:
# occlusion, translational symmetry

# description:
# In the input you will see a translationally symmetric pattern randomly occluded by black pixels.
# To make the output, remove the occluding black pixels to reveal the translationally symmetric pattern.

def transform_0dfd9992(input_grid):
    # Plan:
    # 1. Find the translational symmetries
    # 2. Reconstruct the sprite by ignoring the black pixels and exploiting the symmetry

    w, h = input_grid.shape

    # Identify the translational symmetries. Note that there is no background color for this problem.
    translations = detect_translational_symmetry(input_grid, ignore_colors=[Color.BLACK], background=None)
    assert len(translations) > 0, "No translational symmetry found"

    # Reconstruct the occluded black pixels by replacing them with colors found in the orbit of the symmetries
    output_grid = np.copy(input_grid)
    for x in range(w):
        for y in range(h):
            if output_grid[x, y] == Color.BLACK:
                # Use the translational symmetry to fill in the occluded pixels
                # to do this we compute the ORBIT of the current pixel under the translations
                # and take the most common non-black color in the orbit

                # Compute the orbit into the output
                orbit_pixels = orbit(output_grid, x, y, translations)
                orbit_colors = {input_grid[transformed_x, transformed_y]
                                for transformed_x, transformed_y in orbit_pixels}
                
                # occluded by black, so whatever color it is, black doesn't count
                orbit_colors = orbit_colors - {Color.BLACK}

                # Copy the color
                assert len(orbit_colors) == 1, "Ambiguity: multiple colors in the orbit"
                output_grid[x, y] = orbit_colors.pop()
    
    return output_grid

""" ==============================
Puzzle 0e206a2e

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k k k k
 k k k t k k k k k k k k k k k k k k
 k k g t b k k k k k k k k k k y k k
 k k t y t k k k k k k k k k k k k k
 k k k k k k k k k k k k k g k k k b
 k k k k k k k k k k k k k k k k k k
 k k k k k k k g k k k k k k k k k k
 k k k k k k k t k t k k k k k k k k
 k k k k k k k t t y k k k k k k k k
 k k b k k k k t k t k k k k k k k k
 k k k y k k k b k k k k k k k k k k
 k k g k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k t y t k
 k k k k k k k k k k k k k k k t k k
 k k k k k k k k k k k k k g t t t b
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k b t k k k k k k k k k k k k k k
 k t t y k k k k k k k k k k k k k k
 k k g t k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k r k k k k k k k k k k
 k k k y g g k k k k k k k k k
 k k k k g k k k k k k k k k k
 k k k k g k k k k k k k k k k
 k k k k g k k k k k k k k k k
 k k k g b g k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k y k
 k k k k k k k k k b k k k k r
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k g k k k y k
 k k k k k k k k k b g g g g r
 k k k k k k k k k g k k k g k
 k k k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k y k k k k
 k k k k k t k k k t k k k k
 k k k k k b t t t r t k k k
 k k k k k k k k k t k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k b k k k r k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k y k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k t k k k k k k k k
 k b t t t r t k k k k k k k
 k t k k k t k k k k k k k k
 k k k k k y k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
] """

# concepts:
# symmetries, objects

# description:
# In the input you will see one or two medium-sized multicolor objects, and some multicolor pixels sprinkled around in one or two clusters
# To make the output, take each of the medium-sized object and move it so that it perfectly covers some of the pixels sprinkled around, matching colors wherever they overlap.
# You can rotate and flip the objects in order to make them fit the pixels, but they have to match colors wherever they overlap.

def transform_0e206a2e(input_grid):
    # Plan:
    # 1. Break the input into 1-2 big object(s) and a mask containing the remaining pixels
    # 2. For each object, create rotations and flips of it
    # 3. For each object, find the best way to place one of its rotations/flips so that it covers the most number of the pixels (remember the pixel has to match object color)

    # Extract the objects from the input and categorize them into objects and pixels
    # Objects can be multicolored
    connected_components = find_connected_components(input_grid, monochromatic=False)
    objects = [ cc for cc in connected_components if np.count_nonzero(cc != Color.BLACK) > 4 ]
    pixels = [ cc for cc in connected_components if np.count_nonzero(cc != Color.BLACK) <= 4 ]

    # Make the pixel mask, which shows where the pixels are. These guide the placement of objects.
    pixel_mask = np.full(input_grid.shape, Color.BLACK)
    for pixel_object in pixels:
        blit_object(pixel_mask, pixel_object, background=Color.BLACK)
    
    output_grid = np.full(input_grid.shape, Color.BLACK)
    
    # For each object, find the best way to place it so that it covers the most number of the pixels
    for obj in objects:
        # The object can be rotated and flipped to match the pixels
        # First, convert it to a sprite before transforming, because these operations are independent of position
        sprite = crop(obj)
        sprite_variations = [sprite, np.rot90(sprite), np.rot90(sprite, 2), np.rot90(sprite, 3), np.flipud(sprite), np.fliplr(sprite), np.flipud(np.rot90(sprite)), np.fliplr(np.rot90(sprite))]

        # We are going to optimize the position and variation, so we need to keep track of the best placement so far
        best_output, best_pixels_covered = None, 0
        for x, y, sprite_variation in [(x, y, variant) for x in range(input_grid.shape[0]) for y in range(input_grid.shape[1]) for variant in sprite_variations]:
            test_grid = np.copy(output_grid)
            blit_sprite(test_grid, sprite_variation, x, y, background=Color.BLACK)
            # Check if there was any color mismatch: A colored pixel in the mask which is different from what we just made
            # If there is a mismatch, we can't place the object here
            if np.any((pixel_mask != Color.BLACK) & (test_grid != Color.BLACK) & (pixel_mask != test_grid)):
                continue
            num_covered_pixels = np.count_nonzero((pixel_mask != Color.BLACK) & (test_grid != Color.BLACK))
            if num_covered_pixels > best_pixels_covered:
                best_output, best_pixels_covered = test_grid, num_covered_pixels
        output_grid = best_output

    return output_grid

""" ==============================
Puzzle 10fcaaa3

Train example 1:
Input1 = [
 k k k k
 k e k k
]
Output1 = [
 t k t k t k t k
 k e k k k e k k
 t k t k t k t k
 k e k k k e k k
]

Train example 2:
Input2 = [
 k k p k
 k k k k
 k p k k
]
Output2 = [
 k k p k k k p k
 t t t t t t t t
 k p k t k p k t
 t k p k t k p k
 t t t t t t t t
 k p k k k p k k
]

Train example 3:
Input3 = [
 k k k
 k y k
 k k k
 k k k
 y k k
]
Output3 = [
 t k t t k t
 k y k k y k
 t k t t k t
 k t t k t k
 y k k y k k
 t t t t t t
 k y k k y k
 t k t t k t
 k t t k t k
 y k k y k k
] """

# concepts:
# Coloring diagonal pixels, repetition

# description:
# Given an input grid of arbitrary size, with some small number of colored pixels on it.
# To produce the output, replicate the input grid 4 times, 2 on the top and 2 on the bottom. 
# Color all the diagonal pixels adjacent to a colored pixel teal if the diagonal pixels are black. 

def transform_10fcaaa3(input_grid):
   # Replicate input grid 4 times to initialize output grid
   output_grid = np.zeros((2*input_grid.shape[0], 2* input_grid.shape[1]),dtype=int)
   for i in range(2):
      for j in range(2):
         blit_sprite(output_grid, input_grid, i*input_grid.shape[0], j*input_grid.shape[1])
  
   # Create diagonal directions
   diagonal_dx_dy = [(1,1),(-1,1),(1,-1),(-1,-1)]

   # Color diagonal pixels 
   for y in range(output_grid.shape[1]):
      for x in range(output_grid.shape[0]):
         if output_grid[x,y] != Color.BLACK and output_grid[x,y] != Color.TEAL:
            for dx,dy in diagonal_dx_dy:
               # Color diagonal pixel teal if it is black
               if x+dx >= 0 and x+dx < output_grid.shape[0] and y+dy >= 0 and y+dy < output_grid.shape[1] and output_grid[x+dx,y+dy] == Color.BLACK:
                  output_grid[x+dx,y+dy] = Color.TEAL
  
   return output_grid

""" ==============================
Puzzle 11852cab

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k g k t k k k k k
 k k k r k r k k k k
 k k t k g k t k k k
 k k k r k r k k k k
 k k k k t k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k g k t k g k k k
 k k k r k r k k k k
 k k t k g k t k k k
 k k k r k r k k k k
 k k g k t k g k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k r k g k k k k k
 k k k y k y k k k k
 k k g k y k g k k k
 k k k y k y k k k k
 k k k k g k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k r k g k r k k k
 k k k y k y k k k k
 k k g k y k g k k k
 k k k y k y k k k k
 k k r k g k r k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k k t k t k t k k
 k k k k y k k k k k
 k k k t k b k t k k
 k k k k k k k k k k
 k k k t k t k t k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k
 k k k t k t k t k k
 k k k k y k y k k k
 k k k t k b k t k k
 k k k k y k y k k k
 k k k t k t k t k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
] """

# concepts:
# symmetry detection, occlusion

# description:
# In the input you will see an object that is almost rotationally symmetric, except that some of it has been removed (covered in black pixels)
# To make the output fill in the missing parts of the object to make it rotationally symmetric


def transform_11852cab(input_grid):
    # Plan:
    # 1. Find the center of rotation
    # 2. Rotate each colored pixel (4 times, rotating around the center of rotation) and fill in any missing pixels
    output_grid = input_grid.copy()

    # Find the rotational symmetry
    sym = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK])

    # Find the colored pixels
    colored_pixels = np.argwhere(input_grid != Color.BLACK)

    # Do the rotations and fill in the missing colors
    for x, y in colored_pixels:
        # Get the color, which is going to be copied to the rotated positions
        color = input_grid[x, y]
        
        # Loop over all rotations, going 90 degrees each time (so four times)
        for i in range(1, 4):
            # Calculate rotated coordinate
            rotated_x, rotated_y = sym.apply(x, y, iters=i)

            # Fill in the missing pixel
            if output_grid[rotated_x, rotated_y] == Color.BLACK:
                output_grid[rotated_x, rotated_y] = color
            else:
                assert output_grid[rotated_x, rotated_y] == color, "The object is not rotationally symmetric"

    return output_grid

""" ==============================
Puzzle 1190e5a7

Train example 1:
Input1 = [
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 o o o o o o o o o o o o o o o
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
 g o g g g g g g g g o g g o g
]
Output1 = [
 g g g g
 g g g g
]

Train example 2:
Input2 = [
 b b b b t b b b b b b
 b b b b t b b b b b b
 b b b b t b b b b b b
 t t t t t t t t t t t
 b b b b t b b b b b b
 b b b b t b b b b b b
 b b b b t b b b b b b
 b b b b t b b b b b b
 b b b b t b b b b b b
 t t t t t t t t t t t
 b b b b t b b b b b b
]
Output2 = [
 b b
 b b
 b b
]

Train example 3:
Input3 = [
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 b b b b b b b b b b b b b b b b b b b b b b b b b b b
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 b b b b b b b b b b b b b b b b b b b b b b b b b b b
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 b b b b b b b b b b b b b b b b b b b b b b b b b b b
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 b b b b b b b b b b b b b b b b b b b b b b b b b b b
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 b b b b b b b b b b b b b b b b b b b b b b b b b b b
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
 g g g g g g b g g g g g g g g g g g g g g b g b g b g
]
Output3 = [
 g g g g g
 g g g g g
 g g g g g
 g g g g g
 g g g g g
 g g g g g
] """

# concepts:
# horizontal/vertical bars, counting

# description:
# In the input, you will see horizontal and vertical bars, dividing the input into a grid of rectangular regions, on a non-black background.
# To make the output produce a monochromatic image whose width is the number of background-colored regions going left-to-right, and whose height is the number of regions going top-to-bottom.
# The output should have the same background.


def transform_1190e5a7(input_grid):
    # Plan:
    # 1. Find the color of the background and the bars
    # 2. Count the number of regions going left-to-right
    # 3. Count the number of regions going top-to-bottom

    # Find bar color and background color
    for x in range(input_grid.shape[0]):
        bar_slice = input_grid[x, :]
        if np.all(bar_slice == bar_slice[0]):
            bar_color = bar_slice[0]
            break

    # background is whatever color isn't the bar color
    background = [ color for color in input_grid.flatten() if color != bar_color ][0]

    # Count the number of regions going left-to-right
    n_horizontal = 1
    for x in range(input_grid.shape[0]):
        if input_grid[x, 0] == bar_color:
            n_horizontal += 1
    
    # Count the number of regions going top-to-bottom
    n_vertical = 1
    for y in range(input_grid.shape[1]):
        if input_grid[0, y] == bar_color:
            n_vertical += 1
    
    # Create output grid
    output_grid = np.full((n_horizontal, n_vertical), background)

    return output_grid

""" ==============================
Puzzle 137eaa0f

Train example 1:
Input1 = [
 k k k k k k k k k k k
 k k k k k k p p k k k
 k k k e k k k e k k k
 k k y y k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k o k k k k
 k k k k k e o k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
]
Output1 = [
 p p o
 k e o
 y y k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k
 k k k k k k k k k k k
 k p k k k k k k k k k
 k k e k o e o k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k e k k k
 k k r r k k g g g k k
 k k e k k k k k k k k
 k k k k k k k k k k k
]
Output2 = [
 p r r
 o e o
 g g g
]

Train example 3:
Input3 = [
 k k k k k k k k k k k
 k k k k k k k k k k k
 k b b k k k k k k k k
 b e k k k k k k k k k
 k k k k k e r k k k k
 k k k k k k r k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k e k k k
 k k k k k k m m k k k
 k k k k k k k k k k k
]
Output3 = [
 k b b
 b e r
 m m r
] """

# concepts:
# objects, alignment by color

# description:
# In the input you will see some objects scattered around on a black grid. Each object has a single grey pixel, but everything else is a single other color.
# To make the output, place each object into the output grid such that the grey pixel is in the center of the output.
# Equivalently, move the objects to line up all their grey pixels so they overlap.
# The output grid should be the smallest possible size that contains all the objects (after they have been placed correctly), which for all the inputs here is 3x3.

def transform_137eaa0f(input_grid):
    # Plan:
    # 1. Extract the objects from the input, convert them into sprites by cropping them
    # 2. Make a big output grid
    # 3. Place each sprite into the output grid such that the grey pixel is in the center of the output
    # 4. Make the output as small as you can to contain all the objects

    # Extract the objects from the input. It is not monochromatic because the grey pixel is different, and they can be connected on the diagonals (connectivity=8)
    objects = find_connected_components(input_grid, monochromatic=False, connectivity=8)

    # Convert the objects into sprites by cropping them
    sprites = [crop(obj, background=Color.BLACK) for obj in objects]

    # Make a big output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Place each sprite into the output grid such that the grey pixel is in the center of the output
    for sprite in sprites:
        # Find the grey pixel
        grey_pixel_x, grey_pixel_y = np.argwhere(sprite == Color.GREY)[0]

        # Find the center of the output. We want the grey pixel to end up here.
        center_x, center_y = output_grid.shape[0] // 2, output_grid.shape[1] // 2

        # Calculate the offset to ensure the grey pixel ends up in the center of the output
        x, y = center_x - grey_pixel_x, center_y - grey_pixel_y
        
        # Place the sprite into the output grid
        blit_sprite(output_grid, sprite, x, y, background=Color.BLACK)

    # Make the output as small as you can to contain all the objects
    output_grid = crop(output_grid)

    return output_grid

""" ==============================
Puzzle 150deff5

Train example 1:
Input1 = [
 k k k k k k k k k k k
 k k e e k k k k k k k
 k k e e e e e k k k k
 k k k e e e k k k k k
 k k k e e e e e k k k
 k k k e k k e e k k k
 k k k k k e e e k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k
 k k t t k k k k k k k
 k k t t r r r k k k k
 k k k r t t k k k k k
 k k k r t t t t k k k
 k k k r k k t t k k k
 k k k k k r r r k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k e e e e e e k k k
 k e e e e e e k k k
 k k k e k k e k k k
 k k k k e e e k k k
 k k k k e e e k k k
 k k k k e k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k t t r t t r k k k
 k t t r t t r k k k
 k k k r k k r k k k
 k k k k r t t k k k
 k k k k r t t k k k
 k k k k r k k k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k
 k e e e e e k k k
 k k k k e e k k k
 k k k e k k k k k
 k k k e e e k k k
 k k k e e e k k k
 k k k k k k k k k
 k k k k k k k k k
]
Output3 = [
 k k k k k k k k k
 k r r r t t k k k
 k k k k t t k k k
 k k k r k k k k k
 k k k r t t k k k
 k k k r t t k k k
 k k k k k k k k k
 k k k k k k k k k
] """

# concepts:
# decomposition, color change

# description:
# In the input you will see grey-colored regions on a medium sized black canvas. These regions are comprised of 2x2 squares and 1x3/3x1 rectangles, but this might be hard to see because regions might be touching.
# To make the output, decompose the input into 2x2 squares and 1x3/3x1 rectangles, and color them as follows:
# 1. Color teal the 2x2 squares
# 2. Color red the 1x3/3x1 rectangles

def transform_150deff5(input_grid: np.ndarray) -> np.ndarray:

    # Decompose the grid into non-overlapping grey regions
    # We need a custom predicate because otherwise the regions are also allowed to include background pixels, and we want it all grey
    decomposition = detect_objects(input_grid, background=Color.BLACK, colors=[Color.GREY],
                                   allowed_dimensions=[(2, 2), (3, 1), (1, 3)], # 2x2 squares and 1x3/3x1 rectangles
                                   predicate=lambda sprite: np.all(sprite == Color.GREY))

    output_grid = np.full(input_grid.shape, Color.BLACK)

    for obj in decomposition:
        x, y, w, h = bounding_box(obj, background=Color.BLACK)
        sprite = crop(obj, background=Color.BLACK)

        # Color change based on dimensions: 2x2 -> teal, 1x3/3x1 -> red        
        if w == 2 and h == 2:
            sprite[sprite == Color.GREY] = Color.TEAL
        elif (w == 3 and h == 1) or (w == 1 and h == 3):
            sprite[sprite == Color.GREY] = Color.RED
        else:
            assert 0, "Invalid object found"
        
        # Copy the sprite back into the output grid
        blit_sprite(output_grid, sprite, x, y, background=Color.BLACK)

    return output_grid

""" ==============================
Puzzle 178fcbfb

Train example 1:
Input1 = [
 k k k k k k k k k
 k k k k k k k k k
 k k r k k k k k k
 k k k k k k k k k
 k k k k k k k g k
 k k k k k k k k k
 k k k b k k k k k
 k k k k k k k k k
 k k k k k k k k k
]
Output1 = [
 k k r k k k k k k
 k k r k k k k k k
 k k r k k k k k k
 k k r k k k k k k
 g g g g g g g g g
 k k r k k k k k k
 b b b b b b b b b
 k k r k k k k k k
 k k r k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k
 k g k k k k k k
 k k k k k k k k
 k k k k k k k k
 k k k g k k k k
 k k k k k k k k
 k b k k k k k k
 k k k k k r k k
 k k k k k k k k
 k k k k k k k k
]
Output2 = [
 k k k k k r k k
 g g g g g g g g
 k k k k k r k k
 k k k k k r k k
 g g g g g g g g
 k k k k k r k k
 b b b b b b b b
 k k k k k r k k
 k k k k k r k k
 k k k k k r k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k
 k b k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k g k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k g k k k k k k k k
 k k k k k k k k k k k
 k k k r k k k k k k k
 k k k k k k k k k r k
]
Output3 = [
 k k k r k k k k k r k
 b b b b b b b b b b b
 k k k r k k k k k r k
 g g g g g g g g g g g
 k k k r k k k k k r k
 k k k r k k k k k r k
 g g g g g g g g g g g
 k k k r k k k k k r k
 k k k r k k k k k r k
 k k k r k k k k k r k
] """

# concepts:
# vertical lines, growing

# description:
# In the input you will see individual pixels sprinkled on a black background that are either red, green, or blue
# Turn each red pixel into a vertical bar, and each green or blue pixel into a horizontal bar

def transform_178fcbfb(input_grid):

    # extract the pixels of each color
    red_pixels = (input_grid == Color.RED)
    green_pixels = (input_grid == Color.GREEN)
    blue_pixels = (input_grid == Color.BLUE)

    # prepare a blank output grid, because we don't need to reuse anything from the input (we're not drawing on top of the input)
    output_grid = np.zeros_like(input_grid)

    # turn red pixels into vertical bars
    red_locations = np.argwhere(red_pixels)
    for x, y in red_locations:
        # vertical means the same X value
        output_grid[x, :] = Color.RED
    
    # turn green and blue pixels into horizontal bars
    green_locations = np.argwhere(green_pixels)
    blue_locations = np.argwhere(blue_pixels)
    for x, y in green_locations:
        # horizontal means the same Y value
        output_grid[:, y] = Color.GREEN
    for x, y in blue_locations:
        # horizontal means the same Y value
        output_grid[:, y] = Color.BLUE
    
    return output_grid

""" ==============================
Puzzle 1a07d186

Train example 1:
Input1 = [
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k y k g k k g k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k y k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k r k k k k y k k k k k k
 k k k g k k k k k k k k y k k k g k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
]
Output1 = [
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g g k k k k k k y y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k y y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g g k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
 k k k g k k k k k k k k y k k k k k k
]

Train example 2:
Input2 = [
 k k k r k k k k k k k k k k
 k k k k k k k k k y k k k k
 k k k k k k k k k k k k k k
 r r r r r r r r r r r r r r
 k k k k k k k k k k k k k k
 k k k k k k k k k k b k k k
 k k k k k k k k k k k k k k
 k k k k k k r k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 b b b b b b b b b b b b b b
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k b k k k k k k k r k k k
 k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k r k k k k k k k k k k
 r r r r r r r r r r r r r r
 k k k k k k r k k k r k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k b k k k
 b b b b b b b b b b b b b b
 k k b k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k k
 k k k b k k k t k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 t t t t t t t t t t t t t t t t
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k t k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k t k k k k k k k k k k k k
 k k k k k k k k k k k k b k k k
 k k k k k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k t k k k k k k k k
 t t t t t t t t t t t t t t t t
 k k k t k k k k k k k t k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
] """

# concepts:
# collision detection, sliding objects, horizontal/vertical bars

# description:
# In the input you will see horizontal/vertical bars and individual coloured pixels sprinkled on a black background
# Move each colored pixel to the bar that has the same colour until the pixel touches the bar.
# If a colored pixel doesn't have a corresponding bar, it should be deleted.

def transform_1a07d186(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.zeros_like(input_grid)

    # each object is either a bar or pixel, all uniform color
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # separate the bars from the pixels
    bars, pixels = [], []
    for obj in objects:
        w, h = crop(obj).shape
        if w == input_grid.shape[0] or h == input_grid.shape[1]:
            bars.append(obj)
        else:
            pixels.append(obj)
    
    # copy the bars to the output grid
    for bar in bars:
        blit_object(output_grid, bar, background=Color.BLACK)
    
    # slide each pixel until it just barely touches the bar with the matching color
    for pixel in pixels:
        color = np.unique(pixel)[1]
        matching_bars = [bar for bar in bars if np.unique(bar)[1] == color]

        # if there is no matching bar, delete the pixel
        if len(matching_bars) == 0:
            continue

        # consider sliding in the 4 cardinal directions, and consider sliding as far as possible
        possible_displacements = [ (slide_distance*dx, slide_distance*dy)
                                   for slide_distance in range(max(input_grid.shape))
                                   for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] ]
        for dx, dy in possible_displacements:
            new_pixel = translate(pixel, dx, dy, background=Color.BLACK)
            if contact(object1=matching_bars[0], object2=new_pixel):
                blit_object(output_grid, new_pixel, background=Color.BLACK)
                break
    
    return output_grid

""" ==============================
Puzzle 1b2d62fb

Train example 1:
Input1 = [
 k m m b m m m
 k k m b m m k
 m k m b m m k
 k k k b m k k
 k m m b m m m
]
Output1 = [
 k k k
 k k k
 k k k
 k t t
 k k k
]

Train example 2:
Input2 = [
 k k k b m k k
 m k m b m m m
 k m m b m m m
 k k k b m m m
 k m m b m m m
]
Output2 = [
 k t t
 k k k
 k k k
 k k k
 k k k
]

Train example 3:
Input3 = [
 m k k b m k m
 m k k b k m k
 m k k b m k k
 k m m b k m m
 k k m b k m k
]
Output3 = [
 k t k
 k k t
 k t t
 t k k
 t k k
] """

# concepts:
# boolean logical operations, bitmasks with separator

# description:
# In the input you will see two maroon bitmasks separated by a blue vertical bar
# To make the output, color teal the pixels that are not set in either bitmasks (logical NOR)

def transform_1b2d62fb(input_grid: np.ndarray) -> np.ndarray:
    # Find the blue vertical bar. Vertical means constant X
    for x_bar in range(input_grid.shape[0]):
        if np.all(input_grid[x_bar, :] == Color.BLUE):
            break

    left_mask = input_grid[:x_bar, :]
    right_mask = input_grid[x_bar+1:, :]

    output_grid = np.zeros_like(left_mask)
    output_grid[(left_mask != Color.MAROON) & (right_mask != Color.MAROON)] = Color.TEAL
    
    return output_grid

""" ==============================
Puzzle 1b60fb0c

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k b b b k k k k
 k k k k b b k k k k
 k k k k b k k k b k
 k k k k b b b b b k
 k k k k b b k b b k
 k k k k k b k k k k
 k k k k b b k k k k
 k k k k b b b k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k b b b k k k k
 k k k k b b k k k k
 k k k k b k k k b k
 k r r k b b b b b k
 k r r r b b k b b k
 k r k k k b k k k k
 k k k k b b k k k k
 k k k k b b b k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k b b b b b k k
 k k k b b b b b k k
 k k k k k b k k b b
 k k k k k b k k b b
 k k k k k b b b b b
 k k k k k b k k b b
 k k k k k b k k b b
 k k k b b b b b k k
 k k k b b b b b k k
]
Output2 = [
 k k k k k k k k k k
 k k k b b b b b k k
 k k k b b b b b k k
 k r r k k b k k b b
 k r r k k b k k b b
 k r r r r b b b b b
 k r r k k b k k b b
 k r r k k b k k b b
 k k k b b b b b k k
 k k k b b b b b k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k k b b b b b k k
 k k k k k b k k k k
 k k k k b b b k k b
 k k k k k b k b k b
 k k k k k b b b b b
 k k k k k b k b k b
 k k k k b b b k k b
 k k k k k b k k k k
 k k k b b b b b k k
]
Output3 = [
 k k k k k k k k k k
 k k k b b b b b k k
 k k k k k b k k k k
 k r k k b b b k k b
 k r k r k b k b k b
 k r r r r b b b b b
 k r k r k b k b k b
 k r k k b b b k k b
 k k k k k b k k k k
 k k k b b b b b k k
] """

# concepts:
# symmetry

# description:
# In the input you will see an image containing blue pixels that is almost rotationally symmetric, except that it is missing the section either north, south, east, or west that would make it rotationally symmetric
# Color red all the pixels that would need to be colored in order to make the image rotationally symmetric (when rotating clockwise)

def transform_1b60fb0c(input_grid):

    # The goal is to make the object rotationally symmetric, *not* to make the whole grid rotationally symmetric
    # We have to extract the object from the grid and then rotate it to construct the missing section
    blue_sprite = crop(input_grid)
    rotated_blue_sprite = np.rot90(blue_sprite)
    
    # We need to find the optimal location for placing the rotated sprite
    # This will make the resulting object radially symmetric
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            
            test_grid = np.copy(input_grid)
            blit_sprite(test_grid, rotated_blue_sprite, x, y, background=Color.BLACK)
            test_blue_sprite = crop(test_grid)

            # Check if the resulting object is radially symmetric
            if np.array_equal(test_blue_sprite, np.rot90(test_blue_sprite)):
                # Save what the input would look like if it were perfectly symmetric
                perfectly_symmetric_grid = test_grid
                break

    # The missing section is the part of the input grid that would have been blue if it were perfectly symmetric
    missing_pixels = np.where((input_grid == Color.BLACK) & (perfectly_symmetric_grid == Color.BLUE))

    # Color the missing section red
    output_grid = np.copy(input_grid)
    output_grid[missing_pixels] = Color.RED

    return output_grid

def transform_1b60fb0c(input_grid):
    # This also works, and uses the library function `detect_rotational_symmetry``
    
    # Plan:
    # 1. Detect the (x,y) point that the object is rotated around
    # 2. Rotate each blue colored pixel around that point. If the rotated pixel is not colored, color it red.

    output_grid = np.copy(input_grid)

    # Find the symmetry
    sym = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK], background=Color.BLACK)
    
    # Rotate the blues and color red as needed
    blues = np.argwhere(input_grid == Color.BLUE)
    for x, y in blues:
        rotated_x, rotated_y = sym.apply(x, y, iters=1)        

        if input_grid[rotated_x, rotated_y] == Color.BLACK:
            output_grid[rotated_x, rotated_y] = Color.RED
    
    return output_grid

""" ==============================
Puzzle 1bfc4729

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k p k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k o k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 p p p p p p p p p p
 p k k k k k k k k p
 p p p p p p p p p p
 p k k k k k k k k p
 p k k k k k k k k p
 o k k k k k k k k o
 o k k k k k k k k o
 o o o o o o o o o o
 o k k k k k k k k o
 o o o o o o o o o o
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k b k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k y k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 b b b b b b b b b b
 b k k k k k k k k b
 b b b b b b b b b b
 b k k k k k k k k b
 b k k k k k k k k b
 y k k k k k k k k y
 y k k k k k k k k y
 y y y y y y y y y y
 y k k k k k k k k y
 y y y y y y y y y y
] """

# concepts:
# pattern generation

# description:
# In the input you will see a grid with two colored pixels.
# To make the output, you should draw a pattern from the two pixels.
# step 1: draw a horizontal line outward from the two pixels in their respective colours.
# step 2: draw a border around the canvas whose top half is colored to match the top pixel and whose bottom half matches the bottom pixel.


def transform_1bfc4729(input_grid):
    # Plan:
    # 1. Parse the input and determine which pixel is top in which is bottom
    # 2. Draw horizontal lines
    # 3. Draw border, colored appropriately

    # 1. Input parsing
    # Detect the two pixels, sorting by Y coordinate so top one is first, bottom one is second
    background = Color.BLACK
    pixels = find_connected_components(input_grid, monochromatic=True, background=background)
    pixels.sort(key=lambda obj: object_position(obj, background=background)[1])

    # 2. Draw a horizontal lines outward pixels
    output_grid = np.full_like(input_grid, background)
    for pixel in pixels:
        x, y = object_position(pixel, background=background)
        color = object_colors(pixel, background=background)[0]
        draw_line(output_grid, x=0, y=y, direction=(1, 0), color=color)
    
    # 3. Make a border colored appropriately
    top_color = object_colors(pixels[0], background=background)[0]
    bottom_color = object_colors(pixels[1], background=background)[0]
    draw_line(output_grid, x=0, y=0, direction=(1, 0), color=top_color)
    draw_line(output_grid, x=0, y=0, direction=(0, 1), color=top_color)
    draw_line(output_grid, x=0, y=output_grid.shape[1] - 1, direction=(1, 0), color=bottom_color)
    draw_line(output_grid, x=output_grid.shape[0] - 1, y=0, direction=(0, 1), color=bottom_color)
    # Everything below the midline is bottom color, everything above is top color
    # Recolor to enforce this
    width, height = output_grid.shape
    top = output_grid[:, :height//2]
    top[top!=background] = top_color
    bottom = output_grid[:, height//2:]
    bottom[bottom!=background] = bottom_color

    return output_grid

""" ==============================
Puzzle 1caeab9d

Train example 1:
Input1 = [
 k r r k k k k k k k
 k r r k k k k b b k
 k k k k y y k b b k
 k k k k y y k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k r r k y y k b b k
 k r r k y y k b b k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k y y y
 k k k k k k k y y y
 k r r r k k k k k k
 k r r r k k k k k k
 k k k k k k k k k k
 k k k k b b b k k k
 k k k k b b b k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k r r r b b b y y y
 k r r r b b b y y y
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k k r k k k k k k
 k b k r k k k k k k
 k b k k k k y k k k
 k k k k k k y k k k
]
Output3 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k b k r k k y k k k
 k b k r k k y k k k
 k k k k k k k k k k
] """

# concepts:
# alignment, objects

# description:
# In the input you will see a red, blue, and yellow shape. Each are the same shape (but different color). They occur left to right in the input grid on a black background, but at different vertical heights.
# The output is the same as the input, but with the vertical heights of the red and yellow shapes adjusted to match the height of the blue shape.

def transform_1caeab9d(input_grid):
    # find the blue shape, red shape, and yellow shape
    blue_coords = np.where(input_grid == Color.BLUE)
    red_coords = np.where(input_grid == Color.RED)
    yellow_coords = np.where(input_grid == Color.YELLOW)

    # set the vertical height of the red and yellow shape to match
    red_coords = (red_coords[0], blue_coords[1])
    yellow_coords = (yellow_coords[0], blue_coords[1])

    # make output grid with the colored shapes at their new locations
    output_grid = np.full_like(input_grid, Color.BLACK)
    output_grid[blue_coords] = Color.BLUE
    output_grid[red_coords] = Color.RED
    output_grid[yellow_coords] = Color.YELLOW

    return output_grid

""" ==============================
Puzzle 1cf80156

Train example 1:
Input1 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k r r r k k k k k
 k k k k k r k k k k k k
 k k k r r r k k k k k k
 k k k r k r k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]
Output1 = [
 k r r r
 k k r k
 r r r k
 r k r k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k
 k k b k k k k k k k k k
 k k b b k k k k k k k k
 k k k b k k k k k k k k
 k k b b b k k k k k k k
 k k k k b k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]
Output2 = [
 b k k
 b b k
 k b k
 b b b
 k k b
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k t k t k k k k k
 k k k t t t t k k k k k
 k k k k k k t t k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]
Output3 = [
 k t k t k
 t t t t k
 k k k t t
] """

# concepts:
# cropping

# description:
# In the input you will see a single colored shape, around 4x6 in size, floating in a 12x12 grid of black.
# To make the output, crop the background out of the image - so the output grid has the same dimensions as the shape.

def transform_1cf80156(input_grid):
    return crop(input_grid, background=Color.BLACK)

""" ==============================
Puzzle 1e32b0e9

Train example 1:
Input1 = [
 k k k k k t k k k k k t k k k k k
 k k r k k t k k k k k t k k k k k
 k r r r k t k k k k k t k r r r k
 k k r k k t k k k k k t k k k k k
 k k k k k t k k k k k t k k k k k
 t t t t t t t t t t t t t t t t t
 k k k k k t k k k k k t k k k k k
 k k k k k t k k r k k t k k k k k
 k k k k k t k r k r k t k k k k k
 k k k k k t k k r k k t k k k k k
 k k k k k t k k k k k t k k k k k
 t t t t t t t t t t t t t t t t t
 k k k k k t k k k k k t k k k k k
 k k r k k t k k k k k t k k k k k
 k r r r k t k k k k k t k k k k k
 k k r k k t k k k k k t k k k k k
 k k k k k t k k k k k t k k k k k
]
Output1 = [
 k k k k k t k k k k k t k k k k k
 k k r k k t k k t k k t k k t k k
 k r r r k t k t t t k t k r r r k
 k k r k k t k k t k k t k k t k k
 k k k k k t k k k k k t k k k k k
 t t t t t t t t t t t t t t t t t
 k k k k k t k k k k k t k k k k k
 k k t k k t k k r k k t k k t k k
 k t t t k t k r t r k t k t t t k
 k k t k k t k k r k k t k k t k k
 k k k k k t k k k k k t k k k k k
 t t t t t t t t t t t t t t t t t
 k k k k k t k k k k k t k k k k k
 k k r k k t k k t k k t k k t k k
 k r r r k t k t t t k t k t t t k
 k k r k k t k k t k k t k k t k k
 k k k k k t k k k k k t k k k k k
]

Train example 2:
Input2 = [
 k k k k k r k k k k k r k k k k k
 k b b b k r k k k k k r k k b b k
 k b b b k r k b b k k r k k k k k
 k b b b k r k k k k k r k k k k k
 k k k k k r k k k k k r k k k k k
 r r r r r r r r r r r r r r r r r
 k k k k k r k k k k k r k k k k k
 k k k k k r k k k k k r k k k k k
 k k k k k r k k k k k r k k b k k
 k k k k k r k k k k k r k k k k k
 k k k k k r k k k k k r k k k k k
 r r r r r r r r r r r r r r r r r
 k k k k k r k k k k k r k k k k k
 k k k k k r k k b k k r k k k k k
 k k k k k r k b k b k r k k k k k
 k k k k k r k k b k k r k k k k k
 k k k k k r k k k k k r k k k k k
]
Output2 = [
 k k k k k r k k k k k r k k k k k
 k b b b k r k r r r k r k r b b k
 k b b b k r k b b r k r k r r r k
 k b b b k r k r r r k r k r r r k
 k k k k k r k k k k k r k k k k k
 r r r r r r r r r r r r r r r r r
 k k k k k r k k k k k r k k k k k
 k r r r k r k r r r k r k r r r k
 k r r r k r k r r r k r k r b r k
 k r r r k r k r r r k r k r r r k
 k k k k k r k k k k k r k k k k k
 r r r r r r r r r r r r r r r r r
 k k k k k r k k k k k r k k k k k
 k r r r k r k r b r k r k r r r k
 k r r r k r k b r b k r k r r r k
 k r r r k r k r b r k r k r r r k
 k k k k k r k k k k k r k k k k k
]

Train example 3:
Input3 = [
 k k k k k b k k k k k b k k k k k
 k k g k k b k k k k k b k k k k k
 k g k g k b k g k k k b k k k g k
 k k g k k b k k k k k b k k k k k
 k k k k k b k k k k k b k k k k k
 b b b b b b b b b b b b b b b b b
 k k k k k b k k k k k b k k k k k
 k k g k k b k k k k k b k k k k k
 k g k k k b k k k k k b k k k k k
 k k k k k b k k k k k b k k k k k
 k k k k k b k k k k k b k k k k k
 b b b b b b b b b b b b b b b b b
 k k k k k b k k k k k b k k k k k
 k k k k k b k k k k k b k k k k k
 k k k k k b k k k k k b k k k g k
 k k k k k b k k k k k b k k g k k
 k k k k k b k k k k k b k k k k k
]
Output3 = [
 k k k k k b k k k k k b k k k k k
 k k g k k b k k b k k b k k b k k
 k g k g k b k g k b k b k b k g k
 k k g k k b k k b k k b k k b k k
 k k k k k b k k k k k b k k k k k
 b b b b b b b b b b b b b b b b b
 k k k k k b k k k k k b k k k k k
 k k g k k b k k b k k b k k b k k
 k g k b k b k b k b k b k b k b k
 k k b k k b k k b k k b k k b k k
 k k k k k b k k k k k b k k k k k
 b b b b b b b b b b b b b b b b b
 k k k k k b k k k k k b k k k k k
 k k b k k b k k b k k b k k b k k
 k b k b k b k b k b k b k b k g k
 k k b k k b k k b k k b k k g k k
 k k k k k b k k k k k b k k k k k
] """

# concepts:
# pattern reconstruction

# description:
# In the input you will see 9 squares seperated by 4 lines. The top-left square contains the original pattern.
# Each square contains either a small portion of pattern or remains empty.
# To make the output, you should detect the pattern on the top-left square and fill each square 

def transform_1e32b0e9(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)  

    # Detect the color of the lines
    for x, row in enumerate(output_grid):
        # Find the line
        all_equal = np.unique(row).size == 1
        if all_equal:
            line_color = row[0]
            break
    
    # Get all the squares seperated by lines in the grid
    squares = find_connected_components(grid=output_grid, background=line_color, monochromatic=False, connectivity=4)

    # Get all squares' bounding box and cropped pattern
    cropped_squares  = []
    for obj in squares:
        x, y, width, height = bounding_box(grid=obj, background=line_color)
        square = crop(grid=obj, background=line_color)
        cropped_squares.append({'x': x, 'y': y, 'len': width, 'pattern': square})

    # Sort the squares by their position
    cropped_squares = sorted(cropped_squares, key=lambda x: (x['x'], x['y']))

    # The top-left square contains the original pattern
    template_pattern = cropped_squares[0]['pattern']
    other_patterns = cropped_squares[1:]

    # Fill the missing pattern compared to template square with line color
    for square in other_patterns:
        x, y = square['x'], square['y']
        square_pattern = square['pattern']

        # Fill the missing pattern compared to template square with line color
        for i, j in np.argwhere(template_pattern != Color.BLACK):
            if template_pattern[i, j] != square_pattern[i, j]:
                square_pattern[i, j] = line_color

        # Place the reconstructed pattern on the output grid
        output_grid = blit_sprite(grid=output_grid, sprite=square_pattern, x=x, y=y)

    return output_grid

""" ==============================
Puzzle 1f642eb9

Train example 1:
Input1 = [
 k k k k m k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k t t k k k k
 k k k k t t k k k k
 k k k k t t k k k k
 p k k k t t k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k y k k k k
]
Output1 = [
 k k k k m k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k m t k k k k
 k k k k t t k k k k
 k k k k t t k k k k
 p k k k p y k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k y k k k k
]

Train example 2:
Input2 = [
 k k k k o k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 p k k t t t k k k k
 k k k t t t k k k k
 k k k t t t k k k r
 k k k t t t k k k k
 g k k t t t k k k k
 k k k k k k k k k k
 k k k k k b k k k k
]
Output2 = [
 k k k k o k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 p k k p o t k k k k
 k k k t t t k k k k
 k k k t t r k k k r
 k k k t t t k k k k
 g k k g t b k k k k
 k k k k k k k k k k
 k k k k k b k k k k
]

Train example 3:
Input3 = [
 k k k y k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k t t t k k k p
 g k k t t t k k k k
 k k k t t t k k k k
 r k k t t t k k k k
 k k k t t t k k k r
 k k k k k k k k k k
 k k k o k k k k k k
]
Output3 = [
 k k k y k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k y t p k k k p
 g k k g t t k k k k
 k k k t t t k k k k
 r k k r t t k k k k
 k k k o t r k k k r
 k k k k k k k k k k
 k k k o k k k k k k
] """

# concepts:
# alignment, copy to object border

# description:
# In the input you will see a teal object on a black background, and several colored pixels on the border of canvas.
# To make the output grid, you should copy the colored pixels horizontally/vertically so that they are just barely overlapping/colliding with the teal object.

def transform_1f642eb9(input_grid):
    # Plan:
    # 1. Detect the teal object
    # 2. Detect the colored pixels on the border
    # 3. Slide the colored pixels in the 4 cardinal directions until we find how to make them overlapping with the teal object

    output_grid = np.copy(input_grid)

    # Detects the rectangle in the input grid that is TEAL
    teal_objects = detect_objects(grid=input_grid, colors=[Color.TEAL], monochromatic=True, connectivity=4)
    
    # There should only be one rectangle of the color TEAL has been detected in the grid.
    assert len(teal_objects) == 1
    teal_object = teal_objects[0]

    # colored pixels are NOT black and NOT TEAL.
    colors_except_teal = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    
    # Detects all other colored pixels in the grid 
    pixels = detect_objects(grid=input_grid,
                            # Exclude teal from the search
                            colors=colors_except_teal, 
                            # only consider single pixels
                            allowed_dimensions=[(1,1)], 
                            monochromatic=True, connectivity=4)

    # Copy the colored pixels to the teal object by moving them either vertically or horizontally.
    for pixel in pixels:
        # consider translating the pixel in the 4 cardinal directions, and consider translating as far as possible
        possible_displacements = [ (slide_distance*dx, slide_distance*dy)
                                   # We could slide as far as the maximum grid extent
                                   for slide_distance in range(max(input_grid.shape))
                                   # (dx, dy) ranges over 4 cardinal directions
                                   for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] ]
        for dx, dy in possible_displacements:
            # check if the objects are colliding/overlapping after translating
            translated_pixel = translate(pixel, dx, dy, background=Color.BLACK)
            if collision(object1=teal_object, object2=translated_pixel):
                # put the red object where it belongs
                blit_object(output_grid, translated_pixel, background=Color.BLACK)
                break
    
    return output_grid

""" ==============================
Puzzle 1fad071e

Train example 1:
Input1 = [
 k k k k r r k k b
 k b b k r r k k k
 k b b k k k k r r
 k k k k k k k r r
 b k r r k k k k k
 k k r r k b b k k
 k k k k k b b k k
 k k k k k k k k k
 k b k k k k k k b
]
Output1 = [
 b b k k k
]

Train example 2:
Input2 = [
 b b k r k k k k r
 b b k k k b b k k
 k k k r k b b k k
 k k k k k k k k b
 k b b k r r k k k
 k b b k r r k k r
 k k k k k k k k k
 k k k r r k b b k
 k b k r r k b b k
]
Output2 = [
 b b b b k
]

Train example 3:
Input3 = [
 r r k b b k k k k
 r r k b b k k b b
 b k k k k k k b b
 k r r k k k k k k
 k r r k b b k b k
 k k k k b b k k k
 k k k k r k k k k
 k b b k k k k r r
 k b b k k b k r r
]
Output3 = [
 b b b b k
] """

# concepts:
# counting

# description:
# In the input you will see multiple 2x2 blue squares, multiple 2x2 red squares, and some red or blue dots sprinkled about.
# To make the output, fill a 1x5 grid with blue pixels from left to right for each 2x2 blue square in the input (counting the number of blue 2x2 squares).

def transform_1fad071e(input_grid):
    # make a counter to count the number of blue squares
    blue_square_count = 0

    # scan the grid for blue squares and count them up
    for x in range(input_grid.shape[0]-1):
        for y in range(input_grid.shape[1]-1):
            if input_grid[x,y] == input_grid[x+1,y] == input_grid[x,y+1] == input_grid[x+1,y+1] == Color.BLUE:
                blue_square_count += 1
    
    # make a 1x5 output grid
    output_grid = np.zeros((5,1), dtype=int)

    # add the number of blue squares to the array from left to right with each pixel representing one blue block
    output_grid[:blue_square_count, :] = Color.BLUE

    return output_grid

""" ==============================
Puzzle 2204b7a8

Train example 1:
Input1 = [
 b k k k k k k k k r
 b k k k k k g k k r
 b k k k k k k k k r
 b k k k k k k k k r
 b k k k g k k k k r
 b k k k k k k k k r
 b k g k k k k k k r
 b k k k k k k k k r
 b k k k k k k k k r
 b k k k k k k k k r
]
Output1 = [
 b k k k k k k k k r
 b k k k k k r k k r
 b k k k k k k k k r
 b k k k k k k k k r
 b k k k b k k k k r
 b k k k k k k k k r
 b k b k k k k k k r
 b k k k k k k k k r
 b k k k k k k k k r
 b k k k k k k k k r
]

Train example 2:
Input2 = [
 y y y y y y y y y y
 k k k k k k k k k k
 k g k k k k k g k k
 k k k g k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k g k k k k k
 k k k k k k k k g k
 k k k k k k k k k k
 o o o o o o o o o o
]
Output2 = [
 y y y y y y y y y y
 k k k k k k k k k k
 k y k k k k k y k k
 k k k y k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k o k k k k k
 k k k k k k k k o k
 k k k k k k k k k k
 o o o o o o o o o o
]

Train example 3:
Input3 = [
 t t t t t t t t t t
 k k k k k k k k k k
 k k k k g k k k k k
 k g k k k k k g k k
 k k k k k k k k k k
 k k k k k k k k k k
 k g k k k k g k k k
 k k k g k k k k k k
 k k k k k k k k k k
 m m m m m m m m m m
]
Output3 = [
 t t t t t t t t t t
 k k k k k k k k k k
 k k k k t k k k k k
 k t k k k k k t k k
 k k k k k k k k k k
 k k k k k k k k k k
 k m k k k k m k k k
 k k k m k k k k k k
 k k k k k k k k k k
 m m m m m m m m m m
] """

# concepts:
# proximity, color change, horizontal/vertical bars

# description:
# In the input you will see a pair of lines on the edge of the canvas that are either horizontal or vertical, and also green pixels randomly placed in between the lines
# Change the color of each green pixel to match the color of the line it is closest to

def transform_2204b7a8(input_grid: np.ndarray) -> np.ndarray:
    # find the two lines by removing the green pixels
    lines = np.copy(input_grid)
    lines[input_grid == Color.GREEN] = Color.BLACK

    # lines now contains only the lines, which are going to be used to assign color to the green pixels
    output_grid = np.copy(input_grid)

    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] == Color.GREEN:
                # find the closest point on the lines
                closest_x, closest_y = min([(i, j) for i in range(input_grid.shape[0]) for j in range(input_grid.shape[1]) if lines[i, j] != Color.BLACK],
                                           key=lambda p: abs(p[0] - x) + abs(p[1] - y))
                color_of_closest_line = lines[closest_x, closest_y]
                output_grid[x, y] = color_of_closest_line
    
    return output_grid

""" ==============================
Puzzle 228f6490

Train example 1:
Input1 = [
 o k k k k k k k o o
 k e e e e e k k k k
 k e k k e e k p p k
 k e k k e e k k k k
 k e e e e e k k k k
 k e e e e e k k o k
 k k k k k k k k k k
 k k k k o e e e e e
 k t t k k e e k k e
 k t t k k e e e e e
]
Output1 = [
 o k k k k k k k o o
 k e e e e e k k k k
 k e t t e e k k k k
 k e t t e e k k k k
 k e e e e e k k k k
 k e e e e e k k o k
 k k k k k k k k k k
 k k k k o e e e e e
 k k k k k e e p p e
 k k k k k e e e e e
]

Train example 2:
Input2 = [
 e e e e e k k k k k
 e k k k e k m m m m
 e e e k e k m m m m
 e e e e e k k k k k
 k k k k k k k p k p
 g g g k k k p p k k
 k k g e e e e e e k
 k k k e k k k k e k
 p p k e k k k k e k
 p p k e e e e e e k
]
Output2 = [
 e e e e e k k k k k
 e g g g e k k k k k
 e e e g e k k k k k
 e e e e e k k k k k
 k k k k k k k p k p
 k k k k k k p p k k
 k k k e e e e e e k
 k k k e m m m m e k
 p p k e m m m m e k
 p p k e e e e e e k
]

Train example 3:
Input3 = [
 r r k k e e e e e e
 r r r k e k k k e e
 k k k k e e e k k e
 k y y k e e e e e e
 k k y k k y k k k k
 e e e e e k k y y k
 e e e e e k k k k k
 e k k e e k k k k y
 e k k k e k t t t k
 e e e e e k k k t t
]
Output3 = [
 k k k k e e e e e e
 k k k k e t t t e e
 k k k k e e e t t e
 k y y k e e e e e e
 k k y k k y k k k k
 e e e e e k k y y k
 e e e e e k k k k k
 e r r e e k k k k y
 e r r r e k k k k k
 e e e e e k k k k k
] """

# concepts:
# holes, objects, distracters, topology, puzzle piece

# description:
# In the input you will see multiple grey rectangular objects, each with a black hole inside of it. There are also monochromatic objects somewhere exactly the same shape as each hole, and random other distracter objects (distracters all the same color). 
# To make the output, check to see if each non-distracter object perfectly fits inside the black hole inside of a gray object, like it is a puzzle piece. If it does, place it inside the black hole. If it doesn't, leave it where it is.

def transform_228f6490(input_grid):
    # Plan:
    # 1. Parse the input into objects, sprites, and black holes inside the grey objects
    # 2. Identify color of distracter objects.
    # 3. Turn each object into a sprite
    # 4. Check if each sprite can be moved into a black hole, and if so, move it there
    
    # Parse, separating greys from other objects
    grey_input = input_grid.copy()
    grey_input[input_grid != Color.GREY] = Color.BLACK
    grey_objects = find_connected_components(grey_input, background=Color.BLACK, connectivity=4, monochromatic=True)

    # extracting a mask for the black region is tricky, because black is also the color of the background
    # get the black region inside the object by getting the interior mask, then just the black pixels
    interior_black_regions = [ object_interior(obj, background=Color.BLACK) & (obj == Color.BLACK)
                               for obj in grey_objects ]

    not_grey_input = input_grid.copy()
    not_grey_input[input_grid == Color.GREY] = Color.BLACK
    not_grey_objects = find_connected_components(not_grey_input, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Get the sprites
    not_grey_sprites = [ crop(obj, background=Color.BLACK) for obj in not_grey_objects ]

    # Get the color of the distracter objects
    # This is the most common color of the non-grey objects
    color_of_each_object = [ np.unique(obj[obj!=Color.BLACK])[0] for obj in not_grey_objects ]
    distracter_color = max(set(color_of_each_object), key=color_of_each_object.count)
    
    # Check if each sprite perfectly fits in a black hole/black interior region
    # do this by checking if it has the same shape as a black interior region
    # if it does, place it there
    output_grid = np.copy(input_grid)
    for sprite, obj, color in zip(not_grey_sprites, not_grey_objects, color_of_each_object):
        # Try to find a perfect fit (if it is not the distracter color)
        if color == distracter_color:
            continue
        
        for interior_obj_mask in interior_black_regions:
            # check the sprite masks are the same, meaning that they have the same shape
            # to convert a sprite to a mask you check if it is not background (black)
            sprite_mask = sprite != Color.BLACK
            # to convert an object to a spright you crop it
            interior_sprite_mask = crop(interior_obj_mask, background=Color.BLACK)
            perfect_fit = np.array_equal(sprite_mask, interior_sprite_mask)

            if perfect_fit:
                # remove the object from its original location
                object_mask = obj != Color.BLACK
                output_grid[object_mask] = Color.BLACK

                # place the sprite in the black hole by blitting it
                interior_x, interior_y, interior_width, interior_height = bounding_box(interior_obj_mask)
                blit_sprite(output_grid, sprite, interior_x, interior_y, background=Color.BLACK)
                break

    return output_grid

""" ==============================
Puzzle 23581191

Train example 1:
Input1 = [
 k k k k k k k k k
 k k k k k k k k k
 k k t k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k o k k
 k k k k k k k k k
 k k k k k k k k k
]
Output1 = [
 k k t k k k o k k
 k k t k k k o k k
 t t t t t t r t t
 k k t k k k o k k
 k k t k k k o k k
 k k t k k k o k k
 o o r o o o o o o
 k k t k k k o k k
 k k t k k k o k k
]

Train example 2:
Input2 = [
 k k k k k k k k k
 k k k t k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k o k k
 k k k k k k k k k
]
Output2 = [
 k k k t k k o k k
 t t t t t t r t t
 k k k t k k o k k
 k k k t k k o k k
 k k k t k k o k k
 k k k t k k o k k
 k k k t k k o k k
 o o o r o o o o o
 k k k t k k o k k
] """

# concepts:
# lines, intersection

# description:
# In the input, you will see a grid with a single orange pixel and a single teal pixel.
# To make the output, draw an orange vertical line and an orange horizontal line that intersect at the orange pixel, and draw a teal vertical line and a teal horizontal line that intersect at the teal pixel. The lines should go from edge to edge of the grid.
# Lastly, draw a red pixel where the teal and orange lines intersect.

def transform_23581191(input_grid):
    # make output grid
    output_grid = np.copy(input_grid)

    # get the index of the orange pixel
    orange = np.where(input_grid == Color.ORANGE)
    x, y = orange[0][0], orange[1][0]

    # get the index of the teal pixel
    teal = np.where(input_grid == Color.TEAL)
    x2, y2 = teal[0][0], teal[1][0]

    # draw lines from one edge of the grid through the orange and teal pixels and across to the other edge of the grid:
    # draw orange vertical line
    output_grid[x, :] = Color.ORANGE # Can also use draw_line(output_grid, x, 0, length=None, color=Color.ORANGE, direction=(0, 1))
    # draw orange horizontal line
    output_grid[:, y] = Color.ORANGE # Can also use draw_line(output_grid, 0, y, length=None, color=Color.ORANGE, direction=(1, 0))
    # draw teal vertical line
    output_grid[x2, :] = Color.TEAL # Can also use draw_line(output_grid, x2, 0, length=None, color=Color.TEAL, direction=(0, 1))
    # draw teal horizontal line
    output_grid[:, y2] = Color.TEAL # Can also use draw_line(output_grid, 0, y2, length=None, color=Color.TEAL, direction=(1, 0))
    

    # draw both intersection points
    output_grid[x, y2] = Color.RED
    output_grid[x2, y] = Color.RED

    return output_grid

""" ==============================
Puzzle 239be575

Train example 1:
Input1 = [
 k k t k t
 r r t k k
 r r k k t
 k k k r r
 t t k r r
]
Output1 = [
 k
]

Train example 2:
Input2 = [
 k t k k k k k
 r r k t t t k
 r r t t k r r
 k k t k k r r
 k t k k t k k
]
Output2 = [
 t
]

Train example 3:
Input3 = [
 t r r t t k k
 k r r k k k t
 k t t k k t k
 k k t k k k t
 t k t t t r r
 t k k k k r r
]
Output3 = [
 t
] """

# concepts:
# connectivity

# description:
# In the input image you will see several teal pixels and two 2x2 red squares on the black background.
# If the 2x2 red squares are connected by a path of teal pixels, then output a 1x1 teal grid, otherwise, output a 1x1 black grid. 

def transform_239be575(input_grid):
    # make output grid
    output_grid = np.zeros((1,1), dtype=int)

    # get just the red squares
    red_squares = np.zeros_like(input_grid)
    red_squares[input_grid == Color.RED] = Color.RED

    # get all components that are connected, regardless of color
    connected_components = find_connected_components(input_grid, connectivity=4, monochromatic=False)

    # check each connected component to see if it contains both red squares
    for connected_component in connected_components:
        # if it contains both red squares, output teal grid
        if np.all(connected_component[red_squares == Color.RED] == Color.RED):         
            output_grid[:,:] = Color.TEAL
            return output_grid

    # if none of the connected components contain both red squares, output black grid
    output_grid[:,:] = Color.BLACK
    return output_grid

""" ==============================
Puzzle 25d487eb

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r k k k k k k k k k k k
 k k k r r k k k k k k k k k k
 k k k b r r k k k k k k k k k
 k k k r r k k k k k k k k k k
 k k k r k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r k k k k k k k k k k k
 k k k r r k k k k k k k k k k
 k k k b r r b b b b b b b b b
 k k k r r k k k k k k k k k k
 k k k r k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k t k k k k k
 k k k k k t t t k k k k
 k k k k t t t t t k k k
 k k k t t t g t t t k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]
Output2 = [
 k k k k k k g k k k k k
 k k k k k k g k k k k k
 k k k k k k g k k k k k
 k k k k k k g k k k k k
 k k k k k k g k k k k k
 k k k k k k t k k k k k
 k k k k k t t t k k k k
 k k k k t t t t t k k k
 k k k t t t g t t t k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k g g r g g k k k k k
 k k k g g g k k k k k k
 k k k k g k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k g g r g g k k k k k
 k k k g g g k k k k k k
 k k k k g k k k k k k k
 k k k k r k k k k k k k
 k k k k r k k k k k k k
 k k k k r k k k k k k k
 k k k k r k k k k k k k
 k k k k r k k k k k k k
 k k k k r k k k k k k k
 k k k k r k k k k k k k
 k k k k r k k k k k k k
 k k k k r k k k k k k k
 k k k k r k k k k k k k
] """

# concepts:
# patterns, growing, horizontal/vertical bars

# description:
# In the input you will see a colored triangle with a single pixel centered in the base of the triangle that is a different color.
# Grow a bar out of and away from the triangle, as if shot out of the tip opposite the differently colored pixel. This bar is the same color as that base pixel.

def transform_25d487eb(input_grid):
    # get output grid ready
    output_grid = input_grid

    # find the differently colored pixel
    colors, counts = np.unique(input_grid, return_counts=True)
    base_color = colors[np.argmin(counts)]

    # find the base location and coordinates from it
    base = np.argwhere(input_grid == base_color).flatten()
    [base_x, base_y] = base

    # find which side of the base is not in the triangle
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        # check if this direction would be outside the grid
        x0, y0 = base_x + dx, base_y + dy
        if 0 <= x0 < len(input_grid) and 0 <= y0 < len(input_grid[0]):
            # check if this direction is in the triangle or not
            if input_grid[x0, y0] != Color.BLACK:
                continue
        # this direction is the opposite of the tip, so grow the bar in the opposite direction from the first black pixel
        x0, y0 = base_x - dx, base_y - dy
        while 0 <= x0 < len(input_grid) and 0 <= y0 < len(input_grid[0]):
            if input_grid[x0, y0] == Color.BLACK:
                output_grid[x0, y0] = base_color
            x0, y0 = x0 - dx, y0 - dy

        return output_grid

    assert 0, "No valid slide found"

""" ==============================
Puzzle 25d8a9c8

Train example 1:
Input1 = [
 y y y
 r g r
 r g g
]
Output1 = [
 e e e
 k k k
 k k k
]

Train example 2:
Input2 = [
 o g g
 p p p
 g o o
]
Output2 = [
 k k k
 e e e
 k k k
]

Train example 3:
Input3 = [
 r m r
 y y y
 m m m
]
Output3 = [
 k k k
 e e e
 e e e
] """

# concepts:
# patterns, horizontal bars

# description:
# In the input you will see a colored pattern in a 3x3 grid.
# For each row of the input, if that row is a single color, color that row in the output grey. Otherwise, output black.

def transform_25d8a9c8(input_grid):
    # get input grid shape
    n, m = input_grid.shape

    # get output grid ready
    output_grid = np.zeros((n, m), dtype=int)

    # look at each row of the input grid
    for y in range(m):
        # check if each pixel in the row is the same color
        base_color = input_grid[0][y]
        all_same_color = True
        for color in input_grid[1:, y]:
            if color != base_color:
                all_same_color = False

        # if they are all the same color, change the output row to grey
        if all_same_color:
            for x in range(n):
                output_grid[x][y] = Color.GREY

    return output_grid

""" ==============================
Puzzle 25ff71a9

Train example 1:
Input1 = [
 b b b
 k k k
 k k k
]
Output1 = [
 k k k
 b b b
 k k k
]

Train example 2:
Input2 = [
 k k k
 b b b
 k k k
]
Output2 = [
 k k k
 k k k
 b b b
]

Train example 3:
Input3 = [
 k b k
 b b k
 k k k
]
Output3 = [
 k k k
 k b k
 b b k
] """

# concepts:
# sliding objects

# description:
# In the input you will see a 3x3 grid with a contiguous shape on it.
# Slide the shape down by one pixel.

def transform_25ff71a9(input_grid):
    # find the connected component, which is a monochromatic object
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=True)
    obj = objects[0]

    # translate the object down by one pixel
    output_grid = translate(obj, 0, 1, background=Color.BLACK)

    return output_grid

""" ==============================
Puzzle 264363fd """

# concepts:
# patterns, growing, horizontal/vertical bars

# description:
# In the input you will see a 30x30 grid with at least two rectangles, each with at least one special pixel of a different color, and a crosshair-type pattern outside these rectangles.
# For each of these special pixels, apply the crosshair pattern and extend the points inside the rectangle that special pixel is in, until it reaches the edge.

def transform_264363fd(input_grid):
    # first get the grid size
    n, m = input_grid.shape

    # figure out background color by assuming it is the most common color
    background_color = np.bincount(input_grid.flatten()).argmax()

    # with this background color, find the polychromatic objects
    objects = find_connected_components(input_grid, background=background_color, connectivity=8, monochromatic=False)

    # sort the objects by their size in terms of area, the crosshair is the smallest object
    sorted_objects = sorted(objects, key=lambda x: np.count_nonzero(x != background_color))
    crosshair_object = sorted_objects[0]
    rectangles = sorted_objects[1:]

    # now find the crosshair sprite coordinates
    crosshair_sprite = crop(crosshair_object, background=background_color)
    width, height = crosshair_sprite.shape
    
    # if the crosshair is wider than it is tall, it extends horizontally, vertically if taller, both if square
    horizontal = True
    vertical = True
    if width > height:
        vertical = False
        point_color = crosshair_sprite[0, height // 2]
    elif height > width:
        horizontal = False
        point_color = crosshair_sprite[width // 2, 0]
    else:
        point_color = crosshair_sprite[width // 2, 0]

    # now we prepare the output grid
    output_grid = np.full_like(input_grid, background_color)

    # for each rectangle, crop it to just the rectangle, find the special pixel, extend the crosshair pattern from it, then add it back to the grid
    for rectangle in rectangles:
        # crop the rectangle to just the rectangle, while preserving its position in the grid
        rec_x, rec_y, w, h = bounding_box(rectangle, background=background_color)
        cropped_rectangle = crop(rectangle, background=background_color)

        # find the special color, it is the least common color in the rectangle
        colors, counts = np.unique(cropped_rectangle, return_counts=True)
        # colors are sorted by their frequency, so choose least common as the special color
        special_color = colors[-1]
        rectangle_color = colors[-2]

        # for each special pixel, extend the crosshair pattern
        for x, y in np.argwhere(cropped_rectangle == special_color):
            # first color the special pixel with the crosshair sprite centered on it
            cropped_rectangle = blit_sprite(cropped_rectangle, crosshair_sprite, x - width // 2, y - height // 2, background=background_color)

            # then extend the points in the crosshair pattern until they reach the edge of the rectangle
            if horizontal:
                for x0 in range(w):
                    if cropped_rectangle[x0, y] == rectangle_color:
                        cropped_rectangle[x0, y] = point_color
            if vertical:
                for y0 in range(h):
                    if cropped_rectangle[x, y0] == rectangle_color:
                        cropped_rectangle[x, y0] = point_color
        
        # add the rectangle back to the grid
        blit_sprite(output_grid, cropped_rectangle, rec_x, rec_y, background=background_color)

    return output_grid

""" ==============================
Puzzle 28e73c20

Train example 1:
Input1 = [
 k k k k k k
 k k k k k k
 k k k k k k
 k k k k k k
 k k k k k k
 k k k k k k
]
Output1 = [
 g g g g g g
 k k k k k g
 g g g g k g
 g k g g k g
 g k k k k g
 g g g g g g
]

Train example 2:
Input2 = [
 k k k k k k k k
 k k k k k k k k
 k k k k k k k k
 k k k k k k k k
 k k k k k k k k
 k k k k k k k k
 k k k k k k k k
 k k k k k k k k
]
Output2 = [
 g g g g g g g g
 k k k k k k k g
 g g g g g g k g
 g k k k k g k g
 g k g g k g k g
 g k g g g g k g
 g k k k k k k g
 g g g g g g g g
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output3 = [
 g g g g g g g g g g g g g g g
 k k k k k k k k k k k k k k g
 g g g g g g g g g g g g g k g
 g k k k k k k k k k k k g k g
 g k g g g g g g g g g k g k g
 g k g k k k k k k k g k g k g
 g k g k g g g g g k g k g k g
 g k g k g k k k g k g k g k g
 g k g k g k g g g k g k g k g
 g k g k g k k k k k g k g k g
 g k g k g g g g g g g k g k g
 g k g k k k k k k k k k g k g
 g k g g g g g g g g g g g k g
 g k k k k k k k k k k k k k g
 g g g g g g g g g g g g g g g
] """

# concepts:
# geometric pattern, repetition, spiral

# description:
# In the input you will see an empty black grid.
# To make the output, you should draw a spiral pattern of green pixels starting from the top left corner and going to the right.

def transform_28e73c20(input_grid):
    # get the grid size
    width, height = input_grid.shape

    # start from the top left corner of the grid
    x, y = 0, 0
    output_grid = input_grid.copy()
    output_grid[x, y] = Color.GREEN

    # we define our initial direction as going to the right, which is (1, 0)
    direction = (1, 0)

    # we also make a helper function to turn the direction clockwise
    def turn_clockwise(direction):
        if direction[0] == 0:
            return -direction[1], 0
        return 0, direction[0]

    # continue spiralling until we cannot anymore
    while True:
        # First, check if we hit the border, if so, we turn clockwise
        if x + direction[0] >= width or y + direction[1] >= height or x + direction[0] < 0 or y + direction[1] < 0:
            direction = turn_clockwise(direction)
            continue

        # Then, check if the square after the current one is green, if so,
        # we are already spiralling, so stop here
        if output_grid[x + direction[0], y + direction[1]] == Color.GREEN:
            break

        # Last, check if the square after the current one is green, if so, we turn clockwise
        # We do this to draw the spiral pattern
        if (0 <= x + 2 * direction[0] < width and 0 <= y + 2 * direction[1] < height 
            and output_grid[x + 2 * direction[0], y + 2 * direction[1]] == Color.GREEN):
            direction = turn_clockwise(direction)
            continue
        
        # then we move to the next square and color it green
        x += direction[0]
        y += direction[1]
        output_grid[x, y] = Color.GREEN

    return output_grid

""" ==============================
Puzzle 29623171

Train example 1:
Input1 = [
 k k k e k k k e k k k
 b k k e k k k e k b k
 k k k e k k b e k k k
 e e e e e e e e e e e
 k k k e k k b e k k k
 k k k e k k k e k b k
 k k k e k k k e k k k
 e e e e e e e e e e e
 k k k e k k k e b k k
 k b k e k k k e k k b
 k k k e k k k e k k k
]
Output1 = [
 k k k e k k k e k k k
 k k k e k k k e k k k
 k k k e k k k e k k k
 e e e e e e e e e e e
 k k k e k k k e k k k
 k k k e k k k e k k k
 k k k e k k k e k k k
 e e e e e e e e e e e
 k k k e k k k e b b b
 k k k e k k k e b b b
 k k k e k k k e b b b
]

Train example 2:
Input2 = [
 k k k e k r k e r k k
 r k k e k k k e k k r
 k k k e k k k e k k k
 e e e e e e e e e e e
 r k k e k k k e k k k
 r k k e k k r e k k k
 k k k e k k k e k r k
 e e e e e e e e e e e
 k k k e k k k e k k k
 r k k e k k r e k k r
 k k k e k k k e k k k
]
Output2 = [
 k k k e k k k e r r r
 k k k e k k k e r r r
 k k k e k k k e r r r
 e e e e e e e e e e e
 r r r e k k k e k k k
 r r r e k k k e k k k
 r r r e k k k e k k k
 e e e e e e e e e e e
 k k k e k k k e k k k
 k k k e k k k e k k k
 k k k e k k k e k k k
]

Train example 3:
Input3 = [
 g g k e k k k e k k k
 k k k e k k k e k g k
 k k k e k k k e k k k
 e e e e e e e e e e e
 k k k e k k k e k k k
 k g k e k g k e k k k
 k k k e g k k e k k k
 e e e e e e e e e e e
 k k k e k k k e k k k
 k g k e g k k e g g k
 k k k e k k k e k k g
]
Output3 = [
 k k k e k k k e k k k
 k k k e k k k e k k k
 k k k e k k k e k k k
 e e e e e e e e e e e
 k k k e k k k e k k k
 k k k e k k k e k k k
 k k k e k k k e k k k
 e e e e e e e e e e e
 k k k e k k k e g g g
 k k k e k k k e g g g
 k k k e k k k e g g g
] """

# concepts:
# counting, dividers, filling

# description:
# In the input you will see grey horizontal and vertical bars that divide rectangular regions. Each rectangular region is black with some colored pixels added.
# To make the output, fill the rectangular region with the most colored pixels. Fill it with is color. Fill the other rectangular regions with black.   

def transform_29623171(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)  

    # Get all the rectangular regions seperated by horizontal and vertical dividers
    # The dividers are colored grey, but more generally their color is just whatever color stretches all the way horizontally or vertically
    for x in range(input_grid.shape[0]):
        if np.all(input_grid[x, :] == input_grid[x, 0]):
            divider_color = input_grid[x, 0]
            break
    # For this problem you could also do: divider_color = Color.GRAY
    regions = find_connected_components(grid=output_grid, background=divider_color, monochromatic=False, connectivity=4)

    # Find the region with the most colored pixels inside of it
    num_colored_pixels = [ np.sum((region != divider_color) & (region != Color.BLACK)) for region in regions ]
    max_colored_pixels = max(num_colored_pixels)

    # Fill the region with the most colored pixels with its color
    # Fill the other regions with black
    for region_obj in regions:
        # Figure out if it is one of the max colored regions to determine what the target color is that we are going to fill with
        num_colored_pixels_in_this_region = np.sum((region_obj != divider_color) & (region_obj != Color.BLACK))
        if num_colored_pixels_in_this_region == max_colored_pixels:
            colors = [ color for color in object_colors(region_obj, background=divider_color) if color != Color.BLACK ]
            assert len(colors) == 1, "Each region should have only one color"
            target_color = colors[0]
        else:
            target_color = Color.BLACK

        # Fill the region with the target color
        output_grid[region_obj != divider_color] = target_color

    return output_grid

""" ==============================
Puzzle 29c11459

Train example 1:
Input1 = [
 k k k k k k k k k k k
 b k k k k k k k k k r
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k
 b b b b b e r r r r r
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 g k k k k k k k k k o
 k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 g g g g g e o o o o o
 k k k k k k k k k k k
] """

# concepts:
# attraction, magnetism

# description:
# In the input you will individual pixels are the sides of a grid. Each pixel is in a matching pair, which is a different color, but on the opposite side of the grid.
# To make the output, make each pixel copy its color toward its matching pair until they meet in the middle. Turn the middle point grey.

def transform_29c11459(input_grid):
    # Plan:
    # 1. Detect the pixels
    # 2. Associate each pixel with its matching pair
    # 3. Make them attract/march toward each other until they meet in the middle, leaving a trail of their own color

    # 1. Find the location of the pixels
    pixel_objects = find_connected_components(input_grid, connectivity=4, background=Color.BLACK)
    assert len(pixel_objects) % 2 == 0, "There should be an even number of pixels"

    # we're going to draw on top of the input grid
    output_grid = input_grid.copy()

    for obj in pixel_objects:
        # 2. Associate each pixel with its matching pair
        # Find the matching position, which is either the opposite x or the opposite y
        x, y = object_position(obj, background=Color.BLACK, anchor='center')
        if x in [0, input_grid.shape[0] - 1]:
            opposite_x = input_grid.shape[0] - 1 - x
            opposite_y = y
        else:
            opposite_x = x
            opposite_y = input_grid.shape[1] - 1 - y

        # get the unit vector pointing from one to the other
        dx, dy = np.sign([opposite_x - x, opposite_y - y], dtype=int)

        color = input_grid[x, y]
        other_color = input_grid[opposite_x, opposite_y]

        # 3. Make them attract/march toward each other until they meet in the middle (grey when they touch), leaving a trail of their own color
        while (x, y) != (opposite_x, opposite_y):
            # Draw a trail of color
            if output_grid[x, y] == Color.BLACK:
                output_grid[x, y] = color
            if output_grid[opposite_x, opposite_y] == Color.BLACK:
                output_grid[opposite_x, opposite_y] = other_color
            x += dx
            y += dy
            opposite_x -= dx
            opposite_y -= dy

            # Make sure we haven't fallen out of bounds
            if not (0 <= x < input_grid.shape[0] and 0 <= y < input_grid.shape[1] and 0 <= opposite_x < input_grid.shape[0] and 0 <= opposite_y < input_grid.shape[1]):
                break
        # when they meet, turn the middle point grey
        output_grid[x, y] = Color.GREY
    
    return output_grid

""" ==============================
Puzzle 2bcee788

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k y r k k k k
 k k y y y r k k k k
 k k k k y r k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 g g g g g g g g g g
 g g g g g g g g g g
 g g g g g g g g g g
 g g g g y y g g g g
 g g y y y y y y g g
 g g g g y y g g g g
 g g g g g g g g g g
 g g g g g g g g g g
 g g g g g g g g g g
 g g g g g g g g g g
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k r r k k k k k
 k k k p p k k k k k
 k k k k p k k k k k
 k k k k p p k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 g g g g g g g g g g
 g g g g p p g g g g
 g g g g p g g g g g
 g g g p p g g g g g
 g g g p p g g g g g
 g g g g p g g g g g
 g g g g p p g g g g
 g g g g g g g g g g
 g g g g g g g g g g
 g g g g g g g g g g
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k r o k k k k k
 k k k r o o k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output3 = [
 g g g g g g g g g g
 g g g g g g g g g g
 g g g g g g g g g g
 g g g g g g g g g g
 g g g o o g g g g g
 g g o o o o g g g g
 g g g g g g g g g g
 g g g g g g g g g g
 g g g g g g g g g g
 g g g g g g g g g g
] """

# concepts:
# mirror symmetry, non-black background, indicator pixels

# description:
# In the input you will see an object with some red pixels attached to it on one side.
# To make the output, mirror the object to cover the red pixels. Then change the background to green.
 
def transform_2bcee788(input_grid):
    # Plan:
    # 1. Find the main object and the red pixels
    # 2. Calculate the axis over which to mirror depending on which side the red pixels are on
    # 3. Do the mirroring
    # 4. Change background to green
    
    # 1. Input parsing
    background = Color.BLACK
    objects = find_connected_components(input_grid, connectivity=8, background=background, monochromatic=True)
    assert len(objects) == 2, "There should be exactly two objects"

    # Find the main object
    main_object = next(obj for obj in objects if Color.RED not in object_colors(obj, background=background))
    # Find the red pixels
    red_pixels = next(obj for obj in objects if Color.RED in object_colors(obj, background=background))

    # 2. Axis calculation

    # Figure out what side of the object the red pixels are on
    x1, y1 = object_position(main_object, anchor="upper left")
    x2, y2 = object_position(main_object, anchor="lower right")
    # on the right?
    if collision(object1=translate(main_object, x=1, y=0), object2=red_pixels):
        # the +/- 0.5 is to clobber the red pixels, otherwise we'd reflect over them and leave them be, which would also be a reasonable thing to do
        symmetry = MirrorSymmetry(mirror_x=x2+0.5, mirror_y=None)
    # on the left?
    elif collision(object1=translate(main_object, x=-1, y=0), object2=red_pixels):
        symmetry = MirrorSymmetry(mirror_x=x1-0.5, mirror_y=None)
    # on the top?
    elif collision(object1=translate(main_object, x=0, y=-1), object2=red_pixels):
        symmetry = MirrorSymmetry(mirror_x=None, mirror_y=y1-0.5)
    # on the bottom?
    elif collision(object1=translate(main_object, x=0, y=1), object2=red_pixels):
        symmetry = MirrorSymmetry(mirror_x=None, mirror_y=y2+0.5)
    else:
        assert False, "Red pixels are not on any side of the main object"
    
    # 3. Mirror the main object
    output_grid = np.full_like(input_grid, background)
    blit_object(output_grid, main_object)
    for x, y in np.argwhere(main_object != background):
        for x2, y2 in orbit(output_grid, x, y, symmetries=[symmetry]):
            if 0 <= x2 < output_grid.shape[0] and 0 <= y2 < output_grid.shape[1]:
                output_grid[x2, y2] = main_object[x, y]
    
    # 4. Change the background to green
    output_grid[output_grid == background] = Color.GREEN

    return output_grid

""" ==============================
Puzzle 2c608aff

Train example 1:
Input1 = [
 t t t t t t t t t t t t
 t t g g g t t t t t t t
 t t g g g t t t t t t t
 t t g g g t t t t y t t
 t t g g g t t t t t t t
 t t t t t t t t t t t t
 t t t t t t t t t t t t
 t t t t t t t y t t t t
 t t t t t t t t t t t t
]
Output1 = [
 t t t t t t t t t t t t
 t t g g g t t t t t t t
 t t g g g t t t t t t t
 t t g g g y y y y y t t
 t t g g g t t t t t t t
 t t t t t t t t t t t t
 t t t t t t t t t t t t
 t t t t t t t y t t t t
 t t t t t t t t t t t t
]

Train example 2:
Input2 = [
 r r r r r r r r r r r r
 r r r r r r r r r r r r
 r r r b b b r r r r r r
 r r r b b b r r r r r r
 r r r b b b r r r r r r
 r r r r r r r r r r r r
 r r r r r r r r r r r r
 r r r r r r r r r r r r
 r r r t r r r r r r r r
 r r r r r r r r r r r r
]
Output2 = [
 r r r r r r r r r r r r
 r r r r r r r r r r r r
 r r r b b b r r r r r r
 r r r b b b r r r r r r
 r r r b b b r r r r r r
 r r r t r r r r r r r r
 r r r t r r r r r r r r
 r r r t r r r r r r r r
 r r r t r r r r r r r r
 r r r r r r r r r r r r
]

Train example 3:
Input3 = [
 b b b b r b b b b b b b
 b b b b b b b b b b b b
 b b b b b b b b b b b b
 b b b b b b b b b b b b
 b b b y y y y b b b b b
 b b b y y y y b b b b b
 b b b y y y y b b b r b
 b b b y y y y b b b b b
 b b b b b b b b b b b b
 b b b b b b b b b b b b
 b b b b b b b b b b b b
 b r b b b b b b b b b b
 b b b b b b b b b b b b
 b b b b b b b b b b b b
]
Output3 = [
 b b b b r b b b b b b b
 b b b b r b b b b b b b
 b b b b r b b b b b b b
 b b b b r b b b b b b b
 b b b y y y y b b b b b
 b b b y y y y b b b b b
 b b b y y y y r r r r b
 b b b y y y y b b b b b
 b b b b b b b b b b b b
 b b b b b b b b b b b b
 b b b b b b b b b b b b
 b r b b b b b b b b b b
 b b b b b b b b b b b b
 b b b b b b b b b b b b
] """

# concepts:
# attraction, objects, non-black background

# description:
# In the input you will see a non-black background with a colored rectangle and some colored pixels sprinkled randomly.
# To make the output, draw a horizontal or vertical line connecting each colored pixel to the rectangle (whenever possible: the rectangle and pixel have to be lined up). Color the line the same as the pixel.

def transform_2c608aff(input_grid):
    # Plan:
    # 1. Find the background color
    # 2. Extract objects, separating the pixels from the rectangle
    # 3. For each pixel, draw a line to the rectangle

    # The background is the most common color
    background = np.bincount(input_grid.flatten()).argmax()

    objects = find_connected_components(input_grid, connectivity=4, monochromatic=True, background=background)
    # The rectangle is the largest object
    rectangle_object = max(objects, key=lambda obj: np.sum(obj != background))
    # The pixels are the rest
    pixel_objects = [obj for obj in objects if obj is not rectangle_object]

    for pixel_object in pixel_objects:
        for x, y in np.argwhere(pixel_object != background):
            pixel_color = pixel_object[x, y]

            # Check if the pixel is on a horizontal or vertical line with the rectangle
            # Do this by trying to move the pixel up/down/left/right by different amounts until there is contact
            # After finding contact, double check that going one step further would lead to overlap (collision) to avoid glancing contact
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]: # up, right, down, left
                for distance in range(max(input_grid.shape)):
                    translated_pixel = translate(pixel_object, distance * dx, distance * dy, background=background)
                    if contact(object1=translated_pixel, object2=rectangle_object, background=background) and \
                        collision(object1=translate(pixel_object, (distance + 1) * dx, (distance + 1) * dy, background=background), object2=rectangle_object, background=background):
                        # Draw the line
                        end_x, end_y = x + distance * dx, y + distance * dy
                        draw_line(input_grid, x, y, end_x=end_x, end_y=end_y, color=pixel_color)
                        break
    
    return input_grid

""" ==============================
Puzzle 2dd70a9a

Train example 1:
Input1 = [
 k k k k t t t t t k k t k t t t k t k t
 k t k k k k k t k k k k k t k t k k k k
 t t t t t k t k t k k k t t t k k r k k
 t k t t k k k k k k t t t t t t k r k k
 t k k t t k k k k k k t k t k k k k k k
 k k k t k k k k k k k k k t k k k k k k
 t k k k k k k k k k k k k k k k k k t k
 k k t k k k k k k k k k k k k k k k t t
 t k k k k k k k k k k k t k k k k k t t
 k k k k k k k k k t k k t k k k k k t t
 k t k k k k t t t k t k k t k t t k k k
 t k k k k t k k k k t t t t t t t t t t
 k k k k k k t t t k k t t t k t k k t t
 k k k k k k t t k k k k t k k k t k k t
 k k k g k k k t k t k t k k t k k t k t
 k k k g k k t t t k k k t t t t k k k k
 k t k k k k k t k t t k t k t k t k k k
 k k k k k k k t k k k t k k k k k t t k
 k k k t k k k t k t k k t t t k k k k t
 k k k k t t t t k k t k k k k t t t k k
]
Output1 = [
 k k k k t t t t t k k t k t t t k t k t
 k t k k k k k t k k k k k t k t k k k k
 t t t t t k t k t k k k t t t k k r k k
 t k t t k k k k k k t t t t t t k r k k
 t k k t t k k k k k k t k t k k k g k k
 k k k t k k k k k k k k k t k k k g k k
 t k k g g g g g g g g g g g g g g g t k
 k k t g k k k k k k k k k k k k k k t t
 t k k g k k k k k k k k t k k k k k t t
 k k k g k k k k k t k k t k k k k k t t
 k t k g k k t t t k t k k t k t t k k k
 t k k g k t k k k k t t t t t t t t t t
 k k k g k k t t t k k t t t k t k k t t
 k k k g k k t t k k k k t k k k t k k t
 k k k g k k k t k t k t k k t k k t k t
 k k k g k k t t t k k k t t t t k k k k
 k t k k k k k t k t t k t k t k t k k k
 k k k k k k k t k k k t k k k k k t t k
 k k k t k k k t k t k k t t t k k k k t
 k k k k t t t t k k t k k k k t t t k k
]

Train example 2:
Input2 = [
 k k k k k k k k k t
 k g t k k k k k k k
 k g k k k k k t k k
 k k k k k k t k k t
 k t k t k k k k k k
 k k k t k k k k k k
 k t t k k r k k k k
 k k t k k r k k k k
 k k t k k k k k k k
 t k k k k k k k k k
]
Output2 = [
 k k k k k k k k k t
 k g t k k k k k k k
 k g k k k k k t k k
 k g g g g g t k k t
 k t k t k g k k k k
 k k k t k g k k k k
 k t t k k r k k k k
 k k t k k r k k k k
 k k t k k k k k k k
 t k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k t k t k k t k k t k
 k k k t k k t k k k k t k t t
 t k k k t t t k k k k t t t k
 k k k k k t k t k k k k k k k
 k k k k k k k k k t k k k t k
 k g g k k k k k t k k k t k k
 k k k k k k k k k k k t k t k
 k t t k k t k k t k t t k k k
 k t k k k k k k k k k k k k k
 t r r k k k k k k k k k k t k
 t k k k k k k t t t k k k k k
 k t k k t k t k k k t t t t k
 k k k k k t k t k k k k k k k
 k k k k k t t k t k k t k k t
 k t k k t t k k k k k k k k k
]
Output3 = [
 k k k k k t k t k k t k k t k
 k k k t k k t k k k k t k t t
 t k k k t t t k k k k t t t k
 k k k k k t k t k k k k k k k
 k k k k k k k k k t k k k t k
 k g g g g g g g t k k k t k k
 k k k k k k k g k k k t k t k
 k t t k k t k g t k t t k k k
 k t k k k k k g k k k k k k k
 t r r g g g g g k k k k k t k
 t k k k k k k t t t k k k k k
 k t k k t k t k k k t t t t k
 k k k k k t k t k k k k k k k
 k k k k k t t k t k k t k k t
 k t k k t t k k k k k k k k k
] """

# concepts:
# path finding


# description:
# In the input you will see teal pixels and a short green line and a short red line.
# Find a path starting from the green line and ending at the red line and color that path green, with the following constraints:
# You can't go through a teal pixel; you can only change direction when you hit a teal pixel; you have to start in the direction of the green line.

def transform_2dd70a9a(input_grid):
    # Plan:
    # 1. Find the start and end points of the pathfinding problem
    # 2. Define the state space, initial state(s), successor function, and goal test
    # 3. Run bfs to find the shortest path from start to end
    # 4. Color the path green

    # 1. Parse the input, based on color
    # There is the start object, end object, and barriers object
    background = Color.BLACK
    start_object = input_grid.copy()
    start_object[start_object != Color.GREEN] = background
    end_object = input_grid.copy()
    end_object[end_object != Color.RED] = background
    barriers_object = input_grid.copy()
    barriers_object[barriers_object != Color.TEAL] = background

    # Determine the orientation of the start object
    x_coordinates = {x for x, y in np.argwhere(start_object == Color.GREEN)}
    y_coordinates = {y for x, y in np.argwhere(start_object == Color.GREEN)}
    # vertical line?
    if len(x_coordinates) == 1:
        possible_orientations = [(0, 1), (0, -1)]
    # horizontal line?
    elif len(y_coordinates) == 1:
        possible_orientations = [(1, 0), (-1, 0)]
    else:
        assert False, "Start object is not horizontal/vertical"
    
    # 2. Define the state space, initial state(s), successor function, and goal test
    # A state is a tuple of (x, y, orientation)
    # orientation is a tuple of (dx, dy)
        
    # Initially we begin at a point on the line, along the orientation of the line
    initial_states = [(x, y, orientation)
                      for x, y in np.argwhere(start_object == Color.GREEN)
                      for orientation in possible_orientations]
    

    def successors(state):
        x, y, orientation = state
        dx, dy = orientation

        if not (0 <= x + dx < input_grid.shape[0] and 0 <= y + dy < input_grid.shape[1]):
            return

        if barriers_object[x + dx, y + dy] == background:
            yield (x + dx, y + dy, orientation)
        if barriers_object[x + dx, y + dy] != background:
            # right angle turns
            new_orientations = [(dy, dx), (-dy, -dx)]
            for new_orientation in new_orientations:
                yield (x, y, new_orientation)
    
    def is_goal(state):
        x, y, (dx, dy) = state
        if not (0 <= x + dx < end_object.shape[0] and 0 <= y + dy < end_object.shape[1]):
            return False
        return end_object[x + dx, y + dy] == Color.RED
    
    # 3. Run bfs to find the shortest path from start to end
    queue = list(initial_states)
    visited = set(initial_states)
    parent = {}
    while queue:
        state = queue.pop(0)        
        if is_goal(state):
            break        
        for successor in successors(state):
            if successor not in visited:
                visited.add(successor)
                parent[successor] = state
                queue.append(successor)

    assert is_goal(state), "No path found"
    
    path = []
    while state in parent:
        path.append(state)
        state = parent[state]

    # 4. Color the path green
    # draw on top of the input grid
    output_grid = input_grid.copy()
    for x, y, _ in path:
        output_grid[x, y] = Color.GREEN

    return output_grid

""" ==============================
Puzzle 2dee498d

Train example 1:
Input1 = [
 y e b b e y y e b
 e e e e e e e e e
 b e y y e b b e y
]
Output1 = [
 y e b
 e e e
 b e y
]

Train example 2:
Input2 = [
 r k k b r k k b r k k b
 y r b y y r b y y r b y
 y b r y y b r y y b r y
 b k k r b k k r b k k r
]
Output2 = [
 r k k b
 y r b y
 y b r y
 b k k r
]

Train example 3:
Input3 = [
 r b r b r b
 r g r g r g
]
Output3 = [
 r b
 r g
] """

# concepts:
# translational symmetry, symmetry detection, reflection

# description:
# In the input, you will see a sprite repeated horizontally, and some of those repetitions might be reflected top/down/right/left.
# To make the output, just extract the repeated sprite.

def transform_2dee498d(input_grid):
    # Find the period, remembering that we need to consider reflections
    for period in range(1, input_grid.shape[0]):
        # Extract the sprite and all of its repeated translated versions
        sprite = input_grid[:period]
        repetitions = [ input_grid[i*period:(i+1)*period] for i in range(input_grid.shape[0]//period) ]

        # Check that every repetition matches the sprite, or a reflection of the sprite
        valid = True
        for rep in repetitions:
            reflections = [rep, np.flip(rep, 0), np.flip(rep, 1)]
            if not any([np.array_equal(sprite, r) for r in reflections]):
                valid = False

        if valid:
            return sprite
        
    assert False, "No valid period found"

""" ==============================
Puzzle 31aa019c

Train example 1:
Input1 = [
 k k k b k k k e k k
 k k k k k k k k k k
 r k k k k r k k k b
 k k b k k k k k k e
 k k t k k k k k k k
 k k k k k k k k k k
 k y k k k k k k k k
 k e b k b k k k k k
 k t b k k k b k g k
 k k k k k k k g k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 r r r k k k k k k k
 r y r k k k k k k k
 r r r k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 r o o b k g k k k g
 k k k m k k k k g o
 k k k b k k k p k m
 k k k k k k k b k k
 m k k k k k k k k k
 k k k k r k k k g k
 k e k o g k k k b k
 y y k k k b k k k e
 k k k k k k k e g k
 k k k k y e k k k k
]
Output2 = [
 k k k k k k k k k k
 k k k k k k r r r k
 k k k k k k r p r k
 k k k k k k r r r k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 p k k k t k k k k k
 k k k k k k k k r t
 k o k k r k e k r k
 k m k b k k k k k k
 k m k k k k k k k b
 k k k k k p k k k k
 k b k o k k k k k k
 k k k k k k k k k k
 k k k k k k g k k k
 k k e k k k k k k k
]
Output3 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k r r r k k
 k k k k k r g r k k
 k k k k k r r r k k
] """

# concepts:
# counting, uniqueness, surrounding

# description:
# In the input, you will see a grid with a black background and colored pixels sprinkled on it. Exactly one color occurs only one time.
# To make the output, find the cell whose color is unique (color occurs only one time), and surround that cell with red pixels. Make all the other pixels black.

def transform_31aa019c(input_grid: np.ndarray) -> np.ndarray:
    # Plan:
    # 1. Create a blank new canvas (so that the non-unique colors don't get copied)
    # 2. Find the unique cell
    # 3. Surround the unique cell with red pixels

    output_grid = np.zeros_like(input_grid)

    # 2. Find the unique cell
    unique_color = None
    for color in Color.NOT_BLACK:
        if np.count_nonzero(input_grid == color) == 1:
            unique_color = color
            break
    
    # 3. Surround the unique cell with red pixels
    # First get the coordinates of the unique cell
    x, y, width, height = bounding_box(input_grid == unique_color)
    # Copy red over the region around the unique cell (but this will accidentally delete the unique cell, so be copied back)
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if 0 <= i < len(input_grid) and 0 <= j < len(input_grid[0]):
                output_grid[i, j] = Color.RED
    # Copy the unique cell back
    output_grid[x, y] = unique_color

    return output_grid

""" ==============================
Puzzle 3345333e

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k p p k k p p k k k k k k k
 k k k p p k b b b b k k k k k k
 k k k k p p b b b b k k k k k k
 k k k k k p b b b b k k k k k k
 k k k k k p p k k k k k k k k k
 k k k k p p p p k k k k k k k k
 k k k k p k k p k k k k k k k k
 k k k k p p p p k k k k k k k k
 k k k k k p p k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k p p k k p p k k k k k k k
 k k k p p k k p p k k k k k k k
 k k k k p p p p k k k k k k k k
 k k k k k p p k k k k k k k k k
 k k k k k p p k k k k k k k k k
 k k k k p p p p k k k k k k k k
 k k k k p k k p k k k k k k k k
 k k k k p p p p k k k k k k k k
 k k k k k p p k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k r k k r k k k k k k k k k
 g g g g r r r r k k k k k k k k
 g g g g r r r r r k k k k k k k
 g g g g r r k r k k k k k k k k
 g g g g k k k r k k k k k k k k
 k k r r r r r r k k k k k k k k
 k r r k r r k r r k k k k k k k
 k r r k k k k r r k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k r k k r k k k k k k k k k
 k k r r r r r r k k k k k k k k
 k r r r r r r r r k k k k k k k
 k k r k r r k r k k k k k k k k
 k k r k k k k r k k k k k k k k
 k k r r r r r r k k k k k k k k
 k r r k r r k r r k k k k k k k
 k r r k k k k r r k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
] """

# concepts:
# symmetry, occlusion

# description:
# In the input you will see a left-right symmetric monochromatic object occluded by a colored rectangle
# To make the output, remove the colored rectangle and fill in the missing parts of the object to make it left-right symmetric

def transform_3345333e(input_grid):
    # Plan:
    # 1. Extract and separate the rectangle from the symmetric object
    # 2. Find the left-right symmetry
    # 3. Fill in the missing parts of the object

    background_color = Color.BLACK

    # Each object has a different color, so we can look at connected components by color
    objects = detect_objects(input_grid, monochromatic=True, connectivity=8, background=background_color)
    sprites = [ crop(obj, background=Color.BLACK) for obj in objects ]
    # Find the rectangle
    for obj, sprite in zip(objects, sprites):
        # the rectangle will be completely filled, so we can check if its total area is the whole rectangular region
        if sprite.shape[0] * sprite.shape[1] == np.sum(sprite != Color.BLACK):
            rectangle = obj
            break

    # Find the color of the rectangle, because it is the occluder
    rectangle_color = object_colors(rectangle, background=background_color)[0]
    
    # Delete the rectangle
    rectangle_mask = rectangle != Color.BLACK
    output_grid = input_grid.copy()
    output_grid[rectangle_mask] = Color.BLACK

    # Find the symmetry
    # The occluder is rectangle_color, so we ignore it. In contrast, Color.BLACK is places where the object *actually* isn't located, so we can't ignore that.
    mirrors = detect_mirror_symmetry(input_grid, ignore_colors=[rectangle_color], background=Color.BLACK)

    # Mirror each colored pixel
    for x, y in np.argwhere(output_grid != Color.BLACK):
        for mirror in mirrors:
            source_color = output_grid[x,y]
            destination = mirror.apply(x, y)
            output_grid[destination] = source_color

    return output_grid

""" ==============================
Puzzle 3428a4f5

Train example 1:
Input1 = [
 k k k r r
 k k r k r
 r k k r r
 r r k k r
 k k k k r
 k r k k k
 y y y y y
 r k k k k
 r r k k k
 r k r k k
 k k r k k
 k k k r r
 r k k r k
]
Output1 = [
 g k k g g
 g g g k g
 k k g g g
 g g g k g
 k k k g k
 g g k g k
]

Train example 2:
Input2 = [
 k r r r r
 k k k k r
 r k r r r
 k k r r k
 r r r r k
 r r k k r
 y y y y y
 k k k k k
 k k r k k
 r k k k r
 k k k r k
 k r k r k
 k r r r k
]
Output2 = [
 k g g g g
 k k g k g
 k k g g k
 k k g k k
 g k g k k
 g k g g g
]

Train example 3:
Input3 = [
 r r k r r
 r k r r r
 r k k k k
 k r k r k
 r r r k r
 r k r k k
 y y y y y
 r k k r r
 k k r k r
 r r k k k
 k k r k r
 k r k r r
 k r r k r
]
Output3 = [
 k g k k k
 g k k g k
 k g k k k
 k g g g g
 g k g g k
 g g k k g
] """

# concepts:
# bitmasks with separator, boolean logical operations

# description:
# Compute the XOR operation of where the two grids are red, turning the output green in those locations.
# In the input, you should see two 6x5 red patterns on top and bottom separated a horizontal yellow line in the middle of the grid.
# To make the output, you have to overlap the two patterns. If the overlapping cells are the same color, then the corresponding cell is colored black; otherwise, 
# if the overlapping cells are not the same color, then the corresponding cell is colored green

def transform_3428a4f5(input_grid):

    width, height = input_grid.shape
   
    # Find the yellow horizontal line/bar
    for y_bar in range(height):
        if np.all(input_grid[:, y_bar] == Color.YELLOW):
            break
    
    # extract left and right patterns
    left_pattern = input_grid[:, :y_bar]
    right_pattern = input_grid[:, y_bar+1:] 

    output_grid = np.zeros_like(left_pattern)

    # applying the XOR pattern, which is where they are different
    output_grid[(left_pattern!=right_pattern)] = Color.GREEN
    output_grid[(left_pattern==right_pattern)] = Color.BLACK

    return output_grid

""" ==============================
Puzzle 3618c87e

Train example 1:
Input1 = [
 k k k k k
 k k k k k
 k k b k k
 k k e k k
 e e e e e
]
Output1 = [
 k k k k k
 k k k k k
 k k k k k
 k k e k k
 e e b e e
]

Train example 2:
Input2 = [
 k k k k k
 k k k k k
 k b k b k
 k e k e k
 e e e e e
]
Output2 = [
 k k k k k
 k k k k k
 k k k k k
 k e k e k
 e b e b e
]

Train example 3:
Input3 = [
 k k k k k
 k k k k k
 k b k k b
 k e k k e
 e e e e e
]
Output3 = [
 k k k k k
 k k k k k
 k k k k k
 k e k k e
 e b e e b
] """

# concepts:
# color, falling

# description:
# In the input, you should see a gray baseline at the bottom. For each gray baseline pixel, there may or may not be gray pixels above it. If there are gray pixels above it, you can see a blue pixel above the gray pixels.
# To make the output, make the blue pixels fall downward, falling through the gray baseline until they hit the bottom.

def transform_3618c87e(input_grid):
    output_grid = np.copy(input_grid)

    width, height = output_grid.shape

    # Find the color of the bottom baseline
    baseline = output_grid[:, -1]
    baseline_colors = np.unique(baseline)
    assert len(baseline_colors) == 1
    baseline_color = baseline_colors[0]

    # Find the color of the background, which is the most common color
    background_color = np.argmax(np.bincount(output_grid.flatten()))

    # Now make all the other colors fall down
    for x in range(width):
      for y in range(height):
          if output_grid[x, y] != background_color and output_grid[x, y] != baseline_color:
              # Make it fall to the bottom
              # Do this by finding the background/baseline spot below it which is closest to the bottom
              possible_y_values = [ possible_y for possible_y in range(y+1, height)
                                   if output_grid[x, possible_y] == background_color or output_grid[x, possible_y] == baseline_color]
              if len(possible_y_values) > 0:
                  closest_to_bottom_y = max(possible_y_values)
                  output_grid[x, closest_to_bottom_y] = output_grid[x, y]
                  output_grid[x, y] = background_color                  
  
    return output_grid

""" ==============================
Puzzle 3ac3eb23

Train example 1:
Input1 = [
 k r k k k t k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k r k k k t k k k k
 r k r k t k t k k k
 k r k k k t k k k k
 r k r k t k t k k k
 k r k k k t k k k k
 r k r k t k t k k k
]

Train example 2:
Input2 = [
 k y k k k k k
 k k k k k k k
 k k k k k k k
 k k k k k k k
 k k k k k k k
 k k k k k k k
]
Output2 = [
 k y k k k k k
 y k y k k k k
 k y k k k k k
 y k y k k k k
 k y k k k k k
 y k y k k k k
] """

# concepts:
# pixel pattern generation, falling downward

# description:
# In the input you will see a grid with several colored pixels at the top.
# To make the output, you should draw a pattern downward from each pixel:
# Color the diagonal corners, and then color downward with a vertical period of 2 from those corners and from the original pixel, making the pattern fall downward.

def transform_3ac3eb23(input_grid):
    # Plan:
    # 1. Find the pixels and make the output
    # 2. Grow the pixel pattern downward from each pixel

    # Extract the pixels
    pixels = find_connected_components(input_grid, monochromatic=True, background=Color.BLACK)

    # Create output grid
    output_grid = np.full_like(input_grid, Color.BLACK)
    width, height = input_grid.shape

    # 2. Grow the pixel pattern downward from each pixel
    for pixel in pixels:
        pixel_x, pixel_y = object_position(pixel, background=Color.BLACK)
        pixel_color = object_colors(pixel)[0]

        # We do the diagonal corners *and* also the original pixel, so one of the offsets is 0,0
        for offset_x, offset_y in [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            x, y = offset_x + pixel_x, offset_y + pixel_y

            # Fall downward (w/ period 2)
            while 0 <= x < width and 0 <= y < height:
                output_grid[x, y] = pixel_color
                # Vertical period of 2
                y += 2
            
    return output_grid

""" ==============================
Puzzle 3bdb4ada

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k k k k k k k k k k k k k k k k
 y y y y y y y y y y y y y y y y y y y y y y y y y y y y y k
 y y y y y y y y y y y y y y y y y y y y y y y y y y y y y k
 y y y y y y y y y y y y y y y y y y y y y y y y y y y y y k
 k k k k k k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k t t t t t t t t t t t t t k k k k k
 k k k k k k k k k k k k t t t t t t t t t t t t t k k k k k
 k k k k k k k k k k k k t t t t t t t t t t t t t k k k k k
]
Output1 = [
 k k k k k k k k k k k k k k k k k k k k k k k k k k k k k k
 y y y y y y y y y y y y y y y y y y y y y y y y y y y y y k
 y k y k y k y k y k y k y k y k y k y k y k y k y k y k y k
 y y y y y y y y y y y y y y y y y y y y y y y y y y y y y k
 k k k k k k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k t t t t t t t t t t t t t k k k k k
 k k k k k k k k k k k k t k t k t k t k t k t k t k k k k k
 k k k k k k k k k k k k t t t t t t t t t t t t t k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k k k k k
 k b b b b b b b b b k k k k k k k k k k
 k b b b b b b b b b k k k k k k k k k k
 k b b b b b b b b b k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k o o o o o o o o o o o k k
 k k k k k k k o o o o o o o o o o o k k
 k k k k k k k o o o o o o o o o o o k k
]
Output2 = [
 k k k k k k k k k k k k k k k k k k k k
 k b b b b b b b b b k k k k k k k k k k
 k b k b k b k b k b k k k k k k k k k k
 k b b b b b b b b b k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k o o o o o o o o o o o k k
 k k k k k k k o k o k o k o k o k o k k
 k k k k k k k o o o o o o o o o o o k k
] """

# concepts:
# punching holes, geometric pattern

# description:
# In the input you will see a grid with some rectangles. Each rectangle is three pixels along its short side, and is monochromatic.
# To make the output, you should draw black pixels along the middle of each rectangle with a period of 2 starting one pixel in.
# Effectively punching holes in the middle of each rectangle, skipping alternating pixels.

def transform_3bdb4ada(input_grid):
    # Plan:
    # 1. Extract the rectangles from the input grid
    # 2. Canonicalize the rectangles: ensure that they are horizontal (remember to rotate back otherwise)
    # 3. Draw black pixels inside each rectangle (horizontally, skipping every other pixel)
    # 4. Rotate the rectangle back if it was not originally horizontal

    # 1. Extract the rectangles from the input grid
    rectangle_objects = find_connected_components(input_grid, monochromatic=True)

    output_grid = np.full_like(input_grid, Color.BLACK)
    for rectangle_object in rectangle_objects:
        # 2. Canonicalize the rectangle sprite
        original_x, original_y, width, height = bounding_box(rectangle_object, background=Color.BLACK)
        # crop to convert object to sprite
        rectangle_sprite = crop(rectangle_object, background=Color.BLACK)

        # Flip it to be horizontal if it isn't already
        is_horizontal = width > height
        if not is_horizontal:
            rectangle_sprite = np.rot90(rectangle_sprite)
            width, height = height, width

        # 3. Punch holes through the middle of the rectangle
        # The inner row is y=height//2
        for x in range(1, width, 2):
            rectangle_sprite[x, height//2] = Color.BLACK
        
        # 4. Rotate back if it was originally vertical, and then draw it to the output grid
        if not is_horizontal:
            rectangle_sprite = np.rot90(rectangle_sprite, k=-1)
        
        # draw it back in its original location
        blit_sprite(output_grid, rectangle_sprite, original_x, original_y)

    return output_grid

""" ==============================
Puzzle 3befdf3e

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k y y y k k k k
 k k k y p y k k k k
 k k k y y y k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k y y y k k k k
 k k y p p p y k k k
 k k y p y p y k k k
 k k y p p p y k k k
 k k k y y y k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k r r r r k k k
 k k k r o o r k k k
 k k k r o o r k k k
 k k k r r r r k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k k k r r r r k k k
 k k k r r r r k k k
 k r r o o o o r r k
 k r r o r r o r r k
 k r r o r r o r r k
 k r r o o o o r r k
 k k k r r r r k k k
 k k k r r r r k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k g g g g k k k k
 k k g b b g k k k k
 k k g b b g k k k k
 k k g g g g k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k
 k k g g g g k k k k
 k k g g g g k k k k
 g g b b b b g g k k
 g g b g g b g g k k
 g g b g g b g g k k
 g g b b b b g g k k
 k k g g g g k k k k
 k k g g g g k k k k
 k k k k k k k k k k
] """

# concepts:
# growing, color change

# description:
# In the input you will see square(s) with a single pixel border around them in a different color.
# To make the output, swap the color between the square and the border, 
# and then put rectangles on the edges of the border whose width is the same as the length of the inner square, 
# and whose color is the same as the border in the input image.

def transform_3befdf3e(input_grid: np.ndarray) -> np.ndarray:
    # Get the output grid with the same size as the input grid.
    output_grid = np.copy(input_grid)

    # Detect the square pattern in the input grid.
    entire_square = find_connected_components(grid=input_grid, connectivity=4, monochromatic=False)
    for each_square in entire_square:
        # Detect the inner and outer squares in the pattern.
        object_square = find_connected_components(grid=each_square, connectivity=4, monochromatic=True)

        # Split the square pattern into inner and outer squares.
        split_objects = []
        for obj in object_square:
            x, y, width, height = bounding_box(grid=obj)
            shape = crop(grid=obj)
            color = obj[x, y]
            split_objects.append({'x': x, 'y': y, 'len': width, 'color': color, 'shape': shape})

        # Get the outer and inner square by comparing their size.
        if split_objects[0]['len'] > split_objects[1]['len']:
            outer_square = split_objects[0]
            inner_square = split_objects[1]
        else:
            outer_square = split_objects[1]
            inner_square = split_objects[0]
        
        # Swap the color between the inner and outer squares.
        outer_square_pattern, inner_square_pattern = outer_square['shape'], inner_square['shape']
        outer_square_pattern[outer_square_pattern == outer_square['color']] = inner_square['color']
        inner_square_pattern[inner_square_pattern == inner_square['color']] = outer_square['color']

        # Draw the inner and outer squares after swaping on the output grid.
        output_grid = blit_sprite(grid=output_grid, sprite=outer_square_pattern, x=outer_square['x'], y=outer_square['y'])
        output_grid = blit_sprite(grid=output_grid, sprite=inner_square_pattern, x=inner_square['x'], y=inner_square['y'])

        # Draw the rectangle as growing from outer square, the width is the same as the length of the inner square.
        # The rectangle is the same color as the original outer square.
        rectangle_width, rectangle_height = outer_square['len'], inner_square['len']  
        
        # Create the rectangle pattern for the edges of the outer square.
        rectangle_up_down = np.full((rectangle_width, rectangle_height), outer_square['color'])
        rectangle_left_right = np.full((rectangle_height, rectangle_width), outer_square['color'])

        # Draw the rectangle on the four edges of the outer square.
        output_grid = blit_sprite(grid=output_grid, sprite=rectangle_up_down, x=outer_square['x'], y=outer_square['y'] - inner_square['len'])
        output_grid = blit_sprite(grid=output_grid, sprite=rectangle_up_down, x=outer_square['x'], y=outer_square['y'] + outer_square['len'])
        output_grid = blit_sprite(grid=output_grid, sprite=rectangle_left_right, x=outer_square['x'] - inner_square['len'], y=outer_square['y'])
        output_grid = blit_sprite(grid=output_grid, sprite=rectangle_left_right, x=outer_square['x'] + outer_square['len'], y=outer_square['y'])

    return output_grid

""" ==============================
Puzzle 3de23699

Train example 1:
Input1 = [
 k k k k k k k
 k y k k k y k
 k k k r k k k
 k k r r r k k
 k k k r r k k
 k y k k k y k
 k k k k k k k
]
Output1 = [
 k y k
 y y y
 k y y
]

Train example 2:
Input2 = [
 k k k k k k k k k
 k g k k k k k g k
 k k k r r k k k k
 k k k r r k r k k
 k k r k k r k k k
 k g k k k k k g k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
]
Output2 = [
 k g g k k
 k g g k g
 g k k g k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k p k k k k p k k k k k
 k k k k k t k k k k k k k k
 k k k k k t k k k k k k k k
 k k k k t t t k k k k k k k
 k k k k k t t k k k k k k k
 k k k p k k k k p k k k k k
 k k k k k k k k k k k k k k
]
Output3 = [
 k p k k
 k p k k
 p p p k
 k p p k
] """

# concepts:
# pattern extraction, color matching

# description:
# In the input you will see a grid with a central pattern with four differently-colored pixels at the corners.
# To make the output, you should extract the central pattern (removing the differently-colored corners), 
# and change the color of the central pattern to match the corner pixels.

def transform_3de23699(input_grid: np.ndarray) -> np.ndarray:
    # Extract the central pattern by the four corner squares
    output_grid = np.copy(input_grid)

    # Crop the pattern out
    output_grid = crop(grid=output_grid)

    # Get the color of the corner squares
    corner_color = output_grid[0, 0]

    # Change the color of the central pattern to match the corner squares
    output_grid[output_grid != Color.BLACK] = corner_color

    # Remove the one pixel border around the central pattern
    output_grid = output_grid[1:-1, 1:-1]
    return output_grid

""" ==============================
Puzzle 3e980e27

Train example 1:
Input1 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k r b k k k k k k k k
 k k k b b k k k k k k k k
 k k k k k b k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k r k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k r b k k k k k k k k
 k k k b b k k k k k k k k
 k k k k k b k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k b r k k k k k
 k k k k k k b b k k k k k
 k k k k k b k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k
 k k k k k k y k k k k k k
 k k k k k y g y k k k k k
 k k k k k y y k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k g k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k g k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k
 k k k k k k y k k k k k k
 k k k k k y g y k k k k k
 k k k k k y y k k k k k k
 k k k k k k k k k k k k k
 k k y k k k k k k k k k k
 k y g y k k k k k k k k k
 k y y k k k k k k k k k k
 k k k k k k k k k y k k k
 k k k k k k k k y g y k k
 k k k k k k k k y y k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k r k k
 k k g t t k k k k k k k k
 k k t k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k y r k k k k k
 k g k k k k k y y k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k r y k
 k k g t t k k k k y y k k
 k k t k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k y r k k k k k
 k g t t k k k y y k k k k
 k t k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
] """

# concepts:
# pattern recognition, rotation, color correspondence, pattern reconstruction

# description:
# In the input you will see one or two color pattern with a red or green pixel as an indicator
# and a set of red and green pixels. 
# To make the output, you should reconstruct the pattern with the red and green pixels 
# indicates the pattern's position. If the indicator is red, the pattern should be flipped by x-axis before reconstructing.

def transform_3e980e27(input_grid):
    # Detect the continuous object in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=8)
    pixels = []
    original_pattern = []

    # Find out the original pattern and the pixel indicator for reconstruction
    for obj in objects:
        cropped_obj = crop(grid=obj, background=Color.BLACK)
        if cropped_obj.shape == (1,1):
            pixels.append(obj)
        else:
            original_pattern.append(cropped_obj)
    output_grid = input_grid.copy()

    for pattern in original_pattern:
        # If the indicator color is red, flip the pattern by x-axis
        if np.any(pattern == Color.RED):
            indicator_color = Color.RED
            pattern = np.flipud(pattern)
        else:
            indicator_color = Color.GREEN
        
        # Find the relative position of the indicator pixel in the pattern
        rela_x, rela_y = np.where(pattern == indicator_color)
        rela_x, rela_y = rela_x[0], rela_y[0]

        for pixel in pixels:
            color_pixel = Color.RED if np.any(pixel == Color.RED) else Color.GREEN

            # Find the position of the indicator pixel in the input grid
            if color_pixel == indicator_color:
                x, y = np.where(pixel == color_pixel)
                x, y = x[0], y[0]
                x -= rela_x
                y -= rela_y

                # Place the pattern in correct position using the indicator pixel, finish the reconstruction
                output_grid = blit_sprite(x=x, y=y, grid=output_grid, sprite=pattern, background=Color.BLACK)
    return output_grid

""" ==============================
Puzzle 3eda0437

Train example 1:
Input1 = [
 e b b b b b e k k k k k k k k k k b b k b k k b k k k k b k
 k b k b k k k b b b b k b b b k k k k b b b k b k b b b b b
 b b k k k b k b k b b b k b b k k k k b k k b b k k b k k k
]
Output1 = [
 e b b b b b e k k k k k k k k k k b b k b k k b k k k k b k
 k b k b k k k b b b b k b b b p p p p b b b k b k b b b b b
 b b k k k b k b k b b b k b b p p p p b k k b b k k b k k k
]

Train example 2:
Input2 = [
 b b b k k b b k b b b b k k k k k b k b
 b b b k b b k k b k b k b b k k k b b b
 k k b b b k k b k b k b b k b k b b b k
 k b k b k k k b b k b b b k k k b b b b
]
Output2 = [
 b b b k k b b k b b b b k k p p p b k b
 b b b k b b k k b k b k b b p p p b b b
 k k b b b k k b k b k b b k b k b b b k
 k b k b k k k b b k b b b k k k b b b b
]

Train example 3:
Input3 = [
 b b k k k k k k k b k k b k k b k b k b
 k b k k k k k b b b b k k b b k k k k k
]
Output3 = [
 b b p p p p p k k b k k b k k b k b k b
 k b p p p p p b b b b k k b b k k k k k
] """

# concepts:
# rectangle detection

# description:
# In the input you will see a grid with random pixels on it (mostly blue pixels).
# To make the output, you should find the largest rectangular area (of height/width >= 2, i.e. not a line) of black cells and turn it into pink.

def transform_3eda0437(input_grid: np.ndarray) -> np.ndarray:
    # Plan:
    # 1. Enumerate all rectangular regions
    # 2. For each region, filter it out if it isn't all black
    # 3. Find the biggest region remaining by area
    # 4. Turn the biggest region into pink

    # 1. Enumerate all rectangular regions
    regions = { (x, y, w, h) for x in range(len(input_grid)) for y in range(len(input_grid[0])) for w in range(2, len(input_grid) - x + 1) for h in range(2, len(input_grid[0]) - y + 1) }

    # 2. For each region, filter it out if it isn't all black
    regions = { (x, y, w, h) for x, y, w, h in regions if np.all(input_grid[x:x+w, y:y+h] == Color.BLACK) }

    # 3. Find the biggest region remaining by area
    largest_region = max(regions, key=lambda region: region[2] * region[3])
    x, y, w, h = largest_region

    # 4. Turn the biggest region into pink
    output_grid = np.copy(input_grid)
    output_grid[x:x+w, y:y+h] = Color.PINK
    
    return output_grid

""" ==============================
Puzzle 3f7978a0

Train example 1:
Input1 = [
 k k k k k k k k k
 k t k k k t k k t
 k e k k k e k k k
 k e k t k e k t k
 k e k k k e k k k
 k t k k k t k k k
 k k k k k k k k k
 k k k t k k k t k
 k t k k k k k k k
]
Output1 = [
 t k k k t
 e k k k e
 e k t k e
 e k k k e
 t k k k t
]

Train example 2:
Input2 = [
 k t k k k k k k k k k
 k k t k k k t k k k t
 k k k k k k k k k t t
 k k t k k k k k t k k
 t k e k k k k k e k k
 k k e k k t t k e k k
 k k e k k k k k e k k
 k k t k t k k k t k k
 k t k k k k k k t t k
]
Output2 = [
 t k k k k k t
 e k k k k k e
 e k k t t k e
 e k k k k k e
 t k t k k k t
]

Train example 3:
Input3 = [
 k k k k k k k k k t t k k
 k k k k k k k k k k k k k
 k k k t k k k t k k k k k
 k k k e k k k e k t k k k
 k k t e k t k e k k k k k
 k k k e k k k e t k k k k
 k k t e k t k e k k k k k
 k k k t k k k t k k t k k
 k k k k k k k k k k t k k
 k k t t k k k t k k k k k
 k k k k k k k k t k k k k
]
Output3 = [
 t k k k t
 e k k k e
 e k t k e
 e k k k e
 e k t k e
 t k k k t
] """

# concepts:
# boundary detection, object extraction

# description:
# In the input you will see several teal pixels and two vertical parallel gray lines with four teal pixels indicates the boundary of the output grid.
# To make the output grid, you should extract the part of grid that is bounded by the two vertical parallel gray lines and four teal pixels in each corner.

def transform_3f7978a0(input_grid):
    # Detect the vertical parallel gray lines.
    vertical_lines = detect_objects(grid=input_grid, colors=[Color.GRAY], monochromatic=True, connectivity=4)
    pos_list = []
    for vertical_line in vertical_lines:
        pos_x, pos_y, length_v, height_v = bounding_box(grid=vertical_line)
        pos_list.append({'x': pos_x, 'y': pos_y, 'length': length_v, 'height': height_v})
    
    # Get the left upper position and width, length of the extract part.
    pos_list.sort(key=lambda pos: pos['x'])
    x1, y1 = pos_list[0]['x'], pos_list[0]['y']
    x2, y2 = pos_list[1]['x'], pos_list[1]['y'] + pos_list[1]['height'] - 1

    # Grow the bounding box 1 pixel up and one pixel down.
    y1 = y1 - 1
    y2 = y2 + 1

    # Extract the bounded part of the grid.
    output_grid = input_grid[x1:x2 + 1, y1:y2 + 1]
    return output_grid

""" ==============================
Puzzle 4093f84a

Train example 1:
Input1 = [
 k k k k k k k k r k k k k k
 k k k k k k k k k k k k k k
 k k r k k k k k k k k k k k
 k k k k k k k k k k r k k k
 k k k k k k k k k k k k k k
 e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e
 k k k k k k k k k r k k k k
 k r k k k k k k k k k k k k
 k k k k r k k k k k k k k k
 k k k k k k k k k k k r k k
 k k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k e k k k k k e k e k k k
 e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e
 k e k k e k k k k e k e k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k e e e e e k k k g k
 k k k k e e e e e g k k k k
 k k k g e e e e e k k g k k
 k k k k e e e e e k k k k k
 g k k k e e e e e k k k k k
 k k k k e e e e e k k k k k
 k k k k e e e e e k g k g k
 k g k k e e e e e k k k k k
 k k k k e e e e e k k k k k
 k k g k e e e e e k k k g k
 k k k k e e e e e k k k k k
 k k k k e e e e e k k k k k
 k k k k e e e e e k g k k k
 k k k k e e e e e k k k k k
]
Output2 = [
 k k k k e e e e e e k k k k
 k k k k e e e e e e k k k k
 k k k e e e e e e e k k k k
 k k k k e e e e e k k k k k
 k k k e e e e e e k k k k k
 k k k k e e e e e k k k k k
 k k k k e e e e e e e k k k
 k k k e e e e e e k k k k k
 k k k k e e e e e k k k k k
 k k k e e e e e e e k k k k
 k k k k e e e e e k k k k k
 k k k k e e e e e k k k k k
 k k k k e e e e e e k k k k
 k k k k e e e e e k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k
 k k k k k k k b k k k k k k
 k k k k k k k k k k k k k k
 k k k b k k k b k k k k b k
 k k k k k k k k k k k k k k
 k k k k k k k k b k k k k k
 k k k k k k k k k k k k k k
 e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e
 k b k k k k k k k k k k k k
 k k k k k k k k k k k b k k
 k k k b k k k k b k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k e k k k k k k
 k k k e k k k e e k k k e k
 e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e
 k e k e k k k k e k k e k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
] """

# concepts:
# attraction, magnetism, color change

# description:
# In the input you will see a grey rectangle and colored pixels scattered around it.
# To make the output, move each colored pixel toward the grey rectangle until it touches, then turn its color to gray. If multiple colored pixels collide, they stack.

def transform_4093f84a(input_grid):
    # Plan:
    # 1. Detect the objects; separate the gray rectangle from the other pixels
    # 2. Move each colored pixel toward the gray rectangle until it touches
    # 3. Change its color once it touches

    objects = find_connected_components(input_grid, connectivity=4, background=Color.BLACK, monochromatic=True)

    grey_objects = [ obj for obj in objects if Color.GREY in object_colors(obj, background=Color.BLACK) ]
    other_objects = [ obj for obj in objects if Color.GREY not in object_colors(obj, background=Color.BLACK) ]

    assert len(grey_objects) == 1, "There should be exactly one grey object"
    
    grey_object = grey_objects[0]

    # Make the output grid: Start with the gray object, then add the colored pixels one-by-one
    output_grid = np.full_like(input_grid, Color.BLACK)
    blit_object(output_grid, grey_object)

    # Move the colored objects and change their color once they hit grey
    for colored_object in other_objects:
        # First calculate what direction we have to move in order to contact the grey object
        # Consider all displacements, starting with the smallest translations first
        possible_displacements = [ (i*dx, i*dy) for i in range(0, 30) for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)] ]

        # Only keep the displacements that cause a contact between the colored object and the grey object
        valid_displacements = [ displacement for displacement in possible_displacements
                                if contact(object1=translate(colored_object, *displacement), object2=grey_object) ]
        assert valid_displacements, "There should be at least one valid displacement"

        # Pick the smallest valid displacement
        displacement = min(valid_displacements, key=lambda displacement: sum(abs(x) for x in displacement))

        # Extract the direction from the displacement
        direction = np.sign(displacement, dtype=int)

        # Now move the colored object in that direction until there is a collision with something else
        if not all( delta == 0 for delta in direction ):
            while not collision(object1=translate(colored_object, *direction), object2=output_grid):
                colored_object = translate(colored_object, *direction)
        
        # Finally change the color of the colored object to grey anne draw it onto the outlet
        colored_object[colored_object != Color.BLACK] = Color.GREY
        blit_object(output_grid, colored_object)
    
    return output_grid

""" ==============================
Puzzle 41e4d17e

Train example 1:
Input1 = [
 t t t t t t t t t t t t t t t
 t t t b b b b b t t t t t t t
 t t t b t t t b t t t t t t t
 t t t b t t t b t t t t t t t
 t t t b t t t b t t t t t t t
 t t t b b b b b t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
]
Output1 = [
 t t t t t p t t t t t t t t t
 t t t b b b b b t t t t t t t
 t t t b t p t b t t t t t t t
 p p p b p p p b p p p p p p p
 t t t b t p t b t t t t t t t
 t t t b b b b b t t t t t t t
 t t t t t p t t t t t t t t t
 t t t t t p t t t t t t t t t
 t t t t t p t t t t t t t t t
 t t t t t p t t t t t t t t t
 t t t t t p t t t t t t t t t
 t t t t t p t t t t t t t t t
 t t t t t p t t t t t t t t t
 t t t t t p t t t t t t t t t
 t t t t t p t t t t t t t t t
]

Train example 2:
Input2 = [
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t
 t t t b b b b b t t t t t t t
 t t t b t t t b t t t t t t t
 t t t b t t t b t t t t t t t
 t t t b t t t b t t t t t t t
 t t t b b b b b t t t t t t t
 t t t t t t t t t t t t t t t
 t t t t t t t t b b b b b t t
 t t t t t t t t b t t t b t t
 t t t t t t t t b t t t b t t
 t t t t t t t t b t t t b t t
 t t t t t t t t b b b b b t t
 t t t t t t t t t t t t t t t
]
Output2 = [
 t t t t t p t t t t p t t t t
 t t t t t p t t t t p t t t t
 t t t t t p t t t t p t t t t
 t t t b b b b b t t p t t t t
 t t t b t p t b t t p t t t t
 p p p b p p p b p p p p p p p
 t t t b t p t b t t p t t t t
 t t t b b b b b t t p t t t t
 t t t t t p t t t t p t t t t
 t t t t t p t t b b b b b t t
 t t t t t p t t b t p t b t t
 p p p p p p p p b p p p b p p
 t t t t t p t t b t p t b t t
 t t t t t p t t b b b b b t t
 t t t t t p t t t t p t t t t
] """

# concepts:
# non-black background, occlusion

# description:
# In the input you will see a non-black background (teal background) and the outlines of 5x5 blue rectangles
# To make the output, draw pink horizontal/vertical bars at the center of each rectangle. The bars should be underneath the rectangles, and they should reach the edges of the canvas.

def transform_41e4d17e(input_grid):
    # Plan:
    # 1. Find the background color; check that it is teal
    # 2. Find the rectangles
    # 3. Draw the pink bars
    # 4. Ensure the rectangles are on top of the bars by drawing the rectangles last

    # The background is the most common color
    background = np.bincount(input_grid.flatten()).argmax()
    assert background == Color.TEAL

    # Extract the objects, which are the outlines of rectangles
    objects = find_connected_components(input_grid, connectivity=4, monochromatic=False, background=background)

    # Each object gets pink bars at its center, but these are going to be drawn over the object, which we have to undo later by redrawing the objects
    for obj in objects:
        center_x, center_y = object_position(obj, anchor='center', background=background)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            draw_line(input_grid, center_x, center_y, direction=(dx, dy), color=Color.PINK)
    
    # Redraw the objects
    for obj in objects:
        blit_object(input_grid, obj, background=background)
    
    return input_grid

""" ==============================
Puzzle 4258a5f9

Train example 1:
Input1 = [
 k k k k k k k k k
 k k k k k k e k k
 k k k k k k k k k
 k k k k k k k k k
 k k k e k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k e k k k k k k k
 k k k k k k k k k
]
Output1 = [
 k k k k k b b b k
 k k k k k b e b k
 k k k k k b b b k
 k k b b b k k k k
 k k b e b k k k k
 k k b b b k k k k
 b b b k k k k k k
 b e b k k k k k k
 b b b k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k
 k k k k k k k e k
 k k k e k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k e k
 k k k k k k k k k
 k k k e k k k k k
 k k k k k k k k k
]
Output2 = [
 k k k k k k b b b
 k k b b b k b e b
 k k b e b k b b b
 k k b b b k k k k
 k k k k k k b b b
 k k k k k k b e b
 k k b b b k b b b
 k k b e b k k k k
 k k b b b k k k k
] """

# concepts:
# surrounding

# description:
# surround every gray pixel with blue pixels

def transform_4258a5f9(input_grid):
    output_grid = np.zeros_like(input_grid)

    for i in range(len(input_grid)):
        for j in range(len(input_grid[i])):
            if input_grid[i, j] == Color.GRAY:
                # if the current pixel is gray, then we need to surround it with blue
                output_grid[max(0, i-1):min(len(input_grid), i+2), max(0, j-1):min(len(input_grid[i]), j+2)] = Color.BLUE

    # but we need to keep the gray center: so copy over all the gray pixels
    output_grid[input_grid == Color.GRAY] = Color.GRAY
            
    return output_grid


# create a 9x9 grid of black (0) and then sparsely populate it with gray

""" ==============================
Puzzle 444801d8

Train example 1:
Input1 = [
 k k k k k k k k k k
 k b b k b b k k k k
 k b k k k b k k k k
 k b k r k b k k k k
 k b k k k b k k k k
 k b b b b b k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k r r r r r k k k k
 k b b r b b k k k k
 k b r r r b k k k k
 k b r r r b k k k k
 k b r r r b k k k k
 k b b b b b k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k b b k b b k k k k
 k b k r k b k k k k
 k b k k k b k k k k
 k b b b b b k k k k
 k k k k k k k k k k
 k k k k b b k b b k
 k k k k b k g k b k
 k k k k b b b b b k
]
Output2 = [
 k k k k k k k k k k
 k r r r r r k k k k
 k b b r b b k k k k
 k b r r r b k k k k
 k b r r r b k k k k
 k b b b b b k k k k
 k k k k g g g g g k
 k k k k b b g b b k
 k k k k b g g g b k
 k k k k b b b b b k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k b b k b b k k k k
 k b k p k b k k k k
 k b k k k b k k k k
 k b b b b b k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k b b k b b k
 k k k k b k t k b k
 k k k k b b b b b k
]
Output3 = [
 k p p p p p k k k k
 k b b p b b k k k k
 k b p p p b k k k k
 k b p p p b k k k k
 k b b b b b k k k k
 k k k k k k k k k k
 k k k k t t t t t k
 k k k k b b t b b k
 k k k k b t t t b k
 k k k k b b b b b k
] """

# concepts:
# cups, filling

# description:
# In the input you will see several blue "cups", meaning an almost-enclosed shape with a small opening at the top, and empty space (black pixels) inside, as well as a single colored pixel inside.
# To make the output grid, you should fill the interior of each cup with the same color as the colored pixel inside it. 
# Also, put a single layer of colored pixels above the cup with the same color as what's inside.

def transform_444801d8(input_grid):
    # Plan:
    # 1. Detect all the blue cups
    # 2. For each cup, find the mask of what is inside of it
    # 3. Find the color of the single pixel inside the cup
    # 4. Fill the cup with the color
    # 5. Put a single layer of colored pixels above the cup with the same color as what's inside
    
    # Detect all the blue cups
    blue_cups = detect_objects(grid=input_grid, colors=[Color.BLUE], monochromatic=True, connectivity=4)

    output_grid = input_grid.copy()

    # For each cup object...
    for obj in blue_cups:
        # Extract what's inside the cup (as its own object), which is everything in the bounding box that is not the object itself
        cup_x, cup_y, cup_width, cup_height = bounding_box(obj)
        inside_cup_mask = np.zeros_like(input_grid, dtype=bool)
        inside_cup_mask[cup_x:cup_x+cup_width, cup_y:cup_y+cup_height] = True
        inside_cup_mask = inside_cup_mask & (obj != Color.BLUE)
        object_inside_cup = np.where(inside_cup_mask, input_grid, Color.BLACK)        

        # Find the color of the single pixel inside the cup
        colors = object_colors(object_inside_cup, background=Color.BLACK)
        assert len(colors) == 1, "There should be exactly one color inside the cup"
        color = colors[0]

        # Fill the cup with the color
        output_grid[inside_cup_mask] = color

        # Put a single layer of colored pixels above the cup with the same color as what's inside
        top_y = cup_y - 1
        output_grid[cup_x:cup_x+cup_width, top_y] = color

    return output_grid

""" ==============================
Puzzle 44d8ac46

Train example 1:
Input1 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k e e e e k k k k k k k
 k e k k e k k k k k k k
 k e k k e k k e e e e k
 k e e e e k k e k e e k
 k k k k k k k e k k e k
 k k k k k k k e e e e k
 k k e e e e k k k k k k
 k k e e e e k k k k k k
 k k e k e e k k k k k k
 k k e e e e k k k k k k
]
Output1 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k e e e e k k k k k k k
 k e r r e k k k k k k k
 k e r r e k k e e e e k
 k e e e e k k e k e e k
 k k k k k k k e k k e k
 k k k k k k k e e e e k
 k k e e e e k k k k k k
 k k e e e e k k k k k k
 k k e r e e k k k k k k
 k k e e e e k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k
 k e e e e k k k k k k k
 k e e k e k k k k k k k
 k e e e e k e e e e e e
 k e e e e k e k k k k e
 k k k k k k e k k k k e
 k k k k k k e k k k k e
 e e e e e k e k k k k e
 e e e e e k e e e e e e
 e k k e e k k k k k k k
 e k k e e k k k k k k k
 e e e e e k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k
 k e e e e k k k k k k k
 k e e r e k k k k k k k
 k e e e e k e e e e e e
 k e e e e k e r r r r e
 k k k k k k e r r r r e
 k k k k k k e r r r r e
 e e e e e k e r r r r e
 e e e e e k e e e e e e
 e r r e e k k k k k k k
 e r r e e k k k k k k k
 e e e e e k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k
 k e e e e e k k k k k k
 k e k k k e k k k k k k
 k e k k k e k e e e e k
 k e k k k e k e k k e k
 k e k k k e k e k k e k
 k e e e e e k e e e e k
 k k k k k k k k k k k k
 k k k e e e e e e k k k
 k k k e k k k k e k k k
 k k k e k k k k e k k k
 k k k e e e e e e k k k
]
Output3 = [
 k k k k k k k k k k k k
 k e e e e e k k k k k k
 k e k k k e k k k k k k
 k e k k k e k e e e e k
 k e k k k e k e r r e k
 k e k k k e k e r r e k
 k e e e e e k e e e e k
 k k k k k k k k k k k k
 k k k e e e e e e k k k
 k k k e k k k k e k k k
 k k k e k k k k e k k k
 k k k e e e e e e k k k
] """

# concepts:
# filling, topology

# description:
# The input is a black 12x12 grid containing a few grey squares. Each square has a "hole" in it, a contiguous black region of pixels.
# To create the output, fill in the hole of each grey object with red if the hole is a square. Otherwise, leave the hole as is.

def transform_44d8ac46(input_grid):
    # get the grey squares
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # create an output grid to store the result
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # for each grey square, fill in the hole if it is a square
    for obj in objects:
        # to check if the grey object contains a square hole, we can check if the bounding box of the hole is a square.
        # To do so, first crop the object, then find the black hole inside
        sprite = crop(obj, background=Color.BLACK)
        hole_mask = (sprite == Color.BLACK) & (object_interior(sprite, background=Color.BLACK))

        # check if the mask is square
        def is_square(thing):
            """thing can be a mask or a sprite or an object"""
            thing = crop(thing)
            return np.sum(thing != Color.BLACK) == thing.shape[0] * thing.shape[1] and thing.shape[0] == thing.shape[1]
        
        if is_square(hole_mask):
            sprite[hole_mask] = Color.RED

        # get location of object so we can blit the possibly edited sprite back into the grid
        x, y = object_position(obj, background=Color.BLACK)
        blit_sprite(output_grid, sprite, x, y)

    return output_grid

""" ==============================
Puzzle 44f52bb0

Train example 1:
Input1 = [
 r k r
 k r k
 r k r
]
Output1 = [
 b
]

Train example 2:
Input2 = [
 r k k
 r k k
 k r k
]
Output2 = [
 o
]

Train example 3:
Input3 = [
 r k r
 r k r
 r k r
]
Output3 = [
 b
] """

# concepts:
# symmetry detection, boolean indicator

# description:
# In the input you will see a 3x3 grid with red pixels scattered randomly.
# To make the output grid, you should recognize if the input grid has mirror symmetry along the x-axis.
# If the input grid has mirror symmetry along the x-axis, output a 1x1 grid with a blue pixel.
# Otherwise, output a 1x1 grid with an orange pixel.

def transform_44f52bb0(input_grid):
    # Check if the input grid has mirror symmetry along the middle x-axis.
    width, height = input_grid.shape
    middle_x = width // 2
    
    # If the input grid has mirror symmetry along the middle x-axis, output a blue pixel.
    # Otherwise, output an orange pixel.
    if np.all(input_grid[0: middle_x] == input_grid[middle_x + 1:][::-1]):
        output_grid = np.full((1,1), Color.BLUE)
    else:
        output_grid = np.full((1,1), Color.ORANGE)
    
    return output_grid

""" ==============================
Puzzle 469497ad

Train example 1:
Input1 = [
 k k k k g
 k t t k g
 k t t k g
 k k k k g
 g g g g g
]
Output1 = [
 r k k k k k k r g g
 k r k k k k r k g g
 k k t t t t k k g g
 k k t t t t k k g g
 k k t t t t k k g g
 k k t t t t k k g g
 k r k k k k r k g g
 r k k k k k k r g g
 g g g g g g g g g g
 g g g g g g g g g g
]

Train example 2:
Input2 = [
 k k k k o
 y y k k o
 y y k k p
 k k k k p
 o o p p p
]
Output2 = [
 k k k k k k k k r k k k o o o
 k k k k k k k r k k k k o o o
 k k k k k k r k k k k k o o o
 y y y y y y k k k k k k o o o
 y y y y y y k k k k k k o o o
 y y y y y y k k k k k k o o o
 y y y y y y k k k k k k p p p
 y y y y y y k k k k k k p p p
 y y y y y y k k k k k k p p p
 k k k k k k r k k k k k p p p
 k k k k k k k r k k k k p p p
 k k k k k k k k r k k k p p p
 o o o o o o p p p p p p p p p
 o o o o o o p p p p p p p p p
 o o o o o o p p p p p p p p p
]

Train example 3:
Input3 = [
 k k k k m
 k b b k m
 k b b k g
 k k k k g
 m m g g y
]
Output3 = [
 r k k k k k k k k k k k k k k r m m m m
 k r k k k k k k k k k k k k r k m m m m
 k k r k k k k k k k k k k r k k m m m m
 k k k r k k k k k k k k r k k k m m m m
 k k k k b b b b b b b b k k k k m m m m
 k k k k b b b b b b b b k k k k m m m m
 k k k k b b b b b b b b k k k k m m m m
 k k k k b b b b b b b b k k k k m m m m
 k k k k b b b b b b b b k k k k g g g g
 k k k k b b b b b b b b k k k k g g g g
 k k k k b b b b b b b b k k k k g g g g
 k k k k b b b b b b b b k k k k g g g g
 k k k r k k k k k k k k r k k k g g g g
 k k r k k k k k k k k k k r k k g g g g
 k r k k k k k k k k k k k k r k g g g g
 r k k k k k k k k k k k k k k r g g g g
 m m m m m m m m g g g g g g g g y y y y
 m m m m m m m m g g g g g g g g y y y y
 m m m m m m m m g g g g g g g g y y y y
 m m m m m m m m g g g g g g g g y y y y
] """

# concepts:
# counting, resizing

# description:
# In the input, you will see a grid with a row of colored blocks on the bottom and the right. 
# There is also a square in the top left that is not touching the other colors.
# To make the output:
# 1. count the number of colors that aren't black
# 2. enlarge every pixel in the input by a factor of the number of colors
# 3. add diagonal red lines coming out of the corners of the square in the top left portion of the grid

def transform_469497ad(input_grid):
    # count the number of colors that aren't black
    num_colors = len(set(input_grid.flatten())) - 1

    # magnify the pixels in input grid onto the output grid
    output_grid = np.repeat(np.repeat(input_grid, num_colors, axis=0), num_colors, axis=1)

    # find the square in the output grid
    objects = find_connected_components(output_grid, connectivity=8, monochromatic=False)
    for obj in objects:
        # the square is the only object not in the bottom right corner
        if obj[-1,-1] == Color.BLACK:
            square = obj
            break
    
    # find the bounding box of the square
    x, y, w, h = bounding_box(square)

    # draw the diagonal red lines
    draw_line(output_grid, x - 1, y - 1, length=None, color=Color.RED, direction=(-1,-1), stop_at_color=Color.NOT_BLACK)
    draw_line(output_grid, x + w, y + h, length=None, color=Color.RED, direction=(1,1), stop_at_color=Color.NOT_BLACK)
    draw_line(output_grid, x - 1, y + h, length=None, color=Color.RED, direction=(-1,1), stop_at_color=Color.NOT_BLACK)
    draw_line(output_grid, x + w, y - 1, length=None, color=Color.RED, direction=(1,-1), stop_at_color=Color.NOT_BLACK)

    return output_grid

""" ==============================
Puzzle 46f33fce

Train example 1:
Input1 = [
 k k k k k k k k k k
 k r k k k k k k k k
 k k k k k k k k k k
 k y k b k k k k k k
 k k k k k k k k k k
 k k k k k g k k k k
 k k k k k k k k k k
 k k k k k k k y k k
 k k k k k k k k k k
 k k k k k k k k k g
]
Output1 = [
 r r r r k k k k k k k k k k k k k k k k
 r r r r k k k k k k k k k k k k k k k k
 r r r r k k k k k k k k k k k k k k k k
 r r r r k k k k k k k k k k k k k k k k
 y y y y b b b b k k k k k k k k k k k k
 y y y y b b b b k k k k k k k k k k k k
 y y y y b b b b k k k k k k k k k k k k
 y y y y b b b b k k k k k k k k k k k k
 k k k k k k k k g g g g k k k k k k k k
 k k k k k k k k g g g g k k k k k k k k
 k k k k k k k k g g g g k k k k k k k k
 k k k k k k k k g g g g k k k k k k k k
 k k k k k k k k k k k k y y y y k k k k
 k k k k k k k k k k k k y y y y k k k k
 k k k k k k k k k k k k y y y y k k k k
 k k k k k k k k k k k k y y y y k k k k
 k k k k k k k k k k k k k k k k g g g g
 k k k k k k k k k k k k k k k k g g g g
 k k k k k k k k k k k k k k k k g g g g
 k k k k k k k k k k k k k k k k g g g g
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k b k g k k k k k k
 k k k k k k k k k k
 k k k y k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k t
 k k k k k k k k k k
 k k k k k k k r k r
]
Output2 = [
 b b b b g g g g k k k k k k k k k k k k
 b b b b g g g g k k k k k k k k k k k k
 b b b b g g g g k k k k k k k k k k k k
 b b b b g g g g k k k k k k k k k k k k
 k k k k y y y y k k k k k k k k k k k k
 k k k k y y y y k k k k k k k k k k k k
 k k k k y y y y k k k k k k k k k k k k
 k k k k y y y y k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k t t t t
 k k k k k k k k k k k k k k k k t t t t
 k k k k k k k k k k k k k k k k t t t t
 k k k k k k k k k k k k k k k k t t t t
 k k k k k k k k k k k k r r r r r r r r
 k k k k k k k k k k k k r r r r r r r r
 k k k k k k k k k k k k r r r r r r r r
 k k k k k k k k k k k k r r r r r r r r
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k g k r k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k b k b k k k k k y
]
Output3 = [
 g g g g r r r r k k k k k k k k k k k k
 g g g g r r r r k k k k k k k k k k k k
 g g g g r r r r k k k k k k k k k k k k
 g g g g r r r r k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 b b b b b b b b k k k k k k k k y y y y
 b b b b b b b b k k k k k k k k y y y y
 b b b b b b b b k k k k k k k k y y y y
 b b b b b b b b k k k k k k k k y y y y
] """

# concepts:
# scaling

# description:
# In the input you will see a grid with different colors of pixels scattered on the grid
# To make the output grid, you should first only scale the pixels by 2 times,
# then scale in whole grid 2 times.

def transform_46f33fce(input_grid):
    # Plan:
    # 1. Detect all the colored pixels
    # 2. Rescale each such sprite
    # 3. Blit the rescaled sprite onto the output grid, taking care to anchor it correctly
    # 4. Rescale the output grid (2x)

    # Detect all the colored pixels in the input grid
    pixel_objects = detect_objects(grid=input_grid, colors=Color.NOT_BLACK,
                            # These are single pixels, so they are 1x1
                            allowed_dimensions=[(1, 1)],
                            monochromatic=True, connectivity=4)

    # Initialize the output grid with the same size as the input grid
    output_grid = np.copy(input_grid)

    scale_factor = 2
    for obj in pixel_objects:
        # Get the position of each colored pixel, and crop it to produce a sprite
        x, y = object_position(obj, background=Color.BLACK, anchor="upper left")
        single_pixel_sprite = crop(obj, background=Color.BLACK)

        # Scale the sprite by `scale_factor` times
        scaled_sprite = scale_sprite(single_pixel_sprite, scale_factor)

        # The coordinate of the scaled pattern (anchored at the upper left)
        new_x, new_y = x - scale_factor + 1, y - scale_factor + 1

        # Put the scaled pattern on the output grid
        output_grid = blit_sprite(grid=output_grid, x=new_x, y=new_y, sprite=scaled_sprite, background=Color.BLACK)
    
    # Scale the whole grid by scale_factor times
    output_grid = scale_sprite(output_grid, scale_factor)

    return output_grid

""" ==============================
Puzzle 48d8fb45

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k b b
 k k k e k k k b b k
 k k k b k k k k b k
 k k b b b k k k k k
 k k k b b k k k k k
 k k k k k k k k k k
 k k k k k k b b k k
 k k k k k b b b k k
 k k k k k k b b k k
]
Output1 = [
 k b k
 b b b
 k b b
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k e k k
 k k k k k k y y k k
 k k y k k k k k y k
 k y k y k k k y k k
 k k y y k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 y y k
 k k y
 k y k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k r r k k k k k k
 k r k r k k k k k k
 k k r k k k k e k k
 k k k k k k k r r k
 k k k k k k r r k k
 k k k k k k k r k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output3 = [
 k r r
 r r k
 k r k
] """

# concepts:
# object extraction, contact, cropping

# description:
# In the input you will see several objects with same color placed in a 10x10 grid, only one of 
# them contact a gray pixel.
# To make the output grid, you should select the object contact the gray pixel, crop it, and then output it.

def transform_48d8fb45(input_grid):
    # Get the color of the pattern
    pattern_color = [color for color in np.unique(input_grid) if color != Color.BLACK and color != Color.GRAY][0]

    # Detect all the patterns with pattern color in the input grid
    pattern_list = detect_objects(grid=input_grid, colors=[pattern_color], connectivity=8, monochromatic=True)

    # Detect the indicator gray pixel
    gray_pixel = detect_objects(grid=input_grid, colors=[Color.GRAY], connectivity=8, monochromatic=True)[0]

    # Find out which pattern has contact the gray pixel
    for pattern in pattern_list:
        cropped_pattern = crop(grid=pattern)
        # Check if the gray pixel contact the pattern
        if contact(object1=pattern, object2=gray_pixel, connectivity=4):
            # Crop the pattern and output it
            output_grid = cropped_pattern
            break

    return output_grid

""" ==============================
Puzzle 4c5c2cf0

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k r k k r k k k k k k k
 k k k r r k r k k k k k k k
 k k k k k r r k k k k k k k
 k k k k r r k k k k k k k k
 k k k r k k y k y k k k k k
 k k k k k k k y k k k k k k
 k k k k k k y k y k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k r k k r k r k k r k k
 k k k r r k r k r k r r k k
 k k k k k r r k r r k k k k
 k k k k r r k k k r r k k k
 k k k r k k y k y k k r k k
 k k k k k k k y k k k k k k
 k k k r k k y k y k k r k k
 k k k k r r k k k r r k k k
 k k k k k r r k r r k k k k
 k k k r r k r k r k r r k k
 k k k r k k r k r k k r k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k t k k k k k
 k k k k k k k t t t k k k k
 k k k k k k t t t k k k k k
 k k k k g k g k k k k k k k
 k k k k k g k k k k k k k k
 k k k k g k g k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k t k k k k k t k k k k k
 k t t t k k k t t t k k k k
 k k t t t k t t t k k k k k
 k k k k g k g k k k k k k k
 k k k k k g k k k k k k k k
 k k k k g k g k k k k k k k
 k k t t t k t t t k k k k k
 k t t t k k k t t t k k k k
 k k t k k k k k t k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k t k t k k k k k
 k k k k k t k k k k k k
 k k k k t k t k k k k k
 k k b b k k k k k k k k
 k b k b k k k k k k k k
 k k b k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k b k k k k k b k k k
 k b k b k k k b k b k k
 k k b b k k k b b k k k
 k k k k t k t k k k k k
 k k k k k t k k k k k k
 k k k k t k t k k k k k
 k k b b k k k b b k k k
 k b k b k k k b k b k k
 k k b k k k k k b k k k
] """

# concepts:
# symmetry, reflection

# description:
# In the input you will see a small monochromatic pattern with reflectional symmetry (horizontal and vertical), as well as another monochromatic object in one of its quadrants.
# To make the output, reflect the other object across the axes of reflectional symmetry of the small pattern.

def transform_4c5c2cf0(input_grid):
    # Plan:
    # 1. Detect the pair of objects and divide them by whether or not they are already symmetric
    # 2. Reflect the non-symmetric object across the axes of symmetry of the symmetric object

    # 1. Object detection and setup
    objects = find_connected_components(input_grid, connectivity=8, background=Color.BLACK)
    assert len(objects) == 2

    # Find the object that is symmetric
    symmetric_object = None
    for obj in objects:
        # Detect all symmetry
        # There is no occlusion so and don't ignore any colors
        # But we know the background is black
        symmetries = detect_mirror_symmetry(obj, ignore_colors=[], background=Color.BLACK)

        # actually finds three symmetries: horizontal, vertical, and diagonal mirroring
        if len(symmetries) >= 2:
            symmetric_object = obj
            break
    assert symmetric_object is not None, "There should be a symmetric object"

    # Find the object that is not symmetric
    non_symmetric_object = next(obj for obj in objects if obj is not symmetric_object)

    # 2. Reflect the non-symmetric object across the axes of symmetry of the symmetric object
    output_grid = input_grid.copy()

    for x, y in np.argwhere(non_symmetric_object != Color.BLACK):
        original_color = non_symmetric_object[x, y]
        for transformed_x, transformed_y in orbit(output_grid, x, y, symmetries=symmetries):
            output_grid[transformed_x, transformed_y] = original_color
    
    return output_grid

""" ==============================
Puzzle 508bd3b6

Train example 1:
Input1 = [
 k k k k k k k k k k r r
 k k k k k k k k k k r r
 k k k k k k k k k k r r
 k k k k k k k k k k r r
 k k k k k k k k k k r r
 k k k k k k k k k k r r
 k k k k k k k k k k r r
 k k k k k k k k k k r r
 k k k k k k k k k k r r
 k k k k k k k k k k r r
 k k k t k k k k k k r r
 k k t k k k k k k k r r
]
Output1 = [
 k k k k k g k k k k r r
 k k k k k k g k k k r r
 k k k k k k k g k k r r
 k k k k k k k k g k r r
 k k k k k k k k k g r r
 k k k k k k k k g k r r
 k k k k k k k g k k r r
 k k k k k k g k k k r r
 k k k k k g k k k k r r
 k k k k g k k k k k r r
 k k k t k k k k k k r r
 k k t k k k k k k k r r
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 t k k k k k k k k k k k
 k t k k k k k k k k k k
 k k t k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 r r r r r r r r r r r r
 r r r r r r r r r r r r
 r r r r r r r r r r r r
]
Output2 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 t k k k k k k k k k k k
 k t k k k k k k k k k g
 k k t k k k k k k k g k
 k k k g k k k k k g k k
 k k k k g k k k g k k k
 k k k k k g k g k k k k
 k k k k k k g k k k k k
 r r r r r r r r r r r r
 r r r r r r r r r r r r
 r r r r r r r r r r r r
]

Train example 3:
Input3 = [
 r r k k k k k k k k k k
 r r k k k k k k k k k k
 r r k k k k k k k k k k
 r r k k k k k k k k k k
 r r k k k k k k k k k k
 r r k k k k k k k k k k
 r r k k k k k k k k k k
 r r k k k k k k k k k k
 r r k k k k k k k k k k
 r r k k k k t k k k k k
 r r k k k k k t k k k k
 r r k k k k k k t k k k
]
Output3 = [
 r r k k k k k g k k k k
 r r k k k k g k k k k k
 r r k k k g k k k k k k
 r r k k g k k k k k k k
 r r k g k k k k k k k k
 r r g k k k k k k k k k
 r r k g k k k k k k k k
 r r k k g k k k k k k k
 r r k k k g k k k k k k
 r r k k k k t k k k k k
 r r k k k k k t k k k k
 r r k k k k k k t k k k
] """

# concepts:
# bouncing

# description:
# In the input you will see a short diagonal teal line pointing at a red rectangle on a black background.
# To make the output, shoot outward from the teal line, but change the color to green. Reflect off the red rectangle.

def transform_508bd3b6(input_grid):
    # Plan:
    # 1. Detect the objects
    # 2. Determine the orientation of the teal line, and its endpoints
    # 3. Shoot a green line outward until it hits the red rectangle
    # 4. Reflect the green line off the red rectangle, continuing in green

    teal_line = detect_objects(input_grid, colors=[Color.TEAL], monochromatic=True, connectivity=8)
    assert len(teal_line) == 1, "There should be exactly one teal line"
    teal_line = list(teal_line)[0]
    red_rectangle = detect_objects(input_grid, colors=[Color.RED], monochromatic=True, connectivity=8)
    assert len(red_rectangle) == 1, "There should be exactly one red rectangle"
    red_rectangle = list(red_rectangle)[0]

    output_grid = input_grid.copy()

    # To get the orientation of a line, find the endpoints and compare their x and y coordinates
    x1, y1 = max( (x, y) for x, y in np.argwhere(teal_line == Color.TEAL) )
    x2, y2 = min( (x, y) for x, y in np.argwhere(teal_line == Color.TEAL) )
    direction12 = (int(np.sign(x2 - x1)), int(np.sign(y2 - y1)))
    direction21 = (-direction12[0], -direction12[1])

    # Try both (direction, x2, y2) and (-direction, x1, y1) as starting points
    for (dx,dy), start_x, start_y in [ (direction12, x2, y2), (direction21, x1, y1) ]:
        start_x += dx
        start_y += dy
        # Loop, shooting lines off of red things, until we run out of the canvas
        while 0 <= start_x < input_grid.shape[0] and 0 <= start_y < input_grid.shape[1]:
            stop_x, stop_y = draw_line(output_grid, start_x, start_y, direction=(dx,dy), color=Color.GREEN, stop_at_color=[Color.RED])

            # reflection geometry depends on if we hit the red rectangle on our left/right/up/down
            # did we hit the red rectangle on our right? 
            if stop_x+1 < output_grid.shape[0] and output_grid[stop_x+1, stop_y] != Color.BLACK:
                dx = -dx
            # did we hit the red rectangle on our left?
            elif stop_x-1 >= 0 and output_grid[stop_x-1, stop_y] != Color.BLACK:
                dx = -dx
            # did we hit the red rectangle on our bottom?
            elif stop_y+1 < output_grid.shape[1] and output_grid[stop_x, stop_y+1] != Color.BLACK:
                dy = -dy
            # did we hit the red rectangle on our top?
            elif stop_y-1 >= 0 and output_grid[stop_x, stop_y-1] != Color.BLACK:
                dy = -dy
            else:
                # didn't do any reflections, so stop
                break

            start_x, start_y = stop_x + dx, stop_y + dy
    
    return output_grid

""" ==============================
Puzzle 5168d44c

Train example 1:
Input1 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 r r r k k k k k k k k k k
 r g r g k g k g k g k g k
 r r r k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k r r r k k k k k k k k
 k g r g r g k g k g k g k
 k k r r r k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k g k k
 k k k k k k k
 k k k k g k k
 k k k r r r k
 k k k r g r k
 k k k r r r k
 k k k k g k k
 k k k k k k k
 k k k k g k k
 k k k k k k k
 k k k k g k k
 k k k k k k k
 k k k k g k k
]
Output2 = [
 k k k k g k k
 k k k k k k k
 k k k k g k k
 k k k k k k k
 k k k k g k k
 k k k r r r k
 k k k r g r k
 k k k r r r k
 k k k k g k k
 k k k k k k k
 k k k k g k k
 k k k k k k k
 k k k k g k k
]

Train example 3:
Input3 = [
 k k g k k k k
 k r r r k k k
 k r g r k k k
 k r r r k k k
 k k g k k k k
 k k k k k k k
 k k g k k k k
]
Output3 = [
 k k g k k k k
 k k k k k k k
 k k g k k k k
 k r r r k k k
 k r g r k k k
 k r r r k k k
 k k g k k k k
] """

# concepts:
# collision, translation

# description:
# In the input you will see a red object overlaid on a track of green dots.
# To make the output, move the red object one green dot to the right (if the track is horizontal) or one green dot down (if the track is vertical).

def transform_5168d44c(input_grid):
    # Plan:
    # 1. Detect the objects
    # 2. Determine the orientation of the track of green dots
    # 3. Move in the appropriate direction until it perfectly fits over the next green dot, meaning there are no collisions

    objects = find_connected_components(input_grid, connectivity=8, background=Color.BLACK, monochromatic=True)

    red_objects = [ obj for obj in objects if Color.RED in object_colors(obj, background=Color.BLACK) ]
    green_objects = [ obj for obj in objects if Color.GREEN in object_colors(obj, background=Color.BLACK) ]

    assert len(red_objects) == 1, "There should be exactly one red object"
    assert len(green_objects) >= 1, "There should be at least one green object"

    red_object = red_objects[0]

    # Determine the orientation of the track of green dots by comparing the positions of two dots
    x1,y1 = min( object_position(obj, anchor="center") for obj in green_objects )
    x2,y2 = max( object_position(obj, anchor="center") for obj in green_objects )
    if x1 == x2:
        # vertical track
        dx, dy = 0, 1
    elif y1 == y2:
        # horizontal track
        dx, dy = 1, 0
    
    # Make the output grid: Start with all the greens, then put the red in the right spot by moving it one-by-one
    output_grid = np.full_like(input_grid, Color.BLACK)
    for green_object in green_objects:
        blit_object(output_grid, green_object)

    for distance in range(1, 100):
        translated_red_object = translate(red_object, dx*distance, dy*distance)
        if not collision(object1=translated_red_object, object2=output_grid):
            blit_object(output_grid, translated_red_object)
            break

    return output_grid

""" ==============================
Puzzle 54d82841

Train example 1:
Input1 = [
 k p p p k k k k
 k p k p k k k k
 k k k k k p p p
 k k k k k p k p
 k k k k k k k k
 k k k k k k k k
 k k k k k k k k
 k k k k k k k k
]
Output1 = [
 k p p p k k k k
 k p k p k k k k
 k k k k k p p p
 k k k k k p k p
 k k k k k k k k
 k k k k k k k k
 k k k k k k k k
 k k y k k k y k
]

Train example 2:
Input2 = [
 k g g g k
 k g k g k
 k k k k k
 k k k k k
 k k k k k
]
Output2 = [
 k g g g k
 k g k g k
 k k k k k
 k k k k k
 k k y k k
]

Train example 3:
Input3 = [
 k k k k k k k
 k t t t k k k
 k t k t p p p
 k k k k p k p
 k k k k k k k
]
Output3 = [
 k k k k k k k
 k t t t k k k
 k t k t p p p
 k k k k p k p
 k k y k k y k
] """

# concepts:
# gravity, falling

# description:
# In the input you will see various monochromatic objects
# To make the output, make each object drop a single yellow pixel below it, centered with the middle of the object

def transform_54d82841(input_grid):
    # Plan:
    # 1. Detect the objects
    # 2. Drop yellow pixels which land in the final row of the grid, centered with the middle of the object

    objects = find_connected_components(input_grid, connectivity=4, background=Color.BLACK, monochromatic=True)

    output_grid = input_grid.copy()

    for obj in objects:
        x, y = object_position(obj, background=Color.BLACK, anchor='center')
        bottom_y = output_grid.shape[1] - 1
        output_grid[x, bottom_y] = Color.YELLOW
    
    return output_grid

""" ==============================
Puzzle 56dc2b01

Train example 1:
Input1 = [
 k g k k k k k k k k r k k k k k
 k g g g k k k k k k r k k k k k
 g g k k k k k k k k r k k k k k
 k g g g k k k k k k r k k k k k
]
Output1 = [
 k k k k k t k g k k r k k k k k
 k k k k k t k g g g r k k k k k
 k k k k k t g g k k r k k k k k
 k k k k k t k g g g r k k k k k
]

Train example 2:
Input2 = [
 k k k k k
 g g k k k
 g k k k k
 g g k g g
 k g g g k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 r r r r r
 k k k k k
]
Output2 = [
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 t t t t t
 g g k k k
 g k k k k
 g g k g g
 k g g g k
 r r r r r
 k k k k k
]

Train example 3:
Input3 = [
 k k k k k
 k k k k k
 k k k k k
 r r r r r
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 g g g g k
 g k k g k
 g g k g k
 k k k k k
 k k k k k
 k k k k k
]
Output3 = [
 k k k k k
 k k k k k
 k k k k k
 r r r r r
 g g g g k
 g k k g k
 g g k g k
 t t t t t
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
 k k k k k
] """

# concepts:
# attraction, magnetism, translation5

# description:
# In the input you will see a green object and a red bar
# To make the output, move the green object to touch the red bar. Finally put a teal bar on the other side of the green object.

def transform_56dc2b01(input_grid):
    # Plan:
    # 1. Detect the objects; separate the green thing from the red bar
    # 2. Move the green object to touch the red bar
    # 3. Add a teal bar on the other side of the green object

    # 1. Object detection and setup
    objects = find_connected_components(input_grid, connectivity=4, background=Color.BLACK, monochromatic=True)

    red_objects = [ obj for obj in objects if Color.RED in object_colors(obj, background=Color.BLACK) ]
    green_objects = [ obj for obj in objects if Color.GREEN in object_colors(obj, background=Color.BLACK) ]

    assert len(red_objects) == 1, "There should be exactly one red object"
    assert len(green_objects) == 1, "There should be exactly one green object"
    
    red_object = red_objects[0]
    green_object = green_objects[0]

    # Make the output grid: Start with the red object, then add the green object and the teal bar
    output_grid = np.full_like(input_grid, Color.BLACK)
    blit_object(output_grid, red_object)

    # 2. Move the green object to touch the red bar
    # First calculate what direction we have to move in order to contact the grey object
    # Consider all displacements, starting with the smallest translations first
    possible_displacements = [ (i*dx, i*dy) for i in range(0, 30) for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)] ]

    # Only keep the displacements that cause a contact between the colored object and the grey object
    valid_displacements = [ displacement for displacement in possible_displacements
                            if contact(object1=translate(green_object, *displacement), object2=red_object) ]
    assert valid_displacements, "There should be at least one valid displacement"

    # Pick the smallest valid displacement
    displacement = min(valid_displacements, key=lambda displacement: sum(abs(x) for x in displacement))

    # Extract the direction from the displacement
    direction = np.sign(displacement, dtype=int)

    # Translate and draw on the canvas
    green_object = translate(green_object, *displacement)
    blit_object(output_grid, green_object)

    # 3. Add a teal bar on the other side of the green object
    # It should be the same shape as the red bar, but teal
    # To place it correctly, it needs to be on the other side so we go in the opposite direction that the green object moved
    teal_object = red_object.copy()
    teal_object[teal_object != Color.BLACK] = Color.TEAL
    opposite_direction = -direction
    # Move the teal object until it doesn't collide with anything
    while collision(object1=teal_object, object2=output_grid):
        teal_object = translate(teal_object, *opposite_direction)
    # Draw the teal object on the canvas
    blit_object(output_grid, teal_object)
    
    return output_grid

""" ==============================
Puzzle 57aa92db

Train example 1:
Input1 = [
 k k k k k k k k k k k k
 k k k g k k k k k k k k
 k k g g b k k k k k k k
 k k k g k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k y y b b k k k
 k k k k k y y b b k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k
 k k k g k k k k k k k k
 k k g g b k k k k k k k
 k k k g k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k y y k k k k k
 k k k k k y y k k k k k
 k k k y y y y b b k k k
 k k k y y y y b b k k k
 k k k k k y y k k k k k
 k k k k k y y k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k k k
 k k k k k t k k k k k k k k k k k k
 k k r t t t k k k k k k k k k k k k
 k k k k k t k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k r p k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k r g k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k k k k k k
 k k k k k t k k k k k k k k k k k k
 k k r t t t k k k k k k k k k k k k
 k k k k k t k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k p k k k k
 k k k k k k k k k k r p p p k k k k
 k k k k k k k k k k k k k p k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k g k k k k k k k k k
 k k k k k r g g g k k k k k k k k k
 k k k k k k k k g k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k b b b k k k k k k k k k k k k k
 k k b y b k k k k k k k k k k k k k
 k k b k b k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k t t t k k k k k k
 k k k k k k k k k t t t k k k k k k
 k k k k k k k k k t t t k k k k k k
 k k k k k k k k k y y y k k k k k k
 k k k k k k k k k y y y k k k k k k
 k k k k k k k k k y y y k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k b b b k k k k k k k k k k k k k
 k k b y b k k k k k k k k k k k k k
 k k b k b k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k t t t t t t t t t k k k
 k k k k k k t t t t t t t t t k k k
 k k k k k k t t t t t t t t t k k k
 k k k k k k t t t y y y t t t k k k
 k k k k k k t t t y y y t t t k k k
 k k k k k k t t t y y y t t t k k k
 k k k k k k t t t k k k t t t k k k
 k k k k k k t t t k k k t t t k k k
 k k k k k k t t t k k k t t t k k k
 k k k k k k k k k k k k k k k k k k
] """

# concepts:
# scaling, puzzle pieces, indicator pixels

# description:
# In the input you will see objects with exactly 2 colors, each with one pixel/rectangle of a different color as an indicator. The indicator color is the same across objects.
# To make the output, one of those objects is a template shape that you are going to translate/recolor/rescale to match indicators with each other object.
# Place the rescaled template on top of the other shape so that the indicators are at the same position, and change color to match what you are placing on top of.

def transform_57aa92db(input_grid):
    # Plan:
    # 1. Parse the input into template object and other objects, and determine the indicator color
    # 2. For each other object, rescale+recolor the template to match indicators

    # 1. Parse the input

    # Extract all the objects from the input grid
    background = Color.BLACK
    objects = find_connected_components(input_grid, background=background, connectivity=8, monochromatic=False)

    # The indicator pixel's color appears in all the objects
    possible_indicator_colors = [ color for color in Color.ALL_COLORS
                                 if all( color in object_colors(obj, background=background) for obj in objects)]
    assert len(possible_indicator_colors) == 1, "There should be exactly one indicator color"
    indicator_color = possible_indicator_colors[0]

    # Find the template object, which is the biggest object after you scale the indicator down to have size 1x1
    object_sizes = [ np.sum(obj != background) for obj in objects]
    indicator_sizes = [ np.sum(obj == indicator_color) for obj in objects]
    rescaled_sizes = [size // indicator_size for size, indicator_size in zip(object_sizes, indicator_sizes)]
    template_index = np.argmax(rescaled_sizes)
    template_object = objects[template_index]
    other_objects = [obj for i, obj in enumerate(objects) if i != template_index]

    template_sprite = crop(template_object, background=background)

    # 2. For each other object, rescale+recolor the template to match indicators
    # Determine the scaling factor by the ratio of the size of the indicator pixel region
    # Determine the color according to the non-indicator color of the object
    # Determine the position so that indicator pixels are overlaid

    # To produce the output we draw on top of the input
    output_grid = input_grid.copy()

    for other_object in other_objects:

        # Find the new shape's color
        new_color = [ color for color in object_colors(other_object, background=background) if color != indicator_color][0]

        # find the new scale, which is the ratio of the size of the indicator pixel in the original shape to the size of the indicator pixel in the new shape
        new_scale = crop(other_object == indicator_color).shape[0] // crop(template_object == indicator_color).shape[0]

        # Scale the original template to the same scale...
        template_sprite_scaled = scale_sprite(template_sprite, new_scale)
        # ...and change its color to the new shape's color
        template_sprite_scaled[(template_sprite_scaled != background) & (template_sprite_scaled != indicator_color)] = new_color

        # Overlay the indicator pixels from the scaled/recolored template sprite with the indicator pixels from the other object
        x = np.min(np.argwhere(other_object == indicator_color)[:,0]) - np.min(np.argwhere(template_sprite_scaled == indicator_color)[:,0])
        y = np.min(np.argwhere(other_object == indicator_color)[:,1]) - np.min(np.argwhere(template_sprite_scaled == indicator_color)[:,1])
        blit_sprite(output_grid, template_sprite_scaled, x=x, y=y)
        
    return output_grid

""" ==============================
Puzzle 5c2c9af4 """

# concepts:
# repeating pattern, connecting colors

# description:
# In the input grid, you will see an all black grid with three dots of the same color in a perfect 45 degree diagonal, but equally spaced apart from each other.
# To create the output grid, connect the outer two of the three dots with a square border shape. The square border contains the two dots as corners, and is centered on the third center dot. Then make another square border that is the same distance (number of background cells) from the existing border as the existing border is from the center dot. Repeat making square borders of the same distance outwards until the grid is filled.

def transform_5c2c9af4(input_grid):
    # Plan:
    # 1. get the dots
    # 2. get the center dot, and the two outer dots
    # 3. calculate the distance from the center dot of the outer dots.
    # 4. make a helper function for drawing a square of a certain distance from the center dot
    # 5. repeat making squares of multiples of that distance until no new cells are filled in on the grid.

    # get a list of locations
    pixel_xs, pixel_ys = np.where(input_grid != Color.BLACK)
    pixel_locations = list(zip(list(pixel_xs), list(pixel_ys)))
    assert len(pixel_locations) == 3
    
    # sort by x coordinate
    pixel0, pixel1, pixel2 = sorted(pixel_locations, key=lambda l: l[0])
    color = input_grid[pixel0[0], pixel0[1]]
    width = pixel1[0] - pixel0[0]

    def in_bounds(grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def draw_square_border(grid, x, y, w, h, color):
        # x, y is the top left corner
        for dx in range(w+1):
            # top border
            if in_bounds(grid, x+dx, y):
                grid[x+dx, y] = color
            # bottom border
            if in_bounds(grid, x+dx, y+h):
                grid[x+dx, y+h] = color
        for dy in range(h+1):
            # left border
            if in_bounds(grid, x, y+dy):
                grid[x, y+dy] = color
            # right border
            if in_bounds(grid, x+w, y+dy):
                grid[x+w, y+dy] = color

    output_grid = input_grid.copy()
    i = 1
    while True:
        top_left_x = pixel1[0] - width * i
        top_left_y = pixel1[1] - width * i
        w = 2 * (width * i)
        old_grid = output_grid.copy()
        draw_square_border(output_grid, top_left_x, top_left_y, w, w, color)
        if not np.any(old_grid != output_grid):
            break
        i += 1

    return output_grid

""" ==============================
Puzzle 5daaa586

Train example 1:
Input1 = [
 k k k k k g k k k r k k k k k k t k k k k k
 b b b b b g b b b b b b b b b b t b b b b b
 k k k k k g k k k k k k k k k k t k k k k k
 k k k k k g k k k k k k k k k k t k k k k k
 r k k k k g k k k k k k k k k k t k k k k k
 r k k k k g k k k k k k k k k r t k k k k k
 k k k r k g k k k k k k k k k k t k r k k k
 k k k k k g k k k r k k k k k k t k k r k k
 k k k k k g k k k k k k k k r k t k k k k k
 k k k k k g k k k k k k k k k k t k k k k k
 k k k k k g k k k k k k k k k k t k k k k k
 k k k k k g k k k k k k k k k k t k k k k k
 k k k r k g k r k k k k k k r k t k k k k k
 k k k k k g k k k k k k k k k k t k k k k k
 k k k k k g r k k k k k k k k k t k k k k k
 r r r r r g r r r r r r r r r r t r r r r r
 k k k k k g k k r k k k k k k k t k k k k k
 r k k k k g k k k k k k k k k k t k k k k k
 k k k k k g k k k k k k k k k k t k k k k k
 k k k k k g k k k k k k k k k k t k k k k k
 k k k k k g k k k k k k k k k k t k k k k k
 r k k k k g k k k k k k k k k k t k k k r k
]
Output1 = [
 g b b b b b b b b b b t
 g k k k k k k k k k k t
 g k k k k k k k k k k t
 g k k k k k k k k k k t
 g k k k k k k k k k r t
 g k k k k k k k k k r t
 g k k k r k k k k k r t
 g k k k r k k k k r r t
 g k k k r k k k k r r t
 g k k k r k k k k r r t
 g k k k r k k k k r r t
 g k r k r k k k k r r t
 g k r k r k k k k r r t
 g r r k r k k k k r r t
 g r r r r r r r r r r t
]

Train example 2:
Input2 = [
 k k y k k k k k k b k k
 k k y k k k k k k b k k
 t t y t t t t t t b t t
 k k y k k k k k k b k k
 k k y k k k k k k b k k
 k k y k k k t k k b k t
 k k y t k k t k k b k k
 k k y k k k k k k b k k
 k k y k k k k t k b k t
 p p p p p p p p p b p p
 k k y k k k t k k b k k
 k t y k k k k t k b k k
]
Output2 = [
 y t t t t t t b
 y t k k t t k b
 y t k k t t k b
 y t k k t t k b
 y t k k t t k b
 y k k k k t k b
 y k k k k t k b
 p p p p p p p b
]

Train example 3:
Input3 = [
 k k y g k k k y k k k y k k k
 k k k g k y k k k k k y k k k
 k k k g k k y k k k k y k k k
 k k k g k y k k k k k y k k k
 k k k g k k k k k k y y y k y
 r r r g r r r r r r r y r r r
 y k k g y y k y k k k y k k k
 k k k g k k k k k k k y k k k
 y k k g k k k k y k y y k k k
 y k k g k k y k k k y y k k k
 t t t g t t t t t t t y t t t
 k k k g k k k k k k y y k k y
 k k k g y k k y k k k y k k k
 k k y g k k k k k y k y k k k
]
Output3 = [
 g r r r r r r r y
 g y y y y y y y y
 g k k k k k k k y
 g k k k k y y y y
 g k k y y y y y y
 g t t t t t t t y
] """

# concepts:
# pattern extraction, pixel expanding

# description:
# In the input you will see four lines of different colors intersecting and forming a rectangle.
# Few pixels of one specific line's color are scattered in the grid.
# To make the output, you should cropped out the rectangle and extend the scatterd pixels to 
# the specific line which has same color as the scattered pixels.

def transform_5daaa586(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    
    # Identify two vertical and two horizontal lines
    horizontal_lines = np.where(np.all(output_grid != Color.BLACK, axis=1))[0]
    vertical_lines = np.where(np.all(output_grid != Color.BLACK, axis=0))[0]
    
    # Find out the color of the scattered pixels
    mask = np.ones(output_grid.shape, dtype=bool)
    mask[horizontal_lines, :] = False
    mask[:, vertical_lines] = False
    color = output_grid[mask & (output_grid != Color.BLACK)][0]
    
    # Crop the grid by the rectangle formed by the four lines
    output_grid = output_grid[horizontal_lines[0] : horizontal_lines[1] + 1, vertical_lines[0] : vertical_lines[1] + 1]
    
    # Extend the scattered pixels to the line that has same color as the scattered pixels
    for x in range(len(output_grid)):
        for y in range(len(output_grid[0])):
            # did we find a scattered pixel? (of color `color`)
            if output_grid[x, y] == color:
                # draw a line to the matching color line, which is going to be either left/right/top/bottom
                # so we need to examine four cases for each location that the matching color line might be

                # Left: x=0 indicates this is the left line
                if output_grid[0, y] == color:
                    draw_line(output_grid, x, y, end_x = 0, end_y = y, color = color)
                # Right: x=len(output_grid) - 1 indicates this is the right line
                if output_grid[len(output_grid) - 1, y] == color:
                    draw_line(output_grid, x, y, end_x = len(output_grid) - 1, end_y = y, color = color)
                # Top: y=0 indicates this is the top line
                if output_grid[x, 0] == color:
                    draw_line(output_grid, x, y, end_x = x, end_y = 0, color = color)
                # Bottom: y=len(output_grid[0]) - 1 indicates this is the bottom line
                if output_grid[x, len(output_grid[0]) - 1] == color:
                    draw_line(output_grid, x, y, end_x = x, end_y = len(output_grid[0]) - 1, color = color)

    return output_grid

""" ==============================
Puzzle 623ea044

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output1 = [
 r k k k k k r k k k k k k k k
 k r k k k r k k k k k k k k k
 k k r k r k k k k k k k k k k
 k k k r k k k k k k k k k k k
 k k r k r k k k k k k k k k k
 k r k k k r k k k k k k k k k
 r k k k k k r k k k k k k k k
 k k k k k k k r k k k k k k k
 k k k k k k k k r k k k k k k
 k k k k k k k k k r k k k k k
 k k k k k k k k k k r k k k k
 k k k k k k k k k k k r k k k
 k k k k k k k k k k k k r k k
 k k k k k k k k k k k k k r k
 k k k k k k k k k k k k k k r
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k o k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k o k k k k k k k k
 k k k k k k k o k k k k k k k
 k k k k k k k k o k k k k k o
 k k k k k k k k k o k k k o k
 k k k k k k k k k k o k o k k
 k k k k k k k k k k k o k k k
 k k k k k k k k k k o k o k k
 k k k k k k k k k o k k k o k
 k k k k k k k k o k k k k k o
 k k k k k k k o k k k k k k k
 k k k k k k o k k k k k k k k
 k k k k k o k k k k k k k k k
 k k k k o k k k k k k k k k k
 k k k o k k k k k k k k k k k
 k k o k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k
 k k k k k k k
 k k k k k k k
 k k t k k k k
 k k k k k k k
 k k k k k k k
 k k k k k k k
]
Output3 = [
 k k k k k t k
 t k k k t k k
 k t k t k k k
 k k t k k k k
 k t k t k k k
 t k k k t k k
 k k k k k t k
] """

# concepts:
# diagonal lines

# description:
# In the input you will see one colored pixel on a black background.
# To make the output, make two diagonal lines that are the same color as the colored pixel and intersect at the location of the colored pixel.

def transform_623ea044(input_grid):
    # make output grid
    output_grid = np.copy(input_grid)

    # get the index of the colored pixel
    x, y, width, height = bounding_box(input_grid != Color.BLACK)
    
    # get color from colored pixel
    color = input_grid[x,y]

    # draw diagonals
    # first diagonal
    draw_line(output_grid, x, y, length=None, color=color, direction=(1, -1))
    draw_line(output_grid, x, y, length=None, color=color, direction=(-1, 1))
    # second diagonal
    draw_line(output_grid, x, y, length=None, color=color, direction=(-1, -1))
    draw_line(output_grid, x, y, length=None, color=color, direction=(1, 1))

    return output_grid

""" ==============================
Puzzle 6455b5f5

Train example 1:
Input1 = [
 k r k k k k r k k k k k k
 r r k k k k r k k k k k k
 k r k k k k r k k k k k k
 k r k k k k r r r r r r r
 k r k k k k r k k r k k k
 k r r r r r r k k r k k k
 k r k k k k r k k r k k k
 k r k k k k r r r r r r r
 k r k k k k r k k k k r k
 r r r r r r r r r r r r r
 k k r k k k k k k k k k k
 k k r k k k k k k k k k k
 k k r k k k k k k k k k k
 k k r k k k k k k k k k k
 k k r k k k k k k k k k k
 k k r k k k k k k k k k k
 k k r k k k k k k k k k k
 k k r k k k k k k k k k k
]
Output1 = [
 t r k k k k r k k k k k k
 r r k k k k r k k k k k k
 k r k k k k r k k k k k k
 k r k k k k r r r r r r r
 k r k k k k r k k r k k k
 k r r r r r r k k r k k k
 k r k k k k r k k r k k k
 k r k k k k r r r r r r r
 k r k k k k r k k k k r t
 r r r r r r r r r r r r r
 k k r b b b b b b b b b b
 k k r b b b b b b b b b b
 k k r b b b b b b b b b b
 k k r b b b b b b b b b b
 k k r b b b b b b b b b b
 k k r b b b b b b b b b b
 k k r b b b b b b b b b b
 k k r b b b b b b b b b b
]

Train example 2:
Input2 = [
 k k k k r k k k k k k k k
 k k k k r r r r r r r r r
 k k k k r k k k k k k k k
 k k k k r k k k k k k k k
 k k k k r k k k k k k k k
 r r r r r k k k k k k k k
 k k k k r k k k k k k k k
 k k k k r k k k k k k k k
 k k k k r k k k k k k k k
 k k k k r k k k k k k k k
 k k k k r k k k k k k k k
]
Output2 = [
 k k k k r t t t t t t t t
 k k k k r r r r r r r r r
 k k k k r b b b b b b b b
 k k k k r b b b b b b b b
 k k k k r b b b b b b b b
 r r r r r b b b b b b b b
 k k k k r b b b b b b b b
 k k k k r b b b b b b b b
 k k k k r b b b b b b b b
 k k k k r b b b b b b b b
 k k k k r b b b b b b b b
]

Train example 3:
Input3 = [
 k k k r k k k r k k k k k k k k
 k k k r k k k r k k k k k k k k
 k k k r r r r r k k k k k k k k
 k k k r k k k r k k k k k k k k
 k k k r k k k r k k k k k k k k
 r r r r r r r r k k k k k k k k
 k k k k k k k r k k k k k k k k
 k k k k k k k r r r r r r r r r
 k k k k k k k r k k r k k k k k
 k k k k k k k r k k r k k k k k
 k k k k k k k r k k r k k k k k
]
Output3 = [
 k k k r t t t r b b b b b b b b
 k k k r t t t r b b b b b b b b
 k k k r r r r r b b b b b b b b
 k k k r t t t r b b b b b b b b
 k k k r t t t r b b b b b b b b
 r r r r r r r r b b b b b b b b
 k k k k k k k r b b b b b b b b
 k k k k k k k r r r r r r r r r
 k k k k k k k r t t r k k k k k
 k k k k k k k r t t r k k k k k
 k k k k k k k r t t r k k k k k
] """

# concepts:
# filling

# description:
# The input consists of a black grid. The grid is divided with red lines into black rectangles of different sizes.
# To produce the output grid, fill in the smallest black rectangles with teal, and fillin in the largest black rectangles with blue.

def transform_6455b5f5(input_grid):
    # to get the black rectangles, find connected components with red as background
    objects = find_connected_components(input_grid, background=Color.RED, connectivity=4, monochromatic=True)

    # get object areas
    object_areas = [np.sum(obj == Color.BLACK) for obj in objects]

    # find the smallest and largest areas
    smallest_area = min(object_areas)
    largest_area = max(object_areas)

    # fill in the smallest rectangles with teal, and the largest rectangles with blue
    new_objects = []
    for obj in objects:
        area = np.sum(obj == Color.BLACK)
        if area == smallest_area:
            obj[obj == Color.BLACK] = Color.TEAL
        elif area == largest_area:
            obj[obj == Color.BLACK] = Color.BLUE
        new_objects.append(obj)

    # create an output grid to store the result
    output_grid = np.full(input_grid.shape, Color.RED)

    # blit the objects back into the grid
    for obj in new_objects:
        blit_object(output_grid, obj, background=Color.RED)

    return output_grid

""" ==============================
Puzzle 681b3aeb

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k g g k k k k k k k
 k g k k k k k k k k
 k g k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k o
 k k k k k k k k o o
 k k k k k k k k o o
]
Output1 = [
 g g o
 g o o
 g o o
]

Train example 2:
Input2 = [
 k k k k k k k k y k
 k k k k k k k k y y
 k k k p p p k k k k
 k k k k p p k k k k
 k k k k k p k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 p p p
 y p p
 y y p
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k g k k k k k
 k k k g g g k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k b b b k k k k k k
 k b k b k k k k k k
]
Output3 = [
 b b b
 b g b
 g g g
] """

# concepts:
# puzzle pieces,

# description:
# In the input you will see two monochromatic objects -- "puzzle pieces"
# To make the output, fit the pieces together so that they form a single tightly-packed rectangular object. The pieces can be translated, but not rotated.

def transform_681b3aeb(input_grid):
    # Plan:
    # 1. Detect the pieces
    # 2. Consider different ways of placing them together
    # 3. Pick the way which makes them the most tightly packed, meaning that there is as little empty pixels remaining as possible

    # 1. Extract puzzle pieces
    objects = find_connected_components(input_grid, connectivity=8, background=Color.BLACK, monochromatic=True)
    assert len(objects) == 2, "There should be exactly 2 objects"
    sprites = [ crop(obj, background=Color.BLACK) for obj in objects ]
    
    width = max(sprite.shape[0] for sprite in sprites) + min(sprite.shape[0] for sprite in sprites)
    height = max(sprite.shape[1] for sprite in sprites) + min(sprite.shape[1] for sprite in sprites)
    output_grid = np.full((width, height), Color.BLACK)

    # 2. Try to fit the pieces together
    possible_placements = [ (x1, x2, y1, y2)
                           for x1 in range(width - sprites[0].shape[0] + 1)
                           for x2 in range(width - sprites[1].shape[0] + 1)
                           for y1 in range(height - sprites[0].shape[1] + 1)
                           for y2 in range(height - sprites[1].shape[1] + 1) 
                           if not collision(object1=sprites[0], object2=sprites[1], x1=x1, x2=x2, y1=y1, y2=y2) ]
    
    def score_placement(x1, x2, y1, y2):
        # We are trying to make the puzzle pieces fit together perfectly
        # Therefore, there shouldn't be very many unfilled (black) pixels remaining after we place the pieces
        # So we are minimizing the number of black pixels
        # Equivalently maximizing the negative number of black pixels
        test_canvas = np.full_like(output_grid, Color.BLACK)
        blit_sprite(test_canvas, sprites[0], x1, y1)
        blit_sprite(test_canvas, sprites[1], x2, y2)
        test_canvas = crop(test_canvas, background=Color.BLACK)
        return -np.sum(test_canvas == Color.BLACK)
    
    # pick the best one
    x1, x2, y1, y2 = max(possible_placements, key=lambda placement: score_placement(*placement))
    blit_sprite(output_grid, sprites[0], x1, y1)
    blit_sprite(output_grid, sprites[1], x2, y2)

    return crop(output_grid, background=Color.BLACK)

""" ==============================
Puzzle 6855a6e4

Train example 1:
Input1 = [
 k k k k e k k k k k k k k k k
 k k k k e k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k r r r r r k k k k k k k k
 k k r k k k r k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k r k k k r k k k k k k k k
 k k r r r r r k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k e e e k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k r r r r r k k k k k k k k
 k k r k k k r k k k k k k k k
 k k k k e k k k k k k k k k k
 k k k k e k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k e e e k k k k k k k k k
 k k r k k k r k k k k k k k k
 k k r r r r r k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r r k k k k k r r k k k
 k k k r k k k k k k k r k e k
 e e k r k k k k k k k r k e e
 e e k r k k k k k k k r k e e
 k k k r k k k k k k k r k e k
 k k k r r k k k k k r r k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r r k k k k k r r k k k
 k k k r k k k k k e k r k k k
 k k k r k e e k e e k r k k k
 k k k r k e e k e e k r k k k
 k k k r k k k k k e k r k k k
 k k k r r k k k k k r r k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r r k k k k r r k k k k
 k e k r k k k k k k r k e k k
 e e k r k k k k k k r k e e k
 k e k r k k k k k k r k k e k
 k k k r r k k k k r r k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r r k k k k r r k k k k
 k k k r k e k k e k r k k k k
 k k k r k e e e e k r k k k k
 k k k r k e k e k k r k k k k
 k k k r r k k k k r r k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
] """

# concepts:
# symmetry, mirror

# description:
# In the input you will see two objects on each outer side of two red frames.
# To make the output, you need to mirror the two objects by flipping them over the symmetry of the red frames, making them inside the red frames. Each object flips over the frame closest to it.

def transform_6855a6e4(input_grid):
    # Extract the framework
    frame_color = Color.RED
    object_color = Color.GRAY 
    background = Color.BLACK

    # Create an empty grid
    n, m = input_grid.shape
    output_grid = np.zeros((n, m), dtype=int)

    # parse the input
    objects = find_connected_components(grid=input_grid, connectivity=8, monochromatic=True, background=background)
    frames = [ obj for obj in objects if frame_color in object_colors(obj, background=background) ]
    mirrored_objects = [ obj for obj in objects if object_color in object_colors(obj, background=background) ]

    # determine if we are doing horizontal or vertical mirroring
    # if all the objects have the same X coordinate, we are doing vertical mirroring
    # if all the objects have the same Y coordinate, we are doing horizontal mirroring
    x_positions = [ object_position(obj, background=background, anchor="center")[0] for obj in objects ]
    y_positions = [ object_position(obj, background=background, anchor="center")[1] for obj in objects ]
    if all(x == x_positions[0] for x in x_positions): orientation = "vertical"
    elif all(y == y_positions[0] for y in y_positions): orientation = "horizontal"
    else: raise ValueError(f"The objects are not aligned in a single axis")

    # Flip each other object over its closest frame
    for mirrored_object in mirrored_objects:
        # Find the closest frame
        def distance_between_objects(obj1, obj2):
            x1, y1 = object_position(obj1, background=background, anchor="center")
            x2, y2 = object_position(obj2, background=background, anchor="center")
            return (x1 - x2)**2 + (y1 - y2)**2        
        closest_frame = min(frames, key=lambda frame: distance_between_objects(frame, mirrored_object))

        # Build a symmetry object for flipping over the closest frame
        frame_x, frame_y = object_position(closest_frame, background=background, anchor="center")
        this_x, this_y = object_position(mirrored_object, background=background, anchor="center")
        # Make it one pixel past the middle of frame
        frame_y += 0.5 if this_y > frame_y else -0.5
        frame_x += 0.5 if this_x > frame_x else -0.5
        symmetry = MirrorSymmetry(mirror_x=frame_x if orientation == "horizontal" else None,
                                  mirror_y=frame_y if orientation == "vertical" else None)
        
        # Flip the object over the symmetry
        for x, y in np.argwhere(mirrored_object != background):
            x2, y2 = symmetry.apply(x, y)
            output_grid[x2, y2] = mirrored_object[x, y]
        
        # Draw the frame
        output_grid = blit_object(output_grid, closest_frame)

    return output_grid

""" ==============================
Puzzle 6a1e5592

Train example 1:
Input1 = [
 r r r r r r r r r r r r r r r
 r k r r r r r r r r r r r r k
 r k k r r r k k k r r r r r k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k e k
 k k k k k k e k k k k k k e k
 k e e e k k e e k k k k k e k
 k e e e k k e e e k k k k e k
]
Output1 = [
 r r r r r r r r r r r r r r r
 r b r r r r r r r r r r r r b
 r b b r r r b b b r r r r r b
 k b b b k k b b b k k k k k b
 k k k k k k k k k k k k k k b
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 r r r r r r r r r r r r r r r
 r r r r k r r r k r r k k r r
 r k k r k r r k k k r k k r r
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k e e k k k k k k k k k k k k
 k e e k k k k k k k e k k k k
 e e e e k k k e k k e k k e e
 k e e k k k e e e k e k e e e
]
Output2 = [
 r r r r r r r r r r r r r r r
 r r r r b r r r b r r b b r r
 r b b r b r r b b b r b b r r
 b b b k b k k k k k b b b b k
 k k k k k k k k k k k b b k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
] """

# concepts:
# puzzle pieces, contact

# description:
# In the input you will see blue puzzles pieces lying beneath a red shape with holes on its underside.
# To make the output, move the gray puzzle pieces upward so that they fit into the holes on the underside. They need to fully plug the holes.
# Finally change the grey puzzle pieces to be blue.

def transform_6a1e5592(input_grid):
    # Plan:
    # 1. Detect the puzzle pieces and red thing
    # 2. Extract the sprites of each puzzle piece
    # 3. Try moving the puzzle pieces (but not so much that they collide with the red thing). You can translate but not rotate. Plug as much of the holes in the red thing as possible.
    # 4. Change the color of the puzzle pieces to blue (just change everything that's grey to blue)

    # 1. Separate puzzle pieces from the red object
    red_objects = detect_objects(grid=input_grid, colors=[Color.RED], monochromatic=True, connectivity=8)
    assert len(red_objects) == 1, "There should be exactly one fully red object"
    red_object = red_objects[0]
    
    puzzle_pieces = detect_objects(grid=input_grid, colors=[Color.GREY], monochromatic=True, connectivity=4)

    # 2. Extract sprites by cropping
    sprites = [ crop(piece, background=Color.BLACK) for piece in puzzle_pieces ]

    # Output begins with just the red and then we add stuff to it
    output_grid = red_object.copy()

    # 3. Try moving the puzzle pieces, plugging as much of the red object is possible
    sprites_to_move = list(sprites)
    while sprites_to_move:
        possible_solutions = [ (x, y, sprite) for sprite in sprites_to_move 
                              for x in range(output_grid.shape[0] - sprite.shape[0] + 1)
                              for y in range(output_grid.shape[1] - sprite.shape[1] + 1) ]
        def score_solution(x, y, sprite):
            # The score is -inf if it collides with the red object
            # Otherwise it is the number of black pixels that are plugged by the sprite

            # Make a canvas by trying putting down the sprite
            test_canvas = np.full_like(output_grid, Color.BLACK)
            blit_sprite(test_canvas, sprite, x, y)

            # Check for collision
            if collision(object1=test_canvas, object2=output_grid):
                return float("-inf")
            
            # Count the number of black pixels that are plugged by the sprite, only counting those within the bounding box of the red object
            red_object_mask = red_object != Color.BLACK
            test_object_mask = test_canvas != Color.BLACK
            plugged_pixels = test_object_mask & ~red_object_mask & bounding_box_mask(red_object)
            
            return np.sum(plugged_pixels)
        
        best_x, best_y, best_sprite = max(possible_solutions, key=lambda solution: score_solution(*solution))

        # Blit the sprite into the output grid
        blit_sprite(output_grid, best_sprite, best_x, best_y)

        # Remove the sprite from the list of sprites to move
        sprites_to_move = [ sprite for sprite in sprites_to_move if sprite is not best_sprite ]

    # 4. grey->blue
    output_grid[output_grid == Color.GREY] = Color.BLUE

    return output_grid

""" ==============================
Puzzle 6aa20dc0 """

# concepts:
# puzzle pieces, rotation, rescaling

# description:
# In the input you will see a non-black background, a small multicolored object, and fragments of that object rescaled and rotated scattered around.
# To make the output, isolate the small multicolored object and then rescale/rotate/translate to cover the fragments as much as possible, matching color whenever the fragment has that color.

def transform_6aa20dc0(input_grid):
    # Plan:
    # 1. Detect the object, separating the small multicolored puzzle piece from its fragments
    # 2. Rescale/rotate/translate the small object
    # 3. Find the transformation covering as much of the fragments as possible, matching colors whenever they overlap
    # 4. Copy the resulting transformation to the output grid; delete the fragments from the input
    # 5. Repeat until all the fragments are gone

    # 1. object detection
    # because the background is not black, set it to be the most common color
    background = max(Color.NOT_BLACK, key=lambda color: np.sum(input_grid == color))
    # detect the objects and figure out which is the template puzzle piece, which is going to be the one with the largest variety of colors
    objects = find_connected_components(input_grid, connectivity=8, background=background, monochromatic=False)
    template_object = max(objects, key=lambda obj: len(object_colors(obj, background=background)))
    template_sprite = crop(template_object, background=background)

    output_grid = np.full_like(input_grid, background)

    # 2. rescale/rotate/translate the small object to cover as much of the fragments as possible, matching colors whenever they overlap
    rescaled_and_rotated = [ np.rot90(scale_sprite(template_sprite, scale), k=rot)
                            for scale in [1, 2, 3, 4]
                            for rot in range(4) ]
    # A placement solution is a tuple of (x, y, sprite) where x, y is the top-left corner of the rotated/scaled sprite
    possible_solutions = [ (x, y, sprite)
                          for sprite in rescaled_and_rotated
                          for x in range(output_grid.shape[0] - sprite.shape[0])
                          for y in range(output_grid.shape[1] - sprite.shape[1]) ]
    
    # Keep on looping until we are out of things to copy to the output
    while np.any(input_grid != background):

        def score_solution(x, y, sprite):
            # The score is -inf if the placement violates non-background colors
            # Otherwise it is the number of pixels that match in color between the sprite and the input
            test_canvas = np.full_like(input_grid, background)
            blit_sprite(test_canvas, sprite, x, y)

            if np.any( (test_canvas != background) & (input_grid != background) & (test_canvas != input_grid) ):
                return float("-inf")
            
            return np.sum( (test_canvas == input_grid) & (input_grid != background) )
        
        # Remove -inf solutions, and zero solutions
        possible_solutions = [ solution for solution in possible_solutions if score_solution(*solution) > 0 ]
        
        best_x, best_y, best_sprite = max(possible_solutions, key=lambda solution: score_solution(*solution))

        # 4. Copy the resulting transformation to the output grid; delete the fragments from the input
        # Copy output
        blit_sprite(output_grid, best_sprite, best_x, best_y)
        # Delete from input
        for dx, dy in np.argwhere(best_sprite != background):
            input_grid[best_x + dx, best_y + dy] = background
        
        # 5. Repeat until all the fragments are gone
    
    return output_grid

""" ==============================
Puzzle 6b9890af

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k t k k k k k k k k k k k k k k
 k k k k k t t t k k k k k k k k k k k k k
 k k k k k k t k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k r r r r r r r r k k k k k k k
 k k k k k k r k k k k k k r k k k k k k k
 k k k k k k r k k k k k k r k k k k k k k
 k k k k k k r k k k k k k r k k k k k k k
 k k k k k k r k k k k k k r k k k k k k k
 k k k k k k r k k k k k k r k k k k k k k
 k k k k k k r k k k k k k r k k k k k k k
 k k k k k k r r r r r r r r k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
]
Output1 = [
 r r r r r r r r
 r k k t t k k r
 r k k t t k k r
 r t t t t t t r
 r t t t t t t r
 r k k t t k k r
 r k k t t k k r
 r r r r r r r r
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
 k k r r r r r k k k k k k k k k k k k k k k
 k k r k k k r k k k k k k k k k k k k k k k
 k k r k k k r k k k k k k k k k k k k k k k
 k k r k k k r k k k k k k k k k k k k k k k
 k k r r r r r k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k b b k k k k k k k k k
 k k k k k k k k k k b k k k k k k k k k k k
 k k k k k k k k k k k b b k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
]
Output2 = [
 r r r r r
 r k b b r
 r b k k r
 r k b b r
 r r r r r
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k k k k k k k k k k
 k k r r r r r r r r r r r k k k k k k k k k k k
 k k r k k k k k k k k k r k k k k k k k k k k k
 k k r k k k k k k k k k r k k k k k k k k k k k
 k k r k k k k k k k k k r k k k k k k k k k k k
 k k r k k k k k k k k k r k k k k k k k k k k k
 k k r k k k k k k k k k r k k k k k k k k k k k
 k k r k k k k k k k k k r k k k k k k k k k k k
 k k r k k k k k k k k k r k k k k k k k k k k k
 k k r k k k k k k k k k r k k k k k k k k k k k
 k k r k k k k k k k k k r k k k k k k k k k k k
 k k r r r r r r r r r r r k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k y y k k k k k k k k
 k k k k k k k k k k k k k y k y k k k k k k k k
 k k k k k k k k k k k k k k k y k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k
]
Output3 = [
 r r r r r r r r r r r
 r k k k y y y y y y r
 r k k k y y y y y y r
 r k k k y y y y y y r
 r y y y k k k y y y r
 r y y y k k k y y y r
 r y y y k k k y y y r
 r k k k k k k y y y r
 r k k k k k k y y y r
 r k k k k k k y y y r
 r r r r r r r r r r r
] """

# concepts:
# object detection, scaling

# description:
# In the input you will see a 3x3 object and a red square n times larger than the 3x3 object.
# To make the output, you should scale the 3x3 object to the size of the red square and place it in the red square.
# Return just the red square (with the rescaled object put into it)

def transform_6b9890af(input_grid):
    # Detect the red frame sqaure and the 3x3 pattern square
    objects = detect_objects(input_grid, monochromatic=True, connectivity=8)
    
    # Extract the object, seperate them into the red frame square and the 3x3 pattern square
    for obj in objects:
        sprite = crop(obj, background=Color.BLACK)
        if Color.RED in object_colors(sprite, background=Color.BLACK):
            outer_sprite = sprite
        else:
            inner_sprite = sprite
    
    # Calculate the scaling factor.
    # You need to subtract 2 because the red frame square has 1 pixel border on each side, and there are 2 sides
    scale = (len(outer_sprite) - 2) // len(inner_sprite)

    # Scale the small thing
    scaled_inner_sprite = scale_sprite(inner_sprite, factor=scale)

    # Put them all down on a new output grid
    output_grid = np.full(outer_sprite.shape, Color.BLACK)
    blit_sprite(output_grid, outer_sprite, x=0, y=0, background=Color.BLACK)
    blit_sprite(output_grid, scaled_inner_sprite, x=1, y=1, background=Color.BLACK)

    return output_grid

""" ==============================
Puzzle 6cdd2623

Train example 1:
Input1 = [
 k k k k k r r k k k k g k k k k k r k k k k
 r k k k k k r k k k k k k r k k k k k k k k
 k k k k k k e r k e r k e k k k k k r k k k
 k k k k e e k k k k k k k k r k k k k k r k
 e k k r k r k k k k k k k k r k k k k k k k
 k k k r k k k r k k r k k k k k k k r k e k
 k k r k k k k k k e e k k e k k k k k r e k
 k k k k k k k k k k k k k r k k k k r k k k
 g k k e e k r e k k k k k k k k k k k k r g
 k k k k k k k k k r k r e k e k k k r k k k
 k k r k k k k k e k k g k k k k k e k e k k
]
Output1 = [
 k k k k k k k k k k k g k k k k k k k k k k
 k k k k k k k k k k k g k k k k k k k k k k
 k k k k k k k k k k k g k k k k k k k k k k
 k k k k k k k k k k k g k k k k k k k k k k
 k k k k k k k k k k k g k k k k k k k k k k
 k k k k k k k k k k k g k k k k k k k k k k
 k k k k k k k k k k k g k k k k k k k k k k
 k k k k k k k k k k k g k k k k k k k k k k
 g g g g g g g g g g g g g g g g g g g g g g
 k k k k k k k k k k k g k k k k k k k k k k
 k k k k k k k k k k k g k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k t k k k k k k k k k k k k k k
 k k k k k k k k k k t k b k k k k k k k
 k k k k k k k k b k k k b k k k k k k k
 r k k k k k k k k k k k k k k k k k k r
 k k k k k k k k k k k k k k k k k t b k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k b k k b k k b k t k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k t k k t k k k k k k k k t b k k k k k
 k k k k k k k k k k k k k k t k k k k b
 k k k k k k k k k k k k k k k k k k k k
 r k k k k k k k t k k b k k k k k k k r
 t k k k k k k k k k k k k k k k k t k k
]
Output2 = [
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 r r r r r r r r r r r r r r r r r r r r
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 r r r r r r r r r r r r r r r r r r r r
 k k k k k k k k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k y y k k k k y k k k k k t k y k
 k k y k y k k k k k k k k k k k k
 k k k k k k k k k k k k k k y k e
 y k k y k k k k k k k k k k k k k
 k k k k y k k k e k k k k k k k k
 k k k k k k k y e k k k k k k k e
 k y k k k e k k k k y k k k y k k
 k k k k k k k k k k k k k k k e y
 k k k k k k k k k k k k y k k k k
 k k k k k k y k k k k y k k k k k
 t k k k k k k k k k k k k k k k t
 y k k k k k k k k k k k k k y k k
 k e k k k k k k k k k k y k k y k
 k k k k k k y k y k k k k k k y k
 y k y k y k k k y e k k k t k k k
]
Output3 = [
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 t t t t t t t t t t t t t t t t t
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
 k k k k k k k k k k k k k t k k k
] """

# concepts:
# draw lines, detect objects

# description:
# In the input you will see three colors scattered on the grid. One color only have four pixels on the boundary of the grid.
# To make the output grid, you should connect the four pixels of the color on the boundary of the grid to make two lines.

def transform_6cdd2623(input_grid):
    # Find the used color
    all_color = np.unique(input_grid)

    # Find the color all on the boundary, which is the lines' color
    def on_boundary(x, y):
        return x == 0 or x == input_grid.shape[0] - 1 or y == 0 or y == input_grid.shape[1] - 1
    
    # Get the color all on the boundary, which is the lines' color
    for color in all_color:
        all_on_boundary = all(on_boundary(x, y) for x, y in np.argwhere(input_grid==color))
        if all_on_boundary:
            line_color = color
    output_grid = np.zeros_like(input_grid)

    # Find the boundary pixels of the line_color and then draw a horizontal/vertical line to its matching pair
    for x, y in np.argwhere(input_grid == line_color):
        # Check if it's left/right edge or top/bottom edge
        if x == 0 or x == input_grid.shape[0] - 1:
            # it's left/right, so draw horizontal
            draw_line(grid=output_grid, x=x, y=y, color=line_color, direction=(1, 0))
        elif y == 0 or y == input_grid.shape[1] - 1:
            # it's top/bottom, so draw vertical
            draw_line(grid=output_grid, x=x, y=y, color=line_color, direction=(0, 1))
    
    return output_grid

""" ==============================
Puzzle 6cf79266 """

# concepts:
# rectangle detection, background shape detection

# description:
# In the input you will see a grid with scattered one color pixels
# To make the output grid, you should detect the 3x3 black square in the random color pattern
# and replace it with a 3x3 blue square

def transform_6cf79266(input_grid):
    # Plan: 
    # 1. Detect the 3x3 regions that are all black
    # 2. Draw a blue 3x3 in those regions

    # 1. Detect the 3x3 regions that are all black
    region_len = 3
    output_grid = np.copy(input_grid)
    matching_regions = [(x, y) for x in range(len(input_grid) - (region_len - 1)) for y in range(len(input_grid[0]) - (region_len - 1)) if np.all(input_grid[x:x + region_len, y:y + region_len] == Color.BLACK)]

    # 2. Draw a blue 3x3 in those regions
    for x, y in matching_regions:
        # Check if the region is all black
        if np.all(output_grid[x:x+region_len, y:y+region_len] == Color.BLACK):
            output_grid[x:x+region_len, y:y+region_len] = Color.BLUE

    return output_grid

""" ==============================
Puzzle 6d58a25d """

# concepts:
# sliding objects

# description:
# In the input grid, you will see a chevron-shaped object of one color in a black grid, with pixels of another color scattered around the grid.
# To produce the output grid, take all pixels located underneath the chevron. For each of these pixels, extend a vertical line of the same color up and down, until reaching the bottom of the grid or the boundary of the chevron.

def transform_6d58a25d(input_grid):
    # 1. find the chevron: it is the largest object by size.
    # 2. get the color of the chevron
    # 3. get the color of the colored pixels in the grid.
    # 4. for each colored pixel, check if the chevron is above it. if so, extend a line of the same color above and below it until we reach the bottom of the grid or the boundary of the chevron.

    # get the chevron
    objects = find_connected_components(input_grid, connectivity=4, monochromatic=True)
    chevron = max(objects, key=lambda o: np.count_nonzero(o))

    # get the color of the chevron
    chevron_color = chevron[chevron != Color.BLACK][0]

    # get the color of the colored pixels (the other color in the grid)
    colors = np.unique(input_grid)
    colors = [c for c in colors if c not in [Color.BLACK, chevron_color]]
    assert len(colors) == 1
    pixel_color = colors[0]

    # for each colored pixel, check if chevron is above it
    # to do so, iterate through the grid and check for pixel_color.
    # then try moving up until we hit the chevron color.
    # if we do, then paint a vertical line onto the output grid.
    output_grid = input_grid.copy()
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] != pixel_color:
                continue
            # try moving up until we hit the chevron color
            dy = 0
            while y + dy >= 0 and input_grid[x, y + dy] != chevron_color:
                dy = dy - 1

            if input_grid[x, y + dy] == chevron_color:
                # make a line from here to the bottom
                output_grid[x, y + dy + 1:] = pixel_color

    return output_grid

""" ==============================
Puzzle 6d75e8bb

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k t t t k k k k k k
 k t k k k k k k k k
 k t t t t k k k k k
 k t t k k k k k k k
 k t t t k k k k k k
 k t k k k k k k k k
 k t t t k k k k k k
 k t t t k k k k k k
 k t t k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k t t t r k k k k k
 k t r r r k k k k k
 k t t t t k k k k k
 k t t r r k k k k k
 k t t t r k k k k k
 k t r r r k k k k k
 k t t t r k k k k k
 k t t t r k k k k k
 k t t r r k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k
 k t t t t t t k
 k t k t t k t k
 k t k t k k t k
 k k k t k t t k
 k k k k k k k k
 k k k k k k k k
]
Output2 = [
 k k k k k k k k
 k t t t t t t k
 k t r t t r t k
 k t r t r r t k
 k r r t r t t k
 k k k k k k k k
 k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k
 k t t t t t k k
 k k k t k t k k
 k k t t t t k k
 k k k t t t k k
 k k k k t t k k
 k k k t t t k k
 k k k k k k k k
 k k k k k k k k
]
Output3 = [
 k k k k k k k k
 k t t t t t k k
 k r r t r t k k
 k r t t t t k k
 k r r t t t k k
 k r r r t t k k
 k r r t t t k k
 k k k k k k k k
 k k k k k k k k
] """

# concepts:
# shape completion

# description:
# In the input you will see an incomplete teal ractangle
# To make the output grid, you should use the red color to complete the rectangle.

def transform_6d75e8bb(input_grid):
    # Find the bounding box of the incomplete rectangle and use it to extra the sprite
    x, y, x_len, y_len = bounding_box(grid=input_grid)
    rectangle = input_grid[x:x + x_len, y:y + y_len]

    # Find the missing parts of the rectangle (which are colored black) and complete it with red color
    rectangle_sprite = np.where(rectangle == Color.BLACK, Color.RED, rectangle)

    # Make the output by copying the sprite to a new canvas
    output_grid = np.copy(input_grid)
    output_grid = blit_sprite(grid=output_grid, sprite=rectangle_sprite, x=x, y=y)

    return output_grid

""" ==============================
Puzzle 6e19193c

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k o k k k k k k k k
 k o o k k k k k k k
 k k k k k k o o k k
 k k k k k k k o k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k o k k k k k
 k k k o k k k k k k
 k o k k k k k k k k
 k o o k k k k k k k
 k k k k k k o o k k
 k k k k k k k o k k
 k k k k k o k k k k
 k k k k o k k k k k
 k k k o k k k k k k
 k k o k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k m m k k k k k
 k k k k m k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k m k k k k k k
 k k k m m k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k k k m m k k k k m
 k k k k m k k k m k
 k k m k k k k m k k
 k m k k k k m k k k
 m k k k k m k k k k
 k k k m k k k k k k
 k k k m m k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
] """

# concepts:
# direction, lines, pointing

# description:
# In the input, you will see several objects of the same color that are in an arrowhead shape and facing different directions.
# The goal is to find the directions of the arrowheads and draw lines that would represent the path they had been moving in to go in that direction.

def transform_6e19193c(input_grid):
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # find the objects in the input grid
    objects = find_connected_components(input_grid, connectivity=8)

    # for each object, find the direction the arrowhead is pointing in by finding the relative mean position of colored and black pixels in the bounding box of the object
    for obj in objects:
        # find the bounding box of the object
        x, y, w, h = bounding_box(obj)

        # crop the object to extract the sprite
        sprite = crop(obj)

        # find the color of the object
        color = np.unique(obj[obj != Color.BLACK])[0]

        # find the mean position of the colored pixels
        mean_pos = np.mean(np.argwhere(sprite != Color.BLACK), axis=0)

        # find the mean position of all the black pixels
        mean_black_pos = np.mean(np.argwhere(sprite == Color.BLACK), axis=0)

        # find the direction the arrowhead is pointing in, it is from the mean position of the colored pixels to the mean position of the black pixels
        direction = np.sign(mean_black_pos - mean_pos).astype(int)

        # draw a line in the direction the arrowhead is pointing in from the corresponding corner of the bounding box
        # list the corners of the bounding box
        corners = [(x - 1, y - 1), (x + w, y - 1), (x - 1, y + h), (x + w, y + h)]
        # compute the center of the object
        center = (x + w / 2, y + h / 2)
        # if the direction of the corner from the center of the object matches the direction we want to draw a line in, then draw a line
        for corner in corners:
            # check if the corner is in the direction that the arrowhead is pointing
            vector_to_corner = np.array(corner) - np.array(center)
            if np.all(np.sign(vector_to_corner) == direction):
                draw_line(output_grid, corner[0], corner[1], length=None, color=color, direction=direction)

    return output_grid

""" ==============================
Puzzle 6e82a1ae

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k e e k
 k e e k k k k e e k
 k k e e k k k k k k
 k k k k k k k k k k
 k k k k k k k k k e
 k k k k k e e k k e
 k e k k k k k k k e
 k e k k e k k k k k
 k k k e e k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k b b k
 k b b k k k k b b k
 k k b b k k k k k k
 k k k k k k k k k k
 k k k k k k k k k r
 k k k k k g g k k r
 k g k k k k k k k r
 k g k k r k k k k k
 k k k r r k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k e e e k k k e k k
 k k k k k k k e k k
 k k k k k k k k k k
 k k k e e k k k k k
 k k k e k k k k k k
 k k k k k k k e k k
 k e e k k k e e e k
 k e e k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k r r r k k k g k k
 k k k k k k k g k k
 k k k k k k k k k k
 k k k r r k k k k k
 k k k r k k k k k k
 k k k k k k k b k k
 k b b k k k b b b k
 k b b k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k e k k k e e k k
 k k e k k k k e k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k e e k k k k
 e k k k k k k k k k
 e e k k k k k k k k
 e k k k k k k k k k
 k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k
 k k g k k k r r k k
 k k g k k k k r k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k g g k k k k
 b k k k k k k k k k
 b b k k k k k k k k
 b k k k k k k k k k
 k k k k k k k k k k
] """

# concepts:
# objects, counting, color

# description:
# In the input you will see grey objects on a black background.
# To make the output, count the number of pixels in each object and color the object green if it has two pixels, red if it has three pixels, and blue if it has four pixels.

def transform_6e82a1ae(input_grid):
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # get the objects in the input grid
    objects = find_connected_components(input_grid)

    # count the number of pixels in each object and color them accordingly
    for obj in objects:
        num_pixels = np.sum(obj == Color.GREY)
        if num_pixels == 2:
            color = Color.GREEN
        elif num_pixels == 3:
            color = Color.RED
        elif num_pixels == 4:
            color = Color.BLUE
        else:
            color = Color.GREY
        output_grid[obj == Color.GREY] = color

    return output_grid

""" ==============================
Puzzle 6ecd11f4

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k k k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k k k k k k b b b b b k k k k k k k k k k
 k k k k k k b b b b b k k k k k k k k k k
 k k k k k k b b b b b k k k k k k k k k k
 k k k k k k b b b b b k k k k k k k k k k
 k k k k k k b b b b b k k k k k k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k b b b b b k k k k k b b b b b k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k g b o k k k k k k k k k k k
 k k k k k k k r t m k k k k k k k k k k k
 k k k k k k k g y p k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
]
Output1 = [
 g k o
 k t k
 g k p
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k g g g k k k g g g k k k k k k k
 k k k k k k k k k g g g k k k g g g k k k k k k k
 k k k k k k k k k g g g k k k g g g k k k k k k k
 k k k k k k k k k g g g g g g k k k k k k k k k k
 k k k k k k k k k g g g g g g k k k k k k k k k k
 k k k k k k k k k g g g g g g k k k k k k k k k k
 k k k k k k k k k k k k g g g g g g k k k k k k k
 k k k k k k k k k k k k g g g g g g k k k k k k k
 k k k k k k k k k k k k g g g g g g k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k r b o k k k k k k k k k k k k k k
 k k k k k k k k y t m k k k k k k k k k k k k k k
 k k k k k k k k t p b k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k k k k
]
Output2 = [
 r k o
 y t k
 k p b
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k k k k k k k k
 k k t t t t k k k k t t t t t t t t k k k k
 k k t t t t k k k k t t t t t t t t k k k k
 k k t t t t k k k k t t t t t t t t k k k k
 k k t t t t k k k k t t t t t t t t k k k k
 k k t t t t k k k k t t t t k k k k k k k k
 k k t t t t k k k k t t t t k k k k k k k k
 k k t t t t k k k k t t t t k k k k k k k k
 k k t t t t k k k k t t t t k k k k k k k k
 k k t t t t k k k k k k k k t t t t k k k k
 k k t t t t k k k k k k k k t t t t k k k k
 k k t t t t k k k k k k k k t t t t k k k k
 k k t t t t k k k k k k k k t t t t k k k k
 k k t t t t t t t t t t t t k k k k k k k k
 k k t t t t t t t t t t t t k k k k k k k k
 k k t t t t t t t t t t t t k k k k k k k k
 k k t t t t t t t t t t t t k k k y b m y k
 k k k k k k k k k k k k k k k k k p g p b k
 k k k k k k k k k k k k k k k k k g e o e k
 k k k k k k k k k k k k k k k k k r y r o k
 k k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k k
]
Output3 = [
 y k m y
 p k p k
 g k k e
 r y r k
] """

# concepts:
# objects, color guide, masking, scaling

# description:
# In the input you will see a large monochromatic object, and a smaller square object with many colors.
# To make the output, first downscale the large object so that it is the same size as the small multicolor square, and then use the large object as a binary mask to zero out pixels in the square. Return the resulting masked sprite.

def transform_6ecd11f4(input_grid):
    # find all the objects in the input grid
    objects = find_connected_components(input_grid, connectivity=8, monochromatic=False)

    # figure out which object is the pattern and which object is the multi-colored square
    pattern = square = None
    for obj in objects:
        # if the object has only one color and black, it is the pattern. Otherwise, it is the multi-colored square.
        if len(set(obj.flatten())) == 2:
            pattern = obj
        else:
            square = obj
    
    # get the square location and cut out the bounding box
    x, y, width, height = bounding_box(square)
    square_sprite = square[x:x+width, y:y+height]

    # cut out the bounding box of the pattern
    x2, y2, width2, height2 = bounding_box(pattern)
    # make sure the pattern is square
    if width2 != height2:
        width2 = height2 = max(width2, height2)
    pattern_sprite = pattern[x2:x2+width2, y2:y2+height2]

    # figure out how much bigger the pattern is than the square
    scale = width2 // width

    # scale down the pattern to fit the square
    scaled_pattern = np.zeros_like(square_sprite) 
    for i in range(width):
        for j in range(height):
            scaled_pattern[i,j] = pattern_sprite[i * scale, j * scale]
    
    # if the pixel is in the pattern, keep the color from the square otherwise make it black in the output grid
    output_grid = np.where(scaled_pattern, square_sprite, Color.BLACK)

    return output_grid

""" ==============================
Puzzle 6f8cd79b

Train example 1:
Input1 = [
 k k k
 k k k
 k k k
]
Output1 = [
 t t t
 t k t
 t t t
]

Train example 2:
Input2 = [
 k k k
 k k k
 k k k
 k k k
]
Output2 = [
 t t t
 t k t
 t k t
 t t t
]

Train example 3:
Input3 = [
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
]
Output3 = [
 t t t t
 t k k t
 t k k t
 t k k t
 t t t t
] """

# concepts:
# borders

# description:
# In the input you will see an empty black grid.
# To make the output, draw a line along the border of the input with a thickness of one pixel. The border should be teal.

def transform_6f8cd79b(input_grid):
    # make the output grid
    n, m = input_grid.shape
    output_grid = np.zeros((n, m), dtype=int)

    # draw the border of the input grid
    draw_line(grid=output_grid, x=0, y=0, length=None, color=Color.TEAL, direction=(1,0))
    draw_line(grid=output_grid, x=n-1, y=0, length=None, color=Color.TEAL, direction=(0,1))
    draw_line(grid=output_grid, x=0, y=0, length=None, color=Color.TEAL, direction=(0,1))
    draw_line(grid=output_grid, x=0, y=m-1, length=None, color=Color.TEAL, direction=(1,0))

    return output_grid

""" ==============================
Puzzle 6fa7a44f

Train example 1:
Input1 = [
 m b y
 m b y
 r b b
]
Output1 = [
 m b y
 m b y
 r b b
 r b b
 m b y
 m b y
]

Train example 2:
Input2 = [
 y t y
 o p o
 t o t
]
Output2 = [
 y t y
 o p o
 t o t
 t o t
 o p o
 y t y
]

Train example 3:
Input3 = [
 o o o
 m e e
 e b o
]
Output3 = [
 o o o
 m e e
 e b o
 e b o
 m e e
 o o o
] """

# concepts:
# reflection

# description:
# In the input you will see a square pattern of random colors except black.
# To make the output, reflect the pattern vertically, and put the reflected pattern beneath the input pattern.

def transform_6fa7a44f(input_grid):
    # take the input pattern
    pattern = input_grid

    # reflect the pattern vertically
    reflected_pattern = pattern[:, ::-1]

    # make the output grid
    output_grid = np.concatenate((pattern, reflected_pattern), axis=1)

    return output_grid

""" ==============================
Puzzle 72ca375d

Train example 1:
Input1 = [
 k k k k k k k k k k
 k r r k k k k k k k
 k k r r r k k o o k
 k k k k k k o k o k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k p p p p k k k
 k k k k p p k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 p p p p
 k p p k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k y y k k k k k k
 k k y y k k t t t k
 k k k k k k t k t t
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k r r r r k k k k
 k r r r k k k k k k
 k k k k k k k k k k
]
Output2 = [
 y y
 y y
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k g g k k k k k k k
 k k g k k e k k e k
 k k g k k e e e e k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k t t t k k k k
 t t t t k t t k k k
 k k k k k k k k k k
]
Output3 = [
 e k k e
 e e e e
] """

# concepts:
# symmetry, sprites

# description:
# In the input you will see several objects. One object represents a symmetic sprite. All the other objects represent non-symmetric sprites.
# The goal is to find the symmetric sprite and return it.

def transform_72ca375d(input_grid):
    # find the objects in the input grid
    objects = find_connected_components(input_grid, connectivity=8)

    # crop out the sprites from the objects
    sprites = [crop(obj) for obj in objects]

    # find the symmetric sprite
    symmetric_sprite = None
    for sprite in sprites:
      # check if the sprite is radially symmetric
      if np.array_equal(sprite, np.rot90(sprite, 1)):
        symmetric_sprite = sprite
        break
      # check if the sprite is vertically symmetric
      elif np.array_equal(sprite, np.fliplr(sprite)):
        symmetric_sprite = sprite
        break
      # check if the sprite is horizontally symmetric
      elif np.array_equal(sprite, np.flipud(sprite)):
        symmetric_sprite = sprite
        break
      # check if the sprite is diagonally symmetric
      elif np.array_equal(sprite, sprite.T) or np.array_equal(np.flipud(sprite), np.fliplr(sprite)):
        symmetric_sprite = sprite
        break

    return symmetric_sprite

""" ==============================
Puzzle 7447852a

Train example 1:
Input1 = [
 r k k k r k k k r k
 k r k r k r k r k r
 k k r k k k r k k k
]
Output1 = [
 r k k k r y y y r k
 y r k r k r y r k r
 y y r k k k r k k k
]

Train example 2:
Input2 = [
 r k k k r k k k r k k k r k k
 k r k r k r k r k r k r k r k
 k k r k k k r k k k r k k k r
]
Output2 = [
 r k k k r y y y r k k k r k k
 y r k r k r y r k r k r y r k
 y y r k k k r k k k r y y y r
]

Train example 3:
Input3 = [
 r k k k r k k k r k k k r k k k r k
 k r k r k r k r k r k r k r k r k r
 k k r k k k r k k k r k k k r k k k
]
Output3 = [
 r k k k r y y y r k k k r k k k r y
 y r k r k r y r k r k r y r k r k r
 y y r k k k r k k k r y y y r k k k
] """

# concepts:
# objects, flood fill, connectivity

# description:
# In the input, you will see a black grid with a red line that starts in the top left corner and bounces off the borders of the grid until it reaches the right side of the grid.
# To make the output, find the black regions separated by the red lines, then, starting with the first region from the left, color every third region yellow.

def transform_7447852a(input_grid):
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # find the black regions in the input grid
    black_regions = find_connected_components(input_grid, connectivity=4, background=Color.RED)

    # sort the regions from left to right using the coordinates of their bounding boxes
    black_regions = sorted(black_regions, key=lambda region: bounding_box(region, background=Color.RED)[0])

    # color every third black region yellow using flood fill
    for i, region in enumerate(black_regions):
        if i % 3 == 0:
            x, y = np.where(region == Color.BLACK)
            flood_fill(output_grid, x[0], y[0], Color.YELLOW)

    return output_grid

""" ==============================
Puzzle 746b3537

Train example 1:
Input1 = [
 b b b
 r r r
 b b b
]
Output1 = [
 b
 r
 b
]

Train example 2:
Input2 = [
 g y p
 g y p
 g y p
]
Output2 = [
 g y p
]

Train example 3:
Input3 = [
 r g g t b
 r g g t b
 r g g t b
]
Output3 = [
 r g t b
] """

# concepts:
# line detection, color extraction

# description:
# In the input you will see a grid consisting of stripes that are either horizontal or vertical.
# To make the output, make a grid with one pixel for each stripe whose color is the same color as that stripe.
# If the stripes are vertical, the output should be vertical, and if the stripes are horizontal, the output should be horizontal. The colors should be in the order they appear in the input.

def transform_746b3537(input_grid):
    # Parse input and then determine the orientation of the stripes
    objects = find_connected_components(input_grid, connectivity=4, monochromatic=True, background=Color.BLACK)
    x_positions = [ object_position(obj, background=Color.BLACK, anchor="center")[0] for obj in objects]
    y_positions = [ object_position(obj, background=Color.BLACK, anchor="center")[1] for obj in objects]
    if all(x == x_positions[0] for x in x_positions):
        orientation = "vertical"
    elif all(y == y_positions[0] for y in y_positions):
        orientation = "horizontal"
    else:
        raise ValueError("The stripes are not aligned in a single axis")
    
    # Sort the objects depending on the orientation
    if orientation == "horizontal":
        objects.sort(key=lambda obj: object_position(obj, background=Color.BLACK, anchor="center")[0])
    else:
        objects.sort(key=lambda obj: object_position(obj, background=Color.BLACK, anchor="center")[1])
    
    # Extract the colors of the stripes
    colors = [ object_colors(obj, background=Color.BLACK)[0] for obj in objects ]

    # Generate the output grid
    if orientation == "horizontal":
        output_grid = np.full((len(colors), 1), Color.BLACK)
        output_grid[:, 0] = colors
    else:
        output_grid = np.full((1, len(colors)), Color.BLACK)
        output_grid[0, :] = colors
    
    return output_grid

""" ==============================
Puzzle 776ffc46 """

# concepts:
# same/different, color change, containment

# description:
# In the input you will see some monochromatic objects. Some of them will be contained by a grey box, and some of them will not.
# To make the output, take each shape contained inside a grey box, and find any other shapes with the same shape (but a different color), and change their color to match the color of the shape inside the grey box.

def transform_776ffc46(input_grid):
    # Plan:
    # 1. Extract the objects and separate them according to if they are grey or not
    # 2. Determine if each non-grey shape is contained by a grey shape
    # 3. Check to see which objects (among grey contained shapes) have another object which has the same shape (not contained by grey)
    # 4. Do the color change when you find these matching objects

    # 1. Extract the objects, separating them by if they are grey or not
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=True)
    grey_objects = [ obj for obj in objects if Color.GREY in object_colors(obj, background=Color.BLACK) ]
    non_grey_objects = [ obj for obj in objects if Color.GREY not in object_colors(obj, background=Color.BLACK) ]

    # 2. Determine if each non-grey shape is contained by a grey shape
    # Divide the non-grey objects into two groups: those contained by grey, and those not contained by grey
    # Make a helper function for checking of one object is contained by another
    def object_contains_another_object(inside_object, outside_object):
        # Using bounding boxes:
        # inside_x, inside_y, inside_width, inside_height = bounding_box(inside_object)
        # outside_x, outside_y, outside_width, outside_height = bounding_box(outside_object)
        # return inside_x >= outside_x and inside_y >= outside_y and inside_x + inside_width <= outside_x + outside_width and inside_y + inside_height <= outside_y + outside_height
        # Using topology+masks:
        inside_object_mask = inside_object != Color.BLACK
        outside_interior_mask = object_interior(outside_object, background=Color.BLACK)
        return np.all(outside_interior_mask >= inside_object_mask)
    objects_contained_by_grey = [ non_grey for non_grey in non_grey_objects
                                  if any( object_contains_another_object(non_grey, grey) for grey in grey_objects ) ]
    objects_not_contained_by_gray = [ non_grey for non_grey in non_grey_objects
                                     if not any( object_contains_another_object(non_grey, grey) for grey in grey_objects ) ]
    
    # 3. Check to see which objects (among grey contained shapes) have another object which has the same shape (not contained by grey)
    output_grid = input_grid.copy()

    # Helper function to check if two objects have the same shape
    def objects_have_same_shape(obj1, obj2):
        mask1 = crop(obj1, background=Color.BLACK) != Color.BLACK
        mask2 = crop(obj2, background=Color.BLACK) != Color.BLACK
        return np.array_equal(mask1, mask2)
    
    for grey_contained in objects_contained_by_grey:
        for non_grey in objects_not_contained_by_gray:
            if objects_have_same_shape(grey_contained, non_grey):
                # 4. Do the color change
                target_color = object_colors(grey_contained, background=Color.BLACK)[0]
                non_grey_mask = non_grey != Color.BLACK
                output_grid[non_grey_mask] = target_color

    return output_grid

""" ==============================
Puzzle 780d0b14

Train example 1:
Input1 = [
 b b b b k k b b k k k t t t t t k t t t
 b b b k b k b b k k k t t t t t t t t t
 b b k b b b k b k k t t t t t t t t t t
 b b k b k b b b k k t k t t t t t t t t
 k b b k b b b b k t k t t k t t t k t t
 b k b b b b k k k t t t t t t t t t k t
 b b k b b b b b k t t t k t t t k t k k
 b b k b b k b b k k t t k t t t k k k t
 k k k k k k k k k k k k k k k k k k k k
 p p p p p k p p k b k b b k b b b k k k
 p p p p p p p k k k b k b b k k b b b k
 k p k p p p k p k b b k k k b k b b k b
 p p p k p p p p k b b k b k b b b k b b
 p k p p k p k p k b b b b k b b k b k b
 p p p p p k p p k b k b k b b b b b b b
 p p p p p k p k k b k b k b b b b b b b
 p p p k p p k p k b b b b b b b k k b b
 k p p p k k p k k k k b b k b b b b b k
 p k k k p k p k k b b b b b k b b b b b
 p p k p k p p p k b k b k b k b b b b k
]
Output1 = [
 b t
 p b
]

Train example 2:
Input2 = [
 y y y y y k k t k t t t k k g g g k k g g g
 y y y k k y k t t t t t k k g g g g k g g k
 y y y y k k k t t k k t k k g g g k g k g g
 y y k k y y k t t t t t t k g g g g k g g g
 y y y y y y k k t t t t t k g k g k g k g k
 k k y y y y k t k t k t k k g k g g g g g g
 y y k y y k k t t t t k t k g k k g g g g k
 k k k k k k k k k k k k k k k k k k k k k k
 k b b b b b k r k r r r r k t k t k k t t t
 b k b b k b k r k r r r k k t t t k k t t t
 b b b k b k k r k r r r k k t t t t t t t t
 b b k b k b k r r r r k r k k k t t t k t t
 b b b k b k k r r k r r k k k t k t t t t k
 b b b b b b k k r r r k r k t t k k t k t t
 b b b k k k k r k r r r r k t t k k k t t t
 b k k b k b k r r k r r k k t k t t k k k t
 b b b b k b k k r r r k r k k t t k k k t k
 b b k b b b k r r r k r k k t k t t k k t t
]
Output2 = [
 y t g
 b r t
]

Train example 3:
Input3 = [
 r r r r r k k k k k k o k k o k k
 r r k k r k r k o k o k o o o o k
 r r r r k r r k k o o k k o o k o
 r k r r k r r k k k o o o o o o k
 r r r k r r r k k o k o o o k k k
 r k r k r r r k o o k o o k k o o
 k k k k k k k k k k k k k k k k k
 k y y y y y k k k t k t t t t t t
 y k y y k y k k t k t t t t t t t
 y k k y k y y k k t k t t k t k t
 y y k k k k y k t t k t t t t t t
 y y y y k k k k t t t t t t t t k
 y y y y k y y k t t t t t t t t t
 y y y y y y k k t t t k k t t t k
 k y y y k y y k t t k t t t t k t
 k k k k y y y k k t k k t k t t t
 k k k k k k k k k k k k k k k k k
 k b b b b b b k p p k p p k p p p
 k b b b b b k k p p p k p p p p k
 b b b b b k b k p p p p k p p p p
 b k k k b b b k p p p k p p p p p
 b k b b b k k k p p p p p k k p p
 b b b b b b b k p p p p p p p p p
]
Output3 = [
 r o
 y t
 b p
] """

# concepts:
# downscaling, rectangular partitions

# description:
# In the input you will see a grid consisting of a chessboard pattern (rectangular partitions) of different colors.
# Each rectangular partition region is incompletely scattered with a color. Regions are separated by black lines, going all the way top-bottom/left-right. 
# To make the output, make a grid with one color pixel for each colored rectangular region of the input.

def transform_780d0b14(input_grid):
    # Plan:
    # 1. Partition the input into rectangular regions by finding all horizontal and vertical black lines
    # 2. For each region, find the color of the region
    # 3. Use one pixel to represent the original region and create the output grid

    # 1. Input parsing
    # Get the shape of the input grid
    width, height = input_grid.shape
    background = Color.BLACK
    # Find all horizontal and vertical lines
    vertical_lines = [ x for x in range(width) if np.all(input_grid[x, :] == background) ]
    horizontal_lines = [ y for y in range(height) if np.all(input_grid[:, y] == background) ]
    
    # Start from (0, 0)
    vertical_lines = [0] + vertical_lines
    horizontal_lines = [0] + horizontal_lines

    # Deduplicate successive lines
    vertical_lines = [x for i, x in enumerate(vertical_lines) if i == 0 or x != vertical_lines[i - 1]]
    horizontal_lines = [y for i, y in enumerate(horizontal_lines) if i == 0 or y != horizontal_lines[i - 1]]

    # use one pixel to represent the original region and create the output grid
    output_width, output_height = len(vertical_lines), len(horizontal_lines)
    output_grid = np.full((output_width, output_height), background) 

    # Initialize the output grid
    for i in range(len(vertical_lines)):
        for j in range(len(horizontal_lines)):
            # Get the region of the color
            x1 = vertical_lines[i]
            x2 = vertical_lines[i + 1] if i + 1 < len(vertical_lines) else width
            y1 = horizontal_lines[j]
            y2 = horizontal_lines[j + 1] if j + 1 < len(horizontal_lines) else height

            # Get the original region
            region = input_grid[x1:x2, y1:y2]
            # Get the color of the region
            color = object_colors(region, background=Color.BLACK)[0]
            # Use one pixel to represent the original region
            output_grid[i, j] = color

    return output_grid

""" ==============================
Puzzle 7837ac64 """

# concepts:
# downscaling

# description:
# In the input you will see horizontal and vertical bars/dividers of a particular color that define rectangular regions, with some of the single-pixel vertices colored differently.
# Some rectangular regions are have same color on the four vertices, and some are not.
# To make the output, find the regions colored differently on all vertices and produce a single output pixel of that color in the corresponding part of the output.
# Ignore regions which just have the color of the horizontal and vertical bars at their vertices.

def transform_7837ac64(input_grid):
    # Plan:
    # 1. Parse the input into  dividers, regions, and vertices
    # 2. Extract the regions colored differently from the divider on all vertices
    # 3. Produce the output grid by representing each region with a single pixel of the color of its vertices, as long as its color is not the divider

    # 1. Parse the input
    # Detect the objects
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    # The divider color is the most frequent non-background color, and the background is black
    divider_color = max(Color.NOT_BLACK, key=lambda color: np.sum(input_grid == color))

    # Detect the single pixels that form the vertices of the regions
    pixels = [ obj for obj in objects if object_colors(obj) != [divider_color] ]
    x_positions = [object_position(obj)[0] for obj in pixels]
    y_positions = [object_position(obj)[1] for obj in pixels]

    # Ignore regions that are not part of those special pixels
    x_min, x_max = min(x_positions), max(x_positions)
    y_min, y_max = min(y_positions), max(y_positions)
    input_grid = input_grid[x_min:x_max+1, y_min:y_max+1]

    # Extract just the black regions delimited by the divider color
    regions = find_connected_components(input_grid, background=divider_color, connectivity=4, monochromatic=True)
    regions = [region for region in regions if object_colors(region, background=divider_color) == [Color.BLACK]]

    # 2. Analyze vertices, which live on the diagonal corners of the regions, to find the color of the regions
    # Determine their colors by the colors of their vertices, so we are going to have to look at the corners
    def diagonal_corners(obj, background):
        x, y, w, h = bounding_box(obj, background)
        return [(x-1, y-1), (x + w, y-1), (x-1, y + h), (x + w, y + h)]
    
    region_colors = []
    for region in regions:
        vertex_colors = { input_grid[x, y] for x, y in diagonal_corners(region, background=divider_color) }
        vertex_colors = set(vertex_colors)
        if len(vertex_colors) == 1 and vertex_colors != {divider_color}:
            region_colors.append(vertex_colors.pop())
        else:
            region_colors.append(Color.BLACK)

    # 3. Produce the output grid, representing each big region as a single pixel
    
    # Find the number distinct X/Y positions of the regions, which tells us the size of the output
    x_positions = sorted({object_position(region, background=divider_color)[0] for region in regions})
    y_positions = sorted({object_position(region, background=divider_color)[1] for region in regions})

    # Make the output
    output_grid = np.full((len(x_positions), len(y_positions)), Color.BLACK)

    for region, color in zip(regions, region_colors):
        x, y = object_position(region, background=divider_color)
        output_grid[x_positions.index(x), y_positions.index(y)] = color
    
    return output_grid

""" ==============================
Puzzle 7c008303

Train example 1:
Input1 = [
 r y t k k k k k k
 b p t k k k k k k
 t t t t t t t t t
 k k t k g k k g k
 k k t g g g g g g
 k k t k g k k g k
 k k t k g k k g k
 k k t g g g g g g
 k k t k g k k g k
]
Output1 = [
 k r k k y k
 r r r y y y
 k r k k y k
 k b k k p k
 b b b p p p
 k b k k p k
]

Train example 2:
Input2 = [
 k k k k k k t b r
 k k k k k k t y b
 t t t t t t t t t
 k k g g k g t k k
 g g k k k k t k k
 g g k g k g t k k
 k k k k g k t k k
 g g g g g g t k k
 k k k k g k t k k
]
Output2 = [
 k k b r k r
 b b k k k k
 b b k r k r
 k k k k b k
 y y y b b b
 k k k k b k
]

Train example 3:
Input3 = [
 k k t k k g k k g
 k k t k k g k k g
 k k t g g k g g k
 k k t k k k k g k
 k k t k g k g k k
 k k t k g k k k g
 t t t t t t t t t
 r y t k k k k k k
 p e t k k k k k k
]
Output3 = [
 k k r k k y
 k k r k k y
 r r k y y k
 k k k k e k
 k p k e k k
 k p k k k e
] """

# concepts:
# color correspondence, object splitting

# description:
# In the input you will see a 9x9 grid with a 6x6 green sprite and a 2x2 sprite with 4 different colors separated by two teal lines.
# To make the output grid, you should separate the 6x6 green sprite into 4 3x3 sub-sprites and color them 
# with the 4 different colors in the 2x2 sprite, with the same relative position.

def transform_7c008303(input_grid):
    # Detect four parts seperated by two intersected teal lines.
    sub_grids = find_connected_components(grid=input_grid, connectivity=4, monochromatic=False, background=Color.TEAL)

    # Find the green pattern and square with four colors as a coloring guidance.
    for sub_grid in sub_grids:
        cropped_sub_grid = crop(grid=sub_grid, background=Color.TEAL)

        # If this part is the color guide, then store the color guide.
        if np.all(cropped_sub_grid != Color.BLACK) and np.all(cropped_sub_grid != Color.GREEN):
            color_guide = cropped_sub_grid

        # If this part is the green pattern, then store the green pattern.s
        elif np.any(cropped_sub_grid == Color.GREEN):
            green_pattern = cropped_sub_grid
    
    # Caculate the size of four sub-sprites on green pattern to be colored.
    width_green, height_green = green_pattern.shape
    width_green_half, height_green_half = width_green // 2, height_green // 2
    
    # Color each sub-sprite on the green pattern follow the color guide: with the color in same relative position.
    green_pattern[0: width_green_half, 0: height_green_half][green_pattern[0: width_green_half, 0: height_green_half] == Color.GREEN] = color_guide[0, 0]
    green_pattern[width_green_half: width_green, 0: height_green_half][green_pattern[width_green_half: width_green, 0: height_green_half] == Color.GREEN] = color_guide[1, 0]
    green_pattern[0: width_green_half, height_green_half: height_green][green_pattern[0: width_green_half, height_green_half: height_green] == Color.GREEN] = color_guide[0, 1]
    green_pattern[width_green_half: width_green, height_green_half: height_green][green_pattern[width_green_half: width_green, height_green_half: height_green] == Color.GREEN] = color_guide[1, 1]

    output_grid = green_pattern
    return output_grid

""" ==============================
Puzzle 7e0986d6

Train example 1:
Input1 = [
 t k k k t k g g g g g t k k
 k k k k k k g g g g g k k k
 k k k k k k g g t g t k k k
 k k g g g k g g g g g k k k
 k k g g g k g t g g g k k k
 k k g g t k k k k k k k k k
 k k k k k k k k k g g g g g
 k t k g g g t g k g g g t g
 k k k g t g g g k g g g g g
 k k k g g g g g k g g g g g
 g g g g t g g g t k k k k k
 g g g k k k k k k k k k k k
 g t g k t k k k k k k k k t
]
Output1 = [
 k k k k k k g g g g g k k k
 k k k k k k g g g g g k k k
 k k k k k k g g g g g k k k
 k k g g g k g g g g g k k k
 k k g g g k g g g g g k k k
 k k g g g k k k k k k k k k
 k k k k k k k k k g g g g g
 k k k g g g g g k g g g g g
 k k k g g g g g k g g g g g
 k k k g g g g g k g g g g g
 g g g g g g g g k k k k k k
 g g g k k k k k k k k k k k
 g g g k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k
 k r r r r k b k k k k k k k k k
 k r r r r k k k k r b r r r r r
 k r r b r k k k k r r r r r r r
 k r r r r k k k k r r r r r r r
 k r r r r b k k k r r r r r b r
 k k k k k k k k k r r r r r r r
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k b k k k
 k b k r r r r r r b r r r k k k
 k k k b r r r r r r r r r k k b
 k k k r r r r r r b r r b k k k
 k k k r r r r r r r r r r k k k
]
Output2 = [
 k k k k k k k k k k k k k k k k
 k r r r r k k k k k k k k k k k
 k r r r r k k k k r r r r r r r
 k r r r r k k k k r r r r r r r
 k r r r r k k k k r r r r r r r
 k r r r r k k k k r r r r r r r
 k k k k k k k k k r r r r r r r
 k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k
 k k k r r r r r r r r r r k k k
 k k k r r r r r r r r r r k k k
 k k k r r r r r r r r r r k k k
 k k k r r r r r r r r r r k k k
] """

# concepts:
# denoising, topology

# description:
# In the input you will see a black background with large rectangles of the same color, and random "noise" pixels added at random locations sometimes on the rectangles and sometimes not.
# To make the output, remove the noise pixels to reveal the rectangles.

def transform_7e0986d6(input_grid):
    # Plan:
    # 1. Extract and classify the objects (rectangles, noise pixels)
    # 2. If a noise pixel is significantly touching a rectangle (it has at least 2 neighbors that are part of the rectangle), then the noise reveals the rectangle color
    # 3. Otherwise, the noise reveals the background, so delete the noise pixel

    objects = find_connected_components(input_grid, monochromatic=True, connectivity=4, background=Color.BLACK)
    rectangle_size_threshold = 4
    noisy_objects = [ obj for obj in objects if np.sum(obj != Color.BLACK) < rectangle_size_threshold ]
    rectangle_objects = [ obj for obj in objects if np.sum(obj != Color.BLACK) >= rectangle_size_threshold ]

    output_grid = np.copy(input_grid)
    for noise_object in noisy_objects:
        noise_object_mask = noise_object != Color.BLACK
        noise_neighbors_mask = object_neighbors(noise_object, connectivity=4, background=Color.BLACK)

        for rectangle_object in rectangle_objects:
            # Check if the noise object has at least 2 neighbors that are part of this rectangle
            rectangle_object_mask = rectangle_object != Color.BLACK
            if np.sum(noise_neighbors_mask & rectangle_object_mask) >= 2:
                rectangle_color = np.argmax(np.bincount(rectangle_object[rectangle_object_mask]))
                output_grid[noise_object_mask] = rectangle_color
                break
        else:
            # Delete this noise object
            output_grid[noise_object_mask] = Color.BLACK

    return output_grid

""" ==============================
Puzzle 7f4411dc

Train example 1:
Input1 = [
 k o k k k k k k o o o o o
 k k k k k k k k o o o o o
 k k k k o k k k o o o o o
 k k o o o o k k k k k k k
 k k o o o o k k k k k k k
 k k o o o o k k k k o k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k o k k k k k k o o o k k
 k k k k k k k k o o o k k
 k k k k k k k k k k k o k
 k k k k k k k k k k k k k
 k k k k o k k k k k k k k
]
Output1 = [
 k k k k k k k k o o o o o
 k k k k k k k k o o o o o
 k k k k k k k k o o o o o
 k k o o o o k k k k k k k
 k k o o o o k k k k k k k
 k k o o o o k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k o o o k k
 k k k k k k k k o o o k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k p k k k k
 k p k k k k p k k k k p p p k k k
 k k k k k k k k k k k p p p k k p
 k k k k p k k k k k k k k k k k k
 k k k p p p p k k k k k k k k k k
 k k k p p p p k k k k k p k k k k
 k k k p p p p k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k p k k k
 k k k p k k k k k p p p p p k k k
 k k k k k k k k k p p p p p k k k
 k k p p k k k p k p p p p p k k k
 k k p p k k k k k k k k k k k k k
 k k p p k k k k k k k k k k k k k
 k k k k k k k k k k k k k p k k k
 k k k k k k k k k k p k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k p p p k k k
 k k k k k k k k k k k p p p k k k
 k k k k k k k k k k k k k k k k k
 k k k p p p p k k k k k k k k k k
 k k k p p p p k k k k k k k k k k
 k k k p p p p k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k p p p p p k k k
 k k k k k k k k k p p p p p k k k
 k k p p k k k k k p p p p p k k k
 k k p p k k k k k k k k k k k k k
 k k p p k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k e k k k k k
 k k k k k k e
 k e e e e k k
 k e e e e k k
 k e e e e e k
 e k k k k k k
 k k k k k e k
]
Output3 = [
 k k k k k k k
 k k k k k k k
 k e e e e k k
 k e e e e k k
 k e e e e k k
 k k k k k k k
 k k k k k k k
] """

# concepts:
# denoising, topology

# description:
# In the input you will see a black background with large rectangles of the same color, and random "noise" pixels added at random locations, in the same color as the rectangles.
# To make the output, remove the noise pixels, leaving only the big rectangles.

def transform_7f4411dc(input_grid):
    # Plan:
    # 1. Find the neighbors of each pixel
    # 2. If its neighbors are mostly colored (>=2 neighbors), it's part of a rectangle
    # 3. Otherwise, it's noise, so delete it

    output_grid = np.copy(input_grid)

    for x, y in np.argwhere(input_grid != Color.BLACK):
        # Turn this single pixel into an object, and get its neighbors
        obj = np.full(input_grid.shape, Color.BLACK)
        obj[x, y] = input_grid[x, y]
        neighbors_mask = object_neighbors(obj, connectivity=4, background=Color.BLACK)

        # If the object has at least 2 colored neighbors, then it is part of a rectangle. Otherwise, it is noise, so delete it.
        colors_of_neighbors = input_grid[neighbors_mask]
        if np.sum(colors_of_neighbors != Color.BLACK) >= 2:
            # has at least 2 colored neighbors, so it's part of a rectangle
            pass
        else:
            # doesn't have at least 2 colored neighbors, so delete it
            output_grid[x, y] = Color.BLACK

    return output_grid

""" ==============================
Puzzle 810b9b61

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k b b b k k
 k k b b b b k k k k b k b k k
 k k b k k b k k k k b k b k k
 k k b b b b k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k b b k
 k k k b k k b b b b k k b k k
 k k k k k k b k k b k k k k k
 k k k k k k b k k b k k k k k
 b b b k k k b b b b k k k k k
 b k b k k k k k k k k k k k k
 b k b k k k k k k k b b b b k
 b b b k k b b k k k b k k b k
 k k k k k k k k k k b b b b k
]
Output1 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k b b b k k
 k k g g g g k k k k b k b k k
 k k g k k g k k k k b k b k k
 k k g g g g k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k b b k
 k k k b k k g g g g k k b k k
 k k k k k k g k k g k k k k k
 k k k k k k g k k g k k k k k
 g g g k k k g g g g k k k k k
 g k g k k k k k k k k k k k k
 g k g k k k k k k k g g g g k
 g g g k k b b k k k g k k g k
 k k k k k k k k k k g g g g k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k b b b k k k b k k k k
 k k k k b k b k k k b k k k k
 k k k k b b b k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k b k k k k b k b b k k
 k k k k k k k k k b k k b k k
 k k k k k k k k k b b b b k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k g g g k k k b k k k k
 k k k k g k g k k k b k k k k
 k k k k g g g k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k b k k k k b k b b k k
 k k k k k k k k k b k k b k k
 k k k k k k k k k b b b b k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k
 k k k k k k k k k
 k b b b b b k k k
 k b k k k b k k k
 k b b b b b k k k
 k k k k k k k k k
 b b k k k k k k k
 k b k k b b k k k
 k b k k k k k k k
]
Output3 = [
 k k k k k k k k k
 k k k k k k k k k
 k g g g g g k k k
 k g k k k g k k k
 k g g g g g k k k
 k k k k k k k k k
 b b k k k k k k k
 k b k k b b k k k
 k b k k k k k k k
] """

# concepts:
# objects, topology

# description:
# In the input grid, you will see various blue objects. Some are "hollow" and contain a fully-enclosed region, while others do not have a middle that is separate from outside the object, and fully enclosed.
# To create the output grid, copy the input grid. Then, change the color of all "hollow" shapes to be green.

def transform_810b9b61(input_grid):
    objects = find_connected_components(input_grid, connectivity=4)
    output_grid = input_grid.copy()
    for object in objects:
        if is_hollow(object):
            object[object != Color.BLACK] = Color.GREEN
        blit_object(output_grid, object, background=Color.BLACK)

    return output_grid

def is_hollow(object):
    # to check if it contains a fully enclosed region, find everything that is enclosed by the object (in its interior), but not actually part of the object
    interior_mask = object_interior(object)
    object_mask = object != Color.BLACK
    hollow_mask = interior_mask & ~object_mask
    return np.any(hollow_mask)

""" ==============================
Puzzle 834ec97d

Train example 1:
Input1 = [
 k r k
 k k k
 k k k
]
Output1 = [
 k y k
 k r k
 k k k
]

Train example 2:
Input2 = [
 k k k k k
 k k k k k
 k k p k k
 k k k k k
 k k k k k
]
Output2 = [
 y k y k y
 y k y k y
 y k y k y
 k k p k k
 k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k m k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
]
Output3 = [
 y k y k y k y k y
 y k y k y k y k y
 y k y k y k y k y
 y k y k y k y k y
 y k y k y k y k y
 k k m k k k k k k
 k k k k k k k k k
 k k k k k k k k k
 k k k k k k k k k
] """

# concepts:
# geometric pattern

# description:
# In the input you will see a grid with a single coloured pixel.
# To make the output, move the colored pixel down one pixel and draw a yellow line from the pixel to the top of the grid.
# Finally repeat the yellow line by repeating it horizontally left/right with a period of 2 pixels.

def transform_834ec97d(input_grid):
    # Plan:
    # 1. Extract the pixel from the input grid
    # 2. Move the pixel one pixel down
    # 3. Draw a yellow line from the pixel to the top of the grid, repeating it horizontally left/right with a period of 2 pixels

    # 1. Extract the pixel
    pixel = find_connected_components(input_grid, monochromatic=True)[0]
    pixel_x, pixel_y = object_position(pixel)
    pixel_color = object_colors(pixel)[0]

    # 2. Move the pixel one pixel down
    output_grid = input_grid.copy()
    output_grid[pixel_x, pixel_y + 1] = pixel_color
    output_grid[pixel_x, pixel_y] = Color.BLACK

    # 3. Draw the vertical line from the pixel to top

    # Draw the line from left to right
    horizontal_period = 2
    for x in range(pixel_x, output_grid.shape[0], horizontal_period):
        draw_line(output_grid, x=x, y=pixel_y, direction=(0, -1), color=Color.YELLOW)

    # Draw the line from left to right
    for x in range(pixel_x, -1, -horizontal_period):
        draw_line(output_grid, x=x, y=pixel_y, direction=(0, -1), color=Color.YELLOW)
    return output_grid

""" ==============================
Puzzle 8403a5d5

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k r k k k k k k k k
]
Output1 = [
 k r e r k r e r k r
 k r k r k r k r k r
 k r k r k r k r k r
 k r k r k r k r k r
 k r k r k r k r k r
 k r k r k r k r k r
 k r k r k r k r k r
 k r k r k r k r k r
 k r k r k r k r k r
 k r k r e r k r e r
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k g k k k k
]
Output2 = [
 k k k k k g e g k g
 k k k k k g k g k g
 k k k k k g k g k g
 k k k k k g k g k g
 k k k k k g k g k g
 k k k k k g k g k g
 k k k k k g k g k g
 k k k k k g k g k g
 k k k k k g k g k g
 k k k k k g k g e g
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k y k k k k k
]
Output3 = [
 k k k k y e y k y e
 k k k k y k y k y k
 k k k k y k y k y k
 k k k k y k y k y k
 k k k k y k y k y k
 k k k k y k y k y k
 k k k k y k y k y k
 k k k k y k y k y k
 k k k k y k y k y k
 k k k k y k y e y k
] """

# concepts:
# pattern generation

# description:
# In the input you will see a grid with a single pixel on the bottom of the grid.
# To make the output, you should draw a geometric pattern starting outward from the pixel:
# step 1: draw vertical bars starting from the pixel and going to the right with a horizontal period of 2.
# step 2: put a grey pixel in between the vertical bars alternating between the top / bottom.

def transform_8403a5d5(input_grid):
    # Output grid is the same size as the input grid
    output_grid = np.zeros_like(input_grid)

    # Detect the pixel on the bottom of the grid
    pixel = find_connected_components(input_grid, monochromatic=True)[0]
    pixel_color = object_colors(pixel)[0]
    pixel_x, pixel_y = object_position(pixel)

    # Get the color of the pattern pixel by observation
    pattern_pixel_color = Color.GRAY
    
    # STEP 1: Draw vertical bar from bottom to top starting from the pixel and going to the right, horizontal period of 2
    horizontal_period = 2
    for x in range(pixel_x, output_grid.shape[0], horizontal_period):
        draw_line(output_grid, x=x, y=pixel_y, direction=(0, -1), color=pixel_color)
    
    # STEP 2: put a grey pixel in between the vertical bars alternating between the top / bottom.
    cur_y = -1 if pixel_y == 0 else 0
    for x in range(pixel_x + 1, output_grid.shape[0], horizontal_period):
        output_grid[x, cur_y] = pattern_pixel_color
        # alternate between top and bottom
        cur_y = 0 if cur_y == -1 else -1

    return output_grid

""" ==============================
Puzzle 8a004b2b

Train example 1:
Input1 = [
 k k y k k k k k k k k k k k k y k
 k k k k k k k k k k k k k k k k k
 k k k g g k k k k k k r r k k k k
 k k k g g k k k k k k r r k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k b b k k k k k k k k
 k k k k k k k b b k k k k k k k k
 k k y k k k k k k k k k k k k y k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k g k k k r k k k k k k k k k k
 k k b b k b b k k k k k k k k k k
 k k k b b b k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k
]
Output1 = [
 y k k k k k k k k k k k k y
 k k k k k k k k k k k k k k
 k g g k k k k k k r r k k k
 k g g k k k k k k r r k k k
 k b b b b k k b b b b k k k
 k b b b b k k b b b b k k k
 k k k b b b b b b k k k k k
 k k k b b b b b b k k k k k
 y k k k k k k k k k k k k y
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k k k
 k k y k k k k k y k k k k k k k k k
 k k k r r k k k k k k k k k k k k k
 k k k r r k k k k k k k k k k k k k
 k k k k k t t k k k k k k k k k k k
 k k k k k t t k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k y k k k k k y k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k r g k k k k k k
 k k k k k k k k k k g t k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
]
Output2 = [
 y k k k k k y
 k r r g g k k
 k r r g g k k
 k g g t t k k
 k g g t t k k
 k k k k k k k
 y k k k k k y
]

Train example 3:
Input3 = [
 k k k y k k k k k k k k k y k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k r r r k k k b b b k k k k k
 k k k k r r r k k k b b b k k k k k
 k k k k r r r k k k b b b k k k k k
 k k k k k k k g g g k k k k k k k k
 k k k k k k k g g g k k k k k k k k
 k k k k k k k g g g k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k y k k k k k k k k k y k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k r b b k k k k k k k k k
 k k k k k k k g b k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
]
Output3 = [
 y k k k k k k k k k y
 k k k k k k k k k k k
 k k k k k k k k k k k
 k r r r b b b b b b k
 k r r r b b b b b b k
 k r r r b b b b b b k
 k k k k g g g b b b k
 k k k k g g g b b b k
 k k k k g g g b b b k
 k k k k k k k k k k k
 y k k k k k k k k k y
] """

# concepts:
# puzzle pieces, rescaling

# description:
# In the input you will see a small multicolored object, and a big region bordered by 4 yellow pixels in the corners. The big region contains some colored things.
# To make the output, rescale and translate the small multicolored object so that it is in the big region and matches as much as possible the colors of the things in the big region.
# The output should be just the big region with its 4 yellow pixels (plus the rescaled and translated thing).

def transform_8a004b2b(input_grid):
    # Plan:
    # 1. Detect the little object, yellow pixels, and big region
    # 2. Rescale and translate the little object to cover as much of the big region as possible, matching colors whenever they overlap

    # 1. Detect little object, yellow pixels, and big region
    objects = find_connected_components(input_grid, connectivity=4, background=Color.BLACK, monochromatic=False)
    yellow_pixel_objects = [ obj for obj in objects if set(object_colors(obj, background=Color.BLACK)) == {Color.YELLOW} and crop(obj).shape == (1, 1) ]
    assert len(yellow_pixel_objects) == 4, "There should be exactly 4 yellow pixels"

    # Find the region indicated by the 4 yellow pixels
    yellow_pixel_min_x, yellow_pixel_max_x, yellow_pixel_min_y, yellow_pixel_max_y = \
        min([ object_position(obj, anchor="center")[0] for obj in yellow_pixel_objects ]), \
        max([ object_position(obj, anchor="center")[0] for obj in yellow_pixel_objects ]), \
        min([ object_position(obj, anchor="center")[1] for obj in yellow_pixel_objects ]), \
        max([ object_position(obj, anchor="center")[1] for obj in yellow_pixel_objects ])
    
    # The little object is what is outside that region
    little_objects = [ obj for obj in objects
                      if object_position(obj, anchor="center")[0] < yellow_pixel_min_x or object_position(obj, anchor="center")[0] > yellow_pixel_max_x or object_position(obj, anchor="center")[1] < yellow_pixel_min_y or object_position(obj, anchor="center")[1] > yellow_pixel_max_y ]
    assert len(little_objects) == 1, "There should be exactly one little object"
    little_object = little_objects[0]

    # The output grid is going to be the region delimited by the yellow pixels
    output_grid = input_grid[yellow_pixel_min_x:yellow_pixel_max_x+1, yellow_pixel_min_y:yellow_pixel_max_y+1]

    # The big region has some colored pixels that we are going to try and match with
    # Extract the content of big region as tuples (x, y, color)
    big_region_pixels = [ (x, y, output_grid[x, y])
                         for x in range(output_grid.shape[0]) for y in range(output_grid.shape[1])
                         if output_grid[x, y] != Color.BLACK and output_grid[x, y] != Color.YELLOW ]

    # 2. Rescale and translate the little object to cover as much of the big region as possible, matching colors whenever they overlap
    little_sprite = crop(little_object, background=Color.BLACK)
    scaled_sprites = [ scale_sprite(little_sprite, factor) for factor in [1, 2, 3, 4, 5, 6] ]
    # A placement solution is a tuple of (x, y, scaled_sprite) where x, y is the top-left corner of the scaled sprite
    possible_solutions = [ (x, y, scaled_sprite) for scaled_sprite in scaled_sprites
                          for x in range(output_grid.shape[0] - scaled_sprite.shape[0])
                          for y in range(output_grid.shape[0] - scaled_sprite.shape[1]) ]
    
    # Filter placement solutions to only those where the colors of the big region match the colors of the scaled+translated sprite
    def valid_solution(x, y, scaled_sprite):
        # Make a canvas to try putting down the scaled sprite
        test_canvas = np.full_like(output_grid, Color.BLACK)
        blit_sprite(test_canvas, scaled_sprite, x, y)
        # Check if every big region color is also in the test canvas
        test_colors = [ (x, y, test_canvas[x, y]) for x in range(output_grid.shape[0]) for y in range(output_grid.shape[1]) if test_canvas[x, y] != Color.BLACK ]
        return all( (x, y, color) in test_colors for x, y, color in big_region_pixels )        
    
    possible_solutions = [ solution for solution in possible_solutions if valid_solution(*solution) ]
    if len(possible_solutions) == 0:
        assert False, "No solution found for the little object"
    
    # Pick the first solution and blit the sprite into the output grid
    x, y, scaled_sprite = list(possible_solutions)[0]
    blit_sprite(output_grid, scaled_sprite, x, y)

    return output_grid

""" ==============================
Puzzle 8d5021e8

Train example 1:
Input1 = [
 k t
 k k
 k t
]
Output1 = [
 t k k t
 k k k k
 t k k t
 t k k t
 k k k k
 t k k t
 t k k t
 k k k k
 t k k t
]

Train example 2:
Input2 = [
 r k
 r r
 r k
]
Output2 = [
 k r r k
 r r r r
 k r r k
 k r r k
 r r r r
 k r r k
 k r r k
 r r r r
 k r r k
]

Train example 3:
Input3 = [
 k k
 k e
 e k
]
Output3 = [
 k e e k
 e k k e
 k k k k
 k k k k
 e k k e
 k e e k
 k e e k
 e k k e
 k k k k
] """

# concepts:
# flip

# description:
# In the input you will see a monochromatic sprite.
# To make the output, 
# 1. flip the grid horizontally with y-axis on the left side of the grid, making the canvas twice larger.
# 2. flip it down with x-axis on the bottom side of the grid.
# 3. concatenate the flipped grid in step 2 to the top and bottom of the grid in step 1.
# In total the output grid is twice as wide and three times as tall as the input.

def transform_8d5021e8(input_grid):
    # Create the output grid twice as wide and three times as tall
    n, m = input_grid.shape
    output_grid = np.zeros((n * 2, m * 3), dtype=int)

    # Step 1: Flip the grid horizontally with y-axis on the left side of the grid, concate it to the left.
    # Place it in the middle of the output grid
    flip_grid = np.flipud(input_grid)
    blit_sprite(output_grid, sprite=flip_grid, x=0, y=m)
    blit_sprite(output_grid, sprite=input_grid, x=n, y=m)

    # Step 2: Flip it down with x-axis on the bottom side of the grid, concate it to the bottom and top.
    original_object = output_grid[:, m :2 * m]
    filp_down_object = np.fliplr(original_object)
    blit_sprite(output_grid, sprite=filp_down_object, x=0, y=m * 2)
    blit_sprite(output_grid, sprite=filp_down_object, x=0, y=0)
    
    return output_grid

""" ==============================
Puzzle 8d510a79

Train example 1:
Input1 = [
 k k b k k k k k k k
 k k k k k k b k k k
 k r k k k k k k k r
 k k k k k k k k k k
 k k k k k k k k k k
 e e e e e e e e e e
 k k k k k k k k k k
 k k k k k k k k k k
 k b k k k r k k b k
 k k k k k k k k k k
]
Output1 = [
 k k b k k k b k k k
 k k k k k k b k k k
 k r k k k k k k k r
 k r k k k k k k k r
 k r k k k k k k k r
 e e e e e e e e e e
 k k k k k r k k k k
 k k k k k r k k k k
 k b k k k r k k b k
 k b k k k k k k b k
]

Train example 2:
Input2 = [
 k r k b k k k k k k
 k k k k k r k b k k
 k k k k k k k k k k
 e e e e e e e e e e
 k k k k k k k k k k
 k r k k k k k k k r
 k k k k b k k k k k
 k k k k k k k k k k
 k k b k k k r k b k
 k k k k k k k k k k
]
Output2 = [
 k r k b k k k b k k
 k r k k k r k b k k
 k r k k k r k k k k
 e e e e e e e e e e
 k r k k k k r k k r
 k r k k k k r k k r
 k k k k b k r k k k
 k k k k b k r k k k
 k k b k b k r k b k
 k k b k b k k k b k
] """

# concepts:
# magnetism, lines

# description:
# In the input, you will see a horizontal grey line on a black background, with red and blue pixels scattered on either side of the line.
# To make the output, draw vertical lines from each of the blue and red pixels, with lines from the red pixels going toward the grey line and lines from the blue pixels going away from the grey line. 
# These lines should stop when they hit the grey line or the edge of the grid.

def transform_8d510a79(input_grid):
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # find the location of the horizontal grey line
    grey_line = np.where(output_grid == Color.GREY)

    # get the unique y-coordinates of the grey line
    grey_line_y = np.unique(grey_line[1])

    # find the red and blue pixels
    red_pixels = np.where(output_grid == Color.RED)
    blue_pixels = np.where(output_grid == Color.BLUE)

    # draw lines from the red pixels toward the grey line
    for i in range(len(red_pixels[0])):
        x, y = red_pixels[0][i], red_pixels[1][i]
        # make sure to handle the case where the red pixel is below the grey line and the case where it is above
        if y < grey_line_y:
            draw_line(output_grid, x, y, length=None, color=Color.RED, direction=(0, 1), stop_at_color=[Color.GREY])
        else:
            draw_line(output_grid, x, y, length=None, color=Color.RED, direction=(0, -1), stop_at_color=[Color.GREY])

    # draw lines from the blue pixels away from the grey line, using draw_line
    for i in range(len(blue_pixels[0])):
        x, y = blue_pixels[0][i], blue_pixels[1][i]
        # make sure to handle the case where the blue pixel is below the grey line and the case where it is above
        if y < grey_line_y:
            draw_line(output_grid, x, y, length=None, color=Color.BLUE, direction=(0, -1), stop_at_color=[Color.GREY])
        else:
            draw_line(output_grid, x, y, length=None, color=Color.BLUE, direction=(0, 1), stop_at_color=[Color.GREY])

    return output_grid

""" ==============================
Puzzle 8e1813be

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 r r r r r r r r r r r r r r r
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 b b b b b b b b b b b b b b b
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 g g g g g g g g g g g g g g g
 k k k k k k k k k k k k k k k
 k e e e e e e k k k k k k k k
 k e e e e e e k y y y y y y y
 k e e e e e e k k k k k k k k
 k e e e e e e k k k k k k k k
 k e e e e e e k t t t t t t t
 k e e e e e e k k k k k k k k
 k k k k k k k k k k k k k k k
 p p p p p p p p p p p p p p p
]
Output1 = [
 r r r r r r
 b b b b b b
 g g g g g g
 y y y y y y
 t t t t t t
 p p p p p p
]

Train example 2:
Input2 = [
 k k k k k r k k y k
 k e e e k r k k y k
 k e e e k r k k y k
 k e e e k r k k y k
 k k k k k r k k y k
 k k b k k r k k y k
 k k b k k r k k y k
 k k b k k r k k y k
 k k b k k r k k y k
 k k b k k r k k y k
 k k b k k r k k y k
 k k b k k r k k y k
]
Output2 = [
 b r y
 b r y
 b r y
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k
 r r r r r k e e e e k r
 k k k k k k e e e e k k
 k k k k k k e e e e k k
 t t t t t k e e e e k t
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 y y y y y y y y y y y y
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 b b b b b b b b b b b b
 k k k k k k k k k k k k
]
Output3 = [
 r r r r
 t t t t
 y y y y
 b b b b
] """

# concepts:
# color stripe, fill the square in order

# description:
# In the input you will see several vertical or horizontal stripes of different colors and a gray square with length equal the number of the stripes.
# To make the output, fill the gray square with the colors of the stripes in the order and direction they appear.

def transform_8e1813be(input_grid):
    # Find the gray square in the input grid and get its position and size
    gray_rectangle = detect_objects(grid=input_grid, colors=[Color.GRAY], monochromatic=True, connectivity=4)[0]
    output_grid = crop(grid=gray_rectangle)

    # Get the color lines in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=8)
    lines = []
    for obj in objects:
        # Get the position and size of the color line
        x_pos, y_pos, x_len, y_len = bounding_box(grid=obj)
        color = obj[x_pos, y_pos]
        if color != Color.GRAY:
            lines.append({'x': x_pos, 'y': y_pos, 'color': color, 'x_len': x_len, 'y_len': y_len})
    
    # Check if the stripes are horizontal or vertical
    is_horizontal = all(line['y_len'] == 1 for line in lines)

    # Sort the color lines by their position
    if is_horizontal:
        lines = sorted(lines, key=lambda x: x['y'])
    else:
        lines = sorted(lines, key=lambda x: x['x'])
    
    # Get the direction of the stripes
    direction = (1, 0) if is_horizontal else (0, 1)

    # Find the colors lines in the input grid in order
    colors_in_square = [c['color'] for i, c in enumerate(lines) if i == 0 or c['color'] != lines[i - 1]['color']]

    # Fill the gray square with the colors of the stripes in the order and direction they appear
    for i in range(len(output_grid)):
        # Draw horizontal line
        if is_horizontal:
            draw_line(grid=output_grid, x=0, y=i, direction=direction, color=colors_in_square[i])
        # Draw vertical line
        else:
            draw_line(grid=output_grid, x=i, y=0, direction=direction, color=colors_in_square[i])
            
    return output_grid

""" ==============================
Puzzle 8e5a5113

Train example 1:
Input1 = [
 b b r e k k k e k k k
 y b b e k k k e k k k
 y y b e k k k e k k k
]
Output1 = [
 b b r e y y b e b y y
 y b b e y b b e b b y
 y y b e b b r e r b b
]

Train example 2:
Input2 = [
 p g g e k k k e k k k
 p g g e k k k e k k k
 p g r e k k k e k k k
]
Output2 = [
 p g g e p p p e r g p
 p g g e g g g e g g p
 p g r e r g g e g g p
]

Train example 3:
Input3 = [
 r o t e k k k e k k k
 o o t e k k k e k k k
 t t t e k k k e k k k
]
Output3 = [
 r o t e t o r e t t t
 o o t e t o o e t o o
 t t t e t t t e t o r
] """

# concepts:
# rotate, position

# description:
# In the input you will see a grid with 3 regions separated by grey horizontal lines. The leftmost region contains a multicolored sprite and the others are empty (black)
# To make the output, rotate the leftmost region 90 degree clockwise and place it in the first empty region, then rotate it a further 90 degrees and put it in the second empty region, etc.

def transform_8e5a5113(input_grid):
    # Get all the regions separated by the divider in the input grid
    divider_color = Color.GRAY
    regions = find_connected_components(input_grid, connectivity=4, background=divider_color, monochromatic=False)

    # Sort the region by x position so that we can get the leftmost, middle, and rightmost regions
    regions.sort(key=lambda region: object_position(region, background=divider_color)[0])

    # We are going to draw on top of the input
    output_grid = input_grid.copy()

    # Get the leftmost region which contains the multicolored sprite
    leftmost_region = regions[0]
    template_sprite = crop(grid=leftmost_region, background=divider_color)

    empty_regions = regions[1:]

    for empty_region in empty_regions:
        # Rotate the template sprite 90 degree clockwise
        template_sprite = np.rot90(template_sprite)

        # Place the rotated template sprite in the empty region
        x, y = object_position(empty_region, background=divider_color)
        blit_sprite(output_grid, sprite=template_sprite, x=x, y=y)
    
    return output_grid

""" ==============================
Puzzle 90c28cc7

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k k k k k k k
 k t t t t t t t t o o o o o o o o o o o o
 k t t t t t t t t o o o o o o o o o o o o
 k t t t t t t t t o o o o o o o o o o o o
 k t t t t t t t t o o o o o o o o o o o o
 k t t t t t t t t o o o o o o o o o o o o
 k t t t t t t t t o o o o o o o o o o o o
 k g g g g g g g g y y y y y y b b b b b b
 k g g g g g g g g y y y y y y b b b b b b
 k g g g g g g g g y y y y y y b b b b b b
 k g g g g g g g g y y y y y y b b b b b b
 k r r r r r r r r e e e e e e e e e e e e
 k r r r r r r r r e e e e e e e e e e e e
 k r r r r r r r r e e e e e e e e e e e e
 k r r r r r r r r e e e e e e e e e e e e
 k r r r r r r r r e e e e e e e e e e e e
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
]
Output1 = [
 t o o
 g y b
 r e e
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k k k k k k
 k r r r r r r r t t t t t t t t k k k k k
 k r r r r r r r t t t t t t t t k k k k k
 k r r r r r r r t t t t t t t t k k k k k
 k r r r r r r r t t t t t t t t k k k k k
 k r r r r r r r t t t t t t t t k k k k k
 k r r r r r r r t t t t t t t t k k k k k
 k r r r r r r r t t t t t t t t k k k k k
 k r r r r r r r t t t t t t t t k k k k k
 k b b b b b b b y y y y y y y y k k k k k
 k b b b b b b b y y y y y y y y k k k k k
 k b b b b b b b y y y y y y y y k k k k k
 k b b b b b b b y y y y y y y y k k k k k
 k b b b b b b b y y y y y y y y k k k k k
 k b b b b b b b y y y y y y y y k k k k k
 k b b b b b b b y y y y y y y y k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
]
Output2 = [
 r t
 b y
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k t t t t t t r r r r r r k k k k k k k
 k k t t t t t t r r r r r r k k k k k k k
 k k t t t t t t r r r r r r k k k k k k k
 k k t t t t t t r r r r r r k k k k k k k
 k k t t t t t t r r r r r r k k k k k k k
 k k g g g g g g g g g g g g k k k k k k k
 k k g g g g g g g g g g g g k k k k k k k
 k k g g g g g g g g g g g g k k k k k k k
 k k g g g g g g g g g g g g k k k k k k k
 k k g g g g g g g g g g g g k k k k k k k
 k k g g g g g g g g g g g g k k k k k k k
 k k y y y y y y b b b b b b k k k k k k k
 k k y y y y y y b b b b b b k k k k k k k
 k k y y y y y y b b b b b b k k k k k k k
 k k y y y y y y b b b b b b k k k k k k k
 k k y y y y y y b b b b b b k k k k k k k
 k k y y y y y y b b b b b b k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
]
Output3 = [
 t r
 g g
 y b
] """

# concepts:
# downscaling, rectangular regions

# description:
# In the input you will see a grid consisting of monochromatic rectangular regions/partitions (chessboard pattern) of different colors.
# Each rectangular region/partition/chessboard cell is filled with a single color, and has a different size.
# To make the output, make a grid with one pixel for each cell of the chessboard.

def transform_90c28cc7(input_grid):
    # Plan:
    # 1. Find the colored regions in the input grid
    # 2. Get the coordinates of the chessboard pattern, which are the x/y positions of the regions
    # 3. Draw the output grid with one pixel for each region (cell of the chessboard)

    # 1. Input parsing
    # Find the colored objects in the input grid
    objects = find_connected_components(grid=input_grid, connectivity=4, monochromatic=True, background=Color.BLACK)

    # 2. Figuring out the chessboard pattern
    # Get the position of the objects
    x_position_list = [object_position(obj)[0] for obj in objects]
    y_position_list = [object_position(obj)[1] for obj in objects]

    # Sort the position list, and get the unique position list since
    # the pattern is separated as chessboard
    x_position_list = sorted(np.unique(x_position_list))
    y_position_list = sorted(np.unique(y_position_list))

    # 3. Drawing the output
    # Get the size of chessboard with one pixel represents the original region
    w_color, h_color = len(x_position_list), len(y_position_list)
    output_grid = np.full((w_color, h_color), Color.BLACK)

    for x_index, x in enumerate(x_position_list):
        for y_index, y in enumerate(y_position_list):
            # Use one pixel to represent the original region
            output_grid[x_index, y_index] = input_grid[x, y]

    return output_grid

""" ==============================
Puzzle 941d9a10

Train example 1:
Input1 = [
 k k e k k k k e k k
 k k e k k k k e k k
 k k e k k k k e k k
 e e e e e e e e e e
 k k e k k k k e k k
 k k e k k k k e k k
 k k e k k k k e k k
 e e e e e e e e e e
 k k e k k k k e k k
 k k e k k k k e k k
]
Output1 = [
 b b e k k k k e k k
 b b e k k k k e k k
 b b e k k k k e k k
 e e e e e e e e e e
 k k e r r r r e k k
 k k e r r r r e k k
 k k e r r r r e k k
 e e e e e e e e e e
 k k e k k k k e g g
 k k e k k k k e g g
]

Train example 2:
Input2 = [
 k k k e k k k k e k
 e e e e e e e e e e
 k k k e k k k k e k
 e e e e e e e e e e
 k k k e k k k k e k
 k k k e k k k k e k
 e e e e e e e e e e
 k k k e k k k k e k
 e e e e e e e e e e
 k k k e k k k k e k
]
Output2 = [
 b b b e k k k k e k
 e e e e e e e e e e
 k k k e k k k k e k
 e e e e e e e e e e
 k k k e r r r r e k
 k k k e r r r r e k
 e e e e e e e e e e
 k k k e k k k k e k
 e e e e e e e e e e
 k k k e k k k k e g
]

Train example 3:
Input3 = [
 k e k k e k e k e k
 k e k k e k e k e k
 k e k k e k e k e k
 e e e e e e e e e e
 k e k k e k e k e k
 k e k k e k e k e k
 e e e e e e e e e e
 k e k k e k e k e k
 k e k k e k e k e k
 k e k k e k e k e k
]
Output3 = [
 b e k k e k e k e k
 b e k k e k e k e k
 b e k k e k e k e k
 e e e e e e e e e e
 k e k k e r e k e k
 k e k k e r e k e k
 e e e e e e e e e e
 k e k k e k e k e g
 k e k k e k e k e g
 k e k k e k e k e g
] """

# concepts:
# objects separated by lines, color correspond to position

# description:
# In the input you will see several gray lines that separate the grid into several parts.
# To make the output grid, you should color the upper left part with blue, the lower right part with green,
# and the middle part with red.

def transform_941d9a10(input_grid):
    # Find all the black rectangles separated by gray lines
    black_rectangles = find_connected_components(grid=input_grid, connectivity=4, monochromatic=False, background=Color.GRAY)

    # Get the bounding box of each black rectangle
    rectangles_lists = []
    for rectangle in black_rectangles:
        x, y, w, h = bounding_box(grid=rectangle, background=Color.GRAY)
        rectangles_lists.append({'x': x, 'y': y, 'w': w, 'h': h})

    # Sort the rectangles by x and y position
    rectangles_lists = sorted(rectangles_lists, key=lambda rec: rec['x'])
    rectangles_lists = sorted(rectangles_lists, key=lambda rec: rec['y'])
    
    left_upper_rectangle, middle_rectangle, right_bottom_rectangle = rectangles_lists[0], rectangles_lists[len(rectangles_lists) // 2], rectangles_lists[-1]

    # Color the left upper part with blue
    blue_grid = np.full((left_upper_rectangle['w'], left_upper_rectangle['h']), Color.BLUE)

    # Color the right lower part with green
    green_grid = np.full((right_bottom_rectangle['w'], right_bottom_rectangle['h']), Color.GREEN)

    # Color the middle part with red
    red_grid = np.full((middle_rectangle['w'], middle_rectangle['h']), Color.RED)

    # Place the blue, green, and red grid on the input grid
    output_grid = input_grid.copy()
    output_grid = blit_sprite(grid=input_grid, sprite=blue_grid, x=left_upper_rectangle['x'], y=left_upper_rectangle['y'])
    output_grid = blit_sprite(grid=input_grid, sprite=green_grid, x=right_bottom_rectangle['x'], y=right_bottom_rectangle['y'])
    output_grid = blit_sprite(grid=input_grid, sprite=red_grid, x=middle_rectangle['x'], y=middle_rectangle['y'])
            
    return output_grid

""" ==============================
Puzzle 97a05b5b

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k k k k k
 k k r r r r r r r r r k k k k k k k k
 k k r k r r r r r k r k k b r b k k k
 k k r k k r r r r k r k k r r r k k k
 k k r r r r r r r k r k k b r b k k k
 k k r r r r r r r r r k k k k k k k k
 k k r r r r r r r r r k k k k k k k k
 k k r r r r k r r r r k k k g g g k k
 k k r r r k k k r r r k k k r r r k k
 k k r r r r k r r r r k k k g g g k k
 k k r r r r r r r r r k k k k k k k k
 k k r r r r r r r r r k k k k k k k k
 k k r r r r r r r r r k k k k k k k k
 k k r r r r r r r r r k k k k k k k k
 k k r k k r r r k r r k k k k k k k k
 k k r r k r r r k r r k k k k k k k k
 k k r k k r r k k k r k k k k k k k k
 k k r r r r r r r r r k k k t t t k k
 k k k k k k k k k k k k k k r t r k k
 k k k k k k k k k k k k k k r r r k k
 k k k y y y k k e e r k k k k k k k k
 k k k y r y k k r r r k k k k k k k k
 k k k r r y k k e e r k k k k k k k k
 k k k k k k k k k k k k k k k k k k k
]
Output1 = [
 r r r r r r r r r
 r r y y r r g r g
 r r r y r r g r g
 r y y y r r g r g
 r r r r r r r r r
 r r r r r r r r r
 r r r b r b r r r
 r r r r r r r r r
 r r r b r b r r r
 r r r r r r r r r
 r r r r r r r r r
 r r r r r r r r r
 r r r r r r r r r
 t r r r r e r e r
 t t r r r e r e r
 t r r r r r r r r
 r r r r r r r r r
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k r r r r r r r r k
 k r r k r r r r r k
 k r k k k r r r r k
 k r r r r r r r r k
 k r r r r k k r r k
 k r r r r k r k r k
 k r r r r r k k r k
 k r r r r r r r r k
 k k k k k k k k k k
 k k k k k k k k k k
 k y r y k k k k k k
 k r r y k k k k k k
 k y r y k r r g k k
 k k k k k r g r k k
 k k k k k g r r k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 r r r r r r r r
 r y r y r r r r
 r r r r r r r r
 r y y y r r r r
 r r r r r r g r
 r r r r r g r r
 r r r r g r r r
 r r r r r r r r
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k
 k k k k k t r t k k k k
 k k k k k r r r k k k k
 k k k k k t r t k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k r r r r r r r r r k k
 k r r r r r r r r r k k
 k r r r k r r r r r k k
 k r r k k k r r r r k k
 k r r r k r r r r r k k
 k r r r r r r r r r k k
 k r r r r r r r r r k k
 k r r r r r r r r r k k
 k r r r r r r r r r k k
 k k k k k k k k k k k k
]
Output3 = [
 r r r r r r r r r
 r r r r r r r r r
 r r t r t r r r r
 r r r r r r r r r
 r r t r t r r r r
 r r r r r r r r r
 r r r r r r r r r
 r r r r r r r r r
 r r r r r r r r r
] """

# concepts:
# puzzle pieces, rotation

# description:
# In the input you will see a red object with holes in it, and several multicolored 3x3 "puzzle pieces"
# To make the output, crop to just the red object and then put each puzzle piece in the red object so that the red pixels in the puzzle piece perfectly fit into the holes in the red object. You can rotate and translate the puzzle pieces to make them fit.

def transform_97a05b5b(input_grid):
    # Plan:
    # 1. Detect the object, separating the puzzle pieces from the red object with holes
    # 2. Crop to just the red object
    # 3. Put each puzzle piece so that its red pixels perfectly overlay holes in the red object (considering translations and rotations)

    # 1. Separate puzzle pieces from the red object (remembering that puzzle pieces might have some red in them also: they are not monochromatic---but the red thing is fully red)
    objects = find_connected_components(input_grid, connectivity=8, background=Color.BLACK, monochromatic=False)
    red_objects = [ obj for obj in objects if set(object_colors(obj, background=Color.BLACK)) == {Color.RED} ]
    puzzle_pieces = [ obj for obj in objects if set(object_colors(obj, background=Color.BLACK)) != {Color.RED} ]

    assert len(red_objects) == 1, "There should be exactly one fully red object"
    red_object = red_objects[0]

    # 2. The output begins by cropping to just the red object
    output_grid = crop(red_object, background=Color.BLACK)

    # 3. Put each puzzle piece so that its red pixels perfectly overlay holes in the red object
    # Put in the pieces which have the most red first (otherwise a small puzzle piece might go in a big hole)
    puzzle_pieces.sort(key=lambda piece: np.sum(piece == Color.RED), reverse=True)

    for puzzle_piece in puzzle_pieces:
        # Extract just the sprite of this piece, and then figure out where it should go, including its position and rotation
        piece_sprite = crop(puzzle_piece, background=Color.BLACK)
        possible_sprites = [ piece_sprite, np.rot90(piece_sprite), np.rot90(piece_sprite, k=2), np.rot90(piece_sprite, k=3) ]

        # A placement solution is a tuple (x, y, sprite) where x, y is the top-left corner of the rotated sprite
        possible_solutions = [ (x, y, sprite)
                              for sprite in possible_sprites
                              for x in range(output_grid.shape[0] - piece_sprite.shape[0] + 1)
                              for y in range(output_grid.shape[1] - piece_sprite.shape[1] + 1) ]
        
        # Filter placement solutions to only those where the red pixels of the piece fit into the holes of the red object
        def valid_solution(x, y, sprite):
            # Make a canvas to try putting down the sprite
            test_canvas = np.full_like(output_grid, Color.BLACK)
            blit_sprite(test_canvas, sprite, x, y)
            # Check if every red pixel in the placed test object is also red in the red object
            red_pixels = [ (x, y) for x, y in np.argwhere(test_canvas == Color.RED) ]
            return all( output_grid[red_x, red_y] == Color.BLACK for red_x, red_y in red_pixels )
                
        possible_solutions = [ solution for solution in possible_solutions if valid_solution(*solution) ]
        if len(possible_solutions) == 0:
            assert False, "No solution found for puzzle piece"

        # Pick the first solution and blit the sprite into the output grid
        x, y, sprite = list(possible_solutions)[0]
        blit_sprite(output_grid, sprite, x, y)
    
    return output_grid

""" ==============================
Puzzle 995c5fa3

Train example 1:
Input1 = [
 e e e e k e e e e k e e e e
 e e e e k e k k e k k e e k
 e e e e k e k k e k k e e k
 e e e e k e e e e k e e e e
]
Output1 = [
 r r r
 t t t
 g g g
]

Train example 2:
Input2 = [
 e e e e k e e e e k e e e e
 k e e k k e e e e k e e e e
 k e e k k e k k e k e e e e
 e e e e k e k k e k e e e e
]
Output2 = [
 g g g
 y y y
 r r r
]

Train example 3:
Input3 = [
 e e e e k e e e e k e e e e
 e k k e k e e e e k e e e e
 e k k e k e e e e k e k k e
 e e e e k e e e e k e k k e
]
Output3 = [
 t t t
 r r r
 y y y
] """

# concepts:
# pattern matching, color correspondence

# description:
# In the input you will see three different 4x4 patterns of gray pixels place horizonlly and seperate by black interval. 
# To make the output grid, you should find out each pattern corresponds to a color: red, teal, yellow, or green, 
# and color the corresponding row in the output grid with the corresponding color in the order from left to right.

def transform_995c5fa3(input_grid):
    # Distinguish four different pattern with different black pixels placing on the gray background.
    b, g = Color.BLACK, Color.GRAY
    red_pattern = np.array([[g, g, g, g], [g, g, g, g], [g, g, g, g], [g, g, g, g]]).transpose()
    teal_patten = np.array([[g, g, g, g], [g, b, b, g], [g, b, b, g], [g, g, g, g]]).transpose()
    yellow_pattern = np.array([[g, g, g, g], [g, g, g, g], [g, b, b, g], [g, b, b, g]]).transpose()
    green_pattern = np.array([[g, g, g, g], [b, g, g, b], [b, g, g, b], [g, g, g, g]]).transpose()

    # Detect the patterns of gray pixels with size 4x4 place horizonlly and seperate by black interval.
    detect_patterns = detect_objects(grid=input_grid, colors=[Color.GRAY], connectivity=8, monochromatic=True)

    # Get the bounding box of each pattern and crop the pattern.
    pattern_lists = []
    for pattern in detect_patterns:
        x, y, w, h = bounding_box(grid=pattern, background=Color.BLACK)
        pattern_shape = crop(grid=pattern, background=Color.BLACK)
        pattern_lists.append({'x': x, 'y': y, 'pattern': pattern_shape})
    pattern_lists = sorted(pattern_lists, key=lambda rec: rec['x'])

    # Find the corresponding color of each pattern from left to right.
    color_list = []
    for pattern in pattern_lists:
        cur_pattern = pattern['pattern']
        if np.array_equal(cur_pattern, red_pattern):
            color_list.append(Color.RED)
        elif np.array_equal(cur_pattern, teal_patten):
            color_list.append(Color.TEAL)
        elif np.array_equal(cur_pattern, yellow_pattern):
            color_list.append(Color.YELLOW)
        elif np.array_equal(cur_pattern, green_pattern):
            color_list.append(Color.GREEN)
        else:
            raise ValueError("Invalid pattern")
    square_number = len(color_list)

    # Color the corresponding row in the output grid with the corresponding color in order.
    output_grid = np.zeros((square_number,square_number), dtype=int)
    for cnt, color in enumerate(color_list):
        draw_line(grid=output_grid, color=color, x=0, y=cnt, direction=(1, 0))
    return output_grid

""" ==============================
Puzzle 9af7a82c

Train example 1:
Input1 = [
 r r b
 r g b
 b b b
]
Output1 = [
 b r g
 b r k
 b r k
 b k k
 b k k
]

Train example 2:
Input2 = [
 g b b y
 r r r y
 y y y y
]
Output2 = [
 y r b g
 y r b k
 y r k k
 y k k k
 y k k k
 y k k k
]

Train example 3:
Input3 = [
 t t r
 g t t
 g g y
 g g y
]
Output3 = [
 g t y r
 g t y k
 g t k k
 g t k k
 g k k k
] """

# concepts:
# counting

# description:
# The input grid consists of a small grid filled completely with different colors.
# To create the output grid, take the colors present in the input, and sort them by number of pixels of that color in the input, greatest to least. Then create an output grid of shape (num_colors, max_num_pixels), where num_colors is the number of colors in the input, and max_num_pixels is the max number of pixels of any color in the input. Then fill each row with the color corresponding to that row's index in the sorted list of colors, filling K pixels from the top downwards, where K is the number of pixels of that color in the input. Leave the remaining pixels in the row black.

def transform_9af7a82c(input_grid):
    # find all unique colors in the input grid
    colors = np.unique(input_grid)

    # track the number of pixels of each color in the input grid
    colors_with_counts = [(c, np.sum(input_grid == c)) for c in colors]
    sorted_colors_with_counts = sorted(colors_with_counts, key=lambda x: x[1], reverse=True)

    # create an output grid of shape (num_colors, max_num_pixels)
    num_colors = len(colors)
    max_num_pixels = max([count for _, count in sorted_colors_with_counts])
    output_grid = np.full((num_colors, max_num_pixels), Color.BLACK)

    # for each color in the list, color K pixels to that color from top to bottom, leaving the remaining pixels black, where K is the number of pixels of that color in the input.
    for i, (color, count) in enumerate(sorted_colors_with_counts):
        output_grid[i, :count] = color

    return output_grid

""" ==============================
Puzzle 9f236235

Train example 1:
Input1 = [
 g g g g r k k k k r k k k k r k k k k
 g g g g r k k k k r k k k k r k k k k
 g g g g r k k k k r k k k k r k k k k
 g g g g r k k k k r k k k k r k k k k
 r r r r r r r r r r r r r r r r r r r
 k k k k r g g g g r k k k k r k k k k
 k k k k r g g g g r k k k k r k k k k
 k k k k r g g g g r k k k k r k k k k
 k k k k r g g g g r k k k k r k k k k
 r r r r r r r r r r r r r r r r r r r
 k k k k r k k k k r g g g g r k k k k
 k k k k r k k k k r g g g g r k k k k
 k k k k r k k k k r g g g g r k k k k
 k k k k r k k k k r g g g g r k k k k
 r r r r r r r r r r r r r r r r r r r
 g g g g r g g g g r g g g g r k k k k
 g g g g r g g g g r g g g g r k k k k
 g g g g r g g g g r g g g g r k k k k
 g g g g r g g g g r g g g g r k k k k
]
Output1 = [
 k k k g
 k k g k
 k g k k
 k g g g
]

Train example 2:
Input2 = [
 k k k k t r r r r t k k k k t k k k k
 k k k k t r r r r t k k k k t k k k k
 k k k k t r r r r t k k k k t k k k k
 k k k k t r r r r t k k k k t k k k k
 t t t t t t t t t t t t t t t t t t t
 r r r r t b b b b t k k k k t k k k k
 r r r r t b b b b t k k k k t k k k k
 r r r r t b b b b t k k k k t k k k k
 r r r r t b b b b t k k k k t k k k k
 t t t t t t t t t t t t t t t t t t t
 k k k k t k k k k t b b b b t k k k k
 k k k k t k k k k t b b b b t k k k k
 k k k k t k k k k t b b b b t k k k k
 k k k k t k k k k t b b b b t k k k k
 t t t t t t t t t t t t t t t t t t t
 k k k k t k k k k t k k k k t g g g g
 k k k k t k k k k t k k k k t g g g g
 k k k k t k k k k t k k k k t g g g g
 k k k k t k k k k t k k k k t g g g g
]
Output2 = [
 k k r k
 k k b r
 k b k k
 g k k k
]

Train example 3:
Input3 = [
 k k k r t t t r k k k
 k k k r t t t r k k k
 k k k r t t t r k k k
 r r r r r r r r r r r
 t t t r t t t r k k k
 t t t r t t t r k k k
 t t t r t t t r k k k
 r r r r r r r r r r r
 k k k r k k k r y y y
 k k k r k k k r y y y
 k k k r k k k r y y y
]
Output3 = [
 k t k
 k t t
 y k k
] """

# concepts:
# downscaling, mirror, horizontal/vertical bars

# description:
# In the input you will see horizontal and vertical bars separating different regions/cells/partitions with each cell containing different colors, like a chessboard.
# Each separated region has a single color.
# To make the output, make a grid with one colored pixel for each region of the chessboard.
# Finally mirror along the x-axis.

def transform_9f236235(input_grid):
    # Plan:
    # 1. Determine the color of the separator between the regions
    # 2. Extract all the regions separated by the separator color
    # 3. Find the region positions and their possible X/Y positions
    # 4. Create the output grid so that one pixel represents the original region, preserving X/Y ordering
    # 5. Mirror the output grid by x-axis

    # 1. Find the color of horizontal and vertical bars that separate the different regions/cells/partitions
    # One way of doing this is to find the connected component which stretches all the way horizontally and vertically over the input
    separator_candidates = [ possible_separator
                            for possible_separator in find_connected_components(grid=input_grid, connectivity=4, monochromatic=True, background=Color.BLACK)
                            if crop(possible_separator).shape == input_grid.shape ]
    assert len(separator_candidates) == 1, "There should be exactly 1 separator partitioning the input"
    separator = separator_candidates[0]
    separator_color = object_colors(separator, background=Color.BLACK)[0]

    # 2. Extract all the regions separated by the separator color
    regions = find_connected_components(grid=input_grid, connectivity=4, monochromatic=True, background=separator_color)

    # 3. Find the region positions
    x_positions = { object_position(obj, background=separator_color)[0] for obj in regions }
    y_positions = { object_position(obj, background=separator_color)[1] for obj in regions }

    # 4. Create the output grid, each region becomes a single pixel

    # Get the size of the output
    width = len(x_positions)
    height = len(y_positions)    

    # Create the output grid
    # Use one pixel to represent the original region
    output_grid = np.full((width, height), Color.BLACK)
    for output_x, input_x in enumerate(sorted(x_positions)):
        for output_y, input_y in enumerate(sorted(y_positions)):
            for region in regions:
                if object_position(region, background=separator_color) == (input_x, input_y):
                    output_grid[output_x, output_y] = object_colors(region, background=separator_color)[0]
                    break

    # 5. Mirror the output grid by x-axis
    output_grid = np.flip(output_grid, axis=0)

    return output_grid

""" ==============================
Puzzle a3df8b1e

Train example 1:
Input1 = [
 k k
 k k
 k k
 k k
 k k
 k k
 k k
 k k
 k k
 b k
]
Output1 = [
 k b
 b k
 k b
 b k
 k b
 b k
 k b
 b k
 k b
 b k
]

Train example 2:
Input2 = [
 k k k
 k k k
 k k k
 k k k
 k k k
 k k k
 k k k
 k k k
 k k k
 b k k
]
Output2 = [
 k b k
 b k k
 k b k
 k k b
 k b k
 b k k
 k b k
 k k b
 k b k
 b k k
]

Train example 3:
Input3 = [
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 b k k k
]
Output3 = [
 k k k b
 k k b k
 k b k k
 b k k k
 k b k k
 k k b k
 k k k b
 k k b k
 k b k k
 b k k k
] """

# concepts:
# bouncing

# description:
# In the input you will see a single blue pixel on a black background
# To make the output, shoot the blue pixel diagonally up and to the right, having it reflect and bounce off the walls until it exits at the top of the grid

def transform_a3df8b1e(input_grid):
    # Plan:
    # 1. Detect the pixel
    # 2. Shoot each line of the reflection one-by-one, bouncing (changing horizontal direction) when it hits a (horizontal) wall/edge of canvas

    # 1. Find the location of the pixel
    objects = find_connected_components(input_grid, connectivity=4, background=Color.BLACK)
    assert len(objects) == 1, "There should be exactly one blue pixel"
    blue_pixel = list(objects)[0]
    blue_pixel_x, blue_pixel_y = object_position(blue_pixel, background=Color.BLACK, anchor='center')

    # 2. do the bounce which requires keeping track of the direction of the ray we are shooting, as well as the tip of the ray
    # initially we are shooting diagonally up and to the right (dx=1, dy=-1)
    # initially the tip of the ray is the blue pixel, x=blue_pixel_x, y=blue_pixel_y
    direction = (1, -1)

    # loop until we fall out of the canvas
    while 0 <= blue_pixel_x < input_grid.shape[0] and 0 <= blue_pixel_y < input_grid.shape[1]:
        stop_x, stop_y = draw_line(input_grid, blue_pixel_x, blue_pixel_y, direction=direction, color=Color.BLUE)
        # Terminate if we failed to make progress
        if stop_x == blue_pixel_x and stop_y == blue_pixel_y:
            break
        blue_pixel_x, blue_pixel_y = stop_x, stop_y
        direction = (-direction[0], direction[1])
    
    return input_grid

""" ==============================
Puzzle a78176bb

Train example 1:
Input1 = [
 o k k k k k k k k k
 k o k k k k k k k k
 k k o k k k k k k k
 k k k o e e k k k k
 k k k k o e k k k k
 k k k k k o k k k k
 k k k k k k o k k k
 k k k k k k k o k k
 k k k k k k k k o k
 k k k k k k k k k o
]
Output1 = [
 o k k k o k k k k k
 k o k k k o k k k k
 k k o k k k o k k k
 k k k o k k k o k k
 k k k k o k k k o k
 k k k k k o k k k o
 k k k k k k o k k k
 k k k k k k k o k k
 k k k k k k k k o k
 k k k k k k k k k o
]

Train example 2:
Input2 = [
 k k k k k m k k k k
 k k k k k e m k k k
 k k k k k e e m k k
 k k k k k e e e m k
 k k k k k e e e e m
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k m k k k k
 m k k k k k m k k k
 k m k k k k k m k k
 k k m k k k k k m k
 k k k m k k k k k m
 k k k k m k k k k k
 k k k k k m k k k k
 k k k k k k m k k k
 k k k k k k k m k k
 k k k k k k k k m k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 r k k k k k k k k k
 k r k k k k k k k k
 k k r e e k k k k k
 k k k r e k k k k k
 k k k e r k k k k k
 k k k e e r k k k k
 k k k e e e r k k k
 k k k k k k k r k k
 k k k k k k k k r k
]
Output3 = [
 k k k r k k k k k k
 r k k k r k k k k k
 k r k k k r k k k k
 k k r k k k r k k k
 k k k r k k k r k k
 k k k k r k k k r k
 r k k k k r k k k r
 k r k k k k r k k k
 k k r k k k k r k k
 k k k r k k k k r k
] """

# concepts:
# diagonal lines

# description:
# In the input you will see a grid with a diagonal line and gray objects touching it. The grey objects are all right triangles.
# To make the output grid, draw additional diagonal lines in the same color emanating from the tip of the grey objects. Delete the grey objects.

def transform_a78176bb(input_grid):
    # Plan:
    # 1. Find the diagonal (it's the only one) and greys
    # 2. For each grey, find the tip
    # 3. ... and draw a diagonal line in the same color and same direction
    # 4. Delete the greys
    
    # 1. Input parsing: Find the grey objects, and then extract by color everything that is not grey
    background = Color.BLACK
    grey_objects = [ obj for obj in find_connected_components(input_grid, connectivity=4, monochromatic=True, background=background)
                     if Color.GREY in object_colors(obj, background=background) ]
    # extracting the diagonal by color: we know it's just everything that's not grey
    diagonal_object = input_grid.copy()
    diagonal_object[diagonal_object == Color.GREY] = background

    # Parse out the color and directionbof the diagonal
    diagonal_color = object_colors(diagonal_object, background=background)[0]
    if crop(diagonal_object, background=background)[0,0] == diagonal_color:
        diagonal_direction = (1,1) # down-right
    else:
        diagonal_direction = (-1,1) # up-right

    # We draw on top of the input, so copy it
    output_grid = input_grid.copy()

    # 2. Find the tips of the grey objects
    for grey_object in grey_objects:
        # The tip is the bordering pixel farthest away from the diagonal
        bordering_pixels_mask = object_neighbors(grey_object, connectivity=8, background=background)
        def distance_to_object(x, y, obj):
            return min( np.linalg.norm([x - x2, y - y2]) for x2, y2 in np.argwhere(obj != background) )
        tip = max( np.argwhere(bordering_pixels_mask), key=lambda xy: distance_to_object(xy[0], xy[1], diagonal_object) )
        tip_x, tip_y = tip

        # 3. Draw the diagonal line
        draw_line(output_grid, tip_x, tip_y, direction=diagonal_direction, color=diagonal_color)
        draw_line(output_grid, tip_x, tip_y, direction=(-diagonal_direction[0], -diagonal_direction[1]), color=diagonal_color)

    # 4. Delete grey
    output_grid[output_grid == Color.GREY] = background

    return output_grid

""" ==============================
Puzzle a79310a0

Train example 1:
Input1 = [
 t t k k k
 t t k k k
 k k k k k
 k k k k k
 k k k k k
]
Output1 = [
 k k k k k
 r r k k k
 r r k k k
 k k k k k
 k k k k k
]

Train example 2:
Input2 = [
 k t k
 k k k
 k k k
]
Output2 = [
 k k k
 k r k
 k k k
]

Train example 3:
Input3 = [
 k k k k k
 k t t t k
 k k k k k
 k k k k k
 k k k k k
]
Output3 = [
 k k k k k
 k k k k k
 k r r r k
 k k k k k
 k k k k k
] """

# concepts:
# translation, color change

# description:
# In the input you will see a grid with a teal object.
# To make the output grid, you should translate the teal object down by 1 pixel and change its color to red.

def transform_a79310a0(input_grid):
    # Plan:
    # 1. Find the object (it's the only one)
    # 2. Change its color to red
    # 3. Translate it downward by 1 pixel

    
    # Get the single teal object
    objects = find_connected_components(input_grid, connectivity=4, monochromatic=False, background=Color.BLACK)
    assert len(objects) == 1
    teal_object = objects[0]

    # Make a blank output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Change its color to red
    teal_object[teal_object != Color.BLACK] = Color.RED

    # Translate it downward by 1 pixel
    teal_object = translate(teal_object, x=0, y=1, background=Color.BLACK)

    # Blit the teal object onto the output grid
    output_grid = blit_object(grid=output_grid, obj=teal_object, background=Color.BLACK)

    return output_grid

""" ==============================
Puzzle a8c38be5

Train example 1:
Input1 = [
 e e e k k k k k k k k k k k
 t e e k k k k k r r r k k k
 t t e k k k k k e r e k k k
 k k r e e k k k e e e k k k
 k k r r e k k k k k k k k k
 k k r e e k e e e k k k k k
 k k k k k k e e e k e e e k
 k e b b k k e e e k e y e k
 k e e b k k k k k k y y y k
 k e e e k k e e g k k k k k
 k k k k k k e g g k k k k k
 e e e k k k e e g k p p e k
 e e m k k k k k k k p e e k
 e m m k k k k k k k e e e k
]
Output1 = [
 p p e r r r e b b
 p e e e r e e e b
 e e e e e e e e e
 r e e e e e e e g
 r r e e e e e g g
 r e e e e e e e g
 e e e e e e e e e
 t e e e y e e e m
 t t e y y y e m m
]

Train example 2:
Input2 = [
 k k k k k k k k k k k e e y
 k e e e k k k k k k k e y y
 k g e e k e t t k k k e e y
 k g g e k e e t k k k k k k
 k k k k k e e e k k k k k k
 k k k k k k k k k k e e e k
 k k k k k k k k k k e e m k
 k k k k k k k k k k e m m k
 k b b b k k e e e k k k k k
 k e b e k k e e e k p e e k
 k e e e k k e e e k p p e k
 k k k k k k k k k k p e e k
 k k k k o o e k k k k k k k
 k k k k o e e k k e e e k k
 k k k k e e e k k e r e k k
 k k k k k k k k k r r r k k
]
Output2 = [
 o o e b b b e t t
 o e e e b e e e t
 e e e e e e e e e
 p e e e e e e e y
 p p e e e e e y y
 p e e e e e e e y
 e e e e e e e e e
 g e e e r e e e m
 g g e r r r e m m
] """

# concepts:
# alignment, sliding objects

# description:
# In the input, you should see a black grid with nine 3x3 grey squares randomly placed in it (some of the squares touch a little bit). Each square contains a colored object of a different color, 3-4 cells in area, except for one which is blank. The colored objects are at the border of the 3x3 shape.
# To make the output, create a 9x9 grey grid. Now place each of the 3x3 squares from the input grid into the output grid. The location of an object is done so that the colored object in the grey square is moved "away" fromm the center square of the output grid in the direction the colored object is in the 3x3 square.

def transform_a8c38be5(input_grid):
    # Plan:
    # 1. Extract the 3x3 grey squares from the input grid (tricky because sometimes they touch, so we can't use connected components; detect_objects works better)
    # 2. Create the output grid
    # 3. Place the 3x3 squares into the output grid by sliding it in the direction of the colored (non-grey) portion

    # step 1: extract the 3x3 squares, which are grey+another color
    square_length = 3
    square_objects = detect_objects(input_grid, background=Color.BLACK, allowed_dimensions=[(square_length, square_length)],
                                    predicate=lambda sprite: np.all(sprite != Color.BLACK) and np.any(sprite == Color.GREY))
    square_sprites = [crop(obj, background=Color.BLACK) for obj in square_objects]

    assert len(square_sprites) == 9, "There should be exactly 9 3x3 grey squares in the input grid"

    # step 2: create the output grid, which is all grey
    output_grid = np.full((9, 9), Color.GREY, dtype=int)

    # step 3: place the 3x3 squares into the output grid
    # for each square, find the "direction" of the colored object in it, and place it in that direction of the output grid.

    # we can ignore the blank square, since the middle is already grey
    square_sprites = [square for square in square_sprites if not (square == Color.GREY).all()]

    def get_direction_between(point1, point2):
        '''
        returns one of (-1, -1), (-1, 0), (-1, 1),
                       (0, -1), (0, 0), (0, 1),
                       (1, -1), (1, 0), (1, 1)

        based on the direction from point1 to point2
        '''
        x1, y1 = point1
        x2, y2 = point2

        dx, dy = x2 - x1, y2 - y1

        def sign(x):
            if x < 0:
                return -1
            elif x > 0:
                return 1
            else:
                return 0

        return (sign(dx), sign(dy))

    for square in square_sprites:
        colored_object_center_of_mass = np.argwhere(square != Color.GREY).mean(axis=0)
        grey_center_of_mass = np.argwhere(square == Color.GREY).mean(axis=0)

        dx, dy = get_direction_between(grey_center_of_mass, colored_object_center_of_mass)

        # start with the square in the middle of the canvas, which has length 9 (we will slide it afterward)
        x, y = (9 - square_length)//2, (9 - square_length)//2
        
        # slide until we can't anymore
        while 0 < x < 9 - square_length and 0 < y < 9 - square_length:
            x += dx
            y += dy

        blit_sprite(output_grid, square, x=x, y=y)

    return output_grid

""" ==============================
Puzzle a9f96cdd

Train example 1:
Input1 = [
 k k k k k
 k r k k k
 k k k k k
]
Output1 = [
 g k p k k
 k k k k k
 t k o k k
]

Train example 2:
Input2 = [
 k k k k k
 k k k k k
 k k k k r
]
Output2 = [
 k k k k k
 k k k g k
 k k k k k
]

Train example 3:
Input3 = [
 k k r k k
 k k k k k
 k k k k k
]
Output3 = [
 k k k k k
 k t k o k
 k k k k k
] """

# concepts:
# constant pattern, diagonal corners

# description:
# In the input you will see one red pixel
# To make the output grid, you should 
# 1. draw a pattern with four different colors centered at the red pixel at its diagonal corners:
#    green in the upper left, pink in the upper right, teal in the lower left, and yellow in the lower right.
# 2. remove the red pixel (equivalently start with a blank canvas and draw the pattern at the red pixel location)

def transform_a9f96cdd(input_grid):
    # Find the red single pixel object
    red_pixel_objects = detect_objects(grid=input_grid, colors=[Color.RED], allowed_dimensions=[(1, 1)], monochromatic=True, connectivity=4)
    assert len(red_pixel_objects) == 1
    red_pixel_object = red_pixel_objects[0]

    # Find out the position of the red pixel
    red_x, red_y = object_position(red_pixel_object, background=Color.BLACK, anchor="upper left")

    # Construct the specific pattern that is going to be drawn where the red pixel was
    pattern = np.array([[Color.GREEN, Color.BLACK, Color.PINK], 
                        [Color.BLACK, Color.BLACK, Color.BLACK],
                        [Color.TEAL, Color.BLACK, Color.ORANGE]]).transpose()
    
    # Because sprites are anchored by the upper left corner, we are going to need to calculate where the pattern's upper left corner should be
    pattern_width, pattern_height = pattern.shape
    pattern_x, pattern_y = red_x - pattern_width//2, red_y - pattern_height//2

    # The output grid is the same size of input grid
    # start with a blank canvas and then lit the pattern
    output_grid = np.full(input_grid.shape, Color.BLACK)
    output_grid = blit_sprite(grid=output_grid, x=pattern_x, y=pattern_y, sprite=pattern, background=Color.BLACK)

    return output_grid

""" ==============================
Puzzle aabf363d

Train example 1:
Input1 = [
 k k k k k k k
 k r r r k k k
 k k r k k k k
 k r r r r k k
 k k r r r k k
 k k k r k k k
 y k k k k k k
]
Output1 = [
 k k k k k k k
 k y y y k k k
 k k y k k k k
 k y y y y k k
 k k y y y k k
 k k k y k k k
 k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k
 k k k g k k k
 k k g g g k k
 k g g g g k k
 k g g k k k k
 k k g g k k k
 p k k k k k k
]
Output2 = [
 k k k k k k k
 k k k p k k k
 k k p p p k k
 k p p p p k k
 k p p k k k k
 k k p p k k k
 k k k k k k k
] """

# concepts:
# color guide, filling, objects

# description:
# In the input, you will see a colored object in the middle and a single pixel in the bottom left corner of a different color.
# To make the output, remove the pixel from bottom left corner and color the object in the middle with the color from the pixel you removed.

def transform_aabf363d(input_grid):
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # get the color of the pixel in the bottom left corner
    color = output_grid[0, -1]

    # remove the pixel from the bottom left corner
    output_grid[0, -1] = Color.BLACK

    # color the object in the middle with the color of the pixel from the bottom left corner
    output_grid = np.where(output_grid, color, output_grid)
    # could also have used flood_fill:
    # x, y = np.where(output_grid != Color.BLACK)
    # flood_fill(output_grid, x[0], y[0], color)

    return output_grid

""" ==============================
Puzzle aba27056

Train example 1:
Input1 = [
 k k k k k k k
 k k k k k k k
 k k k k k k k
 k k k k k k k
 k p p k p p k
 k p k k k p k
 k p p p p p k
]
Output1 = [
 k k k y k k k
 y k k y k k y
 k y k y k y k
 k k y y y k k
 k p p y p p k
 k p y y y p k
 k p p p p p k
]

Train example 2:
Input2 = [
 k k k k k k k k k
 k k k k k k k k k
 k k k k o o o o o
 k k k k o k k k o
 k k k k k k k k o
 k k k k k k k k o
 k k k k k k k k o
 k k k k o k k k o
 k k k k o o o o o
]
Output2 = [
 y k k k k k k k k
 k y k k k k k k k
 k k y k o o o o o
 k k k y o y y y o
 y y y y y y y y o
 y y y y y y y y o
 y y y y y y y y o
 k k k y o y y y o
 k k y k o o o o o
]

Train example 3:
Input3 = [
 g g g g g g
 g k k k k g
 g k k k k g
 g g k k g g
 k k k k k k
 k k k k k k
]
Output3 = [
 g g g g g g
 g y y y y g
 g y y y y g
 g g y y g g
 k y y y y k
 y k y y k y
] """

# concepts:
# cups, filling

# description:
# In the input you will see a cups, meaning an almost-enclosed shape with a small opening on one of its sides, and empty space (black pixels) inside.
# To make the output grid, you should fill the interior of each cup with yellow, then shoot yellow out of the opening of the cup both straight out and diagonally from the edges of the opening.
# 

def transform_aba27056(input_grid):
    # Plan:
    # 1. Detect the cup
    # 2. Find the mask of the inside of the cup
    # 3. Find the mask of the opening of the cup (on one of its sides)
    # 4. Fill the cup with yellow
    # 5. Shoot pixels outward from the opening (straight out)
    # 6. Shoot pixels outward from the opening (diagonally out, from the edges)
    
    # 1. Detect cup
    objects = find_connected_components(input_grid, connectivity=4, background=Color.BLACK)
    assert len(objects) == 1, "There should be exactly one cup"
    obj = list(objects)[0]

    output_grid = input_grid.copy()

    # 2. Extract what's inside the cup (as its own object), which is everything in the bounding box that is not the object itself
    cup_x, cup_y, cup_width, cup_height = bounding_box(obj)
    inside_cup_mask = np.zeros_like(input_grid, dtype=bool)
    inside_cup_mask[cup_x:cup_x+cup_width, cup_y:cup_y+cup_height] = True
    inside_cup_mask = inside_cup_mask & (obj == Color.BLACK)

    # 3. Extract the hole in the cup, which is what's inside and on the boundary of the bounding box
    # what's inside...
    hole_mask = inside_cup_mask.copy()
    # ...and then we need to remove anything not on the boundary
    hole_mask[cup_x+1 : cup_x+cup_width-1, cup_y+1 : cup_y+cup_height-1] = False

    # 4. Fill the cup with yellow
    output_grid[inside_cup_mask] = Color.YELLOW

    # 5. Shoot pixels outward from the opening (straight out)
    # Find the direction of the opening, which is the unit vector that points from the center of the cup to the hole
    hole_x, hole_y = object_position(hole_mask, background=Color.BLACK, anchor='center')
    cup_x, cup_y = object_position(obj, background=Color.BLACK, anchor='center')
    direction = (int(np.sign(hole_x - cup_x)), int(np.sign(hole_y - cup_y)))
    # Loop over every boundary pixel and shoot outward
    for x, y in np.argwhere(hole_mask):
        draw_line(output_grid, x, y, direction=direction, color=Color.YELLOW)

    # 6. Shoot pixels outward from the opening (diagonally out, from the edges)
    # Find the two extremal points on the boundary of the hole, which are the points farthest away from each other
    points_on_boundary = np.argwhere(hole_mask)
    pt1, pt2 = max({ ( (x1,y1), (x2,y2) ) for x1,y1 in points_on_boundary for x2,y2 in points_on_boundary },
                   key=lambda pair: np.linalg.norm(np.array(pair[0]) - np.array(pair[1])))
    
    # For each of those points, shoot diagonal lines in all directions, but stop as soon as you hit something that's not black
    for pt in [pt1, pt2]:
        for direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw_line(output_grid, pt[0]+direction[0], pt[1]+direction[1], direction=direction, color=Color.YELLOW, stop_at_color=Color.NOT_BLACK)
    
    return output_grid

""" ==============================
Puzzle aedd82e4

Train example 1:
Input1 = [
 k r r
 k r r
 r k k
]
Output1 = [
 k r r
 k r r
 b k k
]

Train example 2:
Input2 = [
 r r r k
 k r k k
 k k k r
 k r k k
]
Output2 = [
 r r r k
 k r k k
 k k k b
 k b k k
]

Train example 3:
Input3 = [
 r r k k
 k r k k
 r r k r
 k k k k
 k r r r
]
Output3 = [
 r r k k
 k r k k
 r r k b
 k k k k
 k r r r
] """

# concepts:
# object detection, color change

# description:
# In the input you will see a grid with a red pattern
# To make the output grid, you should find out any single isolated red objects with size of 1x1 and change them to blue.

def transform_aedd82e4(input_grid):
    # Detect all the red objects in the grid, ignoring objects of other colors
    red_objects = detect_objects(grid=input_grid, colors=[Color.RED], monochromatic=True, connectivity=4)

    # Convert 1x1 objects (isolated pixels) into blue
    output_grid = input_grid.copy()
    for object in red_objects:
        x, y, length, width = bounding_box(object, background=Color.BLACK)
        # Find out the single isolated red object with size of 1x1 and change it to blue.
        if length == 1 and width == 1:
            output_grid[x, y] = Color.BLUE

    return output_grid

""" ==============================
Puzzle af902bf9

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k y k y k k k k
 k k k k k k k k k k
 k k k y k y k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k y k y k k k k
 k k k k r k k k k k
 k k k y k y k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k y k k k k y k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k y k k k k y k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k y k k k k y k k k
 k k r r r r k k k k
 k k r r r r k k k k
 k k r r r r k k k k
 k k r r r r k k k k
 k y k k k k y k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k y k y k k k k k k
 k k k k k k k k k k
 k y k y k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k y k k k k y
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k y k k k k y
]
Output3 = [
 k k k k k k k k k k
 k y k y k k k k k k
 k k r k k k k k k k
 k y k y k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k y k k k k y
 k k k k k r r r r k
 k k k k k r r r r k
 k k k k y k k k k y
] """

# concepts:
# filling, surrounding

# description:
# In the input you will see several yellow pixels arranged in groups of 4 so that each group outlines a rectangular shape.
# To make the output, fill the corresponding inner rectangular regions with red.

def transform_af902bf9(input_grid):
    # Detect the rectangular regions by finding groups of four surrounding yellow pixels
    surrounding_color = Color.YELLOW
    rectangle_color = Color.RED
    
    output_grid = np.copy(input_grid) 

    # loop over all the yellows...
    for x, y in np.argwhere(input_grid == surrounding_color):
        # ...and find the other matching yellows forming a rectangle: (x, y), (x, y'), (x', y), (x', y')
        for other_x, other_y in np.argwhere(input_grid == surrounding_color):
            if input_grid[x, other_y] == surrounding_color and input_grid[other_x, y] == surrounding_color:
                # fill the rectangle with red
                output_grid[x+1:other_x, y+1:other_y] = rectangle_color

    return output_grid

""" ==============================
Puzzle b527c5c6

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k g g k k k k k
 k k k g g k k k k k
 k k k r g k g g g g
 k k k g g k g g r g
 k k k g g k k k k k
 k k k g g k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k g g k k k k k
 g g g g g k k k k k
 r r r r g k g g g g
 g g g g g k g g r g
 k k k g g k k g r g
 k k k g g k k g r g
 k k k k k k k g r g
 k k k k k k k g r g
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k g g g g g g g g g g g g g r g g g g k
 k g g g g g g g g g g g g g g g g g g k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g r k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k g r g k k k k
 k k k k k k k k k k k k k g r g k k k k
 k k k k k k k k k k k k k g r g k k k k
 k k k k k k k k k k k k k g r g k k k k
 k g g g g g g g g g g g g g r g g g g k
 k g g g g g g g g g g g g g g g g g g k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g g g g g g g g g g g
 k k k k k k k g g g g g g g g g g g g g
 k k k k k k k g g r r r r r r r r r r r
 k k k k k k k g g g g g g g g g g g g g
 k k k k k k k g g g g g g g g g g g g g
 k k k k k k k g g g k k k k k k k k k k
 k k k k k k k g g g k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 g g g g g r g g g g g g g g k k k k k k
 g g g g g g g g g g g g g g k k k k k k
 g g g g g g g g g g g g g g k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k g g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
 k k k k k k k k k k r g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
]
Output3 = [
 k k k g g r g g k k k k k k k k k k k k
 k k k g g r g g k k k k k k k k k k k k
 k k k g g r g g k k k k k k k k k k k k
 g g g g g r g g g g g g g g k k k k k k
 g g g g g g g g g g g g g g k k k k k k
 g g g g g g g g g g g g g g k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k g g g g g k k k k k
 g g g g g g g g g g g g g g g k k k k k
 g g g g g g g g g g g g g g g k k k k k
 g g g g g g g g g g g g g g g k k k k k
 g g g g g g g g g g g g g g g k k k k k
 r r r r r r r r r r r g g g g k k k k k
 g g g g g g g g g g g g g g g k k k k k
 g g g g g g g g g g g g g g g k k k k k
 g g g g g g g g g g g g g g g k k k k k
 g g g g g g g g g g g g g g g k k k k k
 k k k k k k k k k k g g g g g k k k k k
] """

# concepts:
# growing

# description:
# In the input you will see a grid with some green rectangles. Each rectangle has one red pixel on one of its borders.
# To make the output, you should extend the green region of each rectangle to the border of the grid in the direction of the red pixel, with the extent of the line increasing with the extent of the rectangle.
# Also you should draw a red line from the red pixel to the border of the grid.


def transform_b527c5c6(input_grid):
    # Initialize the output grid
    output_grid = np.copy(input_grid)
    
    rectangle_color = Color.GREEN
    indicator_color = Color.RED
    background = Color.BLACK

    # get all the green rectangles on the grid. because of the red pixel on the border of the rectangle, they are not monochromatic.
    objects = find_connected_components(input_grid, connectivity=4, monochromatic=False, background=background)
    for obj in objects:
        # find the red indicator
        for red_x, red_y in np.argwhere(obj == indicator_color):
            break

        # Get the dimensions of the object, and its position
        x, y, width, height = bounding_box(obj, background=background)  

        # depending on which side of the rectangle that the indicator is on, we draw in different directions
        # left edge: draw to the left
        if red_x == x:
            # Extend the green rectangle to the left until it reaches the border
            output_grid[0:red_x, red_y-(width-1) : red_y+(width-1)+1] = rectangle_color
            # Draw the red line from the red pixel to the border
            draw_line(output_grid, x=red_x, y=red_y, direction=(-1, 0), color=indicator_color)
        elif red_x == x + width - 1:
            # Extend the green rectangle to the right until it reaches the border
            output_grid[red_x:, red_y-(width-1) : red_y+(width-1)+1] = rectangle_color
            # Draw the red line from the red pixel to the border
            draw_line(output_grid, x=red_x, y=red_y, direction=(1, 0), color=indicator_color)
        elif red_y == y:
            # Extend the green rectangle to the top until it reaches the border
            output_grid[red_x-(height-1) : red_x+(height-1)+1, 0:red_y] = rectangle_color
            # Draw the red line from the red pixel to the border
            draw_line(output_grid, x=red_x, y=red_y, direction=(0, -1), color=indicator_color)
        elif red_y == y + height - 1:
            # Extend the green rectangle to the bottom until it reaches the border
            output_grid[red_x-(height-1) : red_x+(height-1)+1, red_y:] = rectangle_color
            # Draw the red line from the red pixel to the border
            draw_line(output_grid, x=red_x, y=red_y, direction=(0, 1), color=indicator_color)
        else:
            assert False, "The red pixel is not on the border of the rectangle"

    return output_grid

""" ==============================
Puzzle b7249182

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k r k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k t k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k r k k k k k
 k k k k r k k k k k
 k k k k r k k k k k
 k k r r r r r k k k
 k k r k k k r k k k
 k k t k k k t k k k
 k k t t t t t k k k
 k k k k t k k k k k
 k k k k t k k k k k
 k k k k t k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k g k k k k k k k k k k b
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k g g b b k k k k
 k k k k k g k k b k k k k
 k g g g g g k k b b b b b
 k k k k k g k k b k k k k
 k k k k k g g b b k k k k
 k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k e k k k k k k k k k k k k t k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k e e t t k k k k k k
 k k k k k k k k e k k t k k k k k k
 k k k e e e e e e k k t t t t t t k
 k k k k k k k k e k k t k k k k k k
 k k k k k k k k e e t t k k k k k k
 k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k
] """

# concepts:
# growing, connecting

# description:
# In the input you will see two pixels of different colors aligned horizontally or vertically.
# To make the output, you need to connect the two pixels with two lines and a hollow rectangle of size 4x5 in the middle.
# Half of the rectangle is colored with the color of the one side's pixel, and the other half with the color of the other side's pixel.

def transform_b7249182(input_grid):
    # Plan:
    # 1. Parse the input
    # 2. Canonicalize the input: because it could be horizontal or vertical, rotate to make horizontal (we will rotate back at the end)
    # 3. Prepare a 4x5 rectangle sprite whose left half is left_color and right half is right_color
    # 4. Place the rectangle in the middle of the two pixels
    # 5. Draw lines to connect the original two pixels with the rectangle
    # 6. Rotate the grid back if it was not originally horizontal

    # 1. Input parsing
    # Extract the two pixels from the input grid
    pixels = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # 2. Canonicalize the input: Ensure that the two pixels are horizontally aligned
    # Check if the two pixels are horizontally aligned
    was_originally_horizontal = object_position(pixels[0])[1] == object_position(pixels[1])[1]
    
    # If the two pixels are not horizontally aligned, rotate the grid for easier processing
    if not was_originally_horizontal:
        input_grid = np.rot90(input_grid)
        pixels = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    # we draw on top of the input
    output_grid = input_grid.copy()

    # Prepare for what follows: extract properties of the input here
    # Sort the two horizontally-aligned pixels by their position, from left to right
    pixels = sorted(pixels, key=lambda x: object_position(x)[0])
    # Get the position of the two pixels
    left_pos = object_position(pixels[0])
    right_pos = object_position(pixels[1])
    left_color = object_colors(pixels[0])[0]
    right_color = object_colors(pixels[1])[0]
    
    # 3. Prepare hollow 4x5 rectangle sprite whose left half is left_color and right half is right_color
    rectangle_width, rectangle_height = 4, 5
    rectangle_sprite = np.full((rectangle_width, rectangle_height), Color.BLACK)
    rectangle_sprite[0, :] = left_color
    rectangle_sprite[-1, :] = right_color
    rectangle_sprite[:rectangle_width//2, 0] = left_color
    rectangle_sprite[rectangle_width//2:, 0] = right_color
    rectangle_sprite[:rectangle_width//2, -1] = left_color
    rectangle_sprite[rectangle_width//2:, -1] = right_color

    # 4. Place the rectangle in the middle of the two pixels
    middle_x = (left_pos[0] + right_pos[0] + 1) // 2
    middle_y = left_pos[1]
    blit_sprite(output_grid, sprite=rectangle_sprite, x=middle_x - rectangle_width // 2, y=middle_y - rectangle_height // 2)

    # 5. Draw lines that connect the original two pixels with the rectangle
    draw_line(grid=output_grid, x=left_pos[0]+1, y=left_pos[1], direction=(1, 0), color=left_color, stop_at_color=Color.NOT_BLACK)
    draw_line(grid=output_grid, x=right_pos[0]-1, y=right_pos[1], direction=(-1, 0), color=right_color, stop_at_color=Color.NOT_BLACK)

    # 6. If the grid is not horizontal, rotate it back
    if not was_originally_horizontal:
        output_grid = np.rot90(output_grid, k=-1)

    return output_grid

""" ==============================
Puzzle b775ac94 """

# concepts:
# symmetry, mirror

# description:
# In the input you will see big objects with a primary color and some single pixels of different colors attached to it.
# To make the output, mirror the primary colored part of each object over the differently colored pixels attached to it, changing the primary color to match the other color.

def transform_b775ac94(input_grid):
    # Plan
    # 1. Parse the input into primary regions and their associated single pixel indicators
    # 2. For each primary region and each attached indicator pixel, change color and mirror.

    # 1. Input parsing
    background = Color.BLACK
    objects = find_connected_components(grid=input_grid, connectivity=8, monochromatic=True, background=background)
    indicator_pixels = [ obj for obj in objects if np.sum(obj != background) == 1 ]
    primary_objects = [ obj for obj in objects if np.sum(obj != background) > 1 ]

    # 2. Output generation

    # Copy the input because we draw on top of it
    output_grid = input_grid.copy()

    # loop over primary objects and every pixel that they are in contact with
    for primary_object in primary_objects:
        for indicator_pixel in indicator_pixels:
            if not contact(object1=primary_object, object2=indicator_pixel, background=background, connectivity=8): continue

            # Recolor
            indicator_color = object_colors(indicator_pixel, background=background)[0]
            recolored_object = np.copy(primary_object)
            recolored_object[primary_object != background] = indicator_color

            # Build the mirroring object
            indicator_x, indicator_y = object_position(indicator_pixel, background=background, anchor="upper left")
            primary_x1, primary_y1 = object_position(primary_object, background=background, anchor="upper left")
            primary_x2, primary_y2 = object_position(primary_object, background=background, anchor="lower right")
            # If it's in the corners, we mirror diagonally (over both x and y)
            # If it's on the left/right side, we mirror horizontally
            # If it's on the top/bottom side, we mirror vertically
            mirror_x, mirror_y = None, None
            if indicator_x == primary_x1-1: mirror_x = primary_x1-0.5
            if indicator_x == primary_x2+1: mirror_x = primary_x2+0.5
            if indicator_y == primary_y1-1: mirror_y = primary_y1-0.5
            if indicator_y == primary_y2+1: mirror_y = primary_y2+0.5
            symmetry = MirrorSymmetry(mirror_x=mirror_x, mirror_y=mirror_y)

            # Mirror the primary object over the indicator pixel
            for x, y in np.argwhere(primary_object != background):
                x2, y2 = symmetry.apply(x, y)
                if 0 <= x2 < output_grid.shape[0] and 0 <= y2 < output_grid.shape[1]:
                    output_grid[x2, y2] = recolored_object[x, y]

    return output_grid

""" ==============================
Puzzle b782dc8a

Train example 1:
Input1 = [
 t k k k k k t t t t t t k t t t k t t k t t t k
 k k t t t k k k k k k t k k k t k t k k t k t k
 t t t k t k t t t t k t t t k t k t t t t k t k
 t k k k t k t k k t k k k t k t k k k k k k t k
 t k t t t k t t k t k t t t k t t k t t t t t k
 t k t k k k k t k t k t k k k k t k t k k k k k
 t k t t t t t t k t k t t t t t t g t t t t t k
 t k k k k k k k k t k k k k k k g r g k k k t k
 t t k t t t k t t t k t t t t t t g t t t k t k
 k t k t k t k t k k k t k k k k t k t k t k t k
 k t t t k t t t k t t t k t t k t t t k t t t k
]
Output1 = [
 t g r g r g t t t t t t k t t t r t t k t t t k
 g r t t t r g r g r g t k k k t g t k k t r t k
 t t t k t g t t t t r t t t k t r t t t t g t k
 t k k k t r t k k t g r g t k t g r g r g r t k
 t k t t t g t t k t r t t t k t t g t t t t t k
 t k t r g r g t k t g t k k k k t r t k k k k k
 t k t t t t t t k t r t t t t t t g t t t t t k
 t k k k k k k k k t g r g r g r g r g r g r t k
 t t k t t t k t t t r t t t t t t g t t t g t k
 k t k t k t k t g r g t k k k k t r t k t r t k
 k t t t k t t t r t t t k t t k t t t k t t t k
]

Train example 2:
Input2 = [
 k k k t k k k t k k k k k t
 t t k t t t k t k t t t k t
 k t k k k t k t k t k t t t
 k t t t t t k t k t k k k k
 k k k k k k k t k t t t k t
 t t t t t t k t k k k t k t
 t k k k k t k t t t k t k t
 t t t t k t k k k t k t k k
 k k k t b t t t t t k t t k
 t t k t y b k k k k k k t k
 k t k t b t t t t t t t t k
 k t t t k t k k k k k k k k
 k k k k k t k t t t t t t t
]
Output2 = [
 k k k t k k k t b y b y b t
 t t k t t t k t y t t t y t
 k t k k k t k t b t k t t t
 k t t t t t k t y t k k k k
 k k k k k k k t b t t t k t
 t t t t t t k t y b y t k t
 t y b y b t k t t t b t k t
 t t t t y t k k k t y t k k
 k k k t b t t t t t b t t k
 t t k t y b y b y b y b t k
 b t k t b t t t t t t t t k
 y t t t y t k k k k k k k k
 b y b y b t k t t t t t t t
] """

# concepts:
# maze, path finding

# description:
# In the input you will see a maze with a path that has two indicator pixels of different colors.
# To make the output, fill all reachable parts of the maze starting with the indicator pixels and alternating colors.

def transform_b782dc8a(input_grid):
    # Output grid draws on top of the input grid
    output_grid = input_grid.copy()

    # Parse the input
    maze_color = Color.TEAL
    indicator_colors = [ color for color in object_colors(input_grid, background=Color.BLACK) if color != maze_color]
    assert len(indicator_colors) == 2, "expected exactly two indicator colors"
    
    # Fill the path with alternating colors in turn
    def fill_maze(cur_color, next_color, x, y, grid):
        width, height = grid.shape
        # Search outward in four directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for direction in directions:
            new_x, new_y = x + direction[0], y + direction[1]
            if 0 <= new_x < width and 0 <= new_y < height and grid[new_x, new_y] == Color.BLACK:
                grid[new_x, new_y] = next_color
                # Change the next color to the current color: swap current and next
                fill_maze(next_color, cur_color, new_x, new_y, grid)
    
    # Fill the path with two colors
    # Start to fill the path with the pixel that already has the path_color
    for x, y in np.argwhere((input_grid != Color.BLACK) & (input_grid != maze_color)):
        cur_color = input_grid[x, y]
        next_color = indicator_colors[0] if cur_color == indicator_colors[1] else indicator_colors[1]
        fill_maze(cur_color, next_color, x, y, output_grid)

    return output_grid

""" ==============================
Puzzle bbc9ae5d

Train example 1:
Input1 = [
 b b k k k k
]
Output1 = [
 b b k k k k
 b b b k k k
 b b b b k k
]

Train example 2:
Input2 = [
 r k k k k k k k
]
Output2 = [
 r k k k k k k k
 r r k k k k k k
 r r r k k k k k
 r r r r k k k k
]

Train example 3:
Input3 = [
 e e e k k k k k k k
]
Output3 = [
 e e e k k k k k k k
 e e e e k k k k k k
 e e e e e k k k k k
 e e e e e e k k k k
 e e e e e e e k k k
] """

# concepts:
# counting, incrementing

# description:
# In the input, you will see a row with partially filled with pixels from the one color from left to right.
# To make the output: 
# 1. Take the input row
# 2. Copy it below the original row with one more colored pixel added to the sequence if there is space
# 3. Repeat until there are half as many rows as there are columns
def transform_bbc9ae5d(input_grid):
    # get the color of the row
    color = input_grid[0]

    # copy the row from the input grid
    row = np.copy(input_grid)

    # make the output grid
    output_grid = np.copy(input_grid)

    # repeat the row on the output grid until there are half as many rows as there are columns
    for _ in range(input_grid.shape[0]//2 - 1):
        # find the rightmost color pixel in the row and add one more pixel of the same color to the right if there is space
        rightmost_color_pixel = np.where(row == color)[0][-1]
        if rightmost_color_pixel < input_grid.shape[0] - 1:
            row[rightmost_color_pixel + 1] = color

        # add the row to the output grid
        output_grid = np.concatenate((output_grid, row), axis=1)

    return output_grid

""" ==============================
Puzzle bc1d5164

Train example 1:
Input1 = [
 k t k k k t k
 t t k k k t t
 k k k k k k k
 t t k k k t t
 k t k k k t k
]
Output1 = [
 k t k
 t t t
 k t k
]

Train example 2:
Input2 = [
 r r k k k r r
 k k k k k k r
 k k k k k k k
 k r k k k r k
 r k k k k k r
]
Output2 = [
 r r r
 k r r
 r k r
]

Train example 3:
Input3 = [
 y y k k k y k
 k k k k k y y
 k k k k k k k
 k k k k k k k
 y k k k k k y
]
Output3 = [
 y y k
 k y y
 y k y
] """

# concepts:
# patterns, positioning, copying

# description:
# In the input you will see a pattern of pixels in the top left corner of the grid, the top right corner of the grid, the bottom left corner of the grid, and the bottom right corner of the grid. All the pixels are the same color, and the patterns are in square regions.
# To make the output, copy the pattern in each corner of the input to the corresponding corner of the output. The output grid is one pixel larger in each dimension than the maximum pattern side length.

def transform_bc1d5164(input_grid):
    # get the patterns from the input
    objects = find_connected_components(input_grid, connectivity=8)

    # find the bounding box of each pattern
    bounding_boxes = [bounding_box(obj) for obj in objects]

    # figure out how big the output grid should be (the pattern is a square and the output should be one pixel larger in each dimension)
    n = m = max([max(pattern[2], pattern[3]) for pattern in bounding_boxes]) + 1

    # make the output grid
    output_grid = np.full((n, m), Color.BLACK)

    # copy the patterns to the output grid
    for obj, (x, y, _, _) in zip(objects, bounding_boxes):
        # adjust the position of the pattern in the output grid if necessary
        if x >= n - 1:
            x = x - input_grid.shape[0] + n
        if y >= m - 1:
            y = y - input_grid.shape[1] + m
        # crop the pattern to remove any extra rows or columns of black pixels
        sprite = crop(obj)
        # copy the pattern to the output grid
        blit_sprite(output_grid, sprite, x=x, y=y, background=Color.BLACK)
    
    return output_grid

""" ==============================
Puzzle bd4472b8

Train example 1:
Input1 = [
 r b y
 e e e
 k k k
 k k k
 k k k
 k k k
 k k k
 k k k
]
Output1 = [
 r b y
 e e e
 r r r
 b b b
 y y y
 r r r
 b b b
 y y y
]

Train example 2:
Input2 = [
 g r b y
 e e e e
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
]
Output2 = [
 g r b y
 e e e e
 g g g g
 r r r r
 b b b b
 y y y y
 g g g g
 r r r r
 b b b b
 y y y y
]

Train example 3:
Input3 = [
 t g
 e e
 k k
 k k
 k k
 k k
]
Output3 = [
 t g
 e e
 t t
 g g
 t t
 g g
] """

# concepts:
# patterns, lines

# description:
# In the input, you will see a top row with a sequence of colored pixels, and right below it is a grey line.
# To make the output, copy the first two rows of the input. 
# Then, starting below the grey line, draw rows one color at a time in the order of the colors in the top row from left to right, with the color of each row matching the color of the corresponding pixel in the top row. 
# Repeat this pattern until you reach the bottom of the grid.

def transform_bd4472b8(input_grid):
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # get the colors from the top row
    colors = input_grid[:, 0]

    # get the number of colors
    num_colors = len(set(colors))

    # get the y-coordinate of the grey line
    grey_line = np.where(input_grid[0] == Color.GREY)[0][-1]

    # draw the rows below the grey line
    for i in range(input_grid.shape[1] - grey_line - 1):
        draw_line(output_grid, 0, grey_line + i + 1, length=None, color=colors[i % num_colors], direction=(1, 0))

    return output_grid

""" ==============================
Puzzle caa06a1f

Train example 1:
Input1 = [
 p o p o p o p g g g g
 o p o p o p o g g g g
 p o p o p o p g g g g
 o p o p o p o g g g g
 p o p o p o p g g g g
 o p o p o p o g g g g
 p o p o p o p g g g g
 g g g g g g g g g g g
 g g g g g g g g g g g
 g g g g g g g g g g g
 g g g g g g g g g g g
]
Output1 = [
 o p o p o p o p o p o
 p o p o p o p o p o p
 o p o p o p o p o p o
 p o p o p o p o p o p
 o p o p o p o p o p o
 p o p o p o p o p o p
 o p o p o p o p o p o
 p o p o p o p o p o p
 o p o p o p o p o p o
 p o p o p o p o p o p
 o p o p o p o p o p o
]

Train example 2:
Input2 = [
 p g p g p g p b
 g p g p g p g b
 p g p g p g p b
 g p g p g p g b
 p g p g p g p b
 g p g p g p g b
 p g p g p g p b
 b b b b b b b b
]
Output2 = [
 g p g p g p g p
 p g p g p g p g
 g p g p g p g p
 p g p g p g p g
 g p g p g p g p
 p g p g p g p g
 g p g p g p g p
 p g p g p g p g
]

Train example 3:
Input3 = [
 e y e y e p
 y e y e y p
 e y e y e p
 y e y e y p
 e y e y e p
 p p p p p p
]
Output3 = [
 y e y e y e
 e y e y e y
 y e y e y e
 e y e y e y
 y e y e y e
 e y e y e y
] """

# concepts:
# translational symmetry, symmetry detection, non-black background

# description:
# In the input you will see a a translationally symmetric pattern that does not extend to cover the entire canvas. The background is not black.
# To make the output, continue the symmetric pattern until it covers the entire canvas, but shift everything right by one pixel.
 
def transform_caa06a1f(input_grid):
    # Plan:
    # 1. Find the background color
    # 2. Find the repeated translation, which is a symmetry
    # 3. Extend the pattern by computing the orbit of each pixel in the pattern

    # Find the background color which is the most common color along the border of the canvas
    pixels_on_border = np.concatenate([input_grid[0, :], input_grid[-1, :], input_grid[:, 0], input_grid[:, -1]])
    background = max(set(pixels_on_border), key=list(pixels_on_border).count)
    
    # Find the repeated translation, which is a symmetry
    symmetries = detect_translational_symmetry(input_grid, ignore_colors=[], background=background)
    assert len(symmetries) > 0, "No translational symmetry found"

    # because we are going to shift everything right by one pixel, we make an output grid which is one pixel wider
    # at the end, we will just remove the leftmost pixels
    width, height = input_grid.shape
    output_grid = np.full((width+1, height), Color.BLACK)
    
    # Copy all of the input pixels to the output, INCLUDING their symmetric copies (i.e. their orbit)
    for x, y in np.argwhere(input_grid != background):
        # Compute the orbit into the output grid
        for x2, y2 in orbit(output_grid, x, y, symmetries):
            output_grid[x2, y2] = input_grid[x, y]
    
    # Shift everything right by one pixel by removing the leftmost pixels
    output_grid = output_grid[1:, :]

    return output_grid

""" ==============================
Puzzle ce602527

Train example 1:
Input1 = [
 b b b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b b b
 b b b b b b b b b b b g g b g g b
 b b b r r r r r b b b g b b b g b
 b b b r b r b r b b b g g g g g b
 b b b b b b b r b b b b b g b b b
 b b b r b r b r b b b g g g g g b
 b b b r r r r r b b b b b b b b b
 b b b b b b b b b b b b b b b b b
 b b b t t t t t t t t t t b b b b
 b b b t t t t t t t t t t b b b b
 b b b t t b b t t b b t t b b b b
 b b b t t b b t t b b t t b b b b
 b b b b b b b b b b b t t b b b b
 b b b b b b b b b b b t t b b b b
 b b b t t b b t t b b t t b b b b
 b b b t t b b t t b b t t b b b b
]
Output1 = [
 r r r r r
 r b r b r
 b b b b r
 r b r b r
 r r r r r
]

Train example 2:
Input2 = [
 t t t t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t t t t
 t t t t g t g t t t t t t t t t t t
 t t t g g g g g t t t t t t t t t t
 t t t t g t g t t t t t t t t t t t
 t t t t t t t t t t t t t t t t t t
 t t t t t t t t t t t t y t t t t t
 t t t t t t t t t t t y y y t t t t
 t p p t t t t t t t t t y t t t t t
 t p p t t t t t t t t y y y t t t t
 p p p p p t t t t t t t y t t t t t
 p p p p p t t t t t t t t t t t t t
 t p p t t t t t t t t t t t t t t t
 t p p t t t t t t t t t t t t t t t
 p p p p p t t t t t t t t t t t t t
 p p p p p t t t t t t t t t t t t t
]
Output2 = [
 t y t
 y y y
 t y t
 y y y
 t y t
]

Train example 3:
Input3 = [
 r r r r r r r r r r r r r r r r r
 r r r r r r r r r r r r r r r r r
 r r r r t t t r r r r r r r r r r
 r r r r t r r r r r r r r r r r r
 r r r r t t t r r r r r r r r r r
 r r r r r r t r r r r r r r r r r
 r r r r t t t r r r r r r r r r r
 r r r r r r r r r r r r r r r r r
 r r r r r r r r r r r r r r r r r
 r r r r r r r r r r r r r r r r r
 r r r r r r r r g r g r g r r r r
 r r r r r r r r g g g g g r r r r
 r r r r r r r r r r g r r r r r r
 b b b b r r r r r r r r r r r r r
 b b b b r r r r r r r r r r r r r
 r r r r r r r r r r r r r r r r r
 r r r r r r r r r r r r r r r r r
 b b b b r r r r r r r r r r r r r
 b b b b r r r r r r r r r r r r r
]
Output3 = [
 t t t
 t r r
 t t t
 r r t
 t t t
] """

# concepts:
# scaling, shape matching, non-black background

# description:
# In the input you should see three objects, two of which are the same shape but different sizes and colors. The third object is a different shape.
# To make the output, you need to find the two objects that are the same shape but different sizes and colors.
# Return the smaller object in the same shape.

def transform_ce602527(input_grid):
    # Plan:
    # 1. Find the background color
    # 2. Extract objects by color
    # 3. Define a helper function to check if two objects are the same shape but different color/scale, remembering that the bigger one might be partially out of the canvas
    # 4. Iterate all candidate objects and check if they are the same shape but different color/scale
    # 5. Return what we find

    # Determine the background color, which is the most common color in the grid
    colors = np.unique(input_grid)
    background = colors[np.argmax([np.sum(input_grid == c) for c in colors])]
    object_colors = [c for c in colors if c != background]

    # Extract the objects, each of which is a different color.
    # This means we can split the canvas by color, instead of using connected components.
    objects = []
    for color in object_colors:
        object = np.copy(input_grid)
        object[input_grid != color] = background
        objects.append(object)
    
    # Define a helper function for checking if two objects are different color/scale but same shape
    # This has to handle the case where the bigger object is partially outside the grid
    def same_shape_different_color_different_scale(obj1, obj2):
        # obj1 is the smaller object
        if np.sum(obj1 != background) > np.sum(obj2 != background): return False

        mask1 = crop(obj1, background=background) != background
        mask2 = crop(obj2, background=background) != background

        # Loop through all possible scale factors
        for scale_factor in range(2, 4):
            scaled_mask1 = scale_sprite(mask1, scale_factor)
            # loop over all possible places that we might put mask2, which starts anywhere in the scaled_mask1
            # note that we are only doing this because there can be objects that fall outside of the canvas
            # otherwise we would just compare the two masks directly
            for dx, dy in np.ndindex(scaled_mask1.shape):
                if np.array_equal(scaled_mask1[dx : dx + mask2.shape[0], dy : dy + mask2.shape[1]], mask2):
                    return True
        return False
    
    output_grid_candidates = []
    # Iterate all candidate objects
    for obj in objects:
        other_objects = [o for o in objects if o is not obj]
        if any( same_shape_different_color_different_scale(obj, other_obj) for other_obj in other_objects ):
            output_grid_candidates.append(obj)

    # Check if the generated input grid is valid
    assert len(output_grid_candidates) == 1, f"Should only have one output grid candidate, have {len(output_grid_candidates)}"

    output_grid = crop(output_grid_candidates[0], background=background)

    return output_grid

""" ==============================
Puzzle cf98881b

Train example 1:
Input1 = [
 k y k y r m m k k r k k k k
 k y k k r k k m m r k b k k
 y k k k r k k k k r b b b k
 y y y y r m k m k r b b k b
]
Output1 = [
 m y k y
 k y m m
 y b b k
 y y y y
]

Train example 2:
Input2 = [
 y y y y r m k m k r k k k b
 y y k k r m m k k r b k k k
 y k y y r k k k m r k b k b
 k k k k r k k m k r b k b k
]
Output2 = [
 y y y y
 y y k k
 y b y y
 b k m k
]

Train example 3:
Input3 = [
 y y y k r m m k m r k b k b
 k y k y r k k m k r k b k k
 k y k y r k k m m r b k k b
 y k y y r m m m k r k k k b
]
Output3 = [
 y y y m
 k y m y
 b y m y
 y m y y
] """

# concepts:
# occlusion

# description:
# In the input you will see three regions separated by red vertical bars. Each region is rectangular and the regions are arranged horizontally, so there is a left region, middle region, and a right region. 
# The regions display a yellow pattern, a maroon pattern, and a blue pattern from left to right.
# To make the output, you have to copy the blue pattern first, then overlay the maroon pattern over that, finally overlay the yellow pattern over that as well.

def transform_cf98881b(input_grid):
    # find the location of the vertical red bars that separate the three sections
    red_bars = np.where(input_grid == Color.RED)

    # get the unique x-coordinates of the red bars
    red_bars_x = np.unique(red_bars[0])

    # get the blue pattern from the third section and copy it to make the base of the output grid
    blue_pattern = input_grid[red_bars_x[1]+1:, :]
    output_grid = blue_pattern
    # could also have used blit_sprite:
    # output_grid = blit_sprite(output_grid, blue_pattern, x=0, y=0)

    # get the maroon pattern from the second section and overlay it on output grid
    maroon_pattern = input_grid[red_bars_x[0]+1:red_bars_x[1], :]
    output_grid = np.where(maroon_pattern, maroon_pattern, output_grid)
    # could also have used blit:
    # output_grid = blit_sprite(output_grid, maroon_pattern, x=0, y=0)

    # get the yellow pattern from the first section and overlay it on output grid
    yellow_pattern = input_grid[0:red_bars_x[0], :]
    output_grid = np.where(yellow_pattern, yellow_pattern, output_grid)
    # could also have used blit:
    # output_grid = blit_sprite(output_grid, yellow_pattern, x=0, y=0)

    return output_grid

""" ==============================
Puzzle d06dbe63

Train example 1:
Input1 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k t k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k e k k k k k k
 k k k k e e e k k k k k k
 k k k k e k k k k k k k k
 k k k k t k k k k k k k k
 k k k k e k k k k k k k k
 k k e e e k k k k k k k k
 k k e k k k k k k k k k k
 e e e k k k k k k k k k k
 e k k k k k k k k k k k k
 e k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k t k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
 k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k e
 k k k k k k k k k k e e e
 k k k k k k k k k k e k k
 k k k k k k k k e e e k k
 k k k k k k k k e k k k k
 k k k k k k e e e k k k k
 k k k k k k e k k k k k k
 k k k k k k t k k k k k k
 k k k k k k e k k k k k k
 k k k k e e e k k k k k k
 k k k k e k k k k k k k k
 k k e e e k k k k k k k k
 k k e k k k k k k k k k k
] """

# concepts:
# staircase pattern

# description:
# In the input you will see a single teal pixel.
# To make the output, draw a staircase from the teal pixel to the upper right and lower left with a step size of 2.

def transform_d06dbe63(input_grid):
    # Find the location of the teal pixel
    teal_x, teal_y = np.argwhere(input_grid == Color.TEAL)[0]

    # staircase is gray
    staircase_color = Color.GRAY

    # we are going to draw on top of the input
    output_grid = input_grid.copy()
    width, height = input_grid.shape

    # Draw stairs from the teal pixel
    STAIR_LEN = 2
    # First draw stair to the upper right
    x, y = teal_x, teal_y
    while 0 <= x < width and 0 <= y < height:
        # go up
        draw_line(output_grid, x, y, length=STAIR_LEN, color=staircase_color, direction=(0, -1))
        y -= STAIR_LEN
        # go right
        draw_line(output_grid, x, y, length=STAIR_LEN, color=staircase_color, direction=(1, 0))
        x += STAIR_LEN
    
    # Then draw stair to the lower left
    x, y = teal_x, teal_y
    while 0 <= x < width and 0 <= y < height:
        # go down
        draw_line(output_grid, x, y, length=STAIR_LEN, color=staircase_color, direction=(0, 1))
        y += STAIR_LEN
        # go left
        draw_line(output_grid, x, y, length=STAIR_LEN, color=staircase_color, direction=(-1, 0))
        x -= STAIR_LEN
    
    # make sure that the teal pixel stays there
    output_grid[teal_x, teal_y] = Color.TEAL

    return output_grid

""" ==============================
Puzzle d2abd087

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k e e e k k k k k
 k k e e e k k k k k
 k k k k k k k k k k
 k k k k k k e e k k
 k k k k k e e e k k
 k e e k k k e k k k
 k e e e k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k r r r k k k k k
 k k r r r k k k k k
 k k k k k k k k k k
 k k k k k k r r k k
 k k k k k r r r k k
 k b b k k k r k k k
 k b b b k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k e k
 k e e k k k e e e k
 e e e e k k k k e k
 k k k k k k k k k k
 k k e e e e k k e k
 k k k k k k k k e k
 k k k k k e e k k k
 k e e k k e e k k k
 k e e k k e e k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k b k
 k r r k k k b b b k
 r r r r k k k k b k
 k k k k k k k k k k
 k k b b b b k k b k
 k k k k k k k k b k
 k k k k k r r k k k
 k b b k k r r k k k
 k b b k k r r k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 e e e k k k k e e e
 k e e k e e k e k k
 k k e k e e k e k k
 k k e k k k k e k k
 k k k k e e k k k e
 k e e k k e e k k e
 k k k k k e e k k e
 k k e k k k k k k k
 k e e e e k k k k k
 k k e e k k k k k k
]
Output3 = [
 b b b k k k k r r r
 k b b k b b k r k k
 k k b k b b k r k k
 k k b k k k k r k k
 k k k k r r k k k b
 k b b k k r r k k b
 k k k k k r r k k b
 k k b k k k k k k k
 k b b b b k k k k k
 k k b b k k k k k k
] """

# concepts:
# counting

# description:
# The input consists of several grey objects in a 10x10 grid.
# To create the output, change the color of all objects of area 6 to red, and all other objects to blue.

def transform_d2abd087(input_grid):
    # extract objects
    objects = find_connected_components(input_grid, connectivity=4)

    # convert each object to the desired color
    for obj in objects:
        if np.sum(obj != Color.BLACK) == 6:
            obj[obj != Color.BLACK] = Color.RED
        else:
            obj[obj != Color.BLACK] = Color.BLUE

    # place new objects back into a grid
    output_grid = np.zeros_like(input_grid)
    for obj in objects:
        output_grid = blit_object(output_grid, obj, background=Color.BLACK)

    return output_grid

""" ==============================
Puzzle d4a91cb9

Train example 1:
Input1 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k t k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k r k k
 k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k t k k k k k k k k k k
 k y k k k k k k k k k k
 k y k k k k k k k k k k
 k y k k k k k k k k k k
 k y k k k k k k k k k k
 k y k k k k k k k k k k
 k y y y y y y y y r k k
 k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k
 k k k k k k k k t k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k r k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k
 k k k k k k k k t k k
 k k k k k k k k y k k
 k k k k k k k k y k k
 k k k k k k k k y k k
 k r y y y y y y y k k
 k k k k k k k k k k k
 k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k
 k k k k k k k k r k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k t k k k k k k k k
 k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k
 k k y y y y y y r k k
 k k y k k k k k k k k
 k k y k k k k k k k k
 k k y k k k k k k k k
 k k y k k k k k k k k
 k k y k k k k k k k k
 k k y k k k k k k k k
 k k y k k k k k k k k
 k k y k k k k k k k k
 k k t k k k k k k k k
 k k k k k k k k k k k
] """

# concepts:
# lines, color

# description:
# In the input you will see a red pixel and a teal pixel.
# To make the output, draw a horizontal yellow line from the red pixel to the column of the teal pixel, then draw a vertical yellow line from there to the teal pixel.

def transform_d4a91cb9(input_grid):
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # find the red and teal pixels
    red_x, red_y = np.where(input_grid == Color.RED)
    teal_x, teal_y = np.where(input_grid == Color.TEAL)

    # draw the horizontal yellow line from the red pixel to the column the teal pixel is
    # figure out the direction of the line
    if red_x[0] < teal_x[0]:
        direction = (1, 0)
    else:
        direction = (-1, 0)
    # draw the line but don't draw over the red pixel
    draw_line(output_grid, red_x[0]+direction[0], red_y[0], length=np.abs(teal_x[0] - red_x[0]), color=Color.YELLOW, direction=direction)

    # draw the vertical yellow line from the end of the horizontal yellow line to the teal pixel
    # figure out the direction of the line
    if red_y[0] < teal_y[0]:
        direction = (0, 1)
    else:
        direction = (0, -1)
    # draw the line
    draw_line(output_grid, teal_x[0], red_y[0], length=None, color=Color.YELLOW, direction=direction, stop_at_color=[Color.TEAL])

    return output_grid

""" ==============================
Puzzle d4f3cd78

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k e e e e e e k k
 k k e k k k k e k k
 k k e k k k k e k k
 k k e k k k k e k k
 k k e k k k k e k k
 k k e e e k e e k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k e e e e e e k k
 k k e t t t t e k k
 k k e t t t t e k k
 k k e t t t t e k k
 k k e t t t t e k k
 k k e e e t e e k k
 k k k k k t k k k k
 k k k k k t k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k e e e k e e k k
 k k e k k k k e k k
 k k e k k k k e k k
 k k e k k k k e k k
 k k e e e e e e k k
]
Output2 = [
 k k k k k t k k k k
 k k k k k t k k k k
 k k k k k t k k k k
 k k k k k t k k k k
 k k k k k t k k k k
 k k e e e t e e k k
 k k e t t t t e k k
 k k e t t t t e k k
 k k e t t t t e k k
 k k e e e e e e k k
] """

# concepts:
# cups, filling

# description:
# In the input you will see grey cups, meaning an almost-enclosed shape with a small opening on one of its sides, and empty space (black pixels) inside.
# To make the output grid, you should fill the interior of each cup with teal, then shoot teal out of the opening of the cup straight out in a line.

def transform_d4f3cd78(input_grid):
    # Plan:
    # 1. Detect the cup
    # 2. Find the mask of the inside of the cup
    # 3. Find the mask of the opening of the cup (on one of its sides)
    # 4. Fill the cup with teal
    # 5. Shoot pixels outward from the opening (straight out)
    
    # 1. Detect cup
    objects = find_connected_components(input_grid, connectivity=4, background=Color.BLACK)
    assert len(objects) == 1, "There should be exactly one cup"
    obj = list(objects)[0]

    output_grid = input_grid.copy()

    # 2. Extract what's inside the cup (as its own object), which is everything in the bounding box that is not the object itself
    cup_x, cup_y, cup_width, cup_height = bounding_box(obj)
    inside_cup_mask = np.zeros_like(input_grid, dtype=bool)
    inside_cup_mask[cup_x:cup_x+cup_width, cup_y:cup_y+cup_height] = True
    inside_cup_mask = inside_cup_mask & (obj == Color.BLACK)

    # 3. Extract the hole in the cup, which is what's inside and on the boundary of the bounding box
    # what's inside...
    hole_mask = inside_cup_mask.copy()
    # ...and then we need to remove anything not on the boundary
    hole_mask[cup_x+1 : cup_x+cup_width-1, cup_y+1 : cup_y+cup_height-1] = False

    # 4. Fill the cup with teal
    output_grid[inside_cup_mask] = Color.TEAL

    # 5. Shoot pixels outward from the opening (straight out)
    # Find the direction of the opening, which is the unit vector that points away from the interior
    for cardinal_direction in [ (0, 1), (0, -1), (1, 0), (-1, 0) ]:
        dx, dy = cardinal_direction
        hole_x, hole_y = object_position(hole_mask, background=Color.BLACK, anchor='center')
        if inside_cup_mask[hole_x - dx, hole_y - dy]:
            direction = cardinal_direction
            break
    # Loop over every boundary pixel and shoot outward
    for x, y in np.argwhere(hole_mask):
        draw_line(output_grid, x, y, direction=direction, color=Color.TEAL)
    return output_grid

""" ==============================
Puzzle d511f180

Train example 1:
Input1 = [
 r o t t t
 e e p e y
 t e e e r
 t t y g p
 p e b m g
]
Output1 = [
 r o e e e
 t t p t y
 e t t t r
 e e y g p
 p t b m g
]

Train example 2:
Input2 = [
 g e b
 y e t
 r y m
]
Output2 = [
 g t b
 y t e
 r y m
]

Train example 3:
Input3 = [
 p e g
 e o e
 t t r
]
Output3 = [
 p t g
 t o t
 e e r
] """

# concepts:
# colors

# description:
# To create the output grid, swap the teal and grey colors in the grid.

def transform_d511f180(input_grid):
    output_grid = input_grid.copy()
    output_grid[input_grid == Color.GREY] = Color.TEAL
    output_grid[input_grid == Color.TEAL] = Color.GREY
    return output_grid

""" ==============================
Puzzle d6ad076f

Train example 1:
Input1 = [
 k k k k k k k k k k
 k r r r r k k k k k
 k r r r r k k k k k
 k r r r r k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 o o o o o o k k k k
 o o o o o o k k k k
 o o o o o o k k k k
]
Output1 = [
 k k k k k k k k k k
 k r r r r k k k k k
 k r r r r k k k k k
 k r r r r k k k k k
 k k t t k k k k k k
 k k t t k k k k k k
 k k t t k k k k k k
 o o o o o o k k k k
 o o o o o o k k k k
 o o o o o o k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k y y y k k k k k k
 k y y y k k k k k k
 k y y y k k k p p p
 k y y y k k k p p p
 k y y y k k k p p p
 k y y y k k k p p p
 k y y y k k k p p p
 k y y y k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k y y y k k k k k k
 k y y y k k k k k k
 k y y y k k k p p p
 k y y y t t t p p p
 k y y y t t t p p p
 k y y y t t t p p p
 k y y y k k k p p p
 k y y y k k k k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 g g g g g g g g g k
 g g g g g g g g g k
 g g g g g g g g g k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k m m m m m m k
 k k k m m m m m m k
]
Output3 = [
 g g g g g g g g g k
 g g g g g g g g g k
 g g g g g g g g g k
 k k k k t t t t k k
 k k k k t t t t k k
 k k k k t t t t k k
 k k k k t t t t k k
 k k k k t t t t k k
 k k k m m m m m m k
 k k k m m m m m m k
] """

# concepts:
# connected components

# description:
# In the input you will see two rectangles separated by a gap.
# To make the output, you need to connect the two rectangles with a teal line.

def transform_d6ad076f(input_grid):
    # Copy the input grid as output
    output_grid = input_grid.copy()

    # Detect the objects
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    for x, y in np.argwhere(input_grid == Color.BLACK):
        # Check if the current position is between the two objects
        # Also ensure it is not between the borders of the objects (padding=1)
        if check_between_objects(obj1=objects[0], obj2=objects[1], x=x, y=y, padding=1):
            output_grid[x, y] = Color.TEAL
    
    return output_grid

""" ==============================
Puzzle d9f24cd1

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k e k k k
 k k k k k k k k k k
 k k e k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k r k k r k r k k k
]
Output1 = [
 k r k k r k k r k k
 k r k k r k k r k k
 k r k k r k k r k k
 k r k k r k e r k k
 k r k k r k r r k k
 k r e k r k r k k k
 k r k k r k r k k k
 k r k k r k r k k k
 k r k k r k r k k k
 k r k k r k r k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k e k k k k
 k k k k k k k k k k
 k e k k k k k k k k
 k k k k k k k k e k
 k k k k k k k k k k
 k k k k k k k k k k
 k r k k r k k r k k
]
Output2 = [
 k k r k r k k r k k
 k k r k r k k r k k
 k k r k r k k r k k
 k k r k r e k r k k
 k k r k r k k r k k
 k e r k r k k r k k
 k r r k r k k r e k
 k r k k r k k r k k
 k r k k r k k r k k
 k r k k r k k r k k
] """

# concepts:
# line drawing, obstacle avoidance

# description:
# In the input you will see several red pixels on the bottom row of the grid, and some gray pixels scattered on the grid.
# To make the output grid, you should draw a red line upward from each red pixel, but avoiding the gray pixels.
# To avoid touching the gray pixels, go right to avoid them until you can go up again.

def transform_d9f24cd1(input_grid):
    # The output grid is the same size as the input grid, and we are going to draw on top of the input, so we copy it
    output_grid = input_grid.copy()
    width, height = input_grid.shape

    # Get the positions of the red pixels on the bottom row
    for x, y in np.argwhere(input_grid == Color.RED):
        # Draw the red line upward, but move to the right to avoid touching gray pixels
        while 0 < y < height and 0 < x < width:
            if output_grid[x, y - 1] == Color.GRAY:
                # If the red line touch the gray pixel, it should go right then up to avoid the gray pixel.
                output_grid[x + 1, y] = Color.RED
                x += 1
            else:
                # Otherwise we go up
                output_grid[x, y - 1] = Color.RED
                y -= 1

    return output_grid

""" ==============================
Puzzle db3e9e38

Train example 1:
Input1 = [
 k k k o k k k
 k k k o k k k
 k k k o k k k
 k k k o k k k
 k k k k k k k
]
Output1 = [
 t o t o t o t
 k o t o t o k
 k k t o t k k
 k k k o k k k
 k k k k k k k
]

Train example 2:
Input2 = [
 k k o k k k k k
 k k o k k k k k
 k k o k k k k k
 k k o k k k k k
 k k o k k k k k
 k k k k k k k k
 k k k k k k k k
]
Output2 = [
 o t o t o t o k
 o t o t o t k k
 o t o t o k k k
 k t o t k k k k
 k k o k k k k k
 k k k k k k k k
 k k k k k k k k
] """

# concepts:
# pixel patterns, pyramid, color alternation

# description:
# In the input you will see a single orange line that connects to the top of the grid.
# To make the output, you should draw a pyramid pattern outward from the orange line.
# The pattern is expanded from the orange line to the left and right of the grid.
# Each line of the pattern is one cell shorter than the previous one, and the color alternates between orange and teal.

def transform_db3e9e38(input_grid):
    # Plan:
    # 1. Parse the input
    # 2. Draw the left side of the pyramid
    # 3. Draw the right side of the pyramid

    # 1. Parse the input
    # Extract the orange line from the input grid
    original_line = find_connected_components(input_grid, monochromatic=True)[0]
    original_x, original_y, width, height = bounding_box(original_line)

    # two color pattern
    color1 = Color.ORANGE
    color2 = Color.TEAL
    # Draw on top of the input
    output_grid = np.copy(input_grid)

    # Draw the pattern from the orange line and expand to left and right
    # Each line is one cell shorter than the previous one
    # The line is colored alternately between color1 and color2

    # 2. draw pattern from left to right
    cur_color = color2
    cur_height = height - 1
    for x in range(original_x + 1, output_grid.shape[0]):
        # If the height of the line is 0, stop drawing
        if cur_height <= 0:
            break
        draw_line(output_grid, x=x, y=original_y, direction=(0, 1), length=cur_height, color=cur_color)
        # pyramid pattern, each line is one pixel shorter than the previous one
        cur_height -= 1
        # colors alternate
        cur_color = color1 if cur_color == color2 else color2
    
    # 3. Then draw pattern from right to left
    cur_color = color2
    cur_height = height - 1
    for x in reversed(range(original_x)):
        # If the height of the line is 0, stop drawing
        if cur_height <= 0:
            break
        draw_line(output_grid, x=x, y=original_y, direction=(0, 1), length=cur_height, color=cur_color)
        # pyramid pattern, each line is one pixel shorter than the previous one
        cur_height -= 1
        # colors alternate
        cur_color = color1 if cur_color == color2 else color2
    
    return output_grid

""" ==============================
Puzzle db93a21d

Train example 1:
Input1 = [
 k k k k k k m m k k
 k k k k k k m m k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k m m k k k k k k k
 k m m k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k g m m g k
 k k k k k g m m g k
 k k k k k g g g g k
 k k k k k k b b k k
 k k k k k k b b k k
 k k k k k k b b k k
 g g g g k k b b k k
 g m m g k k b b k k
 g m m g k k b b k k
 g g g g k k b b k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k m m m m k k k
 k k k k k k k k m m m m k k k
 k k k k k k k k m m m m k k k
 k k k k k k k k m m m m k k k
 k k k m m k k k k k k k k k k
 k k k m m k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k m m
 k k k k k k k k k k k k k m m
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k g g g g g g g g k
 k k k k k k g g g g g g g g k
 k k k k k k g g m m m m g g k
 k k k k k k g g m m m m g g k
 k k k k k k g g m m m m g g k
 k k g g g g g g m m m m g g k
 k k g m m g g g g g g g g g k
 k k g m m g g g g g g g g g k
 k k g g g g k k b b b b g g g
 k k k b b k k k b b b b g m m
 k k k b b k k k b b b b g m m
 k k k b b k k k b b b b g g g
 k k k b b k k k b b b b k b b
 k k k b b k k k b b b b k b b
 k k k b b k k k b b b b k b b
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k m m k k k k k k k k k k k k k k k k
 k k m m k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k m m m m k
 k k k k k k k k k k k k k k k m m m m k
 k k k k k k k k k k k k k k k m m m m k
 k k k k k k k k k k k k k k k m m m m k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k m m m m m m k k k k k k k k
 k k k k k k m m m m m m k k k k k k k k
 k k k k k k m m m m m m k k k k k k k k
 k k k k k k m m m m m m k k k k k k k k
 k k k k k k m m m m m m k k k k k k k k
 k k k k k k m m m m m m k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k k k k k k
 k g g g g k k k k k k k k k k k k k k k
 k g m m g k k k k k k k k k k k k k k k
 k g m m g k k k k k k k k g g g g g g g
 k g g g g k k k k k k k k g g g g g g g
 k k b b k k k k k k k k k g g m m m m g
 k k b b k k k k k k k k k g g m m m m g
 k k b b k k k k k k k k k g g m m m m g
 k k b g g g g g g g g g g g g m m m m g
 k k b g g g g g g g g g g g g g g g g g
 k k b g g g g g g g g g g g g g g g g g
 k k b g g g m m m m m m g g g b b b b k
 k k b g g g m m m m m m g g g b b b b k
 k k b g g g m m m m m m g g g b b b b k
 k k b g g g m m m m m m g g g b b b b k
 k k b g g g m m m m m m g g g b b b b k
 k k b g g g m m m m m m g g g b b b b k
 k k b g g g g g g g g g g g g b b b b k
 k k b g g g g g g g g g g g g b b b b k
] """

# concepts:
# Expanding, Framing, Growing

# description:
# In the input you will see some squares of different sizes and colors.
# To make the output, you need to:
# 1. Expand the squares down to the bottom of the grid using the color BLUE.
# 2. Draw a green frame around the squares. The green square should be twice as long as the original square.
# 3. Put the original square back to the center of the green square.

def transform_db93a21d(input_grid):
    # Plan:
    # 1. Extract the square objects from the input grid
    # 2. Expand the squares down to the bottom using the color BLUE
    # 3. Draw the frame
    # 4. Put the original squares back

    # 1. Input parsing and setup
    # Extract the squares in the input grid
    square_objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Note the frame color and the expansion down color
    frame_color = Color.GREEN
    expand_color = Color.BLUE

    # The output grid is the same size as the input grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # 2. Expand the square down to the bottom use color expand_color
    for square_obj in square_objects:
        x, y, w, h = bounding_box(square_obj)
        # Equivalently:
        # output_grid[x:x+w, y+h:] = expand_color
        for i in range(x, x + w):
            draw_line(grid=output_grid, x=i, y=y + h, color=expand_color, direction=(0, 1))
    
    # 3. Draw a green square frame around the original squares, the green square should be twice as big as original were
    for square_obj in square_objects:
        # The square can be partly outside the canvas
        # This math is to get the (x,y) of the top-left corner of the square, even if it's outside the canvas
        x, y, w, h = bounding_box(square_obj)
        square_len = max(w, h)
        x -= (square_len - w)
        y -= (square_len - h)
        # Make and draw the frame
        frame_len = square_len * 2
        green_frame = np.full((frame_len, frame_len), frame_color)
        blit_sprite(output_grid, green_frame, x - square_len // 2, y - square_len // 2)
    
    # 4. Put the original square back to the center of the green square
    for square_obj in square_objects:
        x, y, w, h = bounding_box(square_obj)
        square_obj = crop(square_obj)
        blit_sprite(output_grid, square_obj, x, y)

    return output_grid

""" ==============================
Puzzle e179c5f4

Train example 1:
Input1 = [
 k k
 k k
 k k
 k k
 k k
 k k
 k k
 k k
 k k
 b k
]
Output1 = [
 t b
 b t
 t b
 b t
 t b
 b t
 t b
 b t
 t b
 b t
]

Train example 2:
Input2 = [
 k k k
 k k k
 k k k
 k k k
 k k k
 k k k
 k k k
 k k k
 k k k
 b k k
]
Output2 = [
 t b t
 b t t
 t b t
 t t b
 t b t
 b t t
 t b t
 t t b
 t b t
 b t t
]

Train example 3:
Input3 = [
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 k k k k
 b k k k
]
Output3 = [
 t t t b
 t t b t
 t b t t
 b t t t
 t b t t
 t t b t
 t t t b
 t t b t
 t b t t
 b t t t
] """

# concepts:
# bouncing

# description:
# In the input you will see a single blue pixel on a black background
# To make the output, shoot the blue pixel diagonally up and to the right, having it reflect and bounce off the walls until it exits at the top of the grid. Finally, change the background color to teal.

def transform_e179c5f4(input_grid):
    # Plan:
    # 1. Detect the pixel
    # 2. Shoot each line of the reflection one-by-one, bouncing (changing horizontal direction) when it hits a (horizontal) wall/edge of canvas
    # 3. Change the background color to teal

    # 1. Find the location of the pixel
    blue_pixel_x, blue_pixel_y = np.argwhere(input_grid == Color.BLUE)[0]

    # 2. do the bounce which requires keeping track of the direction of the ray we are shooting, as well as the tip of the ray
    # initially we are shooting diagonally up and to the right (dx=1, dy=-1)
    # initially the tip of the ray is the blue pixel, x=blue_pixel_x, y=blue_pixel_y
    direction = (1, -1)

    # loop until we fall out of the canvas
    while 0 <= blue_pixel_x < input_grid.shape[0] and 0 <= blue_pixel_y < input_grid.shape[1]:
        stop_x, stop_y = draw_line(input_grid, blue_pixel_x, blue_pixel_y, direction=direction, color=Color.BLUE)
        # Terminate if we failed to make progress
        if stop_x == blue_pixel_x and stop_y == blue_pixel_y:
            break
        blue_pixel_x, blue_pixel_y = stop_x, stop_y
        direction = (-direction[0], direction[1])
    
    old_background = Color.BLACK
    new_background = Color.TEAL
    input_grid[input_grid == old_background] = new_background
    
    return input_grid

""" ==============================
Puzzle e21d9049

Train example 1:
Input1 = [
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k t k k k k k k k k
 k k g k k k k k k k k
 t g r k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
 k k k k k k k k k k k
]
Output1 = [
 k k g k k k k k k k k
 k k r k k k k k k k k
 k k t k k k k k k k k
 k k g k k k k k k k k
 t g r t g r t g r t g
 k k t k k k k k k k k
 k k g k k k k k k k k
 k k r k k k k k k k k
 k k t k k k k k k k k
 k k g k k k k k k k k
 k k r k k k k k k k k
 k k t k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k y k k k k k
 k k k k k k k k t k k k k k
 k k k k k k k r g t y k k k
 k k k k k k k k r k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k y k k k k k
 k k k k k k k k t k k k k k
 k k k k k k k k g k k k k k
 k k k k k k k k r k k k k k
 k k k k k k k k y k k k k k
 k k k k k k k k t k k k k k
 g t y r g t y r g t y r g t
 k k k k k k k k r k k k k k
 k k k k k k k k y k k k k k
 k k k k k k k k t k k k k k
 k k k k k k k k g k k k k k
 k k k k k k k k r k k k k k
 k k k k k k k k y k k k k k
 k k k k k k k k t k k k k k
 k k k k k k k k g k k k k k
] """

# concepts:
# pixel patterns, expansion, color sequence

# description:
# In the input you will see a grid with a cross pattern. Each pixel in the cross has a different color.
# To make the output, you should expand the cross right/left/top/bottom following the original color sequence of the cross.

def transform_e21d9049(input_grid):
    # Plan:
    # 1. Parse the input and create output canvas to draw on top of
    # 2. Extract the vertical and horizontal parts of the cross, and make note of the coordinate of the middle
    # 3. Expand the horizontal part to the right and left, aligned with the middle y coordinate
    # 4. Expand the vertical part to the top and bottom, aligned with the middle x coordinate    

    # 1. Input parsing
    # Extract the cross, which has many colors and so is not monochromatic.
    objects = find_connected_components(input_grid, monochromatic=False)
    assert len(objects) == 1, "exactly one cross expected"
    obj = objects[0]
    cross_x, cross_y = object_position(obj)

    # Create output grid, which we are going to draw on top of, so we start with the input grid
    output_grid = input_grid.copy()
    width, height = input_grid.shape

    # 2. Cross analysis: Extract subsprites, get the middle
    # Extract the horizontal/vertical parts of the cross sprite by figuring out where its middle is (where the horizontal and vertical lines meet)
    sprite = crop(obj)
    cross_width, cross_height = sprite.shape
    # Middle is where they meet
    cross_middle_x = next( x for x in range(cross_width) if np.all(sprite[x, :] != Color.BLACK) )
    cross_middle_y = next( y for y in range(cross_height) if np.all(sprite[:, y] != Color.BLACK) )
    # Extract the horizontal and vertical parts of the cross
    vertical_sprite = sprite[cross_middle_x:cross_middle_x+1, :]
    horizontal_sprite = sprite[:, cross_middle_y:cross_middle_y+1]

    # 3. Expand the horizontal line to the right and left
    x_start, y_start, len_line = cross_x, cross_y + cross_middle_y, cross_width
    for i in range(x_start, width, len_line):
        blit_sprite(output_grid, horizontal_sprite, x=i, y=y_start)
    for i in range(x_start, -(len_line), -len_line):
        blit_sprite(output_grid, horizontal_sprite, x=i, y=y_start)
    
    # 4. Expand the vertical line to the top and bottom
    x_start, y_start, len_line = cross_x + cross_middle_x, cross_y, cross_height
    for i in range(y_start, height, len_line):
        blit_sprite(output_grid, vertical_sprite, x=x_start, y=i)
    for i in range(y_start, -(len_line), -len_line):
        blit_sprite(output_grid, vertical_sprite, x=x_start, y=i)
        
    return output_grid

""" ==============================
Puzzle e48d4e1a

Train example 1:
Input1 = [
 k k k r k k k k k e
 k k k r k k k k k e
 k k k r k k k k k k
 k k k r k k k k k k
 k k k r k k k k k k
 k k k r k k k k k k
 r r r r r r r r r r
 k k k r k k k k k k
 k k k r k k k k k k
 k k k r k k k k k k
]
Output1 = [
 k r k k k k k k k k
 k r k k k k k k k k
 k r k k k k k k k k
 k r k k k k k k k k
 k r k k k k k k k k
 k r k k k k k k k k
 k r k k k k k k k k
 k r k k k k k k k k
 r r r r r r r r r r
 k r k k k k k k k k
]

Train example 2:
Input2 = [
 k k k y k k k k k e
 k k k y k k k k k e
 k k k y k k k k k e
 y y y y y y y y y y
 k k k y k k k k k k
 k k k y k k k k k k
 k k k y k k k k k k
 k k k y k k k k k k
 k k k y k k k k k k
 k k k y k k k k k k
]
Output2 = [
 y k k k k k k k k k
 y k k k k k k k k k
 y k k k k k k k k k
 y k k k k k k k k k
 y k k k k k k k k k
 y k k k k k k k k k
 y y y y y y y y y y
 y k k k k k k k k k
 y k k k k k k k k k
 y k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k p k k e
 k k k k k k p k k e
 k k k k k k p k k e
 k k k k k k p k k k
 p p p p p p p p p p
 k k k k k k p k k k
 k k k k k k p k k k
 k k k k k k p k k k
 k k k k k k p k k k
 k k k k k k p k k k
]
Output3 = [
 k k k p k k k k k k
 k k k p k k k k k k
 k k k p k k k k k k
 k k k p k k k k k k
 k k k p k k k k k k
 k k k p k k k k k k
 k k k p k k k k k k
 p p p p p p p p p p
 k k k p k k k k k k
 k k k p k k k k k k
] """

# concepts:
# counting

# description:
# In the input, you will see a 10x10 black grid. The grid has one row and one column colored solidly in a single color, forming a cross of that color across the grid. The grid also has H grey pixels in the rightmost column at the top of the grid.
# To create the output, convert the H grey pixels to be black. Then shift the "cross" H pixels to the left and H pixels down.

def transform_e48d4e1a(input_grid):
    # 1. Calculate the shift amount: the number of grey pixels in the rightmost column
    H = np.sum(input_grid[-1, :] == Color.GREY)

    # 2. find the row and column of the cross, and its color.
    color = next(c for c in np.unique(input_grid) if c not in [Color.BLACK, Color.GREY])

    # find which column is fully colored with the color
    x = np.where(np.all(input_grid == color, axis=1))[0][0]
    # find which row is fully colored with the color
    y = np.where(np.all(input_grid == color, axis=0))[0][0]

    # shift the column and row down/left by H
    x -= H
    y += H

    # create the output grid by recreating the cross at the new location
    output_grid = np.full(input_grid.shape, Color.BLACK)
    output_grid[x, :] = color
    output_grid[:, y] = color

    return output_grid

""" ==============================
Puzzle e509e548

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k k k g g k k
 k k k k k k k k k k k k k k k k k k g k k
 k k k k k k k k k k k k k k k k k k g k k
 k k k g k k g k k k k k k k k k k k k k k
 k k k g g g g k k k k k g g g k k k k k k
 k k k g k k k k k k k k g k k k k k k k k
 k k k k k k k k k k k k g k k k k k k k k
 k k k k k k k k k k k k g g g k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k g g k k k k k k k k k k k k k k k k k k
 k g k k k k k k k k k k k k k k g g k k k
 k g k k k k k k k k k k k k k k k g k k k
 k g k k k k k k k k k k k k k k g g k k k
 g g g g k k g g g g g k k k k k k k k k k
 k k k k k k g k k k k k k k k k k k k k k
 k k k k k k g k k k k k k k k g g g k k k
 k k k k k k k k k k k k k k k g k g k k k
 k k g k k k k k k k k k k k k g k g k k k
 k g g k k k k k k k k k k k k k k g k k k
]
Output1 = [
 k k k k k k k k k k k k k k k k k b b k k
 k k k k k k k k k k k k k k k k k k b k k
 k k k k k k k k k k k k k k k k k k b k k
 k k k r k k r k k k k k k k k k k k k k k
 k k k r r r r k k k k k p p p k k k k k k
 k k k r k k k k k k k k p k k k k k k k k
 k k k k k k k k k k k k p k k k k k k k k
 k k k k k k k k k k k k p p p k k k k k k
 k k k k k k k k k k k k k k k k k k k k k
 k r r k k k k k k k k k k k k k k k k k k
 k r k k k k k k k k k k k k k k p p k k k
 k r k k k k k k k k k k k k k k k p k k k
 k r k k k k k k k k k k k k k k p p k k k
 r r r r k k b b b b b k k k k k k k k k k
 k k k k k k b k k k k k k k k k k k k k k
 k k k k k k b k k k k k k k k p p p k k k
 k k k k k k k k k k k k k k k p k p k k k
 k k b k k k k k k k k k k k k p k p k k k
 k b b k k k k k k k k k k k k k k p k k k
]

Train example 2:
Input2 = [
 g g g k k k k k k k k
 k k g k k k g g g g k
 k k g k k k g k k g k
 k k k k k k g k k g k
 k k k k k k g k k g k
 k k k k k k k k k k k
 g g g g g k k k k k k
 k k g k k k k k k k k
 k k g k k k k k k k k
 k k g g g k k k k k k
]
Output2 = [
 b b b k k k k k k k k
 k k b k k k p p p p k
 k k b k k k p k k p k
 k k k k k k p k k p k
 k k k k k k p k k p k
 k k k k k k k k k k k
 r r r r r k k k k k k
 k k r k k k k k k k k
 k k r k k k k k k k k
 k k r r r k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k
 k g k k k k k k k g k k
 k g k k k k k k k g g g
 k g k k k g k k k k k k
 k g g g g g k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k g k k k k
 k k k k k g g g k k k k
 k k k k k g k g k k k k
 k k k k k g k g k k k k
 g g g k k k k k k k k k
 k k g k k k k k k g k g
 k k k k k k k k k g g g
]
Output3 = [
 k k k k k k k k k k k k
 k p k k k k k k k b k k
 k p k k k k k k k b b b
 k p k k k p k k k k k k
 k p p p p p k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k r k k k k
 k k k k k r r r k k k k
 k k k k k r k r k k k k
 k k k k k r k r k k k k
 b b b k k k k k k k k k
 k k b k k k k k k p k p
 k k k k k k k k k p p p
] """

# concepts:
# topology, counting

# description:
# In the input you will see a grid with several green objects with different number of bends.
# To make the output, color the objects with one bend with blue, two bends with pink, three bends with red.

def transform_e509e548(input_grid):
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)

    # Find all the green objects in the grid
    object_color = Color.GREEN
    background = Color.BLACK
    objects = find_connected_components(input_grid, monochromatic=True, connectivity=4, background=background)
    for obj in objects:
        # Get the bounding box of the sprite and crop the sprite
        x, y, w, h = bounding_box(obj, background=background)
        sprite = crop(obj, background=Color.BLACK)
        mask = sprite != background

        # Determine how many bends the mask has
        # Do this by building an L-shaped mask, and seeing how many times it appears in the sprite
        bend_mask = np.array([[1, 1], [1, 0]])
        rotated_bend_masks = [bend_mask, np.rot90(bend_mask), np.rot90(bend_mask, 2), np.rot90(bend_mask, 3)]

        
        n_bends = sum( np.sum( correlate(mask*1, filter*1, mode='constant', cval=0) == np.sum(filter*1) ) for filter in rotated_bend_masks )

        # find the new color based on bends
        new_color = {1: Color.BLUE, 2: Color.PINK, 3: Color.RED}[n_bends]
        # color the sprite with the new color
        sprite[sprite == object_color] = new_color
        blit_sprite(output_grid, sprite=sprite, x=x, y=y)

    return output_grid

""" ==============================
Puzzle e73095fd

Train example 1:
Input1 = [
 k k k k k k k k k k k k k e k k k e e
 e e k k k e e e e k k k k e k k k e k
 k e e e e e k k e e e e e e e e e e k
 e e k k k e k k e k k k k e k k k e k
 k k k k k e k k e k k k k e k k k e e
 k k k k k e k k e k k e e e e k k k k
 k k k k k e k k e k k e k k e k k k k
 k k k k k e e e e k k e k k e e e e e
 k k k k k k e k k k k e e e e k k k k
 k k k k k k e k k k k k k e k k k k k
 k k k k k k e k k k k k k e k k k k k
 k k k k k k e k k k k k k e k k k k k
]
Output1 = [
 k k k k k k k k k k k k k e k k k e e
 e e k k k e e e e k k k k e k k k e y
 y e e e e e y y e e e e e e e e e e y
 e e k k k e y y e k k k k e k k k e y
 k k k k k e y y e k k k k e k k k e e
 k k k k k e y y e k k e e e e k k k k
 k k k k k e y y e k k e y y e k k k k
 k k k k k e e e e k k e y y e e e e e
 k k k k k k e k k k k e e e e k k k k
 k k k k k k e k k k k k k e k k k k k
 k k k k k k e k k k k k k e k k k k k
 k k k k k k e k k k k k k e k k k k k
]

Train example 2:
Input2 = [
 k k k k k e k k k k k e k k k k
 k k k k k e k k k k k e k k k k
 k k k k k e k k k k k e k k k k
 k k k k k e k k k k k e k k k k
 k k k e e e e e k k k e k k e e
 k k k e k k k e k k k e k k e k
 k k k e k k k e e e e e e e e k
 k k k e k k k e k k k k k k e k
 k k k e e e e e k k k k k k e e
 k k k k k e k k k k k k k k k k
 k k k k k e k k k k k k k k k k
 e e e e e e k k k k k k k k k k
 k k k k k e k k k k k k k k k k
]
Output2 = [
 k k k k k e k k k k k e k k k k
 k k k k k e k k k k k e k k k k
 k k k k k e k k k k k e k k k k
 k k k k k e k k k k k e k k k k
 k k k e e e e e k k k e k k e e
 k k k e y y y e k k k e k k e y
 k k k e y y y e e e e e e e e y
 k k k e y y y e k k k k k k e y
 k k k e e e e e k k k k k k e e
 k k k k k e k k k k k k k k k k
 k k k k k e k k k k k k k k k k
 e e e e e e k k k k k k k k k k
 k k k k k e k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k e k k k k k k k k k k k
 k k k k k e k k k k k e e e e e k
 e e e e e e e e e e e e k k k e k
 k k k k k e k k k k k e k k k e e
 k k k e e e e k k k k e k k k e k
 e e e e k k e k k k k e e e e e k
 k k k e k k e k k k k k e k k k k
 k k k e k k e k k k k k e k k k k
 k k k e e e e k k k k k e k k k k
 k k k k k e k k k k k k e k k k k
 k k k k k e k k k k e e e e k k k
 e e e e e e e e e e e k k e k k k
 k k k k k e k k k k e k k e k k k
 k k k k k e k k k k e e e e k k k
 k k k k k e k k k k k k e k k k k
]
Output3 = [
 k k k k k e k k k k k k k k k k k
 k k k k k e k k k k k e e e e e k
 e e e e e e e e e e e e y y y e k
 k k k k k e k k k k k e y y y e e
 k k k e e e e k k k k e y y y e k
 e e e e y y e k k k k e e e e e k
 k k k e y y e k k k k k e k k k k
 k k k e y y e k k k k k e k k k k
 k k k e e e e k k k k k e k k k k
 k k k k k e k k k k k k e k k k k
 k k k k k e k k k k e e e e k k k
 e e e e e e e e e e e y y e k k k
 k k k k k e k k k k e y y e k k k
 k k k k k e k k k k e e e e k k k
 k k k k k e k k k k k k e k k k k
] """

# concepts:
# filling

# description:
# The input consists of a black grid containing a few hollow grey rectangles. Each rectangle has 1-3 grey horizontal or vertical lines emanating off of it (at most one per side), either travelling to the border or stopping at another rectangle.
# To create the output, fill in the hollow grey rectangles with yellow.

def transform_e73095fd(input_grid):
    # extract objects using grey as background
    objects = find_connected_components(input_grid, background=Color.GREY, connectivity=4, monochromatic=True)

    # create an output grid to store the result
    output_grid = np.full(input_grid.shape, Color.GREY)

    # for each object, fill it in if it is a rectangle
    for obj in objects:
        # to check if the object is a rectangle,
        # we can check if the cropped object is entirely black
        sprite = crop(obj, background=Color.GREY)
        is_rectangle = np.all(sprite == Color.BLACK)

        if is_rectangle:
            # we also need to make sure the rectangle isn't caused from an emanating line.
            # to do so, check for grey pixels around the border of the grey
            # border adjacent to a corner (aka the x/y value is one less than the max)
            border = object_neighbors(obj, background=Color.GREY, connectivity=8)
            # to get the border of the border, make a copy,
            # add yellow where the border is,
            # then find the border of this new object
            obj2 = obj.copy()
            obj2[border] = Color.YELLOW
            border2 = object_neighbors(obj2, background=Color.GREY, connectivity=8)
            x, y, w, h = bounding_box(obj2, background=Color.GREY)

            pixels_to_check = [
                (x2, y2) for x2 in range(obj.shape[0]) for y2 in range(obj.shape[1])
                if (border2[x2, y2]
                    # check if pixel is adjacent to a corner
                    and (x2 in [x, x + w-1] or y2 in [y, y + h-1]))
            ]

            if not any(input_grid[x, y] == Color.GREY for x, y in pixels_to_check):
                # good rectangle!
                # fill in the original object with yellow
                obj[obj == Color.BLACK] = Color.YELLOW

        blit_object(output_grid, obj, background=Color.GREY)

    return output_grid

""" ==============================
Puzzle e8dc4411

Train example 1:
Input1 = [
 t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t
 t t t t k t t t t t t t t t
 t t t k t k t t t t t t t t
 t t t t k t r t t t t t t t
 t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t
]
Output1 = [
 t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t
 t t t t k t t t t t t t t t
 t t t k t k t t t t t t t t
 t t t t k t r t t t t t t t
 t t t t t r t r t t t t t t
 t t t t t t r t r t t t t t
 t t t t t t t r t r t t t t
 t t t t t t t t r t r t t t
 t t t t t t t t t r t r t t
 t t t t t t t t t t r t r t
]

Train example 2:
Input2 = [
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b k b b b b b b b b
 b b b b b k k k b b b b b b b
 b b b b g b k b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
]
Output2 = [
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b b b b b b b b b b
 b b b b b b k b b b b b b b b
 b b b b b k k k b b b b b b b
 b b b b g b k b b b b b b b b
 b b b g g g b b b b b b b b b
 b b g b g b b b b b b b b b b
 b g g g b b b b b b b b b b b
 g b g b b b b b b b b b b b b
 g g b b b b b b b b b b b b b
]

Train example 3:
Input3 = [
 y y y y y y y y y y y y y y y y
 y y y y y y y y y y y y y y y y
 y y y y y y y y y y y y y y y y
 y y y y y y y y y y y y y y y y
 y y y y y y y y y y y t y y y y
 y y y y y y k k y k k y y y y y
 y y y y y y k k y k k y y y y y
 y y y y y y y y k y y y y y y y
 y y y y y y k k y k k y y y y y
 y y y y y y k k y k k y y y y y
 y y y y y y y y y y y y y y y y
 y y y y y y y y y y y y y y y y
]
Output3 = [
 y y y y y y y y y y y t t y t t
 y y y y y y y y y y y t t y t t
 y y y y y y y y y y y y y t y y
 y y y y y y y y y y y t t y t t
 y y y y y y y y y y y t t y t t
 y y y y y y k k y k k y y y y y
 y y y y y y k k y k k y y y y y
 y y y y y y y y k y y y y y y y
 y y y y y y k k y k k y y y y y
 y y y y y y k k y k k y y y y y
 y y y y y y y y y y y y y y y y
 y y y y y y y y y y y y y y y y
] """

# concepts:
# repeated translation, indicator pixels, non-black background

# description:
# In the input you will see a non-black background with a black object and an indicator pixel touching it of a different color.
# To make the output, repeatedly translate the black object in the direction of the indicator pixel and make the color of these repeated translations match the indicator.

def transform_e8dc4411(input_grid):
    # Plan:
    # 1. Parse the input into the black object and the indicator pixel(s)
    # 2. Determine the direction of translation
    # 3. Put down recolored versions of the black object

    # 1. Parse the input
    # background is most common color
    background = max(Color.ALL_COLORS, key=lambda color: np.sum(input_grid == color))
    # REMEMBER: pass background to everything that needs it, because it isn't by default BLACK
    objects = find_connected_components(input_grid, connectivity=8, background=background, monochromatic=True)
    # indicators are single pixels
    indicator_objects = [ obj for obj in objects if crop(obj, background=background).shape == (1,1) ]
    # other objects are bigger than (1,1)
    template_objects = [ obj for obj in objects if crop(obj, background=background).shape != (1,1) ]

    # We draw on top of the output, so copy it
    output_grid = input_grid.copy()

    # Iterate over every template and indicator which are in contact
    for template_obj in template_objects:
        template_sprite = crop(template_obj, background=background)
        for indicator_obj in indicator_objects:
            if not contact(object1=template_obj, object2=indicator_obj, background=background, connectivity=8): continue

            # 2. Determine the direction of translation
            indicator_x, indicator_y = object_position(indicator_obj, background=background, anchor="center")
            template_x, template_y = object_position(template_obj, background=background, anchor="center")

            dx, dy = np.sign(indicator_x - template_x), np.sign(indicator_y - template_y)
            # Figure out the stride the translation, which is as far as we can go while still covering the indicator pixel
            possible_strides = [ stride for stride in range(1, max(output_grid.shape))
                                if collision(object1=indicator_obj, object2=template_obj, x2=stride*dx, y2=stride*dy, background=background) ]
            stride = max(possible_strides)

            # 3. Put down recolored versions of the black object as much as we can until we fall out of the canvas
            # Prepare a new version of the sprite
            new_color = object_colors(indicator_obj, background=background)[0]
            recolored_template_sprite = template_sprite.copy()
            recolored_template_sprite[recolored_template_sprite != background] = new_color

            # Put down the recolored sprite at every stride
            for i in range(1, 10):
                old_x, old_y = object_position(template_obj, background=background, anchor="upper left")
                new_x, new_y = old_x + i*dx*stride, old_y + i*dy*stride
                blit_sprite(output_grid, recolored_template_sprite, new_x, new_y, background=background)
    
    return output_grid

""" ==============================
Puzzle e9614598

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k b k k k k k b k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k g k k k k k
 k b k g g g k b k k
 k k k k g k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k b k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k b k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k b k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k g k k k k k k
 k k g g g k k k k k
 k k k g k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k b k k k k k k
 k k k k k k k k k k
] """

# concepts:
# object detection, pattern drawing

# description:
# In the input you will see two blue pixels
# To make the output grid, you should place a 3x3 green cross pattern between the two blue pixels exactly halfway between them.

def transform_e9614598(input_grid):
    # Detect the two blue pixels on the grid.
    blue_pixels = detect_objects(grid=input_grid, colors=[Color.BLUE], monochromatic=True, connectivity=4)
    first_pixel, second_pixel = blue_pixels[0], blue_pixels[1]

    # Find the midpoint
    first_x, first_y = object_position(first_pixel, background=Color.BLACK, anchor="center")
    second_x, second_y = object_position(second_pixel, background=Color.BLACK, anchor="center")
    mid_x, mid_y = int((first_x + second_x) / 2), int((first_y + second_y) / 2)

    # Generate the 3x3 green cross pattern between the two blue pixels.
    green_cross_sprite = np.array( [[Color.BLACK, Color.GREEN, Color.BLACK],
                                [Color.GREEN, Color.GREEN, Color.GREEN],
                                [Color.BLACK, Color.GREEN, Color.BLACK]])
    output_grid = input_grid.copy()
    green_cross_width, green_cross_height = 3, 3

    # Put the cross centered at the midpoint
    upper_left_x = mid_x - green_cross_width // 2
    upper_left_y = mid_y - green_cross_height // 2
    blit_sprite(output_grid, green_cross_sprite, x=upper_left_x, y=upper_left_y, background=Color.BLACK)

    return output_grid

""" ==============================
Puzzle ea786f4a

Train example 1:
Input1 = [
 b b b
 b k b
 b b b
]
Output1 = [
 k b k
 b k b
 k b k
]

Train example 2:
Input2 = [
 r r r r r
 r r r r r
 r r k r r
 r r r r r
 r r r r r
]
Output2 = [
 k r r r k
 r k r k r
 r r k r r
 r k r k r
 k r r r k
]

Train example 3:
Input3 = [
 g g g g g g g
 g g g g g g g
 g g g g g g g
 g g g k g g g
 g g g g g g g
 g g g g g g g
 g g g g g g g
]
Output3 = [
 k g g g g g k
 g k g g g k g
 g g k g k g g
 g g g k g g g
 g g k g k g g
 g k g g g k g
 k g g g g g k
] """

# concepts:
# non-black background, diagonal lines

# description:
# In the input you will see a grid there's all the same color except for a single black pixel in the middle. Equivalently, a black pixel on a non-black background.
# To make the output, draw black diagonal lines outward from the single black pixel in all 4 diagonal directions.

def transform_ea786f4a(input_grid):
    # Plan:
    # 1. Find the black pixel
    # 2. Draw diagonal lines outward from the black pixel

    # Find the possible locations that are black, then check that there is exactly one
    black_pixel_locations = np.argwhere(input_grid == Color.BLACK)
    assert len(black_pixel_locations) == 1
    black_pixel_location = black_pixel_locations[0]
    black_x, black_y = black_pixel_location

    # We are going to draw on top of the input grid
    output_grid = input_grid.copy()

    # Draw the diagonal lines
    for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        direction = (dx, dy)
        draw_line(output_grid, black_x, black_y, direction=direction, color=Color.BLACK)

    return output_grid

""" ==============================
Puzzle eb281b96

Train example 1:
Input1 = [
 k k t k k k t k k k t k k k t k k
 k t k t k t k t k t k t k t k t k
 t k k k t k k k t k k k t k k k t
]
Output1 = [
 k k t k k k t k k k t k k k t k k
 k t k t k t k t k t k t k t k t k
 t k k k t k k k t k k k t k k k t
 k t k t k t k t k t k t k t k t k
 k k t k k k t k k k t k k k t k k
 k t k t k t k t k t k t k t k t k
 t k k k t k k k t k k k t k k k t
 k t k t k t k t k t k t k t k t k
 k k t k k k t k k k t k k k t k k
]

Train example 2:
Input2 = [
 k k r k k k r k k k r k k k r k k
 k r k r k r k r k r k r k r k r k
 k r k r k r k r k r k r k r k r k
 r k k k r k k k r k k k r k k k r
]
Output2 = [
 k k r k k k r k k k r k k k r k k
 k r k r k r k r k r k r k r k r k
 k r k r k r k r k r k r k r k r k
 r k k k r k k k r k k k r k k k r
 k r k r k r k r k r k r k r k r k
 k r k r k r k r k r k r k r k r k
 k k r k k k r k k k r k k k r k k
 k r k r k r k r k r k r k r k r k
 k r k r k r k r k r k r k r k r k
 r k k k r k k k r k k k r k k k r
 k r k r k r k r k r k r k r k r k
 k r k r k r k r k r k r k r k r k
 k k r k k k r k k k r k k k r k k
] """

# concepts:
# pattern generation, flipping

# description:
# In the input you will see a grid with wave patterns.
# To make the output, you should flip the input grid horizontally and place it in the output grid.
# And then place the flipped input grid in the output grid again.

def transform_eb281b96(input_grid):
    # Plan:
    # 1. Create output grid
    # 2. Flip the input grid horizontally and put it in the output repeatedly

    # 1. Create output grid
    input_width, input_height = input_grid.shape
    # The output grid is placing and flipping the original pattern 4 times
    output_width, output_height = input_width, (input_height - 1) * 4 + 1
    output_grid = np.full((output_width, output_height), Color.BLACK)

    # 2. Make the output by flipping the input and ultimately putting 4 copies (some flipped) into the output
    # Flip the input grid horizontally
    flipped_input = np.fliplr(input_grid)

    # Place the input and flipped input in the output grid
    blit_sprite(output_grid, input_grid, x=0, y=0)
    blit_sprite(output_grid, flipped_input, x=0, y=input_height - 1)
    blit_sprite(output_grid, input_grid, x=0, y=2 * (input_height - 1))
    blit_sprite(output_grid, flipped_input, x=0, y=3 * (input_height - 1))

    return output_grid

""" ==============================
Puzzle eb5a1d5d

Train example 1:
Input1 = [
 t t t t t t t t t t t t t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t t t t t t t t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t g g g r r r r r r g g g g g g g g g t t t t t t
 t t t g g g r r r r r r g g g g g g g g g t t t t t t
 t t t g g g r r r r r r g g g g g g g g g t t t t t t
 t t t g g g r r r r r r g g g g g g g g g t t t t t t
 t t t g g g r r r r r r g g g g g g g g g t t t t t t
 t t t g g g r r r r r r g g g g g g g g g t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t g g g g g g g g g g g g g g g g g g t t t t t t
 t t t t t t t t t t t t t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t t t t t t t t t t t t t
 t t t t t t t t t t t t t t t t t t t t t t t t t t t
]
Output1 = [
 t t t t t
 t g g g t
 t g r g t
 t g g g t
 t t t t t
]

Train example 2:
Input2 = [
 e e e e e e e e e e e e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e e e e e e e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e p p p p p p p p p p p p p p p e e e e e e
 e e e e e e e e e e e e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e e e e e e e e e e e e
 e e e e e e e e e e e e e e e e e e e e e e e e e
]
Output2 = [
 e e e
 e p e
 e e e
]

Train example 3:
Input3 = [
 g g g g g g g g g g g g g g g g g g g g g g
 g g g g g g g g g g g g g g g g g g g g g g
 g g g g g g g g g g g g g g g g g g g g g g
 g g g t t t t t t t t t t t t t t g g g g g
 g g g t t t t t t t t t t t t t t g g g g g
 g g g t t r r r r r r r r r t t t g g g g g
 g g g t t r r r b b b b b r t t t g g g g g
 g g g t t r r r b b b b b r t t t g g g g g
 g g g t t r r r b b b b b r t t t g g g g g
 g g g t t r r r r r r r r r t t t g g g g g
 g g g t t r r r r r r r r r t t t g g g g g
 g g g t t r r r r r r r r r t t t g g g g g
 g g g t t r r r r r r r r r t t t g g g g g
 g g g t t r r r r r r r r r t t t g g g g g
 g g g t t t t t t t t t t t t t t g g g g g
 g g g t t t t t t t t t t t t t t g g g g g
 g g g t t t t t t t t t t t t t t g g g g g
 g g g g g g g g g g g g g g g g g g g g g g
 g g g g g g g g g g g g g g g g g g g g g g
 g g g g g g g g g g g g g g g g g g g g g g
 g g g g g g g g g g g g g g g g g g g g g g
]
Output3 = [
 g g g g g g g
 g t t t t t g
 g t r r r t g
 g t r b r t g
 g t r r r t g
 g t t t t t g
 g g g g g g g
] """

# concepts:
# downscaling, nesting

# description:
# In the input you will see a grid consisting of nested shapes of different colors.
# To make the output, make a grid with one pixel for each layer of the shapes, 
# with the color from outermost layer to the innermost layer in the same order they appear in the input.

def transform_eb5a1d5d(input_grid):
    # Plan:
    # 1. Parse the input into objects and order them from outermost to innermost by area
    # 2. Draw nested rectangles with the colors of the input objects, each layer has only one pixel length

    # 1. input parsing
    # Find the objects in the input grid
    objects = find_connected_components(input_grid, connectivity=4, monochromatic=True, background=Color.BLACK)

    # Sort the objects from outermost to innermost, using area to determine the order
    objects.sort(key=lambda obj: crop(obj).shape[0] * crop(obj).shape[1], reverse=True)

    # 2. drawing the output
    # Leave only one layer of each color shape
    grid_len = len(objects) * 2 - 1
    output_grid = np.full((grid_len, grid_len), Color.BLACK)

    # Calculate each layer's length, which starts at the outermost layer
    current_len = grid_len

    # Draw nested shapes with the colors of the input objects. Each layer has only one pixel length of the current color
    for i, object in enumerate(objects):
        # Get the color of the current layer
        color = object_colors(object)[0]
        # Fill the region with the current color
        cur_shape = np.full((current_len, current_len), color)
        # Place the current shape in the output grid
        # Make sure each layer has only one pixel length of the current color
        blit_sprite(output_grid, sprite=cur_shape, x=i, y=i)
        current_len -= 2

    return output_grid

""" ==============================
Puzzle f15e1fac

Train example 1:
Input1 = [
 k t k k k t k t k t k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 r k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 r k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
]
Output1 = [
 k t k k k t k t k t k k
 k t k k k t k t k t k k
 k t k k k t k t k t k k
 k t k k k t k t k t k k
 r k t k k k t k t k t k
 k k t k k k t k t k t k
 k k t k k k t k t k t k
 k k t k k k t k t k t k
 k k t k k k t k t k t k
 k k t k k k t k t k t k
 r k k t k k k t k t k t
 k k k t k k k t k t k t
 k k k t k k k t k t k t
 k k k t k k k t k t k t
 k k k t k k k t k t k t
 k k k t k k k t k t k t
 k k k t k k k t k t k t
]

Train example 2:
Input2 = [
 k k t k k k t k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k r
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k r
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k r
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k t k k k t k k k
 k k t k k k t k k k
 k k t k k k t k k k
 k t k k k t k k k r
 k t k k k t k k k k
 k t k k k t k k k k
 k t k k k t k k k k
 t k k k t k k k k r
 t k k k t k k k k k
 t k k k t k k k k k
 t k k k t k k k k k
 k k k t k k k k k r
 k k k t k k k k k k
 k k k t k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 t k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 t k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k k k k k
 t k k k k k k k k k k k
 k k k k k k k k k k k k
 k k k k r k k k r k k k
]
Output3 = [
 k k k k k k k k t t t t
 k k k k t t t t k k k k
 t t t t k k k k k k k k
 k k k k k k k k k k k k
 k k k k k k k k t t t t
 k k k k t t t t k k k k
 t t t t k k k k k k k k
 k k k k k k k k t t t t
 k k k k t t t t k k k k
 t t t t k k k k k k k k
 k k k k k k k k k k k k
 k k k k r k k k r k k k
] """

# concepts:
# magnetism, direction, lines

# description:
# In the input, you will see a black grid with teal pixels scattered along one edge and red pixels scattered along an edge perpendicular to the teal one.
# To make the output, make the teal pixels flow from the edge they are on to the opposite edge. Whenever there is a red pixel in the same column or row as the flow of teal pixels, push the teal pixel's flow one pixel away from the red pixel.

def transform_f15e1fac(input_grid):
    # copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # figure out which edges the red and teal pixels are on and decide the direction of flow and push based on that
    top = output_grid[:,0]
    bottom = output_grid[:, -1]
    left = output_grid[0, :]
    if Color.RED in top:
        # push the teal pixels down
        push = (0, 1)

        # teal can only be on the left or right edge if red is on the top edge
        if Color.TEAL in left:
            flow = (1, 0) # flow right
        else:
            flow = (-1, 0) # flow left    
    elif Color.RED in bottom:
        # push the teal pixels up
        push = (0, -1)

        # teal can only be on the left or right edge if red is on the bottom edge
        if Color.TEAL in left:
            flow = (1, 0) # flow right
        else:
            flow = (-1, 0) # flow left
    elif Color.RED in left:
        # push the teal pixels to the right
        push = (1, 0)

        # teal can only be on the top or bottom edge if red is on the left edge
        if Color.TEAL in top:
            flow = (0, 1) # flow down
        else:
            flow = (0, -1) # flow up
    else: # red is on the right edge
        # push the teal pixels to the left
        push = (-1, 0)
        
        # teal can only be on the top or bottom edge if red is on the right edge
        if Color.TEAL in top:
            flow = (0, 1) # flow down
        else:
            flow = (0, -1) # flow up

    # find the coordinates of the teal and red pixels
    teal = np.where(input_grid == Color.TEAL)
    red = np.where(input_grid == Color.RED)

    # draw the flow of teal pixels
    for i in range(len(teal[0])):
        # start at a teal pixel
        x, y = teal[0][i], teal[1][i]

        # draw across the grid one pixel at a time adjusting for red pixel effects
        while x >= 0 and x < output_grid.shape[0] and y >= 0 and y < output_grid.shape[1]:
            # push the teal pixel away from the red pixel if it is in the same row or column
            if x in red[0] or y in red[1]:
                x += push[0]
                y += push[1]

                # stop this flow if it goes off the grid
                if x < 0 or x >= output_grid.shape[0] or y < 0 or y >= output_grid.shape[1]:
                    break
                
            # draw a teal pixel in the flow
            output_grid[x, y] = Color.TEAL

            # move the flow one pixel in the direction of flow
            x += flow[0]
            y += flow[1]

    return output_grid

""" ==============================
Puzzle f8a8fe49

Train example 1:
Input1 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k r r r r r r r k k k k
 k k k k r k k k k k r k k k k
 k k k k k k e e e k k k k k k
 k k k k k k e k e k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k e k k k k k k k
 k k k k r k k k k k r k k k k
 k k k k r r r r r r r k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k k k k k k
 k k k k k k e k e k k k k k k
 k k k k k k e e e k k k k k k
 k k k k k k k k k k k k k k k
 k k k k r r r r r r r k k k k
 k k k k r k k k k k r k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k r k k k k k r k k k k
 k k k k r r r r r r r k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k e k k k k k k k
 k k k k k k k k k k k k k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r r k k k k r r k k k k
 k k k r k k k k k k r k k k k
 k k k r k e k k e k r k k k k
 k k k r k k e k e k r k k k k
 k k k r k k e k e k r k k k k
 k k k r k e k k e k r k k k k
 k k k r k k k k k k r k k k k
 k k k r r k k k k r r k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r r k k k k r r k k k k
 k k k r k k k k k k r k k k k
 k e k r k k k k k k r k e k k
 e k k r k k k k k k r k e k k
 e k k r k k k k k k r k e k k
 k e k r k k k k k k r k e k k
 k k k r k k k k k k r k k k k
 k k k r r k k k k r r k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r r k k k k r r k k k k
 k k k r k e k k e k r k k k k
 k k k r k e e k e k r k k k k
 k k k r k k e k e k r k k k k
 k k k r r k k k k r r k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k r r k k k k r r k k k k
 k e k r k k k k k k r k e k k
 e e k r k k k k k k r k e k k
 e k k r k k k k k k r k e k k
 k k k r r k k k k r r k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
 k k k k k k k k k k k k k k k
] """

# concepts:
# symmetry, mirror

# description:
# In the input you will see two patterns close by two frames.
# To make the output, you need to split the inner pattern into two parts by the symmetry of the framework. 
# Then, mirror the two parts according to each frame's line and place them in the output grid.

def transform_f8a8fe49(input_grid):
    # Extract the framework
    frame_color = Color.RED

    # Create an empty grid
    n, m = input_grid.shape
    output_grid = np.zeros((n, m), dtype=int)

    # Get two frame that form the framework
    frames = find_connected_components(grid=input_grid, connectivity=4)
    frames = [frame for frame in frames if np.any(frame == frame_color)]
    framework = []
    for frame in frames:
        # Get the frame and its position
        x, y, w, h = bounding_box(grid=frame)
        cropped_frame = crop(grid=frame)
        framework.append({"x": x, "y": y, "w": w, "h": h, "frame": cropped_frame})
        # Place the frame in the output grid
        output_grid = blit_sprite(grid=output_grid, sprite=cropped_frame, x=x, y=y)

    # Sort the framework by position
    framework = sorted(framework, key=lambda x: x["x"])
    framework = sorted(framework, key=lambda x: x["y"])

    # Get the inner pattern
    x_whole, y_whole, w_whole, h_whole = bounding_box(grid=input_grid)
    inner_pattern = input_grid[x_whole + 1 : x_whole + w_whole - 1, y_whole + 1 : y_whole + h_whole - 1]

    # Check if the framework is horizontal or vertical
    if_horizontal = framework[0]["w"] > framework[0]["h"]

    if if_horizontal:
        # Split the inner pattern into two parts by the symmetry of the framework
        pattern_len = inner_pattern.shape[1] // 2
        pattern_part1 = inner_pattern[:, : pattern_len]
        pattern_part2 = inner_pattern[:, pattern_len:]

        # Mirror the two inner patterns according to each frame's line
        pattern_part1 = np.fliplr(pattern_part1)
        pattern_part2 = np.fliplr(pattern_part2)

        # Place the two inner patterns in the output grid
        output_grid = blit_sprite(grid=output_grid, sprite=pattern_part1, x=framework[0]['x'] + 1, y=framework[0]['y'] - pattern_len)
        output_grid = blit_sprite(grid=output_grid, sprite=pattern_part2, x=framework[1]['x'] + 1, y=framework[1]['y'] + 2)
    else:
        # Split the inner pattern into two parts by the symmetry of the framework
        pattern_len = inner_pattern.shape[0] // 2
        pattern_part1 = inner_pattern[: pattern_len, :]
        pattern_part2 = inner_pattern[pattern_len:, :]

        # Mirror the two inner patterns according to each frame's line
        pattern_part1 = np.flipud(pattern_part1)
        pattern_part2 = np.flipud(pattern_part2)

        # Place the two inner patterns in the output grid
        output_grid = blit_sprite(grid=output_grid, sprite=pattern_part1, x=framework[0]['x'] - pattern_len, y=framework[0]['y'] + 1)
        output_grid = blit_sprite(grid=output_grid, sprite=pattern_part2, x=framework[1]['x'] + 2, y=framework[1]['y'] + 1)

    return output_grid

""" ==============================
Puzzle f8b3ba0a

Train example 1:
Input1 = [
 k k k k k k k k k k k k k
 k g g k b b k b b k b b k
 k k k k k k k k k k k k k
 k b b k b b k y y k y y k
 k k k k k k k k k k k k k
 k b b k y y k b b k b b k
 k k k k k k k k k k k k k
 k r r k b b k b b k b b k
 k k k k k k k k k k k k k
 k b b k r r k b b k b b k
 k k k k k k k k k k k k k
 k b b k b b k b b k b b k
 k k k k k k k k k k k k k
]
Output1 = [
 y
 r
 g
]

Train example 2:
Input2 = [
 k k k k k k k k k k k k k k k k
 k p p k t t k t t k t t k t t k
 k k k k k k k k k k k k k k k k
 k t t k t t k r r k p p k t t k
 k k k k k k k k k k k k k k k k
 k b b k t t k b b k t t k t t k
 k k k k k k k k k k k k k k k k
 k t t k b b k t t k t t k t t k
 k k k k k k k k k k k k k k k k
 k t t k t t k p p k t t k p p k
 k k k k k k k k k k k k k k k k
 k t t k t t k t t k t t k t t k
 k k k k k k k k k k k k k k k k
]
Output2 = [
 p
 b
 r
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k g g k g g k g g k
 k k k k k k k k k k
 k b b k g g k g g k
 k k k k k k k k k k
 k g g k t t k g g k
 k k k k k k k k k k
 k g g k t t k g g k
 k k k k k k k k k k
 k g g k r r k r r k
 k k k k k k k k k k
 k r r k g g k g g k
 k k k k k k k k k k
]
Output3 = [
 r
 t
 b
] """

# concepts:
# rectangular cells, counting

# description:
# In the input you will see monochromatic rectangles arranged into an array.
# To make the output, find the most common color of the 2x1 rectangles, second most common color, etc.
# Then, make the output a vertical stripe colored with the second common color, then the third most common color, etc., starting at the top, and skipping the most common color.

def transform_f8b3ba0a(input_grid):
    # Plan:
    # 1. Extract the objects, arranged into the array
    # 2. Count the colors
    # 3. Sort the colors by count
    # 4. Create the output grid, remembering to skip the most common color

    background = Color.BLACK
    
    objects = find_connected_components(input_grid, monochromatic=True, connectivity=4, background=background)
    possible_x_values = [ object_position(obj, background=background, anchor="upper left")[0] for obj in objects ]
    possible_y_values = [ object_position(obj, background=background, anchor="upper left")[1] for obj in objects ]
    object_array = [ [ next(obj for obj in objects if (x, y) == object_position(obj, background=background, anchor="upper left") )
                      for y in sorted(set(possible_y_values)) ]
                    for x in sorted(set(possible_x_values)) ]

    # Extract and count the colors
    object_colors = [ obj[obj!=background][0] for obj in objects ]
    color_counts = { color: sum(1 for object_color in object_colors if object_color == color) for color in set(object_colors) }

    sorted_colors = list(sorted(color_counts, key=lambda color: color_counts[color], reverse=True))
    # skip the most common color
    sorted_colors = sorted_colors[1:]

    # the output is a vertical stripe containing one pixel per color
    output_grid = np.full((1, len(sorted_colors)), background)
    for y, color in enumerate(sorted_colors):
        output_grid[0, y] = color

    return output_grid

""" ==============================
Puzzle f9012d9b

Train example 1:
Input1 = [
 r b r b r
 b b b b b
 r b r b r
 k k b b b
 k k r b r
]
Output1 = [
 b b
 r b
]

Train example 2:
Input2 = [
 t p k p
 p t p t
 t p t p
 p t p t
]
Output2 = [
 t
]

Train example 3:
Input3 = [
 r r e r r e r
 r r e r r e r
 e e e e e e e
 r r e r r e r
 r r e r r e r
 e e e e e k k
 r r e r r k k
]
Output3 = [
 e e
 e r
] """

# concepts:
# symmetry detection, occlusion

# description:
# In the input you will see a translationally symmetric pattern that has been partially occluded by a black rectangle
# The output should be what the black rectangle should be in order to make it perfectly symmetric.
# In other words, the output should be the missing part of the pattern, and it should be the same dimensions as the black rectangle.

def transform_f9012d9b(input_grid):
    # Plan:
    # 1. Find the black rectangle
    # 2. Find the symmetry
    # 3. Fill in the missing part
    # 4. Extract the part that you filled in, which is the final output

    # Find the black rectangle and save where it is
    # Do this first because we will need to know where it was after we fill it in
    occlusion_color = Color.BLACK
    black_rectangle_mask = (input_grid == occlusion_color)

    # Find the symmetry. Notice that black is not the background, but instead is the occlusion color. In fact there is no well-defined background color.
    symmetries = detect_translational_symmetry(input_grid, ignore_colors=[occlusion_color], background=None)

    # Fill in the missing part
    for occluded_x, occluded_y in np.argwhere(black_rectangle_mask):
        for symmetric_x, symmetric_y in orbit(input_grid, occluded_x, occluded_y, symmetries):
            if input_grid[symmetric_x, symmetric_y] != occlusion_color:
                input_grid[occluded_x, occluded_y] = input_grid[symmetric_x, symmetric_y]
                break
    
    # Extract the region that we filled in, ultimately as a 2D sprite
    # first, get just the part of the final image which corresponds to what used to be included
    filled_in_region = np.full_like(input_grid, occlusion_color)
    filled_in_region[black_rectangle_mask] = input_grid[black_rectangle_mask]
    # then, crop it to obtain the sprite
    filled_in_region = crop(filled_in_region, background=occlusion_color)

    return filled_in_region

""" ==============================
Puzzle fcc82909

Train example 1:
Input1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k m m k k k k k k k
 k p p k k k k k k k
 k k k k k k k k k k
 k k k k k t y k k k
 k k k k k o o k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output1 = [
 k k k k k k k k k k
 k k k k k k k k k k
 k m m k k k k k k k
 k p p k k k k k k k
 k g g k k k k k k k
 k g g k k t y k k k
 k k k k k o o k k k
 k k k k k g g k k k
 k k k k k g g k k k
 k k k k k g g k k k
]

Train example 2:
Input2 = [
 k k k k k k k k k k
 k k y t k k k k k k
 k k m y k k k k k k
 k k k k k k k k k k
 k k k k k k r b k k
 k k k k k k b r k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output2 = [
 k k k k k k k k k k
 k k y t k k k k k k
 k k m y k k k k k k
 k k g g k k k k k k
 k k g g k k r b k k
 k k g g k k b r k k
 k k k k k k g g k k
 k k k k k k g g k k
 k k k k k k k k k k
 k k k k k k k k k k
]

Train example 3:
Input3 = [
 k k k k k k k k k k
 k k r y k k m t k k
 k k p o k k t m k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k o p k k k k
 k k k k p p k k k k
 k k k k k k k k k k
 k k k k k k k k k k
 k k k k k k k k k k
]
Output3 = [
 k k k k k k k k k k
 k k r y k k m t k k
 k k p o k k t m k k
 k k g g k k g g k k
 k k g g k k g g k k
 k k g g o p k k k k
 k k g g p p k k k k
 k k k k g g k k k k
 k k k k g g k k k k
 k k k k k k k k k k
] """

# concepts:
# counting

# description:
# In the input, you will see a 10x10 black grid containing a few 2x2 squares. Each square contains 1-4 colors.
# To create the output, draw a green 2xL rectangle just below each 2x2 square, where L is the number of colors in the square.

def transform_fcc82909(input_grid):
    # copy the input grid to the output grid
    output_grid = input_grid.copy()
    # find all connected components in the input grid
    objects = find_connected_components(input_grid, connectivity=4, monochromatic=False)

    # for each object, draw a green rectangle below it of shape 2xL, where L is the number of colors in the square
    for obj in objects:
        obj_x, obj_y, obj_width, obj_height = bounding_box(obj, background=Color.BLACK)
        num_colors = len(np.unique(obj[obj != Color.BLACK]))
        green_rectangle = np.full((obj_width, num_colors), Color.GREEN)
        blit_sprite(output_grid, green_rectangle, obj_x, obj_y + obj_height, background=Color.BLACK)

    return output_grid

""" ==============================
Puzzle feca6190

Train example 1:
Input1 = [
 b k o k k
]
Output1 = [
 k k k k k k k k k b
 k k k k k k k k b k
 k k k k k k k b k o
 k k k k k k b k o k
 k k k k k b k o k k
 k k k k b k o k k k
 k k k b k o k k k k
 k k b k o k k k k k
 k b k o k k k k k k
 b k o k k k k k k k
]

Train example 2:
Input2 = [
 k k r k k
]
Output2 = [
 k k k k k
 k k k k k
 k k k k r
 k k k r k
 k k r k k
]

Train example 3:
Input3 = [
 y k p k t
]
Output3 = [
 k k k k k k k k k k k k k k y
 k k k k k k k k k k k k k y k
 k k k k k k k k k k k k y k p
 k k k k k k k k k k k y k p k
 k k k k k k k k k k y k p k t
 k k k k k k k k k y k p k t k
 k k k k k k k k y k p k t k k
 k k k k k k k y k p k t k k k
 k k k k k k y k p k t k k k k
 k k k k k y k p k t k k k k k
 k k k k y k p k t k k k k k k
 k k k y k p k t k k k k k k k
 k k y k p k t k k k k k k k k
 k y k p k t k k k k k k k k k
 y k p k t k k k k k k k k k k
] """

# concepts:
# pixel patterns, counting, expanding, diagonal lines

# description:
# In the input you will see a line with several colored pixels.
# To make the output, create a square grid and place the input line at the bottom-left. Each colored pixel shoots a diagonal line outward toward the upper right.
# The length of the output grid is the product of the number of colored input pixels and the length of the input line.

def transform_feca6190(input_grid):
    # Plan:
    # 1. Figure out how big the output should be and make a blank output grid
    # 2. Place the input line at the bottom left of the output grid
    # 3. Repeatedly translate it diagonally toward the upper right corner, equivalently shooting diagonal lines from each colored pixel

    # 1. The output grid size is the number of non-black pixels in the input grid times the original grid width
    input_width, input_height = input_grid.shape
    num_different_colors = sum(color in input_grid.flatten() for color in Color.NOT_BLACK )
    output_size = input_width * num_different_colors
    output_grid = np.full((output_size, output_size), Color.BLACK)

    # 2-3. Place the input at the bottom left and then move it upward and rightward
    bottommost_y = output_size - 1
    for iteration in range(output_size):
        blit_sprite(output_grid, input_grid, x=iteration, y=bottommost_y - iteration*input_height)

    return output_grid

""" ==============================
Puzzle ff28f65a

Train example 1:
Input1 = [
 r r k k k
 r r k k k
 k k k k k
 k k k k k
 k k k k k
]
Output1 = [
 b k k
 k k k
 k k k
]

Train example 2:
Input2 = [
 k k k k k
 k r r k k
 k r r k k
 k k k r r
 k k k r r
]
Output2 = [
 b k b
 k k k
 k k k
]

Train example 3:
Input3 = [
 k k k k k k k
 k r r k k k k
 k r r k r r k
 k k k k r r k
 k k r r k k k
 k k r r k k k
 k k k k k k k
]
Output3 = [
 b k b
 k b k
 k k k
] """

# concepts:
# counting, object detection, alternating pattern

# description:
# In the input you will see several 2 x 2 red squares on the grid.
# To make the output grid, you should count the number of red squares
# Then place the same number of 1 x 1 blue squares on the output grid following this pattern in the output:
# First fill the top row, then the next row, but skip every other column. Begin the first/third/fifth/etc row in the first column, but begin the second/forth/etc row in the second column.

def transform_ff28f65a(input_grid):
    # Detect all the 2 x 2 red squares on the grid.
    red_square = detect_objects(grid=input_grid, colors=[Color.RED], monochromatic=True, connectivity=4)

    # Count the number of 2 x 2 red squares.
    num_red_square = len(red_square)

    # Output grid is always 3 x 3.
    output_grid = np.zeros((3, 3), dtype=int)

    # Fill the output grid with red square number follow specific pattern sequence:
    # 1. Fill the top row, then the next row, but skip every other column.
    # 2. Begin the first/third/fifth/etc row in the first column, but begin the second/forth/etc row in the second column.
    pos_list = []
    for i in range(9):
        if i % 2 == 0:
            pos_list.append((i % 3, i // 3))

    # Place the same number of 1 x 1 blue squares on the output grid follow the specific pattern sequence.
    for i in range(num_red_square):
        x, y = pos_list[i]
        output_grid[x, y] = Color.BLUE
        
    return  output_grid
