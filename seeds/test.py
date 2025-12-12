from typing import Tuple, List, Optional
import numpy as np
import random
from dataclasses import dataclass
from seeds.common import *



color_names = [
    "Black",
    "Blue",
    "Red",
    "Green",
    "Yellow",
    "Grey",
    "Pink",
    "Orange",
    "Teal",
    "Maroon",
]

def show_grid(grid, name=None):
    if name: print(name)# + ' = [')
    for row in grid:
        print(' '.join(color_names[c] for c in row))
    if name: print()#']')

def text2grid(text):
    return [[color_names.index(color.replace('Gray', 'Grey')) for color in row.split()] for row in text.split('\n')]

def count_elements(puzzle, section='train', max_pairs=None):
    element_count = 0
    max_pairs = max_pairs if max_pairs is not None else len(puzzle.get(section, []))
    for pair in puzzle.get(section, [])[:max_pairs]:
        for key in ('input', 'output'):
            grid = pair.get(key, [])
            element_count += sum(len(row) for row in grid)
    return element_count

# enrich all_tasks
def _compute_separator_masks(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = arr.shape
    # A separator row/col is uniform across the entire dimension and non-black (color != BLACK)
    row_uniform = (arr == arr[:, [0]]).all(axis=1) if w > 0 else np.zeros(h, dtype=bool)
    col_uniform = (arr == arr[[0], :]).all(axis=0) if h > 0 else np.zeros(w, dtype=bool)
    row_uniform &= (arr[:, 0] != Color.BLACK)
    col_uniform &= (arr[0, :] != Color.BLACK)
    return row_uniform, col_uniform

def _count_blocks(nonsep_mask: np.ndarray) -> int:
    # Count contiguous True runs (content blocks). Expects boolean mask. Minimum is 1.
    m = nonsep_mask
    starts = m & np.concatenate(([True], ~m[:-1]))
    blocks = int(starts.sum())
    return max(blocks, 1)

def analyze_grid(grid) -> Tuple[Tuple[int, int], list, int, float]:
    try:
        arr = np.array(grid)
        if arr.ndim != 2 or min(arr.shape) == 0 or arr.shape[0] == 1 or arr.shape[1] == 1:
            return (1, 1), [], None, 0.0, []
        row_mask, col_mask = _compute_separator_masks(arr)
        num_rows = _count_blocks(~row_mask)
        num_cols = _count_blocks(~col_mask)
        row_idxs = np.flatnonzero(row_mask)
        col_idxs = np.flatnonzero(col_mask)
        row_colors = [int(arr[i, 0]) for i in row_idxs]
        col_colors = [int(arr[0, j]) for j in col_idxs]
        sep_colors = row_colors + col_colors
        sep_color = None
        sep_color_pct = 0.0
        if sep_colors:
            sep_color = Counter(sep_colors).most_common(1)[0][0]
            count = int((arr == sep_color).sum())
            sep_color_pct = count / float(arr.size)

        def _spans(nonsep_mask: np.ndarray):
            spans = []
            start = None
            for i, v in enumerate(nonsep_mask):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    spans.append((start, i))
                    start = None
            if start is not None:
                spans.append((start, len(nonsep_mask)))
            return spans

        row_spans = _spans(~row_mask)
        col_spans = _spans(~col_mask)
        subgrid_shapes = [(r1 - r0, c1 - c0) for r0, r1 in row_spans for c0, c1 in col_spans]
        return (int(num_rows), int(num_cols)), sep_colors, sep_color, float(sep_color_pct), subgrid_shapes
    except Exception:
        return (1, 1), [], None, 0.0, []

def has_subgrids(task: dict) -> bool:
    return (
        all(pair['input_num_subgrids'] != (1, 1) for pair in task['train']) and
        not (any(pair['input_num_subgrids'][0] == 1 for pair in task['train']) and any(pair['input_num_subgrids'][1] == 1 for pair in task['train'])) and
        not any(pair['input_sep_color_pct'] > 0.55 and not (min(pair['input_num_subgrids']) >= 3 and pair['input_num_subgrids'] == pair['output_num_subgrids']) for pair in task['train'])
    )
    
def infer_test_output_shape(task: dict, test_input_shape: Tuple[int, int],
                            test_input_num_subgrids: Tuple[int, int] = None,
                            test_input_subgrid_shapes: Optional[List[Tuple[int, int]]] = None) -> Tuple[int, int]:
    train_input0_shape = np.array(task['train'][0]['input']).shape
    train_output0_shape = np.array(task['train'][0]['output']).shape
    if all(np.array(pair['input']).shape == np.array(pair['output']).shape for pair in task['train']):
        return test_input_shape
    elif len(set(tuple(ns[1]/ns[0] for ns in zip(np.array(pair['input']).shape, np.array(pair['output']).shape)) for pair in task['train'])) == 1:
        return tuple(int(test_input_shape[i] * train_output0_shape[i] / train_input0_shape[i]) for i in range(2))
    elif len(set(tuple(math.log(ns[1], max(ns[0], 1.1)) for ns in zip(np.array(pair['input']).shape, np.array(pair['output']).shape)) for pair in task['train'])) == 1: # only 2 tasks:ad7e01d0 and ccd554ac
        return tuple(int(test_input_shape[i] ** math.log(train_output0_shape[i], train_input0_shape[i])) for i in range(2))
    elif len(set(np.array(pair['output']).shape for pair in task['train'])) == 1:
        return train_output0_shape
    elif has_subgrids(task):     
        if all(np.array(pair['output']).shape == pair['input_num_subgrids'] for pair in task['train']):  # only 3 tasks
            return test_input_num_subgrids
        elif all(all(np.array(pair['output']).shape == shape for shape in pair['input_subgrid_shapes']) for pair in task['train']): # only 2 tasks
            return test_input_subgrid_shapes[0]
    return None
    
def annotate_pair(pair: dict) -> None:
    for key in ("input", "output"):
        subgrids, sep_colors, sep_color, sep_color_pct, subgrid_shapes = analyze_grid(pair.get(key, []))
        pair[f"{key}_num_subgrids"] = subgrids          # (rows, cols)
        pair[f"{key}_sep_colors"] = sep_colors          # set of non-black separator colors
        pair[f"{key}_sep_color"] = sep_color            # most common sep color (or None)
        pair[f"{key}_sep_color_pct"] = sep_color_pct    # overall percentage in grid
        pair[f"{key}_subgrid_shapes"] = subgrid_shapes  # list of (rows, cols) subgrid shapes





def show_puzzle(puzzle, ID=None, show_sections=['train', 'test'], max_pairs=None, max_element_count=2000):
    # print('=' * 16)
    # if ID is not None: print(f'Puzzle {ID}')
    print("Given input-output grid pairs as reference examples, carefully observe the patterns "
          "to predict the output grid for new test input. "
          "Each pair follows the same transformation rule. "
          "Grids are 2D arrays represented as strings, with cells (colors) separated by spaces "
          "and rows by newlines.")
    element_count = sum(count_elements(puzzle, section, max_pairs=max_pairs) for section in show_sections)
    # if max_element_count is not None and element_count > max_element_count: return 0

    for section in show_sections:
        print({'train': 'Here are the input and output grids for the reference examples:', 
               'test': 'Here is the input grid for the test example:'}[section])
        for i, pair in enumerate(puzzle.get(section, [])[:max_pairs]):
            if section == 'train': print(f'Example {i + 1}:')
            show_grid(pair['input'], f'Input:')
            if section == 'train': show_grid(pair['output'], f'Output:')
            print()
    print("Directly provide the output grids corresponding to the given test input grids, "
          "based on the patterns observed in the reference examples.")
    return element_count

def dedup_arrays(arrs):
    out, seen = [], set()
    for a in arrs:
        b = np.ascontiguousarray(a)  # canonicalize memory order (rot/flip may create views/neg strides)
        key = (b.shape, b.dtype, b.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(b.copy())     # detach from shared memory
    return out

def gen_puzzle(gen_fn, n_train=3, n_test=1):
    return {'train': [gen_fn() for _ in range(n_train)], 
        'test': [gen_fn() for _ in range(n_test)]}

def gen_variation_color(special_var_idx):
    n_distractors = 2 # np.random.randint(1, 3)
    n, m = 4, 4 * 5
    input_grid = np.full((n, m), Color.BLACK)
    colors = [Color.BLUE, Color.GREY, Color.TEAL]
    sprite = random_sprite(3, 3, color_palette=colors, density=0.6, symmetry='not_symmetric')
    sprite_variations = [np.rot90(sprite), np.rot90(sprite, 2), #np.rot90(sprite, 3), 
        np.flipud(sprite), np.fliplr(sprite),]# np.flipud(np.rot90(sprite)), np.fliplr(np.rot90(sprite))]
    sprite_variations = dedup_arrays(sprite_variations)
    special_bar_color = random.choice(list(set(Color.NOT_BLACK) - set(colors)))
    other_bar_colors = list(set(Color.NOT_BLACK) - set(colors) - {special_bar_color})
    other_sprites = []
    for i, sv in enumerate(sprite_variations):
        bar_color = special_bar_color if i == special_var_idx else random.choice(other_bar_colors)
        sv = np.vstack([sv, np.full((1, sv.shape[1]), bar_color)])
        if i == special_var_idx: special_sprite = sv
        else: other_sprites.append(sv)
    _ = blit_sprite(input_grid, sprite, 0, 0)
    for s in [special_sprite] + random.sample(other_sprites, n_distractors):
        x, y = random_free_location_for_sprite(input_grid, s, padding=1, border_size=0)
        _ = blit_sprite(input_grid, s, x, y)
    output_grid = np.full((1, 3), special_bar_color)
    return {'input': input_grid, 'output': output_grid}

def gen_variation_color(special_var_idx):
    n_distractors = 2 # np.random.randint(1, 3)
    n, m = 4, 4 * 5
    input_grid = np.full((n, m), Color.BLACK)
    colors = [Color.BLUE, Color.GREY, Color.TEAL]
    sprite = random_sprite(3, 3, color_palette=colors, density=0.6, symmetry='not_symmetric')
    sprite_variations = [np.rot90(sprite), np.rot90(sprite, 2), #np.rot90(sprite, 3), 
        np.flipud(sprite), np.fliplr(sprite),]# np.flipud(np.rot90(sprite)), np.fliplr(np.rot90(sprite))]
    sprite_variations = dedup_arrays(sprite_variations)
    special_bar_color = Color.RED
    other_sprites = []
    for i, sv in enumerate(sprite_variations):
        bar_color = special_bar_color if i == special_var_idx else Color.BLACK
        sv = np.vstack([sv, np.full((1, sv.shape[1]), bar_color)])
        if i == special_var_idx: special_sprite = sv
        else: other_sprites.append(sv)
    sprite = np.vstack([sprite, np.full((1, sprite.shape[1]), special_bar_color)])
    _ = blit_sprite(input_grid, sprite, 0, 0)
    for s in [special_sprite] + random.sample(other_sprites, n_distractors):
        x, y = random_free_location_for_sprite(input_grid, s, padding=1, border_size=0)
        _ = blit_sprite(input_grid, s, x, y)
    output_grid = output_grid = np.copy(input_grid)
    input_grid[-1, 3:] = Color.BLACK
    return {'input': input_grid, 'output': output_grid}