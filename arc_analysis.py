from typing import Tuple, List, Optional
from functools import cached_property
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import NamedTuple
from collections import Counter
import numpy as np
from scipy.ndimage import label, binary_fill_holes, generate_binary_structure

from methodtools import lru_cache

from seeds.common import Color

PURITY_THRESHOLD = 8 / 9 + 1e-4

class PaddedArrayView:
    """Wrapper that returns pad_value for out-of-bounds slice access."""
    
    def __init__(self, array: np.ndarray, pad_value: int = 0):
        self.array = array
        self.pad_value = pad_value
    
    def __getitem__(self, key) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        
        # Convert slices to (start, stop) for each dimension
        slices = []
        out_shape = []
        for i, k in enumerate(key):
            if isinstance(k, slice):
                start = k.start or 0
                stop = k.stop if k.stop is not None else self.array.shape[i]
                slices.append((start, stop))
                out_shape.append(stop - start)
            else:  # integer index
                slices.append((k, k + 1))
                out_shape.append(None)  # will squeeze this dim
        
        # Create output filled with pad_value
        full_shape = [s for s in out_shape if s is not None]
        out = np.full(full_shape, self.pad_value, dtype=self.array.dtype)
        
        # Compute valid regions (clip to array bounds)
        src_slices = []
        dst_slices = []
        for i, (start, stop) in enumerate(slices):
            src_start = max(0, start)
            src_stop = min(self.array.shape[i], stop)
            if src_start >= src_stop:
                return out  # completely out of bounds
            
            dst_start = src_start - start
            dst_stop = dst_start + (src_stop - src_start)
            src_slices.append(slice(src_start, src_stop))
            dst_slices.append(slice(dst_start, dst_stop))
        
        # Copy valid portion
        out[tuple(dst_slices)] = self.array[tuple(src_slices)]
        return out

    @property
    def shape(self): return self.array.shape

    @property
    def size(self): return self.array.size


class Pos(NamedTuple):
    """ Combine np array-like vectorized arithmetic and tuple-like indexing (into np arrays) """
    x: int
    y: int
    
    def __add__(self, other):
        if isinstance(other, int): return Pos(self.x + other, self.y + other)
        if isinstance(other, (tuple, Pos)): return Pos(self.x + other[0], self.y + other[1])
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, int): return Pos(self.x - other, self.y - other)
        if isinstance(other, (tuple, Pos)): return Pos(self.x - other[0], self.y - other[1])
        return NotImplemented
    
    def __neg__(self): return Pos(-self.x, -self.y)
    
    def __mul__(self, other):
        if isinstance(other, int): return Pos(self.x * other, self.y * other)
        if isinstance(other, (tuple, Pos)): return Pos(self.x * other[0], self.y * other[1])
        return NotImplemented

    def __floordiv__(self, other, assert_divisible: bool = True):
        if isinstance(other, int):
            if assert_divisible:
                assert self.x % other == 0 and self.y % other == 0, f"{other} must divide {self}"
            return Pos(self.x // other, self.y // other)
        if isinstance(other, (tuple, Pos)):
            if assert_divisible:
                assert self.x % other[0] == 0 and self.y % other[1] == 0, f"{other} must divide {self}"
            return Pos(self.x // other[0], self.y // other[1])
        return NotImplemented
    
    @property
    def is_odd(self) -> bool: return self.x % 2 == 1 and self.y % 2 == 1

    @classmethod
    def slice_around(cls, center: 'Pos', radius: 'Pos') -> tuple[slice, slice]:
        return (slice(center.x - radius.x, center.x + radius.x + 1),
                slice(center.y - radius.y, center.y + radius.y + 1))

    @classmethod
    def rpositions_to_center(cls, radius: 'Pos', include_center: bool = True):
        for dx in range(-radius.x, radius.x + 1):
            for dy in range(-radius.y, radius.y + 1):
                if include_center or (dx, dy) != (0, 0):
                    yield cls(dx, dy)

    # Transformations
    def rot90(self, k: int = 1):
        x, y = self.x, self.y
        for _ in range(k % 4): x, y = -y, x
        return Pos(x, y)

    def rot90_cw(self, k: int = 1): return self.rot90(-k)
    
    def mirror_x(self): return Pos(self.x, -self.y)
    def mirror_y(self): return Pos(-self.x, self.y)    
    def mirror_diagonal(self): return Pos(self.y, self.x)
    def mirror_anti_diagonal(self): return Pos(-self.y, -self.x)
    
    def manhattan(self, other=(0, 0)): return abs(self.x - other[0]) + abs(self.y - other[1])
    def chebyshev(self, other=(0, 0)): return max(abs(self.x - other[0]), abs(self.y - other[1]))

    def neighbors_4(self): return [self + d for d in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
    def neighbors_8(self):
        return [self + d for d in [(-1,-1), (-1,0), (-1,1), 
                                    (0,-1),          (0,1), 
                                    (1,-1),  (1,0),  (1,1)]]

class IO(IntEnum):  # IntEnum survives autoreload
    INPUT = 0
    OUTPUT = 1

class LineDirection(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    SLASH = 2
    BACKSLASH = 3

class Task(object):
    def __init__(self, puzzle_id: str, task_dict: dict):
        self.puzzle_id = puzzle_id
        self.train = Section(puzzle_id + "/train", task_dict['train'])
        self.test = Section(puzzle_id + "/test", task_dict['test'])

class Section(object):
    def __init__(self, name: str, examples: list[dict]):
        self.name = name
        self.examples = [Example(self, i, np.array(e['input']), np.array(e['output']))
                         for i, e in enumerate(examples)]
        self.is_shape_aligned = all(e.is_shape_aligned for e in self.examples)
        if self.is_shape_aligned:
            top_color_pcts = np.array([e.input.color_pcts[0][1] for e in self.examples])
            if all(e.input.color_pcts[0][0] == Color.BLACK for e in self.examples) or \
                top_color_pcts.min() > 0.6 and top_color_pcts.mean() > 0.7:
                for e in self.examples:
                    e.input.background_color = e.input.color_pcts[0][0]

class Example(object):
    def __init__(self, section: Section, example_id: int, input_array: np.ndarray, output_array: np.ndarray):
        self.section = section
        self.example_id = example_id
        self.input = Grid(input_array, IO.INPUT, example_id)
        self.output = Grid(output_array, IO.OUTPUT, example_id)
        self.is_shape_aligned = input_array.shape == output_array.shape

@dataclass
class SubGridsInfo:
    num: Tuple[int, int]
    shapes: List[Tuple[int, int]]
    is_regular: bool
    sep_colors: List[Color]
    sep_color: Color
    sep_color_pct: float


class Grid(object):
    def __init__(self, array: np.ndarray, io: IO, example_id: int):
        self.array = array
        self.io = io
        self.example_id = example_id
        self.color_pcts = [(c, n / self.array.size) for c, n in Counter(array.flatten()).most_common()]
        self.background_color = None
        # self.detect_subgrids()
        # self.parray = PaddedArrayView(array, pad_value=Color.PAD)
        # self.cells = [Cell(self, pos, color) for pos, color in np.ndenumerate(array)]

    def _compute_separator_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Find uniform non-black rows/columns (separators)."""
        arr = self.array
        row_uniform = (arr == arr[:, [0]]).all(axis=1) & (arr[:, 0] != Color.BLACK)
        col_uniform = (arr == arr[[0], :]).all(axis=0) & (arr[0, :] != Color.BLACK)
        return row_uniform, col_uniform

    @staticmethod
    def _find_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find (start, end) spans of contiguous True runs in a boolean mask."""
        if not mask.any():
            return [(0, len(mask))]
        padded = np.concatenate([[False], mask, [False]])
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        return list(zip(starts.tolist(), ends.tolist()))

    def detect_subgrids(self) -> None:
        arr = self.array
        self.subgrids = None
        if arr.ndim != 2 or min(arr.shape) <= 1: return
        row_mask, col_mask = self._compute_separator_masks()
        if not (row_mask.any() or col_mask.any()): return
        
        row_spans = self._find_spans(~row_mask)
        col_spans = self._find_spans(~col_mask)
        shapes = [(r1 - r0, c1 - c0) for r0, r1 in row_spans for c0, c1 in col_spans]
        
        sep_colors = arr[row_mask, 0].tolist() + arr[0, col_mask].tolist()
        sep_color = Counter(sep_colors).most_common(1)[0][0] if sep_colors else None
        sep_color_pct = (arr == sep_color).sum() / arr.size if sep_color else 0.0
        
        self.subgrids = SubGridsInfo(
            num=(len(row_spans), len(col_spans)),
            shapes=shapes, is_regular=all(shape == shapes[0] for shape in shapes),
            sep_colors=sep_colors, sep_color=sep_color, sep_color_pct=sep_color_pct,
        )
        
        sep_id = -1
        self.subgrid_pos = np.full((2, *arr.shape), sep_id, dtype=np.int32)
        self.pos_in_subgrid = np.full((2, *arr.shape), sep_id, dtype=np.int32)
        
        for row_i, (r0, r1) in enumerate(row_spans):
            for col_i, (c0, c1) in enumerate(col_spans):
                self.subgrid_pos[0, r0:r1, c0:c1] = row_i
                self.subgrid_pos[1, r0:r1, c0:c1] = col_i
                self.pos_in_subgrid[0, r0:r1, c0:c1] = np.arange(r1 - r0)[:, None]
                self.pos_in_subgrid[1, r0:r1, c0:c1] = np.arange(c1 - c0)[None, :]
        

class CCPatch(object):
    def __init__(self, patch: dict[Pos, Color], purity: float = 0., pad_value: Color = Color.PAD):
        self.patch = patch
        self.purity = purity
        self.pad = pad_value

    def __eq__(self, other) -> bool:
        """
        Compare patches with PAD-as-wildcard semantics:
        - Same positions with same colors -> match
        - Color.PAD matches ANY color
        - Superfluous PAD cells (extra positions with PAD) are allowed
        """
        if not isinstance(other, CCPatch): return NotImplemented
        # pure patches provide little information
        if self.purity > PURITY_THRESHOLD and other.purity > PURITY_THRESHOLD: return False
        
        self_pos, other_pos = set(self.patch), set(other.patch)
        return (
            all(self.patch[p] == other.patch[p] or self.pad in (self.patch[p], other.patch[p])
                for p in self_pos & other_pos) and
            all(self.patch[p] == self.pad for p in self_pos - other_pos) and
            all(other.patch[p] == self.pad for p in other_pos - self_pos)
        )

    def __len__(self) -> int: return len(self.patch)

    def transform(self, fn) -> 'CCPatch':
        """Apply transformation function to all positions."""
        return CCPatch({fn(p): c for p, c in self.patch.items()}, self.pad)

    def __getattr__(self, name):
        """Auto-delegate Pos transformation methods (rot90, mirror_x, etc.) to all positions."""
        if hasattr(Pos, name) and callable(getattr(Pos, name)):
            return lambda *a, **kw: self.transform(lambda p: getattr(p, name)(*a, **kw))
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

class Cell:
    connectivity2structure: dict[int, np.ndarray] = {
        4: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        8: np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    }

    def __init__(self, grid: Grid, pos: tuple[int, int], color: Color):
        self.grid = grid
        self.io = grid.io
        self.example_id = grid.example_id
        self.pos = Pos(*pos)
        self.color = color

    def __eq__(self, other) -> bool:
        return self.pos == other.pos and self.color == other.color and self.io == other.io and self.example_id == other.example_id

    def __hash__(self) -> int:
        return hash((self.pos, self.color, self.io, self.example_id))

    def is_background(self) -> bool:
        return self.color == self.grid.background_color

    @lru_cache()
    def cc_neighbors(self, monochromatic: bool, neighborhood_size: tuple[int, int] = (5, 5), ignore_color: bool = False, 
                     connectivity: int = 4, include_center=False) -> dict[tuple[int, int], Color]:
        """connected components neighbors (handles boundary via padded array)"""
        center = radius = (Pos(*neighborhood_size) - 1) // 2
        neighborhood = self.grid.parray[Pos.slice_around(self.pos, radius)]

        mask = neighborhood == self.color if monochromatic else \
            neighborhood != self.grid.background_color
        mask |= neighborhood == self.grid.parray.pad_value
        labeled, n_objects = label(mask, Cell.connectivity2structure[connectivity])

        if ignore_color: assert monochromatic, "ignore_color only works if monochromatic is True"
        p = {rpos: neighborhood[center + rpos] if not ignore_color else 1
            for rpos in Pos.rpositions_to_center(radius, include_center=include_center)
            if labeled[center + rpos] == labeled[center]}
        purity = (len(p) + int(not include_center)) / neighborhood.size if monochromatic else 0.
        return CCPatch(p, purity=purity, pad_value=self.grid.parray.pad_value)

    @lru_cache()
    def is_enclosed(self, connectivity: int = 4) -> bool:
        wall_mask = (self.grid.array != self.grid.background_color) & \
                    (self.grid.array != self.color)
        # same as structure = Cell.connectivity2structure[connectivity]
        structure = generate_binary_structure(2, 1 if connectivity == 4 else 2)
        filled = binary_fill_holes(wall_mask, structure)
        return bool(filled[self.pos] and not wall_mask[self.pos])

    @cached_property
    def subgrid_pos(self) -> Pos:
        if self.grid.subgrids is None: return Pos(None, None)
        return Pos(*self.grid.subgrid_pos[:, self.pos[0], self.pos[1]])
    
    @cached_property
    def pos_in_subgrid(self) -> Pos:
        if self.grid.subgrids is None: return Pos(None, None)
        return Pos(*self.grid.pos_in_subgrid[:, self.pos[0], self.pos[1]])

    # cc_id: int = None

    # line_direction: LineDirection = None
    # line_endpoints: dict[Position, int] = None
    # is_boundary: bool = None




# ('color', 'color')
# ('pos', 'pos')
# ('same_color_neighbors', 'same_color_neighbors')
# ('cc_neighbors', 'cc_neighbors')
# ('is_interior', None)