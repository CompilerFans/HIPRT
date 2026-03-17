#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class ObjWriter:
    vertices: List[Tuple[float, float, float]]
    normals: List[Tuple[float, float, float]]
    faces: List[Tuple[str, Tuple[int, int, int, int]]]

    def __init__(self) -> None:
        self.vertices = []
        self.normals = []
        self.faces = []

    def add_vertex(self, x: float, y: float, z: float) -> int:
        self.vertices.append((x, y, z))
        return len(self.vertices)

    def add_normal(self, x: float, y: float, z: float) -> int:
        self.normals.append((x, y, z))
        return len(self.normals)

    def add_quad(
        self,
        material: str,
        normal: Tuple[float, float, float],
        v0: Tuple[float, float, float],
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float],
        v3: Tuple[float, float, float],
    ) -> None:
        n = self.add_normal(*normal)
        i0 = self.add_vertex(*v0)
        i1 = self.add_vertex(*v1)
        i2 = self.add_vertex(*v2)
        i3 = self.add_vertex(*v3)
        self.faces.append((material, (i0, i1, i2, n)))
        self.faces.append((material, (i0, i2, i3, n)))

    def add_box(
        self,
        material: str,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
    ) -> None:
        self.add_quad(material, (0.0, 0.0, 1.0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
        self.add_quad(material, (0.0, 0.0, -1.0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0))
        self.add_quad(material, (-1.0, 0.0, 0.0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))
        self.add_quad(material, (1.0, 0.0, 0.0), (x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1))
        self.add_quad(material, (0.0, 1.0, 0.0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0))
        self.add_quad(material, (0.0, -1.0, 0.0), (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))

    def write(self, obj_path: Path, mtl_name: str) -> None:
        lines = [f"mtllib {mtl_name}", "o metax_showcase"]
        for v in self.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for n in self.normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

        current_material = None
        for material, (i0, i1, i2, n) in self.faces:
            if material != current_material:
                lines.append(f"usemtl {material}")
                current_material = material
            lines.append(f"f {i0}//{n} {i1}//{n} {i2}//{n}")

        obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple extruded text OBJ/MTL showcase scene.")
    parser.add_argument("--text", required=True, help="Text to extrude.")
    parser.add_argument("--font", required=True, help="Path to a TTF/OTF/TTC font.")
    parser.add_argument("--font-size", type=int, default=96, help="Raster font size in pixels.")
    parser.add_argument("--threshold", type=int, default=32, help="Coverage threshold for solid pixels.")
    parser.add_argument("--pixel-size", type=float, default=0.03, help="World-space size of one raster pixel.")
    parser.add_argument("--depth", type=float, default=0.12, help="Extrusion depth.")
    parser.add_argument("--line-gap", type=int, default=24, help="Extra raster pixels between lines.")
    parser.add_argument("--scene-name", default="metax_title", help="Base name for OBJ/MTL files.")
    parser.add_argument("--output-dir", default="test/common/meshes/metax_showcase", help="Output directory.")
    parser.add_argument("--with-stage", action="store_true", help="Add floor, back wall and top emissive light.")
    parser.add_argument("--with-effects", action="store_true", help="Add glass, mirror, and MetaX accents on floor/wall.")
    return parser.parse_args()


def rasterize_text(text: str, font_path: Path, font_size: int, line_gap: int) -> Image.Image:
    font = ImageFont.truetype(str(font_path), font_size)
    lines = text.split("\\n")
    dummy = Image.new("L", (8, 8), 0)
    draw = ImageDraw.Draw(dummy)

    widths = []
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(max(1, bbox[2] - bbox[0]))
        heights.append(max(1, bbox[3] - bbox[1]))

    width = max(widths) + font_size // 2
    height = sum(heights) + max(0, len(lines) - 1) * line_gap + font_size // 2
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)

    y = font_size // 4
    for line, line_height in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, fill=255, font=font)
        y += line_height + line_gap
    return image


def add_text_mesh(writer: ObjWriter, image: Image.Image, pixel_size: float, depth: float, threshold: int) -> None:
    pixels = image.load()
    width, height = image.size
    center_x = width * pixel_size * 0.5
    center_y = height * pixel_size * 0.5

    for py in range(height):
        px = 0
        while px < width:
            while px < width and pixels[px, py] <= threshold:
                px += 1
            start = px
            while px < width and pixels[px, py] > threshold:
                px += 1
            if start == px:
                continue

            x0 = start * pixel_size - center_x
            x1 = px * pixel_size - center_x
            y0 = (height - py - 1) * pixel_size - center_y
            y1 = (height - py) * pixel_size - center_y
            writer.add_box("TextMetal", x0, y0, 0.0, x1, y1, depth)


def add_text_front(
    writer: ObjWriter,
    image: Image.Image,
    pixel_size: float,
    depth: float,
    threshold: int,
    material: str,
    center_x: float,
    base_y: float,
    base_z: float,
) -> None:
    pixels = image.load()
    width, height = image.size
    offset_x = width * pixel_size * 0.5

    for py in range(height):
        px = 0
        while px < width:
            while px < width and pixels[px, py] <= threshold:
                px += 1
            start = px
            while px < width and pixels[px, py] > threshold:
                px += 1
            if start == px:
                continue

            x0 = center_x + start * pixel_size - offset_x
            x1 = center_x + px * pixel_size - offset_x
            y0 = base_y + (height - py - 1) * pixel_size
            y1 = base_y + (height - py) * pixel_size
            writer.add_box(material, x0, y0, base_z, x1, y1, base_z + depth)


def add_text_floor(
    writer: ObjWriter,
    image: Image.Image,
    pixel_size: float,
    height_thickness: float,
    threshold: int,
    material: str,
    center_x: float,
    floor_y: float,
    center_z: float,
) -> None:
    pixels = image.load()
    width, height = image.size
    offset_x = width * pixel_size * 0.5
    offset_z = height * pixel_size * 0.5

    for py in range(height):
        px = 0
        while px < width:
            while px < width and pixels[px, py] <= threshold:
                px += 1
            start = px
            while px < width and pixels[px, py] > threshold:
                px += 1
            if start == px:
                continue

            x0 = center_x + start * pixel_size - offset_x
            x1 = center_x + px * pixel_size - offset_x
            z0 = center_z + (height - py - 1) * pixel_size - offset_z
            z1 = center_z + (height - py) * pixel_size - offset_z
            writer.add_box(material, x0, floor_y, z0, x1, floor_y + height_thickness, z1)


def add_text_wall(
    writer: ObjWriter,
    image: Image.Image,
    pixel_size: float,
    depth: float,
    threshold: int,
    material: str,
    center_x: float,
    base_y: float,
    wall_z: float,
) -> None:
    pixels = image.load()
    width, height = image.size
    offset_x = width * pixel_size * 0.5

    for py in range(height):
        px = 0
        while px < width:
            while px < width and pixels[px, py] <= threshold:
                px += 1
            start = px
            while px < width and pixels[px, py] > threshold:
                px += 1
            if start == px:
                continue

            x0 = center_x + start * pixel_size - offset_x
            x1 = center_x + px * pixel_size - offset_x
            y0 = base_y + (height - py - 1) * pixel_size
            y1 = base_y + (height - py) * pixel_size
            writer.add_box(material, x0, y0, wall_z, x1, y1, wall_z + depth)


def add_stage(writer: ObjWriter, text_bounds: Tuple[float, float, float, float], depth: float) -> dict:
    min_x, max_x, min_y, max_y = text_bounds
    width = max_x - min_x
    height = max_y - min_y
    pad_x = max(1.5, width * 0.25)
    floor_front = max(2.0, height * 0.45)
    wall_height = max(2.0, height * 0.8)
    light_width = width * 0.42
    light_depth = max(0.35, height * 0.08)
    light_height = max_y + height * 0.35

    writer.add_box("FloorDark", min_x - pad_x, -floor_front, -0.08, max_x + pad_x, max_y + 0.5, 0.0)
    writer.add_box("BackWall", min_x - pad_x, max_y + 0.5, -0.08, max_x + pad_x, max_y + 0.65, wall_height)
    writer.add_box(
        "KeyLight",
        -light_width * 0.5,
        light_height,
        depth * 1.6,
        light_width * 0.5,
        light_height + light_depth,
        depth * 1.6 + 0.02,
    )
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "pad_x": pad_x,
        "floor_y": -0.08,
        "floor_top": 0.0,
        "floor_front": floor_front,
        "wall_y": max_y + 0.5,
        "wall_z": -0.08,
        "wall_front_z": wall_height - 0.08,
        "width": width,
        "height": height,
    }


def add_effects(
    writer: ObjWriter,
    font_path: Path,
    text_bounds: Tuple[float, float, float, float],
    stage_info: dict,
    pixel_size: float,
    threshold: int,
) -> None:
    min_x, max_x, _, max_y = text_bounds
    width = max_x - min_x
    floor_top = stage_info["floor_top"]
    wall_y = stage_info["wall_y"]
    wall_front_z = stage_info["wall_front_z"]

    writer.add_box("GlassCyan", -5.8, -0.05, 2.3, -3.6, 3.1, 2.58)
    writer.add_box("GlassCyan", 2.9, -0.05, 0.9, 5.0, 2.5, 1.18)
    writer.add_box("MirrorChrome", 6.2, -0.05, 1.4, 7.3, 4.1, 3.9)

    accent_image = rasterize_text("MetaX", font_path, 96, 8)
    add_text_floor(
        writer,
        accent_image,
        pixel_size * 0.72,
        0.035,
        threshold,
        "FloorAccent",
        center_x=0.0,
        floor_y=floor_top + 0.01,
        center_z=5.4,
    )
    add_text_wall(
        writer,
        accent_image,
        pixel_size * 0.80,
        0.05,
        threshold,
        "WallAccent",
        center_x=0.0,
        base_y=wall_y + 0.8,
        wall_z=wall_front_z - 0.05,
    )
    add_text_front(
        writer,
        accent_image,
        pixel_size * 0.42,
        0.14,
        threshold,
        "GlassEtch",
        center_x=-4.7,
        base_y=0.45,
        base_z=2.60,
    )


def write_mtl(mtl_path: Path) -> None:
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl TextMetal",
                "Ns 256.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.820000 0.800000 0.760000",
                "Ks 0.200000 0.200000 0.200000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.000000",
                "d 1.000000",
                "illum 2",
                "",
                "newmtl FloorDark",
                "Ns 32.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.060000 0.065000 0.075000",
                "Ks 0.050000 0.050000 0.050000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.000000",
                "d 1.000000",
                "illum 2",
                "",
                "newmtl BackWall",
                "Ns 32.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.100000 0.110000 0.120000",
                "Ks 0.020000 0.020000 0.020000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.000000",
                "d 1.000000",
                "illum 2",
                "",
                "newmtl GlassCyan",
                "Ns 256.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.180000 0.620000 0.760000",
                "Ks 0.900000 0.950000 1.000000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.450000",
                "d 0.350000",
                "illum 2",
                "",
                "newmtl MirrorChrome",
                "Ns 512.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.720000 0.740000 0.780000",
                "Ks 0.980000 0.980000 0.980000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.000000",
                "d 1.000000",
                "illum 2",
                "",
                "newmtl FloorAccent",
                "Ns 64.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.920000 0.380000 0.120000",
                "Ks 0.080000 0.080000 0.080000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.000000",
                "d 1.000000",
                "illum 2",
                "",
                "newmtl WallAccent",
                "Ns 128.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.120000 0.780000 0.980000",
                "Ks 0.080000 0.080000 0.080000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.000000",
                "d 1.000000",
                "illum 2",
                "",
                "newmtl GlassEtch",
                "Ns 96.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.920000 0.970000 1.000000",
                "Ks 0.300000 0.300000 0.300000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.000000",
                "d 1.000000",
                "illum 2",
                "",
                "newmtl KeyLight",
                "Ns 1.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 1.000000 1.000000 1.000000",
                "Ks 0.000000 0.000000 0.000000",
                "Ke 2.800000 2.500000 2.100000",
                "Ni 1.000000",
                "d 1.000000",
                "illum 2",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    font_path = Path(args.font)
    if not font_path.exists():
        raise SystemExit(f"Font file not found: {font_path}")

    image = rasterize_text(args.text, font_path, args.font_size, args.line_gap)
    writer = ObjWriter()
    add_text_mesh(writer, image, args.pixel_size, args.depth, args.threshold)

    width_world = image.size[0] * args.pixel_size
    height_world = image.size[1] * args.pixel_size
    text_bounds = (-width_world * 0.5, width_world * 0.5, -height_world * 0.5, height_world * 0.5)

    stage_info = None
    if args.with_stage:
        stage_info = add_stage(writer, text_bounds, args.depth)

    if args.with_effects:
        if stage_info is None:
            raise SystemExit("--with-effects requires --with-stage")
        add_effects(writer, font_path, text_bounds, stage_info, args.pixel_size, args.threshold)

    obj_path = output_dir / f"{args.scene_name}.obj"
    mtl_path = output_dir / f"{args.scene_name}.mtl"
    write_mtl(mtl_path)
    writer.write(obj_path, mtl_path.name)
    print(obj_path)
    print(mtl_path)


if __name__ == "__main__":
    main()
