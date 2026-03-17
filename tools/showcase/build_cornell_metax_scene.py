#!/usr/bin/env python3
from pathlib import Path


ROOT = Path("/data/HIPRT")
SRC_DIR = ROOT / "test/common/meshes"
OUT_DIR = ROOT / "test/common/meshes/metax_showcase"


def load_obj(path: Path):
    vertices = []
    normals = []
    faces = []
    current_material = None

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()[:4]
            vertices.append((float(x), float(y), float(z)))
        elif line.startswith("vn "):
            _, x, y, z = line.split()[:4]
            normals.append((float(x), float(y), float(z)))
        elif line.startswith("usemtl "):
            current_material = line.split()[1]
        elif line.startswith("f "):
            parts = line.split()[1:]
            tri = []
            for part in parts:
                v, _, n = part.partition("//")
                tri.append((int(v), int(n)))
            faces.append((current_material, tri))

    return vertices, normals, faces


def transform_vertices(vertices, scale, translate):
    out = []
    for x, y, z in vertices:
        out.append(
            (
                x * scale + translate[0],
                y * scale + translate[1],
                z * scale + translate[2],
            )
        )
    return out


def write_scene(out_obj: Path, out_mtl: Path):
    cornell_obj = SRC_DIR / "cornellbox/cornellBox.obj"
    cornell_mtl = SRC_DIR / "cornellbox/cornellBox.mtl"
    metax_obj = OUT_DIR / "metax_letters.obj"
    metax_mtl = OUT_DIR / "metax_letters.mtl"

    cornell_vertices, cornell_normals, cornell_faces = load_obj(cornell_obj)
    metax_vertices, metax_normals, metax_faces = load_obj(metax_obj)

    metax_vertices = transform_vertices(
        metax_vertices,
        scale=0.68,
        translate=(0.10, 0.62, -2.58),
    )

    with out_mtl.open("w", encoding="utf-8") as f:
        f.write(cornell_mtl.read_text(encoding="utf-8"))
        f.write("\n")
        text = metax_mtl.read_text(encoding="utf-8")
        text = text.replace("Kd 0.820000 0.800000 0.760000", "Kd 0.920000 0.910000 0.880000")
        text = text.replace("Ks 0.200000 0.200000 0.200000", "Ks 0.850000 0.820000 0.760000")
        f.write(text)

    lines = [f"mtllib {out_mtl.name}", "o cornell_metax"]

    for v in cornell_vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for v in metax_vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for n in cornell_normals:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    for n in metax_normals:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

    vertex_offset = 0
    normal_offset = 0
    current_material = None

    for material, tri in cornell_faces:
        if material != current_material:
            lines.append(f"usemtl {material}")
            current_material = material
        parts = [f"{v + vertex_offset}//{n + normal_offset}" for v, n in tri]
        lines.append("f " + " ".join(parts))

    vertex_offset = len(cornell_vertices)
    normal_offset = len(cornell_normals)

    for material, tri in metax_faces:
        if material != current_material:
            lines.append(f"usemtl {material}")
            current_material = material
        parts = [f"{v + vertex_offset}//{n + normal_offset}" for v, n in tri]
        lines.append("f " + " ".join(parts))

    out_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_obj = OUT_DIR / "cornell_metax.obj"
    out_mtl = OUT_DIR / "cornell_metax.mtl"
    write_scene(out_obj, out_mtl)
    print(out_obj)
    print(out_mtl)


if __name__ == "__main__":
    main()
