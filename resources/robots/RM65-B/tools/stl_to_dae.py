"""Convert binary STL meshes to COLLADA (.dae) declaring <up_axis>Z_UP</up_axis>.

Why this exists
---------------
Isaac Gym's `asset_options.flip_visual_attachments` (True for this repo) rotates
visual meshes loaded from a file *without* up-axis metadata, but the assimp
loader already converts COLLADA data according to the file's `<up_axis>`.  The
effect, measured against a ground-truth render:

  * STL  + flip=True   -> visuals rotated away from the true geometry
  * .dae + flip=True   -> visuals match the true geometry exactly

The quadruped ships COLLADA meshes (up_axis Z_UP), so the arm needs the same
format for its *visuals* to line up.  Its *collision* geometry must stay STL:
for collision, assimp's up-axis conversion is not undone by the flip, so a
COLLADA collision mesh comes out mirrored.  See the ybt_arm README.

Usage
-----
    python resources/robots/RM65-B/tools/stl_to_dae.py \\
        resources/robots/RM65-B/meshes \\
        resources/robots/RM65-B/meshes/visual
"""
import os
import struct
import sys


def read_stl(path):
    """Binary STL -> (vertices, face normals, triangles of (vertex, normal) ids)."""
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + n * 50:
        raise ValueError("%s: not a binary STL (%d bytes, %d facets)" % (path, len(data), n))

    verts, normals, tris = [], [], []
    vmap, nmap = {}, {}
    for i in range(n):
        off = 84 + i * 50
        nkey = tuple(round(v, 5) for v in struct.unpack_from("<3f", data, off))
        if nkey not in nmap:
            nmap[nkey] = len(normals)
            normals.append(nkey)
        ni = nmap[nkey]
        tri = []
        for k in range(3):
            vkey = tuple(round(v, 6) for v in struct.unpack_from("<3f", data, off + 12 + k * 12))
            if vkey not in vmap:
                vmap[vkey] = len(verts)
                verts.append(vkey)
            tri.append((vmap[vkey], ni))
        tris.append(tri)
    return verts, normals, tris


TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor>
      <author>stl_to_dae.py</author>
      <authoring_tool>stl_to_dae.py</authoring_tool>
    </contributor>
    <created>%(date)s</created>
    <modified>%(date)s</modified>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_effects>
    <effect id="effect_0">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <diffuse>
              <color>0.8 0.8 0.8 1</color>
            </diffuse>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="mat_0">
      <instance_effect url="#effect_0"/>
    </material>
  </library_materials>
  <library_geometries>
    <geometry id="geometry_0">
      <mesh>
        <source id="pos_0">
          <float_array id="pos_0-array" count="%(npos)d">%(pos)s</float_array>
          <technique_common>
            <accessor source="#pos_0-array" count="%(nv)d" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="norm_0">
          <float_array id="norm_0-array" count="%(nnor)d">%(nor)s</float_array>
          <technique_common>
            <accessor source="#norm_0-array" count="%(nml)d" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="uvs_0">
          <float_array id="uvs_0-array" count="0"></float_array>
          <technique_common>
            <accessor source="#uvs_0-array" count="0" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="vert_0">
          <input semantic="POSITION" source="#pos_0"/>
        </vertices>
        <triangles count="%(ntri)d" material="mat_0">
          <input semantic="VERTEX" source="#vert_0" offset="0"/>
          <input semantic="NORMAL" source="#norm_0" offset="1"/>
          <p>%(idx)s</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="DefaultScene">
      <node id="node_0" type="NODE">
        <matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
        <node id="node_0_mesh0" type="NODE">
          <matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
          <instance_geometry url="#geometry_0">
            <bind_material>
              <technique_common>
                <instance_material symbol="mat_0" target="#mat_0"/>
              </technique_common>
            </bind_material>
          </instance_geometry>
        </node>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene>
    <instance_visual_scene url="#DefaultScene"/>
  </scene>
</COLLADA>
"""


def convert(src, dst, date="1970-01-01T00:00:00"):
    verts, normals, tris = read_stl(src)
    fields = {
        "date": date,
        "nv": len(verts),
        "npos": len(verts) * 3,
        "nml": len(normals),
        "nnor": len(normals) * 3,
        "ntri": len(tris),
        "pos": " ".join("%g" % v for p in verts for v in p),
        "nor": " ".join("%g" % v for p in normals for v in p),
        "idx": " ".join("%d %d %d %d %d %d" % (t[0][0], t[0][1], t[1][0], t[1][1],
                                               t[2][0], t[2][1]) for t in tris),
    }
    if not os.path.isdir(os.path.dirname(dst) or "."):
        os.makedirs(os.path.dirname(dst))
    with open(dst, "w") as f:
        f.write(TEMPLATE % fields)
    return len(verts), len(tris)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 1
    src_dir, dst_dir = argv[1], argv[2]
    for name in sorted(os.listdir(src_dir)):
        if not name.lower().endswith(".stl"):
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, os.path.splitext(name)[0] + ".dae")
        nv, nt = convert(src, dst)
        print("%-16s -> %-16s %5d verts %5d tris" % (name, os.path.basename(dst), nv, nt))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
