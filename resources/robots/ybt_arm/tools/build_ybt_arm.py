"""Assemble the YBT quadruped and the RM65-B arm into one URDF.

The arm is welded on top of the quadruped's trunk:

    trunk  --arm_mount_joint-->  arm_base_link  --arm_joint_1-->  ...  --> arm_link_6

Everything else about the two source robots is kept verbatim; only names (an
`arm_` prefix) and mesh paths are rewritten, and the weld joint is inserted.
Re-run this script after editing MOUNT_XYZ / MOUNT_RPY to move the arm:

    python resources/robots/ybt_arm/tools/build_ybt_arm.py

Mesh handling (see resources/robots/ybt_arm/README.md for the measurements):

  * arm visuals   -> meshes/visual/*.dae   (converted Z_UP COLLADA)
  * arm collision -> meshes/*.STL          (original SolidWorks meshes)
  * quadruped     -> ../ybt/meshes/*.dae, unchanged

Isaac Gym is loaded with flip_visual_attachments=True.  That flip is *not*
applied to collision geometry, and it only comes out right for visuals whose
file declares its up axis, so the arm's visuals have to be COLLADA while its
collision geometry has to stay STL.  Using one format for both would put either
the rendered arm or its collision hulls in the wrong place.
"""
import os
import xml.etree.ElementTree as ET

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
YBT_URDF = os.path.join(REPO, "resources/robots/ybt/urdf/ybt.urdf")
ARM_URDF = os.path.join(REPO, "resources/robots/RM65-B/urdf/RM65-B.urdf")
OUT_URDF = os.path.join(REPO, "resources/robots/ybt_arm/urdf/ybt_arm.urdf")

# Pose of the arm's base_link origin, expressed in the trunk frame.  The trunk's
# top deck is a flat plate at z = +0.1012 m, and base_link's origin sits on its
# own mounting face, so this drops the arm straight onto the middle of the back
# with joint_1's axis vertical (zero rpy = the arm standing up as modelled).
MOUNT_XYZ = (0.0, 0.0, 0.1012)
MOUNT_RPY = (0.0, 0.0, 0.0)

PREFIX = "arm_"
ARM_MAT = "arm_white"
ARM_VISUAL_DIR = "../../RM65-B/meshes/visual"
ARM_COLLISION_DIR = "../../RM65-B/meshes"


def indent(elem, level=0):
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def arm_mesh_path(filename, geom_tag):
    """package://RM65-B/meshes/<name>.STL -> a path relative to the target URDF."""
    name = filename.replace("package://RM65-B/meshes/", "")
    stem = os.path.splitext(name)[0]
    if geom_tag == "visual":
        return "%s/%s.dae" % (ARM_VISUAL_DIR, stem)
    return "%s/%s.STL" % (ARM_COLLISION_DIR, stem)


def main():
    arm = ET.parse(ARM_URDF).getroot()
    ybt = ET.parse(YBT_URDF).getroot()

    robot = ET.Element("robot", {"name": "ybt_arm"})

    # materials declared by the quadruped, kept verbatim
    for mat in ybt.findall("material"):
        robot.append(mat)

    # the arm's inline materials have no name of their own; give them one
    robot.append(ET.fromstring('<material name="%s"><color rgba="1 1 1 1"/></material>'
                               % ARM_MAT))

    arm_link_names = [l.get("name") for l in arm.findall("link")]
    arm_joint_names = [j.get("name") for j in arm.findall("joint")]
    rename = {n: PREFIX + n for n in arm_link_names}

    for link in arm.findall("link"):
        link.set("name", rename[link.get("name")])
        for mat in link.iter("material"):
            mat.set("name", ARM_MAT)
        for geom in link.findall("visual") + link.findall("collision"):
            for mesh in geom.iter("mesh"):
                mesh.set("filename", arm_mesh_path(mesh.get("filename"), geom.tag))
        robot.append(link)

    # the quadruped's meshes live one folder further away now
    for link in ybt.findall("link"):
        for mesh in link.iter("mesh"):
            name = mesh.get("filename")
            assert name.startswith("../meshes/"), name
            mesh.set("filename", "../../ybt/meshes/" + name[len("../meshes/"):])
        robot.append(link)

    for joint in arm.findall("joint"):
        joint.set("name", PREFIX + joint.get("name"))
        joint.find("parent").set("link", rename[joint.find("parent").get("link")])
        joint.find("child").set("link", rename[joint.find("child").get("link")])
        robot.append(joint)

    for joint in ybt.findall("joint"):
        robot.append(joint)

    mount = ET.fromstring(
        '<joint name="arm_mount_joint" type="fixed">'
        '<origin xyz="%g %g %g" rpy="%g %g %g"/>'
        '<parent link="trunk"/><child link="arm_base_link"/>'
        '</joint>' % (MOUNT_XYZ + MOUNT_RPY))
    robot.insert(0, mount)   # first, so the weld reads at the top of the file

    indent(robot)
    header = """<?xml version="1.0" ?>
<!-- ===================================================================== -->
<!-- ybt_arm: YBT quadruped + RM65-B arm welded into one asset.            -->
<!--                                                                       -->
<!-- Generated by resources/robots/ybt_arm/tools/build_ybt_arm.py from     -->
<!--   resources/robots/ybt/urdf/ybt.urdf                                  -->
<!--   resources/robots/RM65-B/urdf/RM65-B.urdf                            -->
<!-- Do not hand-edit: change the script and re-run it.                    -->
<!--                                                                       -->
<!-- Arm links and joints carry an `arm_` prefix:                          -->
<!--   base_link -> arm_base_link, link_N -> arm_link_N,                   -->
<!--   joint_N  -> arm_joint_N                                             -->
<!--                                                                       -->
<!-- The arm is bolted to the trunk's flat top deck:                       -->
<!--   arm_mount_joint   trunk -> arm_base_link                            -->
<!--                     xyz="%g %g %g" rpy="%g %g %g"                     -->
<!--                                                                       -->
<!-- Arm visuals are COLLADA, arm collision meshes are STL; both are       -->
<!-- referenced from resources/robots/RM65-B and nothing is duplicated     -->
<!-- here.  See the ybt_arm README for why the formats differ.             -->
<!-- ===================================================================== -->
""" % (MOUNT_XYZ + MOUNT_RPY)

    os.makedirs(os.path.dirname(OUT_URDF), exist_ok=True)
    with open(OUT_URDF, "w") as f:
        f.write(header)
        f.write(ET.tostring(robot, encoding="unicode"))
        f.write("\n")

    print("wrote", OUT_URDF)
    print("mount %s rpy %s" % (MOUNT_XYZ, MOUNT_RPY))
    print("arm links : %d  %s" % (len(arm_link_names), arm_link_names))
    print("arm joints: %d  %s" % (len(arm_joint_names), arm_joint_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
