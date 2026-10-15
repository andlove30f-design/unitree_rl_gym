"""Check the folded arm pose used by the `ybt_arm` task, straight from the URDF.

Read-only, no sim: parses ybt_arm.urdf, runs forward kinematics over the arm's joint
chain, and reads the bounding boxes of the seven arm collision meshes (binary STL) to
report, for the pose you ask about:

  * the lowest point of any arm collision geometry, against the top of the trunk's
    collision box -- below that the arm is inside its own body;
  * the arm's mass centre and its second moment about the trunk origin, which is what
    folding is for (the upright zero pose is already statically balanced, so the fold
    buys inertia and height, not torque);
  * the static gravity torque each arm joint has to hold.

The angles come out of legged_gym/envs/ybt/ybt_arm_config.py, so this cannot drift
away from the task. Run it after changing them:

    python resources/robots/ybt_arm/tools/folded_pose.py           # the task's pose
    python resources/robots/ybt_arm/tools/folded_pose.py --grid    # search q2/q3/q5
    python resources/robots/ybt_arm/tools/folded_pose.py 1.0 2.0 0.8   # try angles

The numbers it prints for the default pose are the ones quoted in README.md.
"""
import argparse
import ast
import itertools
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(os.path.dirname(HERE), "urdf", "ybt_arm.urdf")
CONFIG = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "legged_gym", "envs", "ybt", "ybt_arm_config.py"))
# top face of the trunk's visual deck plate (from trunk.dae); arm_base_link is bolted
# onto it, so it is the height a folded arm has to stay above
DECK_TOP = 0.1012
G = 9.81
ARM_JOINT_PREFIX = "arm_joint_"


def parse_xyz(text, default=(0.0, 0.0, 0.0)):
    return default if text is None else tuple(float(v) for v in text.split())


def rpy_to_R(r, p, y):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec(A, v):
    return tuple(sum(A[i][k] * v[k] for k in range(3)) for i in range(3))


def add(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def rot_axis(R, axis, angle):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    Rax = [[x * x * C + c, x * y * C - z * s, x * z * C + y * s],
           [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
           [z * x * C - y * s, z * y * C + x * s, z * z * C + c]]
    return mat_mul(R, Rax)


def stl_bbox(path):
    """Axis-aligned bounds of a binary STL's vertices."""
    with open(path, "rb") as handle:
        data = handle.read()
    facets = struct.unpack_from("<I", data, 80)[0]
    lo, hi = [1e9] * 3, [-1e9] * 3
    for facet in range(facets):
        offset = 84 + facet * 50 + 12
        for corner in range(3):
            vertex = struct.unpack_from("<3f", data, offset + corner * 12)
            for axis in range(3):
                lo[axis] = min(lo[axis], vertex[axis])
                hi[axis] = max(hi[axis], vertex[axis])
    return tuple(lo), tuple(hi)


def corners(lo, hi):
    return list(itertools.product(*zip(lo, hi)))


class Model(object):
    """Kinematics, masses and collision geometry of the assembled asset."""

    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        base = os.path.dirname(os.path.abspath(urdf))
        self.links, self.joints = {}, {}
        for link in root.findall("link"):
            name = link.get("name")
            inertial = link.find("inertial")
            mass = float(inertial.find("mass").get("value")) if inertial is not None else 0.0
            com = parse_xyz(inertial.find("origin").get("xyz")) if inertial is not None \
                and inertial.find("origin") is not None else (0.0, 0.0, 0.0)
            boxes = []
            for collision in link.findall("collision"):
                origin = collision.find("origin")
                xyz = parse_xyz(origin.get("xyz")) if origin is not None else (0.0, 0.0, 0.0)
                rpy = parse_xyz(origin.get("rpy")) if origin is not None else (0.0, 0.0, 0.0)
                geometry = collision.find("geometry")
                mesh, box = geometry.find("mesh"), geometry.find("box")
                cylinder, sphere = geometry.find("cylinder"), geometry.find("sphere")
                if mesh is not None:
                    bounds = stl_bbox(os.path.join(base, mesh.get("filename")))
                elif box is not None:
                    half = [float(v) / 2 for v in box.get("size").split()]
                    bounds = (tuple(-h for h in half), tuple(half))
                elif cylinder is not None:
                    r, length = float(cylinder.get("radius")), float(cylinder.get("length"))
                    bounds = ((-r, -r, -length / 2), (r, r, length / 2))
                else:
                    r = float(sphere.get("radius"))
                    bounds = ((-r,) * 3, (r,) * 3)
                boxes.append((xyz, rpy, bounds))
            self.links[name] = dict(mass=mass, com=com, boxes=boxes)
        for joint in root.findall("joint"):
            origin, axis = joint.find("origin"), joint.find("axis")
            self.joints[joint.get("name")] = dict(
                parent=joint.find("parent").get("link"), child=joint.find("child").get("link"),
                type=joint.get("type"),
                xyz=parse_xyz(origin.get("xyz")) if origin is not None else (0.0, 0.0, 0.0),
                rpy=parse_xyz(origin.get("rpy")) if origin is not None else (0.0, 0.0, 0.0),
                axis=parse_xyz(axis.get("xyz")) if axis is not None else (1.0, 0.0, 0.0))
        self.parents = {j["child"]: j["parent"] for j in self.joints.values()}
        self.arm_links = sorted((n for n in self.links if n.startswith("arm_link_")),
                                key=lambda n: int(n.rsplit("_", 1)[1]))

    def arm_joint_names(self):
        return sorted((n for n in self.joints if n.startswith(ARM_JOINT_PREFIX)),
                      key=lambda n: int(n.rsplit("_", 1)[1]))

    def is_distal(self, link, ancestor):
        """True if `link` is at or below `ancestor` in the tree."""
        cursor = link
        while cursor in self.parents:
            if cursor == ancestor:
                return True
            cursor = self.parents[cursor]
        return cursor == ancestor

    def fk(self, angles, root="trunk"):
        """{joint: angle} -> {link: (rotation, translation)} in the root frame."""
        poses = {root: ([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], (0.0, 0.0, 0.0))}
        pending = list(self.joints.items())
        while pending:
            remaining = []
            for name, joint in pending:
                if joint["parent"] not in poses:
                    remaining.append((name, joint))
                    continue
                Rp, tp = poses[joint["parent"]]
                rotation = mat_mul(Rp, rpy_to_R(*joint["rpy"]))
                translation = add(tp, mat_vec(Rp, joint["xyz"]))
                if joint["type"] in ("revolute", "continuous"):
                    rotation = rot_axis(rotation, joint["axis"], angles.get(name, 0.0))
                poses[joint["child"]] = (rotation, translation)
            if not remaining:
                break
            if len(remaining) == len(pending):
                raise SystemExit("kinematic tree did not resolve: " + str([n for n, _ in remaining]))
            pending = remaining
        return poses


def trunk_collision_top(model):
    trunk = model.links["trunk"]
    return max(add(xyz, mat_vec(rpy_to_R(*rpy), corner))[2]
               for xyz, rpy, (lo, hi) in trunk["boxes"] for corner in corners(lo, hi))


def evaluate(model, angles):
    poses = model.fk(angles)
    trunk_top = trunk_collision_top(model)

    # arm_link_1 only turns with arm_joint_1, so it sits still through a
    # joint_2/3/5 fold; track the links that actually move separately
    lowest, moving_lowest = None, None
    footprint = [1e9, 1e9, -1e9, -1e9]
    for name in model.arm_links:
        rotation, translation = poses[name]
        link_min = 1e9
        for xyz, rpy, (lo, hi) in model.links[name]["boxes"]:
            Rg = mat_mul(rotation, rpy_to_R(*rpy))
            tg = add(translation, mat_vec(rotation, xyz))
            for corner in corners(lo, hi):
                world = add(tg, mat_vec(Rg, corner))
                link_min = min(link_min, world[2])
                footprint[0] = min(footprint[0], world[0])
                footprint[1] = min(footprint[1], world[1])
                footprint[2] = max(footprint[2], world[0])
                footprint[3] = max(footprint[3], world[1])
        lowest = (link_min, name) if lowest is None or link_min < lowest[0] else lowest
        if name != "arm_link_1":
            moving_lowest = (link_min, name) if moving_lowest is None or link_min < moving_lowest[0] \
                else moving_lowest

    mass = sum(model.links[n]["mass"] for n in model.arm_links)
    com = [0.0, 0.0, 0.0]
    points = {}
    for name in model.arm_links:
        rotation, translation = poses[name]
        point = add(translation, mat_vec(rotation, model.links[name]["com"]))
        points[name] = point
        for axis in range(3):
            com[axis] += model.links[name]["mass"] * point[axis]
    com = tuple(c / mass for c in com)
    inertia = sum(model.links[n]["mass"] * sum(c ** 2 for c in points[n]) for n in model.arm_links)

    torques = {}
    distal = {}
    for name in model.arm_joint_names():
        joint = model.joints[name]
        Rp, tp = poses[joint["parent"]]
        origin = add(tp, mat_vec(Rp, joint["xyz"]))
        axis_world = mat_vec(mat_mul(Rp, rpy_to_R(*joint["rpy"])), joint["axis"])
        norm = math.sqrt(sum(v * v for v in axis_world))
        axis_world = tuple(v / norm for v in axis_world)
        torque = 0.0
        for link, point in points.items():
            # what the joint actually carries: the links at and below its child
            if not model.is_distal(link, joint["child"]):
                continue
            distal[name] = distal.get(name, 0.0) + model.links[link]["mass"]
            r = tuple(point[d] - origin[d] for d in range(3))
            force = (0.0, 0.0, -model.links[link]["mass"] * G)
            cross = (r[1] * force[2] - r[2] * force[1],
                     r[2] * force[0] - r[0] * force[2],
                     r[0] * force[1] - r[1] * force[0])
            torque += sum(axis_world[d] * cross[d] for d in range(3))
        torques[name] = torque
    return dict(trunk_top=trunk_top, lowest=lowest, moving_lowest=moving_lowest,
                footprint=tuple(footprint), com=com, mass=mass, inertia=inertia,
                torques=torques, distal=distal)


def report(model, angles, label):
    result = evaluate(model, angles)
    print("%s: %s" % (label, " ".join("%s=%+.2f" % (k.replace(ARM_JOINT_PREFIX, "q"), v)
                                      for k, v in sorted(angles.items())) or "zero pose"))
    print("  lowest arm collision point %+.3f m (%s), moving links only %+.3f m"
          % (result["lowest"][0], result["lowest"][1], result["moving_lowest"][0]))
    print("  trunk collision box top   %+.3f m   deck plate top %+.3f m"
          % (result["trunk_top"], DECK_TOP))
    print("  footprint x=[%+.3f,%+.3f] y=[%+.3f,%+.3f] m"
          % (result["footprint"][0], result["footprint"][2],
             result["footprint"][1], result["footprint"][3]))
    print("  moving arm mass %.3f kg (arm_link_1..6, without the bolted base)  centre of mass "
          "(%+.3f,%+.3f,%+.3f)  second moment %.3f kg m^2"
          % (result["mass"], result["com"][0], result["com"][1], result["com"][2],
             result["inertia"]))
    print("  static gravity torque: " + " ".join(
        "%s=%+.2f" % (k.replace(ARM_JOINT_PREFIX, "j"), v) for k, v in result["torques"].items()))
    print("  mass carried by each joint: " + " ".join(
        "%s=%.2fkg" % (k.replace(ARM_JOINT_PREFIX, "j"), v) for k, v in result["distal"].items()))
    clear = result["lowest"][0] - result["trunk_top"]
    print("  clearance above the trunk box: %+.3f m %s" % (clear, "OK" if clear > 0 else "COLLIDES"))
    print()
    return result


def configured_angles(path=CONFIG):
    """The six arm angles in YBTFlatArmCfg.init_state.default_joint_angles."""
    with open(path) as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "default_joint_angles"
                   for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        # the legs come in via `**YBTFlatCfg...`, whose value is None here
        return {k.value: float(v.value)
                for k, v in zip(node.value.keys, node.value.values)
                if k is not None and v is not None and k.value.startswith(ARM_JOINT_PREFIX)}
    raise SystemExit("could not find default_joint_angles in " + path)


def search(model):
    """Grid search q2/q3/q5 for a low-inertia pose that clears the trunk."""
    trunk_top = trunk_collision_top(model)
    qs = [round(-2.0 + 0.1 * i, 2) for i in range(41)]
    best = []
    for q2 in qs:
        for q3 in qs:
            for q5 in (0.0, -0.4, -0.8, 0.4, 0.8):
                angles = {"arm_joint_2": q2, "arm_joint_3": q3, "arm_joint_5": q5}
                r = evaluate(model, angles)
                if r["moving_lowest"][0] < trunk_top + 0.04:
                    continue
                if r["footprint"][0] < -0.40 or r["footprint"][2] > 0.42:
                    continue
                if abs(r["footprint"][1]) > 0.18 or abs(r["footprint"][3]) > 0.18:
                    continue
                if max(abs(v) for v in r["torques"].values()) > 15.0:
                    continue
                # keep the arm near the middle of the back in x, then minimise inertia
                best.append((r["inertia"] + 8.0 * r["com"][0] ** 2, q2, q3, q5, r))
    best.sort(key=lambda t: t[0])
    print("clear of the trunk box (top %+.3f), inside the deck, |tau| < 15 N m: %d candidates"
          % (trunk_top, len(best)))
    for _, q2, q3, q5, r in best[:12]:
        print("  q2=%+.2f q3=%+.2f q5=%+.2f  lowest=%+.3f com=(%+.3f,%+.3f,%+.3f) I=%.3f "
              "j2=%+.1f j3=%+.1f" % (q2, q3, q5, r["lowest"][0], r["com"][0], r["com"][1],
                                     r["com"][2], r["inertia"],
                                     r["torques"]["arm_joint_2"], r["torques"]["arm_joint_3"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("angles", nargs="*", type=float,
                        help="arm_joint_2 arm_joint_3 arm_joint_5 (default: the task's)")
    parser.add_argument("--grid", action="store_true", help="search q2/q3/q5 instead")
    parser.add_argument("--urdf", default=URDF)
    args = parser.parse_args(argv)
    model = Model(args.urdf)

    if args.grid:
        report(model, {}, "zero pose")
        search(model)
        return 0

    if args.angles:
        if len(args.angles) != 3:
            parser.error("expected three angles: arm_joint_2 arm_joint_3 arm_joint_5")
        angles = dict(zip(("arm_joint_2", "arm_joint_3", "arm_joint_5"), args.angles))
        label = "asked about"
    else:
        angles = configured_angles()
        label = "ybt_arm task pose"
    report(model, {}, "zero pose")
    result = report(model, angles, label)
    if result["moving_lowest"][0] <= result["trunk_top"]:
        print("FAIL: the arm intersects the trunk's collision box in this pose")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
