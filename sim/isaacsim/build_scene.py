"""Build ``9a_pusht_simulation.usd`` (plan task P2.1) from movensys-simulation's ``3a`` scene.

Run with Isaac Sim's own Python, with system ROS 2 Jazzy sourced so the ROS 2 bridge loads:

    source /opt/ros/jazzy/setup.bash
    ~/isaacsim/6.2/python.sh sim/isaacsim/build_scene.py            # build and save only
    ~/isaacsim/6.2/python.sh sim/isaacsim/build_scene.py --play 60  # then run smoke_check.py

Run ``make_assets.py`` first (it writes the sheet texture the scene references).

What is added to the 3a scene (robot, table, room, joint-state and clock graphs stay as they are):

* ``/World/pusht``                      the sheet frame: aligned with ``world_manipulator``,
                                        origin at the push-area centre on the table top
* ``/World/pusht/target_sheet``         textured quad with the goal outline and start marks
* ``/World/pusht/t_block``              rigid T (two dark blue boxes, 0.15 kg, friction material)
                                        No markers anywhere (plan v2.1): scoring is pixel-based.
* ``.../Link6/pusher_tip``              30 mm rod along the tool axis from the Link6 flange to
                                        0.22 m, collision enabled
* ``/World/camera_top_link/camera_top`` a plain ``Camera`` with the D455 colour optics on the
                                        operator side of the table, looking obliquely back at the
                                        sheet and the robot (``--camera-sheet``,
                                        ``--camera-look-at``): 1280 x 720, 30 Hz on
                                        ``/image_top/rgb`` and ``/image_top/camera_info``
* ``/Graph/pusht_t_block_pose``         ``geometry_msgs/Pose`` of the T in the sheet frame, 30 Hz
* ``/Graph/pusht_t_block_pose_cmd``     teleports the T to a ``Pose`` and zeroes its velocity

What is taken away from the 3a robot (visuals only; the articulation and the joint topics are
unchanged): the gripper body under ``Link6/bracket`` is deactivated and the two finger bodies are
made invisible, because they hid the T from the camera. The wrist camera of earlier builds is
gone with them (2026-11-05); the dataset never recorded it.

``/Graph/ROS_Record_Camera`` and the render product it left behind are dropped (a second render
product costs frame rate; the Camera prim stays for the GUI). ``scene_9a.json`` next to the scene
records every number the ROS side needs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_BASE = (
    Path.home() / "workspaces/movensys-simulation/dobot_cr3a/3a_trajectory_simulation.usd"
)

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="3a scene to start from")
parser.add_argument("--out", type=Path, default=HERE / "9a_pusht_simulation.usd")
parser.add_argument(
    "--center-wm",
    type=float,
    nargs=2,
    default=(-0.03, -0.05),
    metavar=("X", "Y"),
    help="push-area (sheet) centre in world_manipulator (plan v2.5: 0.60 x 0.30 m area)",
)
parser.add_argument(
    "--jetson-xy",
    type=float,
    nargs=2,
    default=(0.40, 0.25),
    metavar=("X", "Y"),
    help="where to park the 3a scene's 0.20 x 0.20 m Jetson box, in world_manipulator "
    "(the 3a default (0.36, 0.18) overlaps the wide sheet)",
)
parser.add_argument(
    "--camera-sheet",
    type=float,
    nargs=3,
    default=(0.015, -0.35, 0.40),
    metavar=("X", "Y", "Z"),
    help="camera position in the sheet frame [m]: x, y on the table, z above the table top. "
    "The default is on the operator side, 5 cm beyond the table edge, 0.40 m up "
    "(decided 2026-11-05 after the straight-down view lost the T behind the forearm)",
)
parser.add_argument(
    "--camera-look-at",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("X", "Y"),
    help="point on the table top the optical axis goes through, in the sheet frame [m]",
)
parser.add_argument(
    "--tip-length",
    type=float,
    default=0.22,
    help="pusher tip rod length [m], from the Link6 flange to the tip bottom",
)
parser.add_argument(
    "--play", type=float, default=0.0, help="after saving, play for this many seconds"
)
args = parser.parse_args()

sys.path.insert(0, str(ROOT))
from isaacsim import SimulationApp  # noqa: E402

from movensys_imitation.tasks.pusht import geometry as g  # noqa: E402

app = SimulationApp({"headless": True})

import omni.graph.core as og  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
app.update()

ROBOT = "/World/movensys_manipulator"
TABLE_MESH = f"{ROBOT}/world_manipulator/collisions/table/mesh"
LINK6 = f"{ROBOT}/Link6"
PUSHT = "/World/pusht"
SHEET = f"{PUSHT}/target_sheet"
BLOCK = f"{PUSHT}/t_block"
LOOKS = f"{PUSHT}/Looks"
# Dark blue T, 0-255; the same value as make_assets.py / make_usd_assets.py.
BLOCK_RGB = (20, 40, 110)
TIP = f"{LINK6}/pusher_tip"
CAM_LINK = "/World/camera_top_link"
CAM_PRIM = f"{CAM_LINK}/camera_top"
# D455 colour optics as authored on the colour camera of Isaac's rsd455.usd asset (focal length
# and horizontal aperture in mm): 90.5 deg horizontal field of view, fx = 634 px at 1280 wide.
# Hard-coded so the scene no longer references the remote asset (its rigid body fell on Play).
D455_FOCAL_MM = 1.93
D455_H_APERTURE_MM = 3.896
# Gripper and its mounting plate, as the robot payload composes them under Link6: the meshes
# (and their colliders) are children of ``Link6/visuals`` and ``Link6/collisions``, the
# ``Link6/bracket/gripper`` Xform is an empty mount frame.
GRIPPER_PRIMS = (
    f"{LINK6}/visuals/gripper",
    f"{LINK6}/visuals/bracket",
    f"{LINK6}/collisions/gripper",
    f"{LINK6}/collisions/bracket",
    f"{LINK6}/bracket/gripper",
)
FINGERS = (f"{ROBOT}/picker_1", f"{ROBOT}/picker_2")  # finger bodies on prismatic joints
WM_IN_WORLD = Gf.Quatf(
    math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)
)  # world_manipulator is /World yawed +90 deg


def log(msg: str) -> None:
    print(f"[build_scene] {msg}", flush=True)


def wm_to_world(x: float, y: float) -> tuple[float, float]:
    """world_manipulator (x, y) to /World (x, y): a +90 degree yaw."""
    return (-y, x)


def set_xform(prim, translate, orient: Gf.Quatf | None = None, scale=(1, 1, 1)):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(orient or Gf.Quatf(1, 0, 0, 0))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*scale))


def quad(stage, path: str, x0, y0, x1, y1, z) -> UsdGeom.Mesh:
    """A textured unit-UV quad in the x-y plane; st (0, 0) at (x0, y0)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y0, z), Gf.Vec3f(x1, y1, z), Gf.Vec3f(x0, y1, z)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y1, z)])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    return mesh


def texture_material(stage, path: str, texture: str, cutout: bool = False) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, f"{path}/surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, f"{path}/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    if cutout:
        surf.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            tex.CreateOutput("a", Sdf.ValueTypeNames.Float)
        )
        surf.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(
        surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    return mat


def srgb_to_linear(rgb8) -> tuple[float, float, float]:
    """8-bit sRGB to linear floats: UsdPreviewSurface's diffuseColor is linear, so an sRGB value
    written as c / 255 renders too light (the (20, 40, 110) T came out about (78, 110, 175))."""
    out = []
    for c in rgb8:
        c = c / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def color_material(
    stage, path: str, rgb, roughness: float = 0.7, metallic: float = 0.0
) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, f"{path}/surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(
        surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    return mat


def physics_material(stage, path: str, static: float, dynamic: float) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return mat


def bind(prim, material: UsdShade.Material, purpose: str = ""):
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if purpose:
        api.Bind(material, UsdShade.Tokens.weakerThanDescendants, purpose)
    else:
        api.Bind(material)


def table_top_z(stage) -> float:
    mesh = stage.GetPrimAtPath(TABLE_MESH)
    assert mesh and mesh.GetTypeName() == "Mesh", f"{TABLE_MESH} not found in {args.base}"
    m = UsdGeom.XformCache().GetLocalToWorldTransform(mesh)
    return max(m.Transform(Gf.Vec3d(p))[2] for p in mesh.GetAttribute("points").Get())


def set_message_type(node_path: str, package: str, name: str) -> None:
    """The generic ROS2Publisher/Subscriber nodes grow their message fields when the type is set
    on an existing node (setting it at creation time does not trigger that), so do it afterwards."""
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(node_path)
    for attr, val in (
        ("messagePackage", package),
        ("messageSubfolder", "msg"),
        ("messageName", name),
    ):
        og.Controller.attribute(f"{node_path}.inputs:{attr}").set(val)
        # The controller's set() does not reach USD here, and the graph is rebuilt from USD on
        # save-as and on every later open, so author the values on the prim as well.
        prim.CreateAttribute(f"inputs:{attr}", Sdf.ValueTypeNames.String).Set(val)
    for _ in range(3):
        app.update()


def connect(src: str, dst: str) -> None:
    ends = []
    for path in (src, dst):
        for _ in range(20):
            attr = og.Controller.attribute(path)
            if attr is not None and attr.is_valid():
                break
            app.update()
        else:
            raise RuntimeError(f"OmniGraph attribute {path} never appeared")
        ends.append(attr)
    og.Controller.connect(*ends)


# ---------------------------------------------------------------------------------------- build
ctx = omni.usd.get_context()
assert args.base.exists(), args.base
assert ctx.open_stage(str(args.base)), f"could not open {args.base}"
stage = ctx.get_stage()
for _ in range(5):
    app.update()

table_z = table_top_z(stage)
cx_wm, cy_wm = args.center_wm
sheet_world = (*wm_to_world(cx_wm, cy_wm), table_z)
log(
    f"table top z = {table_z:.4f} m; sheet centre wm=({cx_wm}, {cy_wm}) "
    f"-> world={tuple(round(v, 4) for v in sheet_world)}"
)
n_room = (
    sum(1 for _ in Usd.PrimRange(stage.GetPrimAtPath("/SimpleRoom")))
    if stage.GetPrimAtPath("/SimpleRoom")
    else 0
)
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
room_box = (
    bb.ComputeWorldBound(stage.GetPrimAtPath("/SimpleRoom")).ComputeAlignedRange()
    if n_room
    else None
)
log(f"/SimpleRoom prims: {n_room}, bbox: {room_box}")

for stale in (
    "/Graph/ROS_Record_Camera",
    "/Render/OmniverseKit/HydraTextures/Replicator",
    "/Render/Vars/LdrColorSD",
):
    if stage.GetPrimAtPath(stale):
        stage.RemovePrim(
            stale
        )  # the recording camera's graph and the render product it left behind

assets_abs = HERE / "assets"
assert (assets_abs / "target_sheet.png").exists(), "missing target_sheet.png: run make_assets.py"

# Jetson box: the 3a scene puts it at (0.36, 0.18) on the table, inside the wide push area. Park
# it in the table corner (table spans x +-0.50, y +-0.35; the box is 0.20 x 0.20 x 0.07 m).
jetson = stage.GetPrimAtPath(f"{ROBOT}/world_manipulator/table/jetson_thor")
if jetson:
    t = jetson.GetAttribute("xformOp:translate")
    old = t.Get()
    t.Set(type(old)(args.jetson_xy[0], args.jetson_xy[1], old[2]))
    log(f"jetson box moved from wm ({old[0]:.2f}, {old[1]:.2f}) to {tuple(args.jetson_xy)}")
else:
    log("no jetson_thor prim in the base scene; nothing to move")

# Sheet frame.
pusht = UsdGeom.Xform.Define(stage, PUSHT)
set_xform(pusht.GetPrim(), sheet_world, WM_IN_WORLD)
UsdGeom.Scope.Define(stage, LOOKS)

# Target sheet (visual only, 0.3 mm above the table so it does not z-fight).
hw, hh = g.SHEET_SIZE[0] / 2, g.SHEET_SIZE[1] / 2
sheet = quad(stage, SHEET, -hw, -hh, hw, hh, 0.0003)
bind(
    sheet.GetPrim(),
    texture_material(stage, f"{LOOKS}/sheet_material", str(assets_abs / "target_sheet.png")),
)

# T block.
block = UsdGeom.Xform.Define(stage, BLOCK)
bprim = block.GetPrim()
x0, y0, yaw0 = g.sample_start_pose(0)
set_xform(bprim, (x0, y0, 0.001), Gf.Quatf(*g.yaw_to_quat_wxyz(yaw0)))
UsdPhysics.RigidBodyAPI.Apply(bprim)
UsdPhysics.MassAPI.Apply(bprim).CreateMassAttr(g.BLOCK_MASS)
block_vis = color_material(stage, f"{LOOKS}/block_material", srgb_to_linear(BLOCK_RGB))
block_phys = physics_material(stage, f"{LOOKS}/block_physics", 0.40, 0.35)
h = g.BLOCK_HEIGHT
for name, size, centre in (
    ("bar", (g.BAR[0], g.BAR[1], h), (0.0, g.BAR[1] / 2, h / 2)),
    ("stem", (g.STEM[0], g.STEM[1], h), (0.0, -g.STEM[1] / 2, h / 2)),
):
    cube = UsdGeom.Cube.Define(stage, f"{BLOCK}/{name}")
    cube.CreateSizeAttr(1.0)
    set_xform(cube.GetPrim(), centre, None, size)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    bind(cube.GetPrim(), block_vis)
    bind(cube.GetPrim(), block_phys, "physics")

# Gripper, mounting plate and fingers: removed from sight and from contact, since they hid the
# T from the camera (the black plate and the gripper housing sit right above the T whenever the
# tip is between the camera and the T) and the finger colliders reached down to the T's height
# and knocked it about (seen 2026-11-05 in the first v2 episodes). ``active = false`` overrides
# in this layer remove the payload's prims. The fingers are rigid bodies on prismatic joints of
# the articulation; removing the bodies would change /joint_states and the gripper service, so
# only their ``visuals`` and ``collisions`` children go and the bodies stay as empty stubs. The
# gripper and plate colliders belong to the Link6 body and never reached the T (their lowest
# point is about 0.10 m above the table at push height); they go with the visuals.
hidden = [*GRIPPER_PRIMS] + [f"{f}/{c}" for f in FINGERS for c in ("visuals", "collisions")]
for path in hidden:
    prim = stage.GetPrimAtPath(path)
    assert prim, f"{path} not found in {args.base}"
    if prim.IsInstanceProxy():
        # ``Link6/visuals`` and ``Link6/collisions`` are instanceable in the payload, and USD
        # refuses overrides on children of an instance. De-instance the root first.
        root = prim.GetParent()
        while root and not root.IsInstance():
            root = root.GetParent()
        assert root, f"{path} is an instance proxy without an instance root"
        root.SetInstanceable(False)
        prim = stage.GetPrimAtPath(path)
    prim.SetActive(False)
    assert not prim.IsActive(), f"{path} still active"
log(f"deactivated {len(hidden)} gripper, plate and finger prims")

# Table top and Jetson box as in the lab (2026-11-05): brushed silver table, black Jetson Thor.
# Both are inside the policy crop, so the sim look should match the real camera's. The payload
# binds its white or grey material on the visual Xforms with strongerThanDescendants, so the
# replacement is bound the same way on the Xform and its geometry, after de-instancing
# ``world_manipulator/visuals``. Neither colour has saturation, so the T threshold ignores them.
WM_VISUALS = f"{ROBOT}/world_manipulator/visuals"
SURFACES = {  # visual Xform -> (material name, sRGB 0-255, roughness, metallic)
    f"{WM_VISUALS}/table": ("table_silver", (185, 185, 190), 0.4, 0.8),
    f"{WM_VISUALS}/mesh_2": ("jetson_black", (20, 20, 22), 0.6, 0.0),  # the Jetson box visual
}
wm_vis = stage.GetPrimAtPath(WM_VISUALS)
if wm_vis.IsInstance():
    wm_vis.SetInstanceable(False)
for path, (mat_name, rgb, rough, metal) in SURFACES.items():
    xform = stage.GetPrimAtPath(path)
    assert xform and not xform.IsInstanceProxy(), f"{path} not editable in {args.base}"
    mat = color_material(
        stage, f"{LOOKS}/{mat_name}", srgb_to_linear(rgb), roughness=rough, metallic=metal
    )
    for prim in (xform, *xform.GetChildren()):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
log("table top silver, Jetson box black")

# Pusher tip under Link6: one rod along the tool axis (+z of Link6) from the flange to the tip
# bottom. With the gripper gone it starts at the Link6 origin; the bottom stays at 0.22 m so the
# guard's z setpoint and task.yaml's link6_to_tip_bottom are unchanged from the gripper build.
TIP_RADIUS = 0.015  # 30 mm rod, one gym-pusht T unit, as the paper's agent (plan v2.6)
tip_top = 0.0  # Link6 flange
tip_offset = tip_top + args.tip_length  # Link6 origin to tip bottom
tip = UsdGeom.Cylinder.Define(stage, TIP)
tip.CreateRadiusAttr(TIP_RADIUS)
tip.CreateHeightAttr(args.tip_length)
tip.CreateAxisAttr("Z")
tip.CreateExtentAttr(
    [
        Gf.Vec3f(-TIP_RADIUS, -TIP_RADIUS, -args.tip_length / 2),
        Gf.Vec3f(TIP_RADIUS, TIP_RADIUS, args.tip_length / 2),
    ]
)
set_xform(tip.GetPrim(), (0.0, 0.0, tip_top + args.tip_length / 2))
UsdPhysics.CollisionAPI.Apply(tip.GetPrim())
bind(tip.GetPrim(), color_material(stage, f"{LOOKS}/tip_material", (0.15, 0.15, 0.18)))
bind(tip.GetPrim(), physics_material(stage, f"{LOOKS}/tip_physics", 0.30, 0.25), "physics")

# Top camera: an oblique view from the operator side of the table. ``camera_top_link`` carries the
# pose (a USD camera looks along its -z with +y up, so the link frame is the camera frame) and
# the plain ``camera_top`` child carries the optics at identity. The image up direction is the
# sheet +y direction (toward the robot) projected into the image plane, so the robot base is at
# the top of the picture as before and image +x is sheet +x.
cam_sheet = Gf.Vec3d(*args.camera_sheet)
cam_pos = Gf.Vec3d(*wm_to_world(cam_sheet[0], cam_sheet[1]), 0.0) + Gf.Vec3d(
    sheet_world[0], sheet_world[1], table_z + cam_sheet[2]
)
look_at = Gf.Vec3d(*wm_to_world(*args.camera_look_at), 0.0) + Gf.Vec3d(*sheet_world)
view = (look_at - cam_pos).GetNormalized()
up_hint = Gf.Vec3d(*wm_to_world(0.0, 1.0), 0.0)  # sheet +y in /World
up = (up_hint - Gf.Dot(up_hint, view) * view).GetNormalized()
right = Gf.Cross(view, up)
# Gf matrices act on row vectors, so the rows are the camera axes in /World.
m_cam = Gf.Matrix3d(*right, *up, *(-view))
q_cam = m_cam.ExtractRotation().GetQuat()
cam_link = UsdGeom.Xform.Define(stage, CAM_LINK)
set_xform(cam_link.GetPrim(), cam_pos, Gf.Quatf(q_cam.GetReal(), Gf.Vec3f(*q_cam.GetImaginary())))
cam = UsdGeom.Camera.Define(stage, CAM_PRIM)
cam.CreateFocalLengthAttr(D455_FOCAL_MM)
cam.CreateHorizontalApertureAttr(D455_H_APERTURE_MM)
cam.CreateVerticalApertureAttr(D455_H_APERTURE_MM * 720 / 1280)  # square pixels at 1280 x 720
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
app.update()
m_world = UsdGeom.XformCache().GetLocalToWorldTransform(cam.GetPrim())
got_view = m_world.TransformDir(Gf.Vec3d(0, 0, -1))
got_up = m_world.TransformDir(Gf.Vec3d(0, 1, 0))
assert (got_view - view).GetLength() < 1e-6 and (got_up - up).GetLength() < 1e-6, (
    f"camera orientation did not round-trip: view {got_view} vs {view}, up {got_up} vs {up}"
)
cam_pitch_deg = math.degrees(math.acos(-view[2]))  # 0 = straight down
hfov_deg = 2 * math.degrees(math.atan(D455_H_APERTURE_MM / 2 / D455_FOCAL_MM))
log(
    f"top camera at sheet {tuple(cam_sheet)} -> world {tuple(round(v, 3) for v in cam_pos)}, "
    f"looking at sheet {tuple(args.camera_look_at)}, {cam_pitch_deg:.1f} deg from vertical, "
    f"view dir {tuple(round(v, 3) for v in got_view)}, image up {tuple(round(v, 3) for v in got_up)}, "
    f"HFOV {hfov_deg:.1f} deg"
)

# --------------------------------------------------------------------------------------- graphs
keys = og.Controller.Keys


def camera_graph(graph_path: str, cam_prim: str, namespace: str, frame_id: str) -> None:
    """RGB + CameraInfo publisher for ``cam_prim`` at 1280 x 720, every other frame (30 Hz)."""
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.SET_VALUES: [
                ("Context.inputs:useDomainIDEnvVar", True),
                ("RenderProduct.inputs:cameraPrim", cam_prim),
                ("RenderProduct.inputs:width", 1280),
                ("RenderProduct.inputs:height", 720),
                ("RGBPublish.inputs:type", "rgb"),
                ("RGBPublish.inputs:topicName", f"{namespace}/rgb"),
                ("RGBPublish.inputs:frameId", frame_id),
                ("RGBPublish.inputs:frameSkipCount", 1),
                ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                ("CameraInfoPublish.inputs:topicName", f"{namespace}/camera_info"),
                ("CameraInfoPublish.inputs:frameId", frame_id),
                ("CameraInfoPublish.inputs:frameSkipCount", 1),
                ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                ("Context.outputs:context", "RGBPublish.inputs:context"),
                ("RenderProduct.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
                (
                    "RenderProduct.outputs:renderProductPath",
                    "CameraInfoPublish.inputs:renderProductPath",
                ),
                ("Context.outputs:context", "CameraInfoPublish.inputs:context"),
            ],
        },
    )


camera_graph("/Graph/ROS_Camera_Top", CAM_PRIM, "/image_top", "camera_top_color_optical_frame")

# T pose publisher: every second physics tick (60 Hz physics -> 30 Hz), pose in the sheet frame.
pub = "/Graph/pusht_t_block_pose"
og.Controller.edit(
    {"graph_path": pub, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("Gate", "isaacsim.core.nodes.IsaacSimulationGate"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("ReadTranslate", "omni.graph.nodes.ReadPrimAttribute"),
            ("ReadOrient", "omni.graph.nodes.ReadPrimAttribute"),
            ("BreakTranslate", "omni.graph.nodes.BreakVector3"),
            # Stock nodes cannot split a quaternion, but the block only yaws, so publish
            # (0, 0, sin(yaw/2), cos(yaw/2)) from the Euler z angle of the local orient.
            ("OrientToEuler", "omni.graph.nodes.GetMatrix4Rotation"),
            ("BreakEuler", "omni.graph.nodes.BreakVector3"),
            ("YawDouble", "omni.graph.nodes.ToDouble"),  # the Euler angles come out as float32
            ("Half", "omni.graph.nodes.ConstantDouble"),
            ("HalfYaw", "omni.graph.nodes.Multiply"),
            ("SinHalfYaw", "omni.graph.nodes.Sin"),
            ("CosHalfYaw", "omni.graph.nodes.Cos"),
            ("Publisher", "isaacsim.ros2.bridge.ROS2Publisher"),
        ],
        keys.SET_VALUES: [
            ("Gate.inputs:step", 2),
            ("OrientToEuler.inputs:rotationOrder", "XYZ"),
            ("Half.inputs:value", 0.5),
            ("Context.inputs:useDomainIDEnvVar", True),
            ("ReadTranslate.inputs:prim", BLOCK),
            ("ReadTranslate.inputs:name", "xformOp:translate"),
            ("ReadOrient.inputs:prim", BLOCK),
            ("ReadOrient.inputs:name", "xformOp:orient"),
            ("Publisher.inputs:topicName", "/pusht/t_block/pose"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "Gate.inputs:execIn"),
            ("Gate.outputs:execOut", "Publisher.inputs:execIn"),
            ("Context.outputs:context", "Publisher.inputs:context"),
            ("OnPlaybackTick.outputs:time", "ReadTranslate.inputs:usdTimecode"),
            ("OnPlaybackTick.outputs:time", "ReadOrient.inputs:usdTimecode"),
            ("ReadTranslate.outputs:value", "BreakTranslate.inputs:tuple"),
            ("ReadOrient.outputs:value", "OrientToEuler.inputs:matrix"),
            ("OrientToEuler.outputs:rotation", "BreakEuler.inputs:tuple"),
            ("BreakEuler.outputs:z", "YawDouble.inputs:value"),
            ("YawDouble.outputs:converted", "HalfYaw.inputs:a"),
            ("Half.inputs:value", "HalfYaw.inputs:b"),
            ("HalfYaw.outputs:product", "SinHalfYaw.inputs:value"),
            ("HalfYaw.outputs:product", "CosHalfYaw.inputs:value"),
        ],
    },
)
app.update()
set_message_type(f"{pub}/Publisher", "geometry_msgs", "Pose")
for axis in "xyz":
    connect(f"{pub}/BreakTranslate.outputs:{axis}", f"{pub}/Publisher.inputs:position:{axis}")
connect(f"{pub}/SinHalfYaw.outputs:value", f"{pub}/Publisher.inputs:orientation:z")
connect(f"{pub}/CosHalfYaw.outputs:value", f"{pub}/Publisher.inputs:orientation:w")

# T teleport: a Pose on /pusht/t_block/pose_cmd sets the transform and zeroes both velocities.
sub = "/Graph/pusht_t_block_pose_cmd"
og.Controller.edit(
    {"graph_path": sub, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("Subscriber", "isaacsim.ros2.bridge.ROS2Subscriber"),
            ("MakeTranslate", "omni.graph.nodes.MakeVector3"),
            (
                "MakeOrient",
                "omni.graph.nodes.MakeVector4",
            ),  # OmniGraph quaternions are (x, y, z, w)
            ("ZeroLinear", "omni.graph.nodes.ConstantFloat3"),
            ("ZeroAngular", "omni.graph.nodes.ConstantFloat3"),
            ("WriteTranslate", "omni.graph.nodes.WritePrimAttribute"),
            ("WriteOrient", "omni.graph.nodes.WritePrimAttribute"),
            ("WriteLinear", "omni.graph.nodes.WritePrimAttribute"),
            ("WriteAngular", "omni.graph.nodes.WritePrimAttribute"),
        ],
        keys.SET_VALUES: [
            ("Context.inputs:useDomainIDEnvVar", True),
            ("Subscriber.inputs:topicName", "/pusht/t_block/pose_cmd"),
            ("ZeroLinear.inputs:value", [0.0, 0.0, 0.0]),
            ("ZeroAngular.inputs:value", [0.0, 0.0, 0.0]),
            ("WriteTranslate.inputs:prim", BLOCK),
            ("WriteTranslate.inputs:name", "xformOp:translate"),
            ("WriteOrient.inputs:prim", BLOCK),
            ("WriteOrient.inputs:name", "xformOp:orient"),
            ("WriteLinear.inputs:prim", BLOCK),
            ("WriteLinear.inputs:name", "physics:velocity"),
            ("WriteAngular.inputs:prim", BLOCK),
            ("WriteAngular.inputs:name", "physics:angularVelocity"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "Subscriber.inputs:execIn"),
            ("Context.outputs:context", "Subscriber.inputs:context"),
            ("Subscriber.outputs:execOut", "WriteTranslate.inputs:execIn"),
            ("WriteTranslate.outputs:execOut", "WriteOrient.inputs:execIn"),
            ("WriteOrient.outputs:execOut", "WriteLinear.inputs:execIn"),
            ("WriteLinear.outputs:execOut", "WriteAngular.inputs:execIn"),
            ("MakeTranslate.outputs:tuple", "WriteTranslate.inputs:value"),
            ("MakeOrient.outputs:tuple", "WriteOrient.inputs:value"),
            ("ZeroLinear.inputs:value", "WriteLinear.inputs:value"),
            ("ZeroAngular.inputs:value", "WriteAngular.inputs:value"),
        ],
    },
)
app.update()
set_message_type(f"{sub}/Subscriber", "geometry_msgs", "Pose")
for axis in "xyz":
    connect(f"{sub}/Subscriber.outputs:position:{axis}", f"{sub}/MakeTranslate.inputs:{axis}")
for axis in "xyzw":
    connect(f"{sub}/Subscriber.outputs:orientation:{axis}", f"{sub}/MakeOrient.inputs:{axis}")
# The rigid body needs authored velocity attributes for the writes to land on something.
UsdPhysics.RigidBodyAPI(bprim).CreateVelocityAttr(Gf.Vec3f(0, 0, 0))
UsdPhysics.RigidBodyAPI(bprim).CreateAngularVelocityAttr(Gf.Vec3f(0, 0, 0))
app.update()

# ----------------------------------------------------------------------------------------- save
args.out.parent.mkdir(parents=True, exist_ok=True)
assert ctx.save_as_stage(str(args.out)), f"save failed: {args.out}"
stage = ctx.get_stage()
stage.GetPrimAtPath(f"{LOOKS}/sheet_material/texture").GetAttribute("inputs:file").Set(
    "./assets/target_sheet.png"
)
stage.GetRootLayer().Save()
log(f"saved {args.out}")

eef_z_wm = table_z + 0.010 + tip_offset
meta = {
    "scene": args.out.name,
    "base_scene": str(args.base),
    "frames": {
        "sheet_prim": PUSHT,
        "sheet_in_world_manipulator": {"xy": [cx_wm, cy_wm], "z": table_z, "yaw": 0.0},
        "world_manipulator_in_World": {"xy": [0.0, 0.0], "z": 0.0, "yaw_deg": 90.0},
        "robot_base_in_world_manipulator": [-0.03, 0.275, 0.05],
    },
    "table_top_z_world_manipulator": table_z,
    "push_area": {
        "size_m": list(g.SHEET_SIZE),
        "axes": "(width x, depth y)",
        "center_wm": [cx_wm, cy_wm],
    },
    "jetson_box_wm": {
        "xy": list(args.jetson_xy),
        "size_m": [0.20, 0.20, 0.07],
        "note": "moved from the 3a default (0.36, 0.18); the manipulator URDF still has it there",
    },
    "block_prim": BLOCK,
    "block_rest_z_in_sheet": 0.0,
    "markers": "none (plan v2.1); real scoring is in overhead pixel space",
    "block_rgb": list(BLOCK_RGB),
    "teleport": {"topic": "/pusht/t_block/pose_cmd", "frame": "sheet", "z_hint": 0.001},
    "pose_topic": {"topic": "/pusht/t_block/pose", "frame": "sheet", "rate_hz": 30},
    "pusher_tip": {
        "prim": TIP,
        "radius_m": TIP_RADIUS,
        "length_m": args.tip_length,
        "link6_to_tip_bottom_m": tip_offset,
        "eef_z_setpoint_wm": eef_z_wm,
        "note": "Link6 z in world_manipulator that puts the tip bottom 10 mm above the table",
    },
    "camera": {
        "link_prim": CAM_LINK,
        "camera_prim": CAM_PRIM,
        "view": "oblique from the operator side, looking back toward the robot (2026-11-05)",
        "position_sheet_m": list(cam_sheet),
        "look_at_sheet_m": list(args.camera_look_at),
        "height_above_table_m": cam_sheet[2],
        "pitch_from_vertical_deg": cam_pitch_deg,
        "position_world_m": list(cam_pos),
        "optics": {
            "focal_length_mm": D455_FOCAL_MM,
            "horizontal_aperture_mm": D455_H_APERTURE_MM,
            "hfov_deg": hfov_deg,
            "fx_px": 1280 * D455_FOCAL_MM / D455_H_APERTURE_MM,
        },
        "topics": ["/image_top/rgb", "/image_top/camera_info"],
        "frame_id": "camera_top_color_optical_frame",
        "resolution": [1280, 720],
        "rate_hz": 30,
        "image_axes": "image +x = sheet +x; image up = sheet +y projected (robot at the top)",
    },
    "appearance": {
        "table_top": {"rgb": [185, 185, 190], "roughness": 0.4, "metallic": 0.8, "note": "brushed silver, as the lab table"},
        "jetson_box": {"rgb": [20, 20, 22], "roughness": 0.6, "metallic": 0.0, "note": "black Jetson Thor, as in the lab"},
        "robot": "payload white (real CR3A is white with grey joint caps)",
    },
    "hidden_robot_parts": {
        "deactivated": hidden,
        "note": "finger bodies, joints and /joint_states are unchanged; gripper, plate and "
        "fingers no longer render or collide",
    },
    "graphs": [
        pub,
        sub,
        "/Graph/ROS_Camera_Top",
        "/Graph/ROS_JointStates",
        "/Graph/ROS_Clock",
    ],
}
meta_path = args.out.with_name("scene_9a.json")
meta_path.write_text(json.dumps(meta, indent=2) + "\n")
log(f"wrote {meta_path}; EEF z setpoint in world_manipulator = {eef_z_wm:.4f} m")

# ----------------------------------------------------------------------------------------- play
if args.play > 0:
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app.update()
    log(f"PLAYING for {args.play:.0f} s")
    t0 = time.time()
    frames = 0
    next_report = 2.0
    while time.time() - t0 < args.play:
        app.update()
        frames += 1
        elapsed = time.time() - t0
        if elapsed > next_report:
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(BLOCK))
            t = xf.GetOrderedXformOps()[0].Get()
            log(
                f"t={elapsed:5.1f}s frames={frames} ({frames / elapsed:.0f} fps) "
                f"block xyz=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})"
            )
            probes = {
                "clock.simTime": "/Graph/ROS_Clock/ReadSimTime.outputs:simulationTime",
                "pose.quat.w": f"{pub}/CosHalfYaw.outputs:value",
                "pose.pos.x": f"{pub}/BreakTranslate.outputs:x",
                "camera.renderProduct": (
                    "/Graph/ROS_Camera_Top/RenderProduct.outputs:renderProductPath"
                ),
            }
            vals = {}
            for k, path in probes.items():
                try:
                    vals[k] = og.Controller.get(og.Controller.attribute(path))
                except Exception as e:  # noqa: BLE001
                    vals[k] = f"ERR {e}"
            log(f"graph probes: {vals}")
            next_report += 10.0
    timeline.stop()
    log("stopped")

app.close()
