"""Build ``9a_pusht_simulation.usd`` (plan task P2.1) from movensys-simulation's ``3a`` scene.

Run with Isaac Sim's own Python, with system ROS 2 Jazzy sourced so the ROS 2 bridge loads:

    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim/python.sh sim/isaacsim/build_scene.py            # build and save only
    ~/isaac-sim/python.sh sim/isaacsim/build_scene.py --play 60     # then run smoke_check.py

Run ``make_assets.py`` first (it writes the sheet texture the scene references).

What is added to the 3a scene (robot, table, room, joint-state and clock graphs stay as they are):

* ``/World/pusht``                      the sheet frame: aligned with ``world_manipulator``,
                                        origin at the push-area centre on the table top
* ``/World/pusht/target_sheet``         textured quad with the goal outline and start marks
* ``/World/pusht/t_block``              rigid T (two dark blue boxes, 0.25 kg, friction material)
                                        No markers anywhere (plan v2.1): scoring is pixel-based.
* ``.../Link6/pusher_tip``              20 mm cylinder along the tool axis, collision enabled
* ``/World/camera_top_link``            RealSense D455 model looking straight down; the render
                                        camera is the plain ``camera_top`` child: 1280 x 720,
                                        30 Hz on ``/image_top/rgb`` and ``/image_top/camera_info``
* ``.../Link6/bracket/camera_hand_link`` wrist D455 model at the ``7a`` scene's mount pose, its
                                        rigid body removed; the render camera is the plain
                                        ``camera_hand`` child: 1280 x 720, 30 Hz on
                                        ``/image_hand/rgb`` and ``/image_hand/camera_info``
* ``/Graph/pusht_t_block_pose``         ``geometry_msgs/Pose`` of the T in the sheet frame, 30 Hz
* ``/Graph/pusht_t_block_pose_cmd``     teleports the T to a ``Pose`` and zeroes its velocity

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
    "--camera-height", type=float, default=0.70, help="camera above the table top [m]"
)
parser.add_argument("--tip-length", type=float, default=0.08, help="pusher tip cylinder length [m]")
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
ASSET_CAM_PRIM = (
    f"{CAM_LINK}/RSD455/Camera_OmniVision_OV9782_Color"  # pose and optics reference only
)
CAM_PRIM = f"{CAM_LINK}/camera_top"
HAND_LINK = f"{LINK6}/bracket/camera_hand_link"
HAND_ASSET_CAM_PRIM = f"{HAND_LINK}/RSD455/Camera_OmniVision_OV9782_Color"
HAND_CAM_PRIM = f"{HAND_LINK}/camera_hand"
# Wrist D455 mount pose in the Link6 bracket frame, copied from 7a_robopoly_simulation.usd.
HAND_IN_BRACKET_T = (0.0, -0.08, 0.038)
HAND_IN_BRACKET_Q = Gf.Quatf(0.5, 0.5, -0.5, 0.5)
RSD455_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Sensors/Intel/RealSense/rsd455.usd"
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


def color_material(stage, path: str, rgb) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, f"{path}/surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
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

# Pusher tip under Link6: a cylinder along the tool axis (+z of Link6), protruding past the fingers.
tip_top = 0.14  # where the gripper body ends along Link6 +z (from the Link6 visual extent)
tip_offset = tip_top + args.tip_length  # Link6 origin to tip bottom
tip = UsdGeom.Cylinder.Define(stage, TIP)
tip.CreateRadiusAttr(0.010)
tip.CreateHeightAttr(args.tip_length)
tip.CreateAxisAttr("Z")
tip.CreateExtentAttr(
    [Gf.Vec3f(-0.01, -0.01, -args.tip_length / 2), Gf.Vec3f(0.01, 0.01, args.tip_length / 2)]
)
set_xform(tip.GetPrim(), (0.0, 0.0, tip_top + args.tip_length / 2))
UsdPhysics.CollisionAPI.Apply(tip.GetPrim())
bind(tip.GetPrim(), color_material(stage, f"{LOOKS}/tip_material", (0.15, 0.15, 0.18)))
bind(tip.GetPrim(), physics_material(stage, f"{LOOKS}/tip_physics", 0.30, 0.25), "physics")

# Overhead camera: D455 model, optical axis straight down, image +x = sheet +x, image up = sheet +y.
cam_link = UsdGeom.Xform.Define(stage, CAM_LINK)
cam_link.GetPrim().GetReferences().AddReference(RSD455_URL)
set_xform(cam_link.GetPrim(), (sheet_world[0], sheet_world[1], table_z + args.camera_height))
for _ in range(5):
    app.update()
asset_cam = stage.GetPrimAtPath(ASSET_CAM_PRIM)
assert asset_cam and asset_cam.GetTypeName() == "Camera", (
    f"{ASSET_CAM_PRIM} missing (asset not loaded?)"
)
q_cam_in_link = UsdGeom.XformCache().GetLocalToWorldTransform(asset_cam).ExtractRotationQuat()
q_desired = Gf.Quatd(
    WM_IN_WORLD.GetReal(), Gf.Vec3d(*WM_IN_WORLD.GetImaginary())
)  # camera +x = world +y, +y = world -x, -z down
want = {"x": Gf.Vec3d(0, 1, 0), "y": Gf.Vec3d(-1, 0, 0), "z": Gf.Vec3d(0, 0, 1)}
best = None
for q_link in (q_desired * q_cam_in_link.GetInverse(), q_cam_in_link.GetInverse() * q_desired):
    r = Gf.Rotation(q_link)
    rc = Gf.Rotation(q_cam_in_link)
    axes = {
        k: r.TransformDir(rc.TransformDir(v))
        for k, v in {"x": Gf.Vec3d(1, 0, 0), "y": Gf.Vec3d(0, 1, 0), "z": Gf.Vec3d(0, 0, 1)}.items()
    }
    err = sum((axes[k] - want[k]).GetLength() for k in want)
    if best is None or err < best[0]:
        best = (err, q_link)
assert best[0] < 1e-3, f"could not orient the camera (error {best[0]})"
q_link = best[1]
set_xform(
    cam_link.GetPrim(),
    (sheet_world[0], sheet_world[1], table_z + args.camera_height),
    Gf.Quatf(q_link.GetReal(), Gf.Vec3f(*q_link.GetImaginary())),
)
app.update()


def render_camera(link, asset_cam, path: str, label: str) -> UsdGeom.Camera:
    """A plain Camera prim under ``link`` at the asset colour camera's pose, with its optics.

    The asset's own camera prims are marked invisible, and a render product on one of them shows
    only the dome backdrop (visible, it shows the inside of the lens housing), so the asset stays
    a visual prop only.
    """
    plain = UsdGeom.Camera.Define(stage, path)
    m_rel = (
        UsdGeom.XformCache().GetLocalToWorldTransform(asset_cam)
        * UsdGeom.XformCache().GetLocalToWorldTransform(link).GetInverse()
    )
    plain.AddTransformOp().Set(m_rel)
    h_ap = asset_cam.GetAttribute("horizontalAperture").Get()
    plain.CreateFocalLengthAttr(asset_cam.GetAttribute("focalLength").Get())
    plain.CreateHorizontalApertureAttr(h_ap)
    plain.CreateVerticalApertureAttr(h_ap * 720 / 1280)  # square pixels at 1280 x 720
    plain.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    app.update()
    m_cam = UsdGeom.XformCache().GetLocalToWorldTransform(plain.GetPrim())
    log(
        f"{label} camera world pos {tuple(round(v, 3) for v in m_cam.ExtractTranslation())}, "
        f"view dir {tuple(round(v, 3) for v in m_cam.TransformDir(Gf.Vec3d(0, 0, -1)))}, "
        f"image up {tuple(round(v, 3) for v in m_cam.TransformDir(Gf.Vec3d(0, 1, 0)))}"
    )
    return plain


render_camera(cam_link.GetPrim(), asset_cam, CAM_PRIM, "top")

# Wrist camera: the 7a scene's D455 mount on the Link6 bracket, same local pose. The asset root
# carries a dynamic rigid body, which PhysX rejects when nested inside the Link6 body (and which
# falls under gravity when it is not), so the body and mass schemas are removed and the model
# rides on the link as plain geometry.
hand_link = UsdGeom.Xform.Define(stage, HAND_LINK)
hand_link.GetPrim().GetReferences().AddReference(RSD455_URL)
set_xform(hand_link.GetPrim(), HAND_IN_BRACKET_T, HAND_IN_BRACKET_Q)
for _ in range(5):
    app.update()
hand_asset_cam = stage.GetPrimAtPath(HAND_ASSET_CAM_PRIM)
assert hand_asset_cam and hand_asset_cam.GetTypeName() == "Camera", (
    f"{HAND_ASSET_CAM_PRIM} missing (asset not loaded?)"
)
hand_model = stage.GetPrimAtPath(f"{HAND_LINK}/RSD455")
for schema in ("PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"):
    hand_model.RemoveAppliedSchema(schema)
assert not hand_model.HasAPI(UsdPhysics.RigidBodyAPI), "hand D455 still has a rigid body"
render_camera(hand_link.GetPrim(), hand_asset_cam, HAND_CAM_PRIM, "hand")

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
camera_graph(
    "/Graph/ROS_Camera_Hand", HAND_CAM_PRIM, "/image_hand", "camera_hand_color_optical_frame"
)

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
        "radius_m": 0.010,
        "length_m": args.tip_length,
        "link6_to_tip_bottom_m": tip_offset,
        "eef_z_setpoint_wm": eef_z_wm,
        "note": "Link6 z in world_manipulator that puts the tip bottom 10 mm above the table",
    },
    "camera": {
        "link_prim": CAM_LINK,
        "camera_prim": CAM_PRIM,
        "model_prim": ASSET_CAM_PRIM,
        "height_above_table_m": args.camera_height,
        "topics": ["/image_top/rgb", "/image_top/camera_info"],
        "frame_id": "camera_top_color_optical_frame",
        "resolution": [1280, 720],
        "rate_hz": 30,
        "image_axes": "image +x = sheet +x, image up = sheet +y (robot at the top of the image)",
    },
    "camera_hand": {
        "link_prim": HAND_LINK,
        "camera_prim": HAND_CAM_PRIM,
        "model_prim": HAND_ASSET_CAM_PRIM,
        "mount_in_link6_bracket": {
            "xyz": list(HAND_IN_BRACKET_T),
            "quat_wxyz": [HAND_IN_BRACKET_Q.GetReal(), *HAND_IN_BRACKET_Q.GetImaginary()],
            "source": "7a_robopoly_simulation.usd",
        },
        "topics": ["/image_hand/rgb", "/image_hand/camera_info"],
        "frame_id": "camera_hand_color_optical_frame",
        "resolution": [1280, 720],
        "rate_hz": 30,
    },
    "graphs": [
        pub,
        sub,
        "/Graph/ROS_Camera_Top",
        "/Graph/ROS_Camera_Hand",
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
                "hand.renderProduct": (
                    "/Graph/ROS_Camera_Hand/RenderProduct.outputs:renderProductPath"
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
