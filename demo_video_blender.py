# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "webcam_smplx",
    "author" : "brusselslee",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "Viewport > Right panel",
    "warning" : "",
    "category" : "webcam_smplx"
}

from collections import deque
import functools
import socket
import threading
import bpy
import queue
from bpy_extras.io_utils import ImportHelper,ExportHelper # ImportHelper/ExportHelper is a helper class, defines filename and invoke() function which calls the file selector.
from mathutils import Vector, Quaternion
from bpy.props import ( BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty )
from bpy.types import ( Context, Event, PropertyGroup )
import cv2

import json
from math import radians
import numpy as np
import os
import pickle

# SMPL-X globals
SMPLX_MODELFILE = "smplx_model_20210421.blend"
SMPLX_MODELFILE_300 = "smplx_model_300_20220615.blend"

SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15
# End SMPL-X globals

def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues

def update_corrective_poseshapes(self, context):
    if self.smplx_corrective_poseshapes:
        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
    else:
        bpy.ops.object.smplx_reset_poseshapes('EXEC_DEFAULT')

def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        # SMPL-X is adding the reference rodrigues rotation to the relaxed hand rodrigues rotation, so we have to do the same here.
        # This means that pose values for relaxed hand model cannot be interpreted as rotations in the local joint coordinate system of the relaxed hand.
        # https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py#L1190
        #   full_pose += self.pose_mean
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

        """
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        angle_rad_reference = rod_reference.length
        axis_reference = rod_reference.normalized()
        quat_reference = Quaternion(axis_reference, angle_rad_reference)

        # Rotate first into reference pose and then add the target pose
        armature.pose.bones[bone_name].rotation_quaternion = quat_reference @ quat
        """
    return

# Property groups for UI
class PG_WEBCAMProperties(PropertyGroup):

    smplx_gender: EnumProperty(
        name = "Model",
        description = "SMPL-X model",
        items = [ ("female", "Female", ""), ("male", "Male", ""), ("neutral", "Neutral", "") ]
    )

    smplx_texture: EnumProperty(
        name = "",
        description = "SMPL-X model texture",
        items = [ ("NONE", "None", ""), ("smplx_texture_f_alb.png", "Female", ""), ("smplx_texture_m_alb.png", "Male", ""), ("smplx_texture_rainbow.png", "Rainbow", ""), ("UV_GRID", "UV Grid", ""), ("COLOR_GRID", "Color Grid", "") ]
    )

    smplx_corrective_poseshapes: BoolProperty(
        name = "Corrective Pose Shapes",
        description = "Enable/disable corrective pose shapes of SMPL-X model",
        update = update_corrective_poseshapes,
        default = True
    )

    smplx_handpose: EnumProperty(
        name = "",
        description = "SMPL-X hand pose",
        items = [ ("relaxed", "Relaxed", ""), ("flat", "Flat", "") ]
    )

    smplx_height: FloatProperty(name="Target Height [m]", default=1.70, min=1.4, max=2.2)

    smplx_weight: FloatProperty(name="Target Weight [kg]", default=60, min=40, max=110)

    # smplx_webcam : BoolProperty(
    #     name="Webcam", 
    #     description='use webcam to capture face to smplx',
    #     default=False
    #     )
    


class AddGender(bpy.types.Operator):
    bl_idname = "scene.smplx_add_gender"
    bl_label = "Add"
    bl_description = ("Add SMPL-X model of selected gender to scene")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in Object Mode
            if (context.active_object is None) or (context.active_object.mode == 'OBJECT'):
                return True
            else: 
                return False
        except: return False

    def execute(self, context):
        gender = context.window_manager.webcam_tool.smplx_gender
        print("Adding gender: " + gender)

        path = os.path.dirname(os.path.realpath(__file__))

        # Use 300 shape model by default if available
        model_path = os.path.join(path, "data", SMPLX_MODELFILE_300)
        if os.path.exists(model_path):
            model_file = SMPLX_MODELFILE_300
        else:
            model_file = SMPLX_MODELFILE

        objects_path = os.path.join(path, "data", model_file, "Object")
        object_name = "SMPLX-mesh-" + gender

        bpy.ops.wm.append(filename=object_name, directory=str(objects_path))

        # Select imported mesh
        object_name = context.selected_objects[0].name
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = bpy.data.objects[object_name]
        bpy.data.objects[object_name].select_set(True)

        # Set currently selected hand pose
        bpy.ops.object.smplx_set_handpose('EXEC_DEFAULT')

        return {'FINISHED'}

class SetTexture(bpy.types.Operator):
    bl_idname = "object.smplx_set_texture"
    bl_label = "Set"
    bl_description = ("Set selected texture")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in active object is mesh
            if (context.object.type == 'MESH'):
                return True
            else:
                return False
        except: return False

    def execute(self, context):
        texture = context.window_manager.webcam_tool.smplx_texture
        print("Setting texture: " + texture)

        obj = bpy.context.object
        if (len(obj.data.materials) == 0) or (obj.data.materials[0] is None):
            self.report({'WARNING'}, "Selected mesh has no material: %s" % obj.name)
            return {'CANCELLED'}

        mat = obj.data.materials[0]
        links = mat.node_tree.links
        nodes = mat.node_tree.nodes

        # Find texture node
        node_texture = None
        for node in nodes:
            if node.type == 'TEX_IMAGE':
                node_texture = node
                break

        # Find shader node
        node_shader = None
        for node in nodes:
            if node.type.startswith('BSDF'):
                node_shader = node
                break

        if texture == 'NONE':
            # Unlink texture node
            if node_texture is not None:
                for link in node_texture.outputs[0].links:
                    links.remove(link)

                nodes.remove(node_texture)

                # 3D Viewport still shows previous texture when texture link is removed via script.
                # As a workaround we trigger desired viewport update by setting color value.
                node_shader.inputs[0].default_value = node_shader.inputs[0].default_value
        else:
            if node_texture is None:
                node_texture = nodes.new(type="ShaderNodeTexImage")

            if (texture == 'UV_GRID') or (texture == 'COLOR_GRID'):
                if texture not in bpy.data.images:
                    bpy.ops.image.new(name=texture, generated_type=texture)
                image = bpy.data.images[texture]
            else:
                if texture not in bpy.data.images:
                    path = os.path.dirname(os.path.realpath(__file__))
                    texture_path = os.path.join(path, "data", texture)
                    image = bpy.data.images.load(texture_path)
                else:
                    image = bpy.data.images[texture]

            node_texture.image = image

            # Link texture node to shader node if not already linked
            if len(node_texture.outputs[0].links) == 0:
                links.new(node_texture.outputs[0], node_shader.inputs[0])

        # Switch viewport shading to Material Preview to show texture
        if bpy.context.space_data:
            if bpy.context.space_data.type == 'VIEW_3D':
                bpy.context.space_data.shading.type = 'MATERIAL'

        return {'FINISHED'}

class MeasurementsToShape(bpy.types.Operator):
    bl_idname = "object.smplx_measurements_to_shape"
    bl_label = "Measurements To Shape"
    bl_description = ("Calculate and set shape parameters for specified measurements")
    bl_options = {'REGISTER', 'UNDO'}

    betas_regressor_female = None
    betas_regressor_male = None
    betas_regressor_neutral = None

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        if self.betas_regressor_female is None:
            path = os.path.dirname(os.path.realpath(__file__))
            regressor_path = os.path.join(path, "data", "smplx_measurements_to_betas_female.json")
            with open(regressor_path) as f:
                data = json.load(f)
                self.betas_regressor_female = (np.asarray(data["A"]).reshape(-1, 2), np.asarray(data["B"]).reshape(-1, 1))

        if self.betas_regressor_male is None:
            path = os.path.dirname(os.path.realpath(__file__))
            regressor_path = os.path.join(path, "data", "smplx_measurements_to_betas_male.json")
            with open(regressor_path) as f:
                data = json.load(f)
                self.betas_regressor_male = (np.asarray(data["A"]).reshape(-1, 2), np.asarray(data["B"]).reshape(-1, 1))

        if self.betas_regressor_neutral is None:
            path = os.path.dirname(os.path.realpath(__file__))
            regressor_path = os.path.join(path, "data", "smplx_measurements_to_betas_neutral.json")
            with open(regressor_path) as f:
                data = json.load(f)
                self.betas_regressor_neutral = (np.asarray(data["A"]).reshape(-1, 2), np.asarray(data["B"]).reshape(-1, 1))

        if "female" in obj.name.lower():
            (A, B) = self.betas_regressor_female
        elif "male" in obj.name.lower():
            (A, B) = self.betas_regressor_male
        elif "neutral" in obj.name.lower():
            (A, B) = self.betas_regressor_neutral
        else:
            self.report({"ERROR"}, f"Cannot derive gender from mesh object name: {obj.name}")
            return {"CANCELLED"}

        # Calculate beta values from measurements
        height_m = context.window_manager.webcam_tool.smplx_height
        height_cm = height_m * 100.0
        weight_kg = context.window_manager.webcam_tool.smplx_weight

        v_root = pow(weight_kg, 1.0/3.0)
        measurements = np.asarray([[height_cm], [v_root]])
        betas = A @ measurements + B

        num_betas = betas.shape[0]
        for i in range(num_betas):
            name = f"Shape{i:03d}"
            key_block = obj.data.shape_keys.key_blocks[name]
            value = betas[i, 0]

            # Adjust key block min/max range to value
            if value < key_block.slider_min:
                key_block.slider_min = value
            elif value > key_block.slider_max:
                key_block.slider_max = value

            key_block.value = value

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}

class RandomShape(bpy.types.Operator):
    bl_idname = "object.smplx_random_shape"
    bl_label = "Random"
    bl_description = ("Sets all shape blend shape keys to a random value")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        randomized_betas = 0
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                beta = np.random.normal(0.0, 1.0)
                beta = np.clip(beta, -1.0, 1.0)
                key_block.value = beta

                randomized_betas += 1
                if randomized_betas >= 16:
                    break

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}

class ResetShape(bpy.types.Operator):
    bl_idname = "object.smplx_reset_shape"
    bl_label = "Reset"
    bl_description = ("Resets all blend shape keys for shape")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                key_block.value = 0.0

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}

class RandomExpressionShape(bpy.types.Operator):
    bl_idname = "object.smplx_random_expression_shape"
    bl_label = "Random Face Expression"
    bl_description = ("Sets all face expression blend shape keys to a random value")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Exp"):
                key_block.value = np.random.uniform(-2, 2)

        return {'FINISHED'}

class ResetExpressionShape(bpy.types.Operator):
    bl_idname = "object.smplx_reset_expression_shape"
    bl_label = "Reset"
    bl_description = ("Resets all blend shape keys for face expression")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Exp"):
                key_block.value = 0.0

        return {'FINISHED'}

class SnapGroundPlane(bpy.types.Operator):
    bl_idname = "object.smplx_snap_ground_plane"
    bl_label = "Snap To Ground Plane"
    bl_description = ("Snaps mesh to the XY ground plane")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return ((context.object.type == 'MESH') or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')

        obj = bpy.context.object
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        # Get vertices with applied skin modifier in object coordinates
        depsgraph = context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)
        mesh_from_eval = object_eval.to_mesh()

        # Get vertices in world coordinates
        matrix_world = obj.matrix_world
        vertices_world = [matrix_world @ vertex.co for vertex in mesh_from_eval.vertices]
        z_min = (min(vertices_world, key=lambda item: item.z)).z
        object_eval.to_mesh_clear() # Remove temporary mesh

        # Adjust height of armature so that lowest vertex is on ground plane.
        # Do not apply new armature location transform so that we are later able to show loaded poses at their desired height.
        armature.location.z = armature.location.z - z_min

        return {'FINISHED'}

class UpdateJointLocations(bpy.types.Operator):
    bl_idname = "object.smplx_update_joint_locations"
    bl_label = "Update Joint Locations"
    bl_description = ("Update joint locations after shape/expression changes")
    bl_options = {'REGISTER', 'UNDO'}

    j_regressor_female = { 10: None, 300: None }
    j_regressor_male = { 10: None, 300: None }
    j_regressor_neutral = { 10: None, 300: None }

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False

    def load_regressor(self, gender, betas):
        path = os.path.dirname(os.path.realpath(__file__))
        if betas == 10:
            suffix = ""
        elif betas == 300:
            suffix = "_300"
        else:
            print(f"ERROR: No betas-to-joints regressor for desired beta shapes [{betas}]")
            return (None, None)

        regressor_path = os.path.join(path, "data", f"smplx_betas_to_joints_{gender}{suffix}.json")
        with open(regressor_path) as f:
            data = json.load(f)
            return (np.asarray(data["betasJ_regr"]), np.asarray(data["template_J"]))

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        # Get beta shapes
        betas = []
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                betas.append(key_block.value)
        num_betas = len(betas)
        betas = np.array(betas)

        # Cache regressor files on first call
        for target_betas in [10, 300]:
            if self.j_regressor_female[target_betas] is None:
                self.j_regressor_female[target_betas] = self.load_regressor("female", target_betas)

            if self.j_regressor_male[target_betas] is None:
                self.j_regressor_male[target_betas] = self.load_regressor("male", target_betas)

            if self.j_regressor_neutral[target_betas] is None:
                self.j_regressor_neutral[target_betas] = self.load_regressor("neutral", target_betas)

        if "female" in obj.name:
            (betas_to_joints, template_j) = self.j_regressor_female[num_betas]
        elif "male" in obj.name:
            (betas_to_joints, template_j) = self.j_regressor_male[num_betas]
        else:
            (betas_to_joints, template_j) = self.j_regressor_neutral[num_betas]

        joint_locations = betas_to_joints @ betas + template_j

        # Set new bone joint locations
        armature = obj.parent
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')

        for index in range(NUM_SMPLX_JOINTS):
            bone = armature.data.edit_bones[SMPLX_JOINT_NAMES[index]]
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.1)

            # Convert SMPL-X joint locations to Blender joint locations
            joint_location_smplx = joint_locations[index]
            bone_start = Vector( (joint_location_smplx[0], -joint_location_smplx[2], joint_location_smplx[1]) )
            bone.translate(bone_start)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = obj

        return {'FINISHED'}

class SetPoseshapes(bpy.types.Operator):
    bl_idname = "object.smplx_set_poseshapes"
    bl_label = "Update Pose Shapes"
    bl_description = ("Sets and updates corrective poseshapes for current pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
    def rodrigues_to_mat(self, rotvec):
        theta = np.linalg.norm(rotvec)
        r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
        cost = np.cos(theta)
        mat = np.asarray([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]], dtype=object)
        return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Calculate weights of pose corrective blend shapes
    # Input is pose of all 55 joints, output is weights for all joints except pelvis
    def rodrigues_to_posecorrective_weight(self, pose):
        joints_posecorrective = NUM_SMPLX_JOINTS
        rod_rots = np.asarray(pose).reshape(joints_posecorrective, 3)
        mat_rots = [self.rodrigues_to_mat(rod_rot) for rod_rot in rod_rots]
        bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
        return(bshapes)

    def execute(self, context):
        obj = bpy.context.object

        # Get armature pose in rodrigues representation
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        pose = [0.0] * (NUM_SMPLX_JOINTS * 3)

        for index in range(NUM_SMPLX_JOINTS):
            joint_name = SMPLX_JOINT_NAMES[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        poseweights = self.rodrigues_to_posecorrective_weight(pose)

        # Set weights for pose corrective shape keys
        for index, weight in enumerate(poseweights):
            obj.data.shape_keys.key_blocks["Pose%03d" % index].value = weight

        # Set checkbox without triggering update function
        context.window_manager.webcam_tool["smplx_corrective_poseshapes"] = True

        return {'FINISHED'}

class ResetPoseshapes(bpy.types.Operator):
    bl_idname = "object.smplx_reset_poseshapes"
    bl_label = "Reset"
    bl_description = ("Resets corrective poseshapes for current pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'ARMATURE':
            obj = bpy.context.object.children[0]

        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Pose"):
                key_block.value = 0.0

        return {'FINISHED'}

class SetHandpose(bpy.types.Operator):
    bl_idname = "object.smplx_set_handpose"
    bl_label = "Set"
    bl_description = ("Set selected hand pose")
    bl_options = {'REGISTER', 'UNDO'}

    hand_poses = None

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        if self.hand_poses is None:
            path = os.path.dirname(os.path.realpath(__file__))
            data_path = os.path.join(path, "data", "smplx_handposes.npz")
            with np.load(data_path, allow_pickle=True) as data:
                self.hand_poses = data["hand_poses"].item()

        hand_pose_name = context.window_manager.webcam_tool.smplx_handpose
        print("Setting hand pose: " + hand_pose_name)

        if hand_pose_name not in self.hand_poses:
            self.report({"ERROR"}, f"Desired hand pose not existing: {hand_pose_name}")
            return {"CANCELLED"}

        (left_hand_pose, right_hand_pose) = self.hand_poses[hand_pose_name]

        hand_pose = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(-1, 3)

        hand_joint_start_index = 1 + NUM_SMPLX_BODYJOINTS + 3
        for index in range(2 * NUM_SMPLX_HANDJOINTS):
            pose_rodrigues = hand_pose[index]            
            bone_name = SMPLX_JOINT_NAMES[index + hand_joint_start_index]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

        # Update corrective poseshapes if used
        if context.window_manager.webcam_tool.smplx_corrective_poseshapes:
            bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

        return {'FINISHED'}

class WritePose(bpy.types.Operator):
    bl_idname = "object.smplx_write_pose"
    bl_label = "Write Pose To Console"
    bl_description = ("Writes SMPL-X flat hand pose thetas to console window")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return (context.object.type == 'MESH') or (context.object.type == 'ARMATURE')
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        # Get armature pose in rodrigues representation
        pose = [0.0] * (NUM_SMPLX_JOINTS * 3)

        for index in range(NUM_SMPLX_JOINTS):
            joint_name = SMPLX_JOINT_NAMES[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        print("\npose = " + str(pose))

        return {'FINISHED'}

class ResetPose(bpy.types.Operator):
    bl_idname = "object.smplx_reset_pose"
    bl_label = "Reset Pose"
    bl_description = ("Resets pose to default zero pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        for bone in armature.pose.bones:
            if bone.rotation_mode != 'QUATERNION':
                bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = Quaternion()

        # Reset corrective pose shapes
        bpy.ops.object.smplx_reset_poseshapes('EXEC_DEFAULT')

        return {'FINISHED'}

class LoadPose(bpy.types.Operator, ImportHelper):
    '''
    pose *.pkl file format use 
    'betas', 'global_orient', 'transl', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'body_pose', 'keypoints_3d', 'pose_embedding', 'gender', 'v'
    e.g agora dataset
    '''
    
    bl_idname = "object.smplx_load_pose"
    bl_label = "Load Pose"
    bl_description = ("Load relaxed-hand model pose from file")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(
        default="*.pkl",
        options={'HIDDEN'}
    )

    update_shape: BoolProperty(
        name="Update shape parameters",
        description="Update shape parameters using the beta shape information in the loaded file",
        default=True
    )

    hand_pose_relaxed = None

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj
            obj = armature.children[0]
            context.view_layer.objects.active = obj # mesh needs to be active object for recalculating joint locations

        if self.hand_pose_relaxed is None:
            path = os.path.dirname(os.path.realpath(__file__))
            data_path = os.path.join(path, "data", "smplx_handposes.npz")
            with np.load(data_path, allow_pickle=True) as data:
                hand_poses = data["hand_poses"].item()
                (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
                self.hand_pose_relaxed = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(-1, 3)

        print("Loading: " + self.filepath)

        translation = None
        global_orient = None
        body_pose = None
        jaw_pose = None
        #leye_pose = None
        #reye_pose = None
        left_hand_pose = None
        right_hand_pose = None
        betas = None
        expression = None
        with open(self.filepath, "rb") as f:
            data = pickle.load(f, encoding="latin1")

            if "transl" in data:
                translation = np.array(data["transl"]).reshape(3)

            if "global_orient" in data:
                global_orient = np.array(data["global_orient"]).reshape(3)

            body_pose = np.array(data["body_pose"])
            if body_pose.shape != (1, NUM_SMPLX_BODYJOINTS * 3):
                print(f"Invalid body pose dimensions: {body_pose.shape}")
                body_data = None
                return {'CANCELLED'}

            body_pose = np.array(data["body_pose"]).reshape(NUM_SMPLX_BODYJOINTS, 3)

            jaw_pose = np.array(data["jaw_pose"]).reshape(3)
            #leye_pose = np.array(data["leye_pose"]).reshape(3)
            #reye_pose = np.array(data["reye_pose"]).reshape(3)
            left_hand_pose = np.array(data["left_hand_pose"]).reshape(-1, 3)
            right_hand_pose = np.array(data["right_hand_pose"]).reshape(-1, 3)

            betas = np.array(data["betas"]).reshape(-1).tolist()
            expression = np.array(data["expression"]).reshape(-1).tolist()

        # Update shape if selected
        if self.update_shape:
            bpy.ops.object.mode_set(mode='OBJECT')
            for index, beta in enumerate(betas):
                key_block_name = f"Shape{index:03}"

                if key_block_name in obj.data.shape_keys.key_blocks:
                    obj.data.shape_keys.key_blocks[key_block_name].value = beta
                else:
                    print(f"ERROR: No key block for: {key_block_name}")

            bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        if global_orient is not None:
            set_pose_from_rodrigues(armature, "pelvis", global_orient)

        for index in range(NUM_SMPLX_BODYJOINTS):
            pose_rodrigues = body_pose[index]
            bone_name = SMPLX_JOINT_NAMES[index + 1] # body pose starts with left_hip
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

        set_pose_from_rodrigues(armature, "jaw", jaw_pose)

        # Left hand
        start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3
        for i in range(0, NUM_SMPLX_HANDJOINTS):
            pose_rodrigues = left_hand_pose[i]
            bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
            pose_relaxed_rodrigues = self.hand_pose_relaxed[i]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

        # Right hand
        start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3 + NUM_SMPLX_HANDJOINTS
        for i in range(0, NUM_SMPLX_HANDJOINTS):
            pose_rodrigues = right_hand_pose[i]
            bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
            pose_relaxed_rodrigues = self.hand_pose_relaxed[NUM_SMPLX_HANDJOINTS + i]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

        if translation is not None:
            # Set translation
            armature.location = (translation[0], -translation[2], translation[1])

        # Activate corrective poseshapes
        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

        # Set face expression
        for index, exp in enumerate(expression):
            key_block_name = f"Exp{index:03}"

            if key_block_name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[key_block_name].value = exp
            else:
                print(f"ERROR: No key block for: {key_block_name}")

        return {'FINISHED'}

class AddAnimation(bpy.types.Operator, ImportHelper):
    bl_idname = "object.smplx_add_animation"
    bl_label = "Add Animation"
    bl_description = ("Load AMASS/SMPL-X animation and create animated SMPL-X body")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(
        default="*.npz",
        options={'HIDDEN'}
    )

    anim_format: EnumProperty(
        name="Format",
        items=(
            ("AMASS", "AMASS", ""),
            ("SMPL-X", "SMPL-X", ""),
        ),
    )

    rest_position: EnumProperty(
        name="Rest position",
        items=(
            ("SMPL-X", "SMPL-X", "Use default SMPL-X rest position (feet below the floor)"),
            ("GROUNDED", "Grounded", "Use feet-on-floor rest position"),
        ),
    )

    keyframe_corrective_pose_weights: BoolProperty(
        name="Use keyframed corrective pose weights",
        description="Keyframe the weights of the corrective pose shapes for each frame. This increases animation load time and slows down editor real-time playback.",
        default=False
    )

    target_framerate: IntProperty(
        name="Target framerate [fps]",
        description="Target framerate for animation in frames-per-second. Lower values will speed up import time.",
        default=30,
        min = 1,
        max = 120
    )

    @classmethod
    def poll(cls, context):
        try:
            # Always enable button
            return True
        except: return False

    def execute(self, context):

        target_framerate = self.target_framerate

        # Load .npz file
        print("Loading: " + self.filepath)
        with np.load(self.filepath) as data:
            # Check for valid AMASS file
            if ("trans" not in data) or ("gender" not in data) or (("mocap_frame_rate" not in data) and ("mocap_framerate" not in data)) or ("betas" not in data) or ("poses" not in data):
                self.report({"ERROR"}, "Invalid AMASS animation data file")
                return {"CANCELLED"}

            trans = data["trans"]
            gender = str(data["gender"])
            mocap_framerate = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else int(data["mocap_framerate"])
            betas = data["betas"]
            poses = data["poses"]

            if mocap_framerate < target_framerate:
                self.report({"ERROR"}, f"Mocap framerate ({mocap_framerate}) below target framerate ({target_framerate})")
                return {"CANCELLED"}

        if (context.active_object is not None):
            bpy.ops.object.mode_set(mode='OBJECT')

        # Add gender specific model
        context.window_manager.webcam_tool.smplx_gender = gender
        context.window_manager.webcam_tool.smplx_handpose = "flat"
        bpy.ops.scene.smplx_add_gender()

        obj = context.view_layer.objects.active
        armature = obj.parent

        # Append animation name to armature name
        armature.name = armature.name + "_" + os.path.basename(self.filepath).replace(".npz", "")

        context.scene.render.fps = target_framerate
        context.scene.frame_start = 1

        # Set shape and update joint locations
        bpy.ops.object.mode_set(mode='OBJECT')
        for index, beta in enumerate(betas):
            key_block_name = f"Shape{index:03}"

            if key_block_name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[key_block_name].value = beta
            else:
                print(f"ERROR: No key block for: {key_block_name}")

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        height_offset = 0
        if self.rest_position == "GROUNDED":
            bpy.ops.object.smplx_snap_ground_plane('EXEC_DEFAULT')
            height_offset = armature.location[2]

            # Apply location offsets to armature and skinned mesh
            bpy.context.view_layer.objects.active = armature
            armature.select_set(True)
            obj.select_set(True)
            bpy.ops.object.transform_apply(location = True, rotation=False, scale=False) # apply to selected objects
            armature.select_set(False)

            # Fix root bone location
            bpy.ops.object.mode_set(mode='EDIT')
            bone = armature.data.edit_bones["root"]
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.1)
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = obj

        # Keyframe poses
        step_size = int(mocap_framerate / target_framerate)

        num_frames = trans.shape[0]
        num_keyframes = int(num_frames / step_size)

        if self.keyframe_corrective_pose_weights:
            print(f"Adding pose keyframes with keyframed corrective pose weights: {num_keyframes}")
        else:
            print(f"Adding pose keyframes: {num_keyframes}")

        if len(bpy.data.actions) == 0:
            # Set end frame if we don't have any previous animations in the scene
            context.scene.frame_end = num_keyframes
        elif num_keyframes > context.scene.frame_end:
            context.scene.frame_end = num_keyframes

        for index, frame in enumerate(range(0, num_frames, step_size)):
            if (index % 100) == 0:
                print(f"  {index}/{num_keyframes}")
            current_frame = index + 1
            current_pose = poses[frame].reshape(-1, 3)
            current_trans = trans[frame]
            for index, bone_name in enumerate(SMPLX_JOINT_NAMES):
                if bone_name == "pelvis":
                    # Keyframe pelvis location

                    if self.rest_position == "GROUNDED":
                        current_trans[1] = current_trans[1] - height_offset # SMPL-X local joint coordinates are Y-Up

                    armature.pose.bones[bone_name].location = Vector((current_trans[0], current_trans[1], current_trans[2]))
                    armature.pose.bones[bone_name].keyframe_insert('location', frame=current_frame)

                # Keyframe bone rotation
                pose_rodrigues = current_pose[index]
                set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)
                armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

            if self.keyframe_corrective_pose_weights:
                # Calculate corrective poseshape weights for current pose and keyframe them.
                # Note: This significantly increases animation load time and also reduces real-time playback speed in Blender viewport.
                bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
                for key_block in obj.data.shape_keys.key_blocks:
                    if key_block.name.startswith("Pose"):
                        key_block.keyframe_insert("value", frame=current_frame)

        if self.anim_format == "AMASS":
            # AMASS target floor is XY ground plane for SMPL-X template in OpenGL Y-up space (XZ ground plane).
            # Since SMPL-X Blender model is Z-up (and not Y-up) for rest/template pose, we need to adjust root node rotation to ensure that the resulting animated body is on Blender XY ground plane.
            bone_name = "root"
            if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
                armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'
            armature.pose.bones[bone_name].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), radians(-90))
            armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=1)

        print(f"  {num_keyframes}/{num_keyframes}")
        context.scene.frame_set(1)

        return {'FINISHED'}

class ExportAlembic(bpy.types.Operator, ExportHelper):
    bl_idname = "object.smplx_export_alembic"
    bl_label = "Export Alembic ABC"
    bl_description = ("Export as Alembic geometry cache")
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".abc"

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return (context.object.type == 'MESH')
        except: return False

    def execute(self, context):
        bpy.ops.wm.alembic_export(filepath=self.filepath, selected=True, packuv=False, face_sets=True)
        print("Exported: " + self.filepath)

        return {'FINISHED'}

class ExportFBX(bpy.types.Operator, ExportHelper):
    bl_idname = "object.smplx_export_fbx"
    bl_label = "Export FBX"
    bl_description = ("Export skinned mesh in FBX format")
    bl_options = {'REGISTER', 'UNDO'}

    # ExportHelper mixin class uses this
    filename_ext = ".fbx"

    export_shape_keys: EnumProperty(
        name = "Blend Shapes",
        description = "Blend shape export settings",
        items = [ ("SHAPE_POSE", "All: Shape + Posecorrectives", "Export shape keys for body shape and pose correctives"), ("SHAPE", "Reduced: Shape space only", "Export only shape keys for body shape"), ("NONE", "None: Apply shape space", "Do not export any shape keys, shape keys for body shape will be baked into mesh") ],
    )


    target_format: EnumProperty(
        name="Format",
        items=(
            ("UNITY", "Unity", ""),
            ("UNREAL", "Unreal", ""),
        ),
    )

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return (context.object.type == 'MESH')
        except: return False

    def execute(self, context):

        obj = bpy.context.object

        armature_original = obj.parent
        skinned_mesh_original = obj

        # Operate on temporary copy of skinned mesh and armature
        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh_original.select_set(True)
        armature_original.select_set(True)
        bpy.context.view_layer.objects.active = skinned_mesh_original
        bpy.ops.object.duplicate()
        skinned_mesh = bpy.context.object
        armature = skinned_mesh.parent

        # Apply armature object location to armature root bone and skinned mesh so that armature and skinned mesh are at origin before export
        context.view_layer.objects.active = armature
        armature_offset = Vector(armature.location)
        armature.location = (0, 0, 0)
        bpy.ops.object.mode_set(mode='EDIT')
        for edit_bone in armature.data.edit_bones:
            if edit_bone.name != "root":
                edit_bone.translate(armature_offset)

        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = skinned_mesh
        mesh_location = Vector(skinned_mesh.location)
        skinned_mesh.location = mesh_location + armature_offset
        bpy.ops.object.transform_apply(location = True)

        # Reset pose
        bpy.ops.object.smplx_reset_pose('EXEC_DEFAULT')

        if self.export_shape_keys != 'SHAPE_POSE':
            # Remove pose corrective shape keys
            print("Removing pose corrective shape keys")
            num_shape_keys = len(skinned_mesh.data.shape_keys.key_blocks.keys())

            current_shape_key_index = 0
            for index in range(0, num_shape_keys):
                bpy.context.object.active_shape_key_index = current_shape_key_index

                if bpy.context.object.active_shape_key is not None:
                    if bpy.context.object.active_shape_key.name.startswith('Pose'):
                        bpy.ops.object.shape_key_remove(all=False)
                    else:
                        current_shape_key_index = current_shape_key_index + 1        

        if self.export_shape_keys == 'NONE':
            # Bake and remove shape keys
            print("Baking shape and removing shape keys for shape")

            # Create shape mix for current shape
            bpy.ops.object.shape_key_add(from_mix=True)
            num_shape_keys = len(skinned_mesh.data.shape_keys.key_blocks.keys())

            # Remove all shape keys except newly added one
            bpy.context.object.active_shape_key_index = 0
            for count in range(0, num_shape_keys):
                bpy.ops.object.shape_key_remove(all=False)

        # Model (skeleton and skinned mesh) needs to have rotation of (90, 0, 0) when exporting so that it will have rotation (0, 0, 0) when imported into Unity
        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh.select_set(True)
        skinned_mesh.rotation_euler = (radians(-90), 0, 0)
        bpy.context.view_layer.objects.active = skinned_mesh
        bpy.ops.object.transform_apply(location = False, rotation = True, scale = False)
        skinned_mesh.rotation_euler = (radians(90), 0, 0)
        skinned_mesh.select_set(False)

        armature.select_set(True)
        armature.rotation_euler = (radians(-90), 0, 0)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.transform_apply(location = False, rotation = True, scale = False)
        armature.rotation_euler = (radians(90), 0, 0)

        if self.target_format == "UNREAL":
            # Scale armature by 100 so that Unreal FBX importer can be used with default scale 1.
            # This ensures that attached objects to imported skeleton in Unreal will keep scale 1.

            armature.scale = (100, 100, 100)

            # Scale keyframed pelvis locations if available
            if armature.animation_data is not None:
                action = armature.animation_data.action
                for fcurve in action.fcurves:
                    if fcurve.data_path.endswith("location"):
                        for keyframe_point in fcurve.keyframe_points:
                            keyframe_point.co[1] = keyframe_point.co[1] * 100
                            keyframe_point.handle_left[1] = keyframe_point.handle_left[1] * 100
                            keyframe_point.handle_right[1] = keyframe_point.handle_right[1] * 100

            bpy.ops.object.transform_apply(location = False, rotation = False, scale = True)

        # Select armature and skinned mesh for export
        skinned_mesh.select_set(True)

        # Rename armature and skinned mesh to not contain Blender copy suffix
        if "female" in skinned_mesh.name:
            gender = "female"
        elif "male" in skinned_mesh.name:
            gender = "male"
        else:
            gender = "neutral"

        target_mesh_name = "SMPLX-mesh-%s" % gender
        target_armature_name = "SMPLX-%s" % gender

        if target_mesh_name in bpy.data.objects:
            bpy.data.objects[target_mesh_name].name = "SMPLX-temp-mesh"
        skinned_mesh.name = target_mesh_name

        if target_armature_name in bpy.data.objects:
            bpy.data.objects[target_armature_name].name = "SMPLX-temp-armature"
        armature.name = target_armature_name

        # Default FBX export settings export all animations. Since we duplicated the armature we have a copy of the animation and the original animation.
        # We avoid export of both by only exporting the active animation for the armature (bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False).
        # Disable keyframe simplification to ensure that exported FBX animation properly matches up with exported Alembic cache.
        bpy.ops.export_scene.fbx(filepath=self.filepath, use_selection=True, apply_scale_options="FBX_SCALE_ALL", add_leaf_bones=False, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False, bake_anim_simplify_factor=0)

        print("Exported: " + self.filepath)

        # Remove temporary copies of armature and skinned mesh
        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh.select_set(True)
        armature.select_set(True)
        bpy.ops.object.delete()

        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh_original.select_set(True)
        bpy.context.view_layer.objects.active = skinned_mesh_original

        if "SMPLX-temp-mesh" in bpy.data.objects:
            bpy.data.objects["SMPLX-temp-mesh"].name = target_mesh_name

        if "SMPLX-temp-armature" in bpy.data.objects:
            bpy.data.objects["SMPLX-temp-armature"].name = target_armature_name

        return {'FINISHED'}
    
    
class WebcamCap(bpy.types.Operator):
    bl_idname = "object.webcam_cap"
    bl_label = "Webcam Capture"
    bl_description = 'start webcam '
    bl_options = {'REGISTER', 'UNDO'}
    
    client_socket = None
    tcp_server_socket = None
    obj  = None
    armature = None
    frame_idx = 0
    # tem = None
    # execution_queue = queue.Queue()
    cap = cv2.VideoCapture('./jimcarrey_cut.mp4')
    
    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False
        
    
    def socket_recv(self, client_socket, tcp_server_socket):
        recv_data_whole = bytes()
        recv_data = client_socket.recv(464)  # (100+10+6)*4 = 464  
        if len(recv_data) == 0 :
            # close socket
            client_socket.close()
            tcp_server_socket.close()
            print('server is close, stop receive!')
            return None
            # client_socket, clientAddr = tcp_server_socket.accept() 
        else:
            recv_data_whole += recv_data
            # print('receive data length :', recv_data_whole.__len__())
            # check data length again
            # (100+10+6)*4 = 464
            if len(recv_data_whole) == 464 :   
                tem_arr = np.frombuffer(recv_data_whole, dtype=np.float32).reshape(1, 116) 
                recv_data_whole = bytes()
            
                # print(tem_arr)

                return tem_arr    
        
        
    def web_thread_update(self):
        context = bpy.context
        # bpy.ops.object.mode_set(mode='OBJECT')

        # obj = bpy.context.object
        # if obj.type == 'ARMATURE':
        #     armature = obj
        #     obj = bpy.context.object.children[0]
        # else:
        #     armature = obj.parent 
        
        # show now frame
        ret_val, img0 = self.cap.read()
        if ret_val :    
            cv2.imshow('img', img0)
            cv2.waitKey(1)
        
        '''
        tem_out : [shape,  exp , pose] 
            100    10     6[neck, jaw]
        '''
        tem_ = self.socket_recv(self.client_socket, self.tcp_server_socket)
        
        if tem_ is None:
            bpy.app.timers.unregister(self.web_thread_update)
            return 1
        
        elif np.array_equal(tem_ ,np.zeros((1, 116))):
            
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')
            bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
            
            return 0
        # tem_ = self.socket_recv(self.client_socket, self.tcp_server_socket)
             
        
        # # smooth the pose
        # if len(queue_tem) == 0:
        #     # pad queue 
        #     for _ in range(queue_n):
        #         # todo: load prepare loc
        #         queue_tem.append(np.empty((1, 116)))
        #     queue_tem.append(np.empty((1, 116)))
        # else:
        #     queue_tem.append(tem_.copy())
            
        # tem_out = np.mean(queue_tem, axis=0)

        tem_shape = tem_[0, :100]
        tem_exp = tem_[0, 100:110]
        tem_pose = tem_[0, 110:]
        # bpy.ops.object.mode_set(mode='OBJECT')
        
        print('update shape!')
        bpy.ops.object.mode_set(mode='OBJECT')
        for index, shape in enumerate(tem_shape):
            key_block_name = f"Shape{index:03}"
            
            if key_block_name in self.obj.data.shape_keys.key_blocks:
                    self.obj.data.shape_keys.key_blocks[key_block_name].value = shape
            else:
                print(f"ERROR: No key block for: {key_block_name}")

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')
        
        print('update pose!')
        
        # use pelvis replace neck pose 
        neck_pose = tem_pose[0:3]
        set_pose_from_rodrigues(self.armature, "neck", neck_pose)
        
        # update jaw pose
        jaw_pose = tem_pose[3:]
        set_pose_from_rodrigues(self.armature, "jaw", jaw_pose)
        
        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')    
        # update face exp
        print('update exp!')
        for index, exp in enumerate(tem_exp):
            key_block_name = f"Exp{index:03}"

            if key_block_name in self.obj.data.shape_keys.key_blocks:
                self.obj.data.shape_keys.key_blocks[key_block_name].value = exp
            else:
                print(f"ERROR: No key block for: {key_block_name}")   
            
        print('------finish update!---------')
        print(f'---now image frame is {self.frame_idx} -----')
        
        self.frame_idx += 1
        # return 0.1 is work
        return 0
            
    # def modal(self, context: Context | socket.Any, event: Event | socket.Any):

    #     if event.type == 'TIMER':

    #     return {'PASS_THROUGH'}
    
            
    
    def execute(self, context):
        #
        # self.execution_queue = queue.Queue()
        
        obj = bpy.context.object
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent 
        
        self.obj = obj
        self.armature = armature
        
        tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        hostname = socket.gethostname()
        port =  8800
        address = (hostname, port)
        # bind
        tcp_server_socket.bind(address)
        tcp_server_socket.listen(5)
        
        # clientAddr  your blind (hostname | ip ，port)
        client_socket, clientAddr = tcp_server_socket.accept()
        
        self.client_socket = client_socket
        self.tcp_server_socket = tcp_server_socket
        # recv_data_whole = bytes()
        
        # queue_tem = deque()
        
        # thread = threading.Thread(target=self.web_thread_update, args=(client_socket, tcp_server_socket, obj, armature))
        # thread.start()
        
        # self.execution_queue.put(self.web_thread_update)
        
        # avoid screen freeze
        bpy.app.timers.register(self.web_thread_update)
        
        
        return {'FINISHED'}
    
    
    # def update_face(self):
        
    #     tem_ = self.socket_recv(self.client_socket, self.tcp_server_socket)
        
    #     while not (tem_ is None):
    #         self.tem = tem_
    #         # function = self.execution_queue.get()
    #         self.web_thread_update(bpy.context)
            
    #     return 1
    
    

        
    
    
class CAP_PT_Webcam(bpy.types.Panel):
    bl_label = "Webcam"
    bl_category = "Webcam"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        
        row = col.row(align=True)
        col.label(text="Webcam:")
        # col.prop(context.window_manager.webcam_tool, "smplx_webcam")
        col.separator()
        col.operator("object.webcam_cap", text="Start")


class CAP_PT_Model(bpy.types.Panel):
    bl_label = "SMPL-X Model"
    bl_category = "Webcam"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):

        layout = self.layout
        col = layout.column(align=True)
        
        row = col.row(align=True)
        col.prop(context.window_manager.webcam_tool, "smplx_gender")
        col.operator("scene.smplx_add_gender", text="Add")
        col.separator()
        
        col.label(text="Texture:")
        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.prop(context.window_manager.webcam_tool, "smplx_texture")
        split.operator("object.smplx_set_texture", text="Set")

class CAP_PT_Shape(bpy.types.Panel):
    bl_label = "Shape"
    bl_category = "Webcam"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.prop(context.window_manager.webcam_tool, "smplx_height")
        col.prop(context.window_manager.webcam_tool, "smplx_weight")
        col.operator("object.smplx_measurements_to_shape")
        col.separator()

        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.operator("object.smplx_random_shape")
        split.operator("object.smplx_reset_shape")
        col.separator()

        col.operator("object.smplx_snap_ground_plane")
        col.separator()

        col.operator("object.smplx_update_joint_locations")
        col.separator()
        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.operator("object.smplx_random_expression_shape")
        split.operator("object.smplx_reset_expression_shape")

class CAP_PT_Pose(bpy.types.Panel):
    bl_label = "Pose"
    bl_category = "Webcam"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.prop(context.window_manager.webcam_tool, "smplx_corrective_poseshapes")
        col.separator()
        col.operator("object.smplx_set_poseshapes")

        col.separator()
        col.label(text="Hand Pose:")
        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.prop(context.window_manager.webcam_tool, "smplx_handpose")
        split.operator("object.smplx_set_handpose", text="Set")

        col.separator()
        col.operator("object.smplx_write_pose")
        col.separator()
        col.operator("object.smplx_load_pose")

class CAP_PT_Animation(bpy.types.Panel):
    bl_label = "Animation"
    bl_category = "Webcam"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("object.smplx_add_animation")

class CAP_PT_Export(bpy.types.Panel):
    bl_label = "Export"
    bl_category = "Webcam"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.operator("object.smplx_export_alembic")
        col.separator()

        col.operator("object.smplx_export_fbx")
        col.separator()

#        export_button = col.operator("export_scene.obj", text="Export OBJ [m]", icon='EXPORT')
#        export_button.global_scale = 1.0
#        export_button.use_selection = True
#        col.separator()

        row = col.row(align=True)
        row.operator("ed.undo", icon='LOOP_BACK')
        row.operator("ed.redo", icon='LOOP_FORWARDS')
        col.separator()

        (year, month, day) = bl_info["version"]
        col.label(text="Version: %s-%s-%s" % (year, month, day))

classes = [
    PG_WEBCAMProperties,
    AddGender,
    SetTexture,
    MeasurementsToShape,
    RandomShape,
    ResetShape,
    RandomExpressionShape,
    ResetExpressionShape,
    SnapGroundPlane,
    UpdateJointLocations,
    SetPoseshapes,
    ResetPoseshapes,
    SetHandpose,
    WritePose,
    LoadPose,
    ResetPose,
    AddAnimation,
    ExportAlembic,
    ExportFBX,
    WebcamCap,
    CAP_PT_Webcam,
    CAP_PT_Model,
    CAP_PT_Shape,
    CAP_PT_Pose,
    CAP_PT_Animation,
    CAP_PT_Export
]

def register():
    from bpy.utils import register_class
    for cls in classes:
        bpy.utils.register_class(cls)

    # Store properties under WindowManager (not Scene) so that they are not saved in .blend files and always show default values after loading
    bpy.types.WindowManager.webcam_tool = PointerProperty(type=PG_WEBCAMProperties)
    

def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        bpy.utils.unregister_class(cls)
        
    del bpy.types.WindowManager.webcam_tool


if __name__ == "__main__":
    register()