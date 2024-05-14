bl_info = {
    "name": "CL Anim Ops",
    "author" : "Christoph Lendenfeld",
    "blender": (2, 81, 0),
    "category": "User",
}


import bpy


class BlendToNeighbour(bpy.types.Operator):
    """Blend current position to keyframes on each side"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "object.blend_to_neighbour"        # Unique identifier for buttons and menu items to reference.
    bl_label = "blend to neighbour"    # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    #overshoot : bpy.props.BoolProperty(name = 'Overshoot', description = 'Allow values out of -1/1 range')
    delta : bpy.props.FloatProperty(name = 'Blend Value', description = 'Amount of blending', min = -1, max = 1)
    leftKey : bpy.props.IntProperty(name = 'Left Key', description= 'Left keyframe neighbour of current time', default = -64000)
    rightKey : bpy.props.IntProperty(name = 'Right Key', description= 'Right keyframe neighbour of current time', default = 64000)



    # calculate delta, interpolate and apply values here
    def execute(self, context):        # execute() is called when running the operator.
        for fcurveInfo in self.fcurveInfos:
            fcurve = fcurveInfo.fcurve
            leftValue = fcurve.evaluate(self.leftKey)
            rightValue = fcurve.evaluate(self.rightKey)
            centerValue = fcurveInfo.startValue

            blendValue = 0
            if self.delta < 0:
                blendValue = (abs(self.delta) * leftValue) + ((1 - abs(self.delta)) * centerValue)
            else:
                blendValue = (self.delta * rightValue) + ((1 - self.delta) * centerValue)

            fcurveInfo.fcurve.keyframe_points.insert(self.currentFrame, blendValue)
            fcurveInfo.key.co.y = blendValue    # TODO this makes the scene update???
            
        return {'FINISHED'}            # Lets Blender know the operator finished successfully.


    # called every frame?
    # that's the interactive part
    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':  # Apply
            self.delta = max(min((event.mouse_x - self.startPosX) / 100, 1),-1) 
            self.execute(context)
        elif event.type == 'LEFTMOUSE':  # Confirm
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:  # Cancel
            # delete key if it didn't exist before, or reset key to default position
            for fcurveInfo in self.fcurveInfos:
                fcurveInfo.key.select_control_point = False     # TODO keys are not deleted as long as they are selected...
                if fcurveInfo.newKey:
                    fcurveInfo.fcurve.keyframe_points.remove(fcurveInfo.key)
                else:
                    fcurveInfo.fcurve.keyframe_points.insert(self.currentFrame, fcurveInfo.startValue)
                    fcurveInfo.key.co.y = fcurveInfo.startValue     # TODO update scene
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


    # called by blender before calling execute
    # add the modal handler, which makes it interactive
    def invoke(self, context, event):
        self.startPosX = event.mouse_x
        self.currentFrame = context.scene.frame_current
        self.fcurveInfos = []

        if context.mode == 'OBJECT':
            self.selection = context.selected_objects
        
        if context.mode == 'POSE':
            self.selection = context.selected_pose_bones

        if not self.selection:
            self.info('Nothing selected')
            return {'FINISHED'}

        for obj in self.selection:
            if not obj.animation_data:
                continue
            if not obj.animation_data.action:
                continue
            for fcurve in obj.animation_data.action.fcurves:
                startValue = fcurve.evaluate(self.currentFrame)
                newKey = self.findClosestFrames(fcurve)
                key = fcurve.keyframe_points.insert(self.currentFrame, startValue)
                self.fcurveInfos.append(FCurveInfo(fcurve, key, startValue, newKey))

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


    def info(self, info):
        self.report({'INFO'}, str(info))


    def findClosestFrames(self, fcurve):
        newKey = True   # False if current position already has a key
        for key in fcurve.keyframe_points:
            if key.co.x > self.leftKey and key.co.x < self.currentFrame:
                self.leftKey = key.co.x
            elif key.co.x < self.rightKey and key.co.x > self.currentFrame:
                self.rightKey = key.co.x
            elif key.co.x == self.currentFrame:
                newKey = False

        return newKey


# container for fcurves and keys
class FCurveInfo():
    def __init__(self, fcurve, key, startValue, newKey):
        self.fcurve = fcurve
        self.key = key
        self.startValue = startValue
        self.newKey = newKey    # bool that determines if the key was newly created






classes = (
    BlendToNeighbour,
)

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)