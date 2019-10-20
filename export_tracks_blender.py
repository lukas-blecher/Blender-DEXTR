
# Taken and modified to an addon from
# https://gist.githubusercontent.com/anonymous/5663418/raw/ac857474130ecb39ec283515f56903d2e193507a/gistfile1.txt

from __future__ import print_function
import bpy
from bpy.types import Operator, Panel, PropertyGroup, WindowManager
from bpy.props import PointerProperty, StringProperty
import os


bl_info = {
    'name': 'Tracking to CSV',
    'category': 'Motion Tracking',
    'location': 'Movie clip Editor > Tools Panel > Solve > Export tracks',
}


class TrackingExport(Operator):
    bl_idname = "tracking.export"
    bl_label = "Tracking export"

    def execute(self, context):

        D = bpy.data

        printFrameNums = True  # include frame numbers in the csv file
        relativeCoords = False  # marker coords will be relative to the dimensions of the clip
        logfile = 'export-markers.log'
        custom_dir = context.scene.out_path
        if custom_dir == '':
            filepath = bpy.data.filepath  
            directory = os.path.dirname(filepath)
            os.makedirs(os.path.join(directory, 'data'), exist_ok=True)
            directory = os.path.join(directory, 'data')
        else: 
            filepath=bpy.path.abspath(custom_dir)  #custom_dir[2:] if custom_dir[:2] == '//' else custom_dir
            directory = os.path.dirname(filepath)
        f2 = open(os.path.join(directory, logfile), 'w')
        print('First line test', file=f2)
        for clip in D.movieclips:
            print('clip {0} found\n'.format(clip.name), file=f2)
            width = clip.size[0]
            height = clip.size[1]
            for ob in clip.tracking.objects:
                print('object {0} found\n'.format(ob.name), file=f2)
                for track in ob.tracks:
                    print('track {0} found\n'.format(track.name), file=f2)
                    fn = '{0}_{1}_tr_{2}.csv'.format(clip.name.split('.')[0], ob.name, track.name)
                    with open(os.path.join(directory, fn), 'w') as f:
                        framenum = 0
                        while framenum < clip.frame_duration:
                            markerAtFrame = track.markers.find_frame(framenum)
                            if markerAtFrame:
                                coords = markerAtFrame.co.xy
                                if relativeCoords:
                                    if printFrameNums:
                                        print('{0},{1},{2}'.format(framenum, coords[0], (1-coords[1])), file=f)
                                    else:
                                        print('{0},{1}'.format(coords[0], (1-coords[1])), file=f)
                                else:
                                    if printFrameNums:
                                        print('{0},{1},{2}'.format(framenum, coords[0]*width, (1-coords[1])*height), file=f)
                                    else:
                                        print('{0},{1}'.format(coords[0]*width, (1-coords[1])*height), file=f)

                            framenum += 1
        f2.close()
        return {'FINISHED'}


class ExportPanel(Panel):
    bl_label = "Export Tracks"
    bl_idname = "export_tracks"
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'TOOLS'
    bl_category = "Solve"

    @classmethod
    def poll(cls, context):
        return (context.area.spaces.active.clip is not None)

    # Draw UI
    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        row = layout.row()
        props = row.operator("tracking.export", text="Export")
        row = layout.column()
        row.prop(context.scene, 'out_path')

        layout.separator()


def register():
    # bpy.utils.register_class(TrackingExport)
    #WindowManager.autotracker_props = PointerProperty(type=TrackingExport)
    bpy.utils.register_module(__name__)
    bpy.types.Scene.out_path = StringProperty(
        name="Path",
        default="",
        description="Define the path where to save the tracking information",
        subtype='DIR_PATH'
    )


def unregister():
    # bpy.utils.unregister_class(TrackingExport)
    bpy.utils.unregister_module(__name__)
    #del WindowManager.autotracker_props
    del bpy.types.Scene.conf_path


if __name__ == "__main__":
    register()
