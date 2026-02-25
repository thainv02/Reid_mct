from map import Map
import cv2


mapper = Map(map_image_path="layout.png", mm_per_pixel_x=23.8, mm_per_pixel_y=23.3)


mapper.add_camera(camera_id="cam1",
                  intrinsic="cam1/intrinsic.yaml",
                  extrinsic="cam1/extrinsic.yaml",
                  origin_px=(234, 389)
                  )
mapper.add_camera(camera_id="cam2", 
                  intrinsic="cam2/intrinsic.yaml",
                  extrinsic="cam2/extrinsic.yaml",
                  origin_px=(678, 382)
                  )
mapper.add_camera(camera_id="cam3",
                  intrinsic="cam3/intrinsic.yaml",
                  extrinsic="cam3/extrinsic.yaml",
                  origin_px=(678, 382)
                  )
mapper.add_camera(camera_id="cam4",
                  intrinsic="cam4/intrinsic.yaml",
                  extrinsic="cam4/extrinsic.yaml",
                  origin_px=(678, 382)
                  )

mapper.load_rois_from_yaml("rois.yaml")

"""
INPUT FORMAT:

camera_points = {
            "cam1": [(point_id, x_pixel, y_pixel), ...],
            "cam2": [(point_id, x_pixel, y_pixel), ...],
            }
            
"""

camera_points = {
"cam1": [(1, 1756, 634), (2, 2066, 736), (3, 2389, 900),  (4, 1422, 850), (5, 1822, 1128)],
# "cam2": [(5, 447, 617), (6, 816, 641)],
# "cam3": [(5, 830, 894), (6, 1427, 1020)],
# "cam4": [(5, 783, 430), (6, 811, 509)],
}

map_img = mapper.update_map(camera_points, distance_mm=130)

cv2.imshow("Live Map", map_img)
cv2.waitKey(0)
