import os, sys
# to import module in sibling folders
dir_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_scripts + "/src") #  code/python/src 

DATA_DIR = dir_scripts + "/data/"

import image_io
import flow_estimate
import flow_io
import flow_vis
import flow_postproc
import flow_evaluate

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False

if __name__ == "__main__":
    padding_size = 0.3

    test_data_root_dir =  DATA_DIR + "replica_360/hotel_0/"

    src_erp_image = image_io.image_read(test_data_root_dir + "0000_rgb_pano.jpg")
    tar_erp_image = image_io.image_read(test_data_root_dir + "0001_rgb_pano.jpg")
    src_forward_flow_gt=  flow_io.read_flow_flo(test_data_root_dir + "0000_opticalflow_forward_pano.flo")

    result_opticalflow_filepath = test_data_root_dir + "0001_rgb_forward.flo"
    result_opticalflow_vis_filepath = test_data_root_dir + "0001_rgb_forward.flo.jpg"

    # 1) estimate optical flow
    flow_estimator = flow_estimate.PanoOpticalFlow()
    flow_estimator.debug_enable = False
    flow_estimator.debug_output_dir = None
    flow_estimator.padding_size_cubemap = padding_size
    flow_estimator.padding_size_ico = padding_size
    flow_estimator.flow2rotmat_method= "3D"
    flow_estimator.tangent_image_width_ico = 480
    optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)

    # 2) evaluate the optical flow and output result
    # output optical flow image
    optical_flow = flow_postproc.erp_of_wraparound(optical_flow)
    flow_io.flow_write(optical_flow, result_opticalflow_filepath)
    optical_flow_vis = flow_vis.flow_to_color(optical_flow, min_ratio=0.2, max_ratio=0.8)
    image_io.image_save(optical_flow_vis, result_opticalflow_vis_filepath)

    # 3) error metric
    epe = flow_evaluate.EPE(src_forward_flow_gt, optical_flow)
    sepe = flow_evaluate.EPE(src_forward_flow_gt, optical_flow,  spherical=True)
    print(f"EPE: {epe}, SEPE: {sepe}")

    aae = flow_evaluate.AAE(src_forward_flow_gt, optical_flow)
    saae = flow_evaluate.AAE(src_forward_flow_gt, optical_flow,  spherical=True)
    print(f"AAE: {aae}, SAAE: {saae}")

    rmse = flow_evaluate.RMSE(src_forward_flow_gt, optical_flow)
    srmse = flow_evaluate.RMSE(src_forward_flow_gt, optical_flow,  spherical=True)
    print(f"RMSE: {rmse}, SRMSE: {srmse}")