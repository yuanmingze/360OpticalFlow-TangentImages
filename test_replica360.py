import sys
import os
# import 36 optical flow module 
dir_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_scripts + "/src")  # code/python/src
import fs_utility
import flow_evaluate
import flow_estimate
import flow_postproc
import flow_vis
import image_io
import flow_io
import platform
import csv
from logger import Logger
log = Logger(__name__)

log.logger.propagate = False
"""The Replica360 dataset convention 
"""

opticalflow_mathod = "our_replica360"

class ReplicaDataset():
    ## NOTE chanage the data's path
    if platform.system() == "Windows":
        pano_dataset_root_dir = "D:/workdata/opticalflow_data_bmvc_2021/"
    elif platform.system() == "Linux":
        pano_dataset_root_dir = "/mnt/sda1/workdata/opticalflow_data_bmvc_2021/"

    # 0) the Replica-Dataset configuration
    # panoramic filename expresion
    replica_pano_rgb_image_filename_exp = "{:04d}_rgb_pano.jpg"
    replica_pano_depthmap_filename_exp = "{:04d}_depth_pano.dpt"
    replica_pano_depthmap_visual_filename_exp = "{:04d}_depth_pano_visual.jpg"
    replica_pano_opticalflow_forward_filename_exp = "{:04d}_opticalflow_forward_pano.flo"
    replica_pano_opticalflow_forward_visual_filename_exp = "{:04d}_opticalflow_forward_pano_visual.jpg"
    replica_pano_opticalflow_backward_filename_exp = "{:04d}_opticalflow_backward_pano.flo"
    replica_pano_opticalflow_backward_visual_filename_exp = "{:04d}_opticalflow_backward_pano_visual.jpg"
    replica_pano_mask_filename_exp = "{:04d}_mask_pano.png"

    # 2) data generating programs
    pano_data_dir = "pano/"
    pano_output_dir = "result/"
    pano_output_csv = "result_replica.csv"
    padding_size = 0.3

    # circle data
    dataset_circ_dirlist = [
        "apartment_0_circ_1k_0",
        "apartment_1_circ_1k_0",
        "apartment_2_circ_1k_0",
        "frl_apartment_0_circ_1k_0",
        "frl_apartment_1_circ_1k_0",
        "frl_apartment_2_circ_1k_0",
        "frl_apartment_3_circ_1k_0",
        "frl_apartment_4_circ_1k_0",
        "frl_apartment_5_circ_1k_0",
        "hotel_0_circ_1k_0",
        "office_0_circ_1k_0",
        "office_1_circ_1k_0",
        "office_2_circ_1k_0",
        "office_3_circ_1k_0",
        "office_4_circ_1k_0",
        "room_0_circ_1k_0",
        "room_1_circ_1k_0",
        "room_2_circ_1k_0",
    ]
    circle_start_idx = 0
    circle_end_idx = 35

    # line data
    dataset_line_dirlist = [
        "apartment_0_line_1k_0",
        "apartment_1_line_1k_0",
        "apartment_2_line_1k_0",
        "frl_apartment_0_line_1k_0",
        "frl_apartment_1_line_1k_0",
        "frl_apartment_2_line_1k_0",
        "frl_apartment_3_line_1k_0",
        "frl_apartment_4_line_1k_0",
        "frl_apartment_5_line_1k_0",
        "hotel_0_line_1k_0",
        "office_0_line_1k_0",
        "office_1_line_1k_0",
        "office_2_line_1k_0",
        "office_3_line_1k_0",
        "office_4_line_1k_0",
        "room_0_line_1k_0",
        "room_1_line_1k_0",
        "room_2_line_1k_0",
    ]
    line_start_idx = 0
    line_end_idx = 9

    # random data
    dataset_rand_dirlist = [
        "apartment_0_rand_1k_0",
        "apartment_1_rand_1k_0",
        "apartment_2_rand_1k_0",
        "frl_apartment_0_rand_1k_0",
        "frl_apartment_1_rand_1k_0",
        "frl_apartment_2_rand_1k_0",
        "frl_apartment_3_rand_1k_0",
        "frl_apartment_4_rand_1k_0",
        "frl_apartment_5_rand_1k_0",
        "hotel_0_rand_1k_0",
        "office_0_rand_1k_0",
        "office_1_rand_1k_0",
        "office_2_rand_1k_0",
        "office_3_rand_1k_0",
        "office_4_rand_1k_0",
        "room_0_rand_1k_0",
        "room_1_rand_1k_0",
        "room_2_rand_1k_0",
    ]
    rand_start_idx = 0
    rand_end_idx = 9


def of_estimate_replica():
    """Get the our and DIS's result in replica. """
    # optical flow esitmator
    flow_estimator = flow_estimate.PanoOpticalFlow()
    flow_estimator.debug_enable = False
    flow_estimator.debug_output_dir = None
    flow_estimator.padding_size_cubemap = ReplicaDataset.padding_size
    flow_estimator.padding_size_ico = ReplicaDataset.padding_size
    flow_estimator.flow2rotmat_method = "3D"
    flow_estimator.tangent_image_width_ico = 480
    dataset_dirlist = ReplicaDataset.dataset_circ_dirlist + ReplicaDataset.dataset_line_dirlist + ReplicaDataset.dataset_rand_dirlist

    # 1) iterate each 360 image dataset
    for pano_image_folder in dataset_dirlist:
        log.info("processing the data folder {} with padding {}".format(pano_image_folder, ReplicaDataset.padding_size))
        # input dir
        input_filepath = ReplicaDataset.pano_dataset_root_dir + pano_image_folder + "/" + ReplicaDataset.pano_data_dir + "/"
        # input index
        if pano_image_folder.find("line") != -1:
            pano_start_idx = ReplicaDataset.line_start_idx
            pano_end_idx = ReplicaDataset.line_end_idx
        elif pano_image_folder.find("circ") != -1:
            pano_start_idx = ReplicaDataset.circle_start_idx
            pano_end_idx = ReplicaDataset.circle_end_idx
        elif pano_image_folder.find("rand") != -1:
            pano_start_idx = ReplicaDataset.rand_start_idx
            pano_end_idx = ReplicaDataset.rand_end_idx
        else:
            log.error("{} folder naming is wrong".format(pano_image_folder))

        # output folder
        output_pano_filepath = ReplicaDataset.pano_dataset_root_dir + pano_image_folder + "/" + ReplicaDataset.pano_output_dir
        # the flo files output folder
        gt_of_dir = ReplicaDataset.pano_dataset_root_dir + pano_image_folder + "/" + ReplicaDataset.pano_data_dir
        if ReplicaDataset.padding_size is None:
            output_dir = output_pano_filepath + "/" + opticalflow_mathod + "/"
        else:
            output_dir = output_pano_filepath + "/" + opticalflow_mathod + "_" + str(ReplicaDataset.padding_size) + "/"
        fs_utility.dir_make(output_pano_filepath)
        fs_utility.dir_make(output_dir)

        # estimate on the all images
        for pano_image_idx in range(pano_start_idx, pano_end_idx + 1):
            # forward and backward optical flow
            for forward_of in [True, False]:
                #
                src_erp_image_filename = ReplicaDataset.replica_pano_rgb_image_filename_exp.format(pano_image_idx)
                if forward_of:
                    tar_erp_image_filename = ReplicaDataset.replica_pano_rgb_image_filename_exp.format((pano_image_idx + 1) % (pano_end_idx + 1))
                    optical_flow_filename = ReplicaDataset.replica_pano_opticalflow_forward_filename_exp.format(pano_image_idx)
                    optical_flow_gt_filename = ReplicaDataset.replica_pano_opticalflow_forward_filename_exp.format(pano_image_idx)
                    optical_flow_vis_filepath = ReplicaDataset.replica_pano_opticalflow_forward_visual_filename_exp.format(pano_image_idx)
                else:
                    tar_erp_image_filename = ReplicaDataset.replica_pano_rgb_image_filename_exp.format((pano_image_idx - 1) % (pano_end_idx + 1))
                    optical_flow_filename = ReplicaDataset.replica_pano_opticalflow_backward_filename_exp.format(pano_image_idx)
                    optical_flow_gt_filename = ReplicaDataset.replica_pano_opticalflow_backward_filename_exp.format(pano_image_idx)
                    optical_flow_vis_filepath = ReplicaDataset.replica_pano_opticalflow_backward_visual_filename_exp.format(pano_image_idx)
                result_opticalflow_filepath = output_dir + optical_flow_filename
                gt_opticalflow_filepath = gt_of_dir + optical_flow_gt_filename

                if pano_image_idx % 1 == 0:
                    print("{}, image index: {}, source Image: {}, target image: {}, output flow file: {}".format(
                        pano_image_folder, pano_image_idx, src_erp_image_filename, tar_erp_image_filename, optical_flow_filename))

                # 0) esitamte the optical flow
                optical_flow = None
                if not os.path.exists(result_opticalflow_filepath):
                    src_erp_image = image_io.image_read(input_filepath + src_erp_image_filename)
                    tar_erp_image = image_io.image_read(input_filepath + tar_erp_image_filename)
                    optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)
                    optical_flow = flow_postproc.erp_of_wraparound(optical_flow)
                    flow_io.flow_write(optical_flow, result_opticalflow_filepath)
                else:
                    log.info("{} exist, skip it.".format(result_opticalflow_filepath))

                # output optical flow image
                result_opticalflow_vis_filepath = output_dir + optical_flow_vis_filepath
                if not os.path.exists(result_opticalflow_vis_filepath):
                    if optical_flow is None:
                        optical_flow = flow_io.read_flow_flo(result_opticalflow_filepath)
                    optical_flow_vis = flow_vis.flow_to_color(optical_flow, min_ratio=0.2, max_ratio=0.8)
                    image_io.image_save(optical_flow_vis, result_opticalflow_vis_filepath)


def summary_error_scene_replica(overwrite=False, dataset_list = ["circ", "rand", "line"]):
    """ Summary the error on replica for each scene."""
    for dataset in dataset_list:
        dataset_dirlist = []
        skip_filelist = []
        if dataset == "circ":
            dataset_dirlist = ReplicaDataset.dataset_circ_dirlist 
        elif dataset == "line":
            dataset_dirlist = ReplicaDataset.dataset_line_dirlist 
            skip_filelist.append("0000_opticalflow_backward_pano.flo")
            skip_filelist.append("0009_opticalflow_forward_pano.flo")
        elif dataset == "rand":
            dataset_dirlist = ReplicaDataset.dataset_rand_dirlist

        # iterate each scene's data
        for pano_image_folder in dataset_dirlist:
            log.info("Evaluate the error of {}".format(pano_image_folder))
            # input dir
            of_gt_dir = ReplicaDataset.pano_dataset_root_dir + pano_image_folder + "/" + ReplicaDataset.pano_data_dir
            if ReplicaDataset.padding_size is None:
                of_eva_dir = ReplicaDataset.pano_dataset_root_dir + pano_image_folder + "/" + ReplicaDataset.pano_output_dir + opticalflow_mathod + "/"
            else:
                of_eva_dir = ReplicaDataset.pano_dataset_root_dir + pano_image_folder + "/" + ReplicaDataset.pano_output_dir + opticalflow_mathod + "_" + str(ReplicaDataset.padding_size) + "/"

            if os.path.exists(of_eva_dir + ReplicaDataset.pano_output_csv) and not overwrite:
                log.warn("{} exist.".format(of_eva_dir + ReplicaDataset.pano_output_csv))
                continue
            log.info("Evaluate optical flow folder {}".format(of_eva_dir))
            flow_evaluate.opticalflow_metric_folder(of_eva_dir, of_gt_dir, mask_filename_exp=ReplicaDataset.replica_pano_mask_filename_exp,
                                                    result_csv_filename=ReplicaDataset.pano_output_csv, visual_of_error=False, of_wraparound=True, skip_list = skip_filelist)


def summary_error_dataset_replica(csv_postfix=None, dataset_list = ["all", "circ", "rand", "line"]):
    """ Summary the error on whole replica. Collect all csv file's number. 
        # output the scv file to root of the whole dataset
    """
    for dataset in dataset_list:

        row_counter = 0
        aae = 0
        epe = 0
        rms = 0
        aae_sph = 0
        epe_sph = 0
        rms_sph = 0

        dataset_dirlist = []
        if dataset == "all":
            dataset_dirlist = ReplicaDataset.dataset_circ_dirlist + ReplicaDataset.dataset_line_dirlist + ReplicaDataset.dataset_rand_dirlist
        elif dataset == "circ":
            dataset_dirlist = ReplicaDataset.dataset_circ_dirlist 
        elif dataset == "line":
            dataset_dirlist = ReplicaDataset.dataset_line_dirlist 
        elif dataset == "rand":
            dataset_dirlist = ReplicaDataset.dataset_rand_dirlist


        opticalflow_mathod = "our_replica360"
        # 1) iterate each scene's data
        for pano_image_folder in dataset_dirlist:
            # load the csv
            if ReplicaDataset.padding_size is None:
                of_error_csv_filepath = ReplicaDataset.pano_dataset_root_dir + pano_image_folder + "/" + ReplicaDataset.pano_output_dir + opticalflow_mathod + "/" + ReplicaDataset.pano_output_csv
            else:
                of_error_csv_filepath = ReplicaDataset.pano_dataset_root_dir + pano_image_folder + "/" + ReplicaDataset.pano_output_dir + opticalflow_mathod + "_" + str(ReplicaDataset.padding_size) + "/" + ReplicaDataset.pano_output_csv
            log.debug("read {}".format(of_error_csv_filepath))
            of_error_csv_file = open(of_error_csv_filepath, "r")
            of_error_csv = csv.DictReader(of_error_csv_file)
            for row in of_error_csv:
                aae += float(row["AAE"])
                epe += float(row["EPE"])
                rms += float(row["RMS"])
                aae_sph += float(row["SAAE"])
                epe_sph += float(row["SEPE"])
                rms_sph += float(row["SRMS"])
                row_counter += 1

            of_error_csv_file.close()

        # 2) output whole dataset summarized error information to file
        of_error_sum_csv_filepath = ReplicaDataset.pano_dataset_root_dir + "00_result_quantity_csv/"
        fs_utility.dir_make(of_error_sum_csv_filepath)

        if ReplicaDataset.padding_size is None:
            log.warn("The padding_size is None!")
            of_error_sum_csv_filepath = of_error_sum_csv_filepath + opticalflow_mathod + "_" + dataset + "_"
        else:
            of_error_sum_csv_filepath = of_error_sum_csv_filepath + opticalflow_mathod + "_" + str(ReplicaDataset.padding_size) + "_" + dataset + "_"

        if csv_postfix is not None:
            of_error_sum_csv_filepath = of_error_sum_csv_filepath + f"{csv_postfix}_"

        of_error_sum_csv_filepath = of_error_sum_csv_filepath + ReplicaDataset.pano_output_csv
        log.info("Output the datasets summary error to {}".format(of_error_sum_csv_filepath))
        log.info("Output the error summary file to {}".format(of_error_sum_csv_filepath))
        msg = ""
        msg += "AAE: {}\n".format(aae / row_counter)
        msg += "EPE: {}\n".format(epe / row_counter)
        msg += "RMS: {}\n".format(rms / row_counter)
        msg += "AAE_SPH: {}\n".format(aae_sph / row_counter)
        msg += "EPE_SPH: {}\n".format(epe_sph / row_counter)
        msg += "RMS_SPH: {}\n".format(rms_sph / row_counter)
        msg += "\n===== Dataset & Optical flow method Information =====\n"
        from datetime import datetime
        msg += f"There are {row_counter} row data.\n"
        msg += "Evaluation Time: {}\n".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        msg += f"Padding size: {ReplicaDataset.padding_size}\n"
        msg += f"Optical flow method: {opticalflow_mathod}\n"
        msg += "\nThe all datasets are:\n".format(rms_sph / row_counter)
        for dataset_name in dataset_dirlist:
            msg += f"\t{dataset_name}\n"
        file = open(of_error_sum_csv_filepath, "w")
        file.write(msg)
        file.close()


if __name__ == "__main__":
    of_estimate_replica()
    summary_error_scene_replica()
    summary_error_dataset_replica()
