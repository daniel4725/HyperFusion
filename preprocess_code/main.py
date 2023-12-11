from nipype import Node, Function, Workflow
from nipype.interfaces.fsl import Reorient2Std, RobustFOV, BET, DilateImage, ApplyMask, Threshold, ExtractROI
from nipype.interfaces import afni as afni
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
import pipeFunctions
import os
from nipype.interfaces.io import SelectFiles
import numpy as np
import pickle
import time
from argparse import ArgumentParser, Namespace


parser = ArgumentParser()
parser.add_argument('--part_num', type=int, default=0)
parser.add_argument('--part_len', type=int, default=100)
args = parser.parse_args()

t_start = time.time()

# scans dir:
path_data = os.path.join(os.getcwd(), "ADNI")

out_dir = os.path.join(os.getcwd(), "processed")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

sub_list = os.listdir(path_data)

sub_list_as_parts = [sub_list[i: i + args.part_len] for i in range(0, len(sub_list), args.part_len)]
if args.part_num >= len(sub_list_as_parts):
     print("part not in parts list")
     exit(0)
else:
     sub_list = sub_list_as_parts[args.part_num]


# checks for existance of subject_list, if not creates it.
# list = out_dir+'sub_list.pickle'
# if not os.path.isfile(list):
#     sub_list = [fn for fn in os.listdir(path_data) if os.path.isdir(path_data + fn)
#                 and os.path.isfile(path_data + fn + '/Anat/Anat.nii.gz')]
#     with open(list, 'wb') as f:
#         pickle.dump(sub_list, f)
# else:
#     with open(list, 'rb') as f:
#         list = pickle.load(f)
# # slices sub_list to desired number of subjects
# sub_list = list[15000:20000]




from nipype import IdentityInterface
infosource = Node(IdentityInterface(fields=['subject_id']),
                  name="infosource")
infosource.iterables = [('subject_id', sub_list)]
# defines scan file template relying on subjects name from sub_list
anat_file = path_data+'/{subject_id}/raw.nii'
templates = {'anat': anat_file}

selectfiles = Node(SelectFiles(templates,
                               base_directory=path_data),
                   name="selectfiles")

#creating complex Nodes
deOblique = Node(afni.Warp(deoblique=True), name="deOblique")

resampleD = Node(afni.Resample(orientation='RPI', outputtype='NIFTI_GZ'), name="resampleD")

robustFov = Node(RobustFOV(), name="robustFov")

robex = Node(Function(input_names=["inputImg"],
                      output_names=["out_stripped", "out_mask"],
                      function=pipeFunctions.RobexImage),
            name='robex')
robex_simple = Node(Function(input_names=["inputImg"],
                      output_names=["out_stripped", "out_mask"],
                      function=pipeFunctions.RobexImage),
            name='robex_simple')
intensity = Node(Function(input_names=["inputImg", "maskFile"],
                          output_names=["normalizedImage"],
                          function=pipeFunctions.imageIntesity),
                 name='intesity')
maxFilter = Node(Function(input_names=["inputImg"],
                          output_names=["filtered_path"],
                          function=pipeFunctions.maximum_filter),
                 name='maxFilter')
maxFilter_simple = Node(Function(input_names=["inputImg"],
                          output_names=["filtered_path"],
                          function=pipeFunctions.maximum_filter),
                 name='maxFilter_simple')

applyMask = Node(ApplyMask(), name="applyMask")

rescaleImage = Node(Function(input_names=["inputImg"],
                     output_names=["resampled_path"],
                     function=pipeFunctions.rescaleImage),
                name='rescaleImage')

biasFieldCorrection = Node(N4BiasFieldCorrection(), name="biasFieldCorrection")

threshold = Node(Threshold(thresh=0.0001, direction='below'), name="threshold")

extract_roi = Node(Function(input_names=(["inputImg"]),
                 output_names=["out_path"],
                 function=pipeFunctions.extract_ROI),
                 name='extract_ROI')

numpySave = Node(Function(input_names=["inputImg"],
                 output_names=["out_path"],
                 function=pipeFunctions.saveAsNumpy),
            name='numpySave')

sliceSave = Node(Function(input_names=["inputImg"],
                 output_names=["out_path"],
                 function=pipeFunctions.sliceSaveComplex),
            name='sliceSave')

delete_and_save = Node(Function(input_names=["sub_dir", "anat_path"],
                 output_names=[],
                 function=pipeFunctions.delete_save_trans_V2),
            name='delete_and_save')

# creates simple Nodes
applyMask_simple = Node(ApplyMask(), name="applyMask_simple")

rescaleImage_simple = Node(Function(input_names=["inputImg"],
                     output_names=["resampled_path"],
                     function=pipeFunctions.rescaleImage),
                name='rescaleImage_simple')

threshold_simple = Node(Threshold(thresh=0.0001, direction='below'), name="threshold_simple")

extract_roi_simple = Node(Function(input_names=(["inputImg"]),
                 output_names=["out_path"],
                 function=pipeFunctions.extract_ROI),
                 name='extract_ROI_simple')

numpySave_simple = Node(Function(input_names=["inputImg"],
                 output_names=["out_path"],
                 function=pipeFunctions.saveAsNumpy),
            name='numpySave_simple')

sliceSave_simple = Node(Function(input_names=["inputImg"],
                 output_names=[],
                 function=pipeFunctions.sliceSaveSimple),
            name='sliceSave_simple')

#defines workflow and connect nodes
projpype = Workflow(name='Preprocess_Pipeline_V2', base_dir=out_dir)
projpype.connect([(infosource, selectfiles, [("subject_id", "subject_id")]),
                  (selectfiles, deOblique, [("anat", "in_file")]),
                  (deOblique,  resampleD, [("out_file", "in_file" )]),
                  (resampleD, robustFov, [("out_file", "in_file")]),
                  (robustFov, biasFieldCorrection, [("out_roi", "input_image")]),
                  (robustFov, applyMask_simple, [("out_roi", "in_file")]),
                  (robustFov, robex, [("out_roi", "inputImg")]),
                  (robex, biasFieldCorrection, [("out_mask", "mask_image")]),
                  (biasFieldCorrection, intensity, [("output_image", "inputImg")]),
                  (robex, maxFilter, [("out_mask", "inputImg")]),
                  (robex, intensity, [("out_mask", "maskFile")]),
                  (maxFilter, applyMask, [("filtered_path", "mask_file")]),
                  (maxFilter, applyMask_simple, [("filtered_path", "mask_file")]),
                  (intensity, applyMask, [("normalizedImage", "in_file")]),
                  (applyMask, rescaleImage, [("out_file", "inputImg")]),
                  (applyMask_simple, rescaleImage_simple, [("out_file", "inputImg")]),
                  (rescaleImage, threshold, [("resampled_path", "in_file")]),
                  (rescaleImage_simple, threshold_simple, [("resampled_path", "in_file")]),
                  (threshold, extract_roi, [("out_file", "inputImg")]),
                  (threshold_simple, extract_roi_simple, [("out_file", "inputImg")]),
                  (extract_roi, numpySave, [("out_path", "inputImg")]),
                  (extract_roi_simple, numpySave_simple, [("out_path", "inputImg")]),
                  (numpySave, sliceSave, [("out_path", "inputImg")]),
                  (sliceSave, delete_and_save, [("out_path", "sub_dir")]),
                  (selectfiles, delete_and_save, [("anat", "anat_path")]),
                  (numpySave_simple, sliceSave_simple, [("out_path", "inputImg")])])

projpype.write_graph(graph2use='colored', dotfilename=out_dir+'graph/pipe_graph')
projpype.run(plugin='MultiProc',plugin_args={'n_procs': 5})

print(f"Finished all {len(sub_list)} subjects in {time.time() - t_start}sec = {(time.time() - t_start)/60}min = {(time.time() - t_start)/60/60}hr")
