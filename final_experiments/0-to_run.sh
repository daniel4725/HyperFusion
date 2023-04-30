# sent:

#gpu1:
sh final_experiments/baseline-concat_1-fset5.sh
sh final_experiments/baseline-DAFT-fset5.sh
sh final_experiments/baseline-Film-fset5.sh
sh final_experiments/age_noembd_lastblockHyp_FFT_fcHyp_2.sh
sh final_experiments/age_noembd_lastblockHyp_TTF_fcHyp_2.sh
sh final_experiments/img_as_hyper_tabMLP_set8_3.sh


#gpu3:
sh final_experiments/image_as_hyper_1to3_set5.sh
sh final_experiments/baseline-DAFT-fset8.sh
sh final_experiments/baseline-Film-fset8.sh
# --------------------- no more here ------------------

# to send:
sh final_experiments/age_noembd_lastblockHyp_FFT_fcHyp_both.sh
sh final_experiments/age_noembd_lastblockHyp_TTF_fcHyp_both.sh

