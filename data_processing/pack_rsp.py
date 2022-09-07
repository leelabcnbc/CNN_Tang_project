import numpy as np

vis_256_shared = np.load('../data/tmp_data/Results/sharecore_low_mode_256_noCropResize_V_0/npyfiles/ModelRsp.npy')
vis_300_shared = np.load('../data/tmp_data/Results/sharecore_low_mode_300_noCropResize_V_0/npyfiles/ModelRsp.npy')
vis_tf = np.load('../data/tmp_data/Results/gs_tf_vis_noCropResize_V_0/npyfiles/ModelRsp.npy')
vis_scnn_a = np.load('../data/tmp_data/Results/daniel_scnn_vis_noCropResize_V_0/npyfiles/ModelRsp.npy')
vis_scnn_b = np.load('../data/tmp_data/Results/gs_torch_vis_noCropResize_V_0/npyfiles/ModelRsp.npy')

tf_vis_rsp = np.zeros((5,299))
tf_vis_rsp[0,:] = vis_256_shared
tf_vis_rsp[1,:] = vis_300_shared
tf_vis_rsp[2,:] = vis_tf
tf_vis_rsp[3,:] = vis_scnn_a
tf_vis_rsp[4,:] = vis_scnn_b
np.save('../data/tmp_data/tf_vis_rsp.npy', tf_vis_rsp)