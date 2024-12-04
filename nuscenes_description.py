
"""both are dict,
dict_keys(['infos', 'metadata'])
dict['metadata']=='v1.0-trainval'
len(dict['infos'])==6019:即6019帧，对应于150个scene_val

description for dict['infos']：
dict['infos']为一列表
val_ego['infos'][0]为字典，其有31个键值对：
    'lidar_path':'./data/nuscenes/samples/LIDAR_TOP/n015-2018-08-02-17-16-37+0800__LIDAR_TOP__1533201470448696.pcd.bin'
    'token': 'fd8420396768425eabec9bdddf7e64b6'
    'prev': ''
    'next': '6eb8a3ff0abf4f3a9380a48f2a0b87ef'
    'can_bus':array([18个小浮点数])
    'sweeps': []
    'frame_idx': 0
    'cams': {'CAM_FRONT':{
                            'data_path': './data/nuscenes/samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460.jpg',
                            'type':'CAM_FRONT', 
                            'sample_data_token':'b8fba7d78cf547b996c431dec1f5ee26', 
                            'sensor2ego_translation': [1.70079118954, 0.0159456324149, 1.51095763913],
                            'sensor2ego_rotation': [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],
                            'ego2global_translation':,
                            'ego2global_rotation':,
                            'timestamp': 1533201470412460,
                            'sensor2lidar_rotation':array([[ 0.99996937,  0.0067556 , -0.0039516 ],[ 0.00382456,  0.01871645,  0.99981752],[ 0.00682833, -0.99980201,  0.01869004]]),
                            'sensor2lidar_translation':array([-0.01271581,  0.76880558, -0.31059456]),
                            'cam_intrinsic': array([[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],[0.00000000e+00, 1.26641720e+03, 4.91507066e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                            }
            'CAM_FRONT_RIGHT':{},'CAM_FRONT_LEFT':{},'CAM_BACK':{},'CAM_BACK_LEFT':{},'CAM_BACK_RIGHT':{}
            } 
    'scene_token': 'e7ef871f77f44331aefdebc24ec034b7'
    'lidar2ego_translation':: [0.943713, 0.0, 1.84023]
    'lidar2ego_rotation': [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]
    'ego2global_translation':[249.89610931430778, 917.5522573162784, 0.0]
    'ego2global_rotation': [0.9984303573176436, -0.008635865272570774, 0.0025833156025800875, -0.05527720957189669]
    'timestamp': 1533201470448696
    'gt_boxes':array([[-7.50550033e+00,  1.66410681e+01,  1.35939091e+00,4.24800000e+00,  1.71000000e+00,  1.52700000e+00,1.57887547e+00]....])
    'gt_names'：array(['car', 'car', 'car', 'car', 'pedestrian', 'car', 'pedestrian',
                      'car', 'pedestrian', 'pedestrian', 'traffic_cone', 'traffic_cone',
                      'car', 'car', 'car', 'car', 'car', 'car', 'car', 'pedestrian',
                      'car', 'car', 'car', 'car', 'car', 'car', 'pedestrian', 'car',
                      'car', 'car', 'traffic_cone', 'car', 'pedestrian', 'pedestrian',
                      'car', 'pedestrian', 'car'], dtype='<U12')
    'gt_velocity':array([[2.35975396e-01, -6.72914107e-01]....:上述物体在x，y轴的真实速度])
    'num_lidar_pts':array([169,  34, 475,   6,   2,   6,   1,   3,   2,   5,   2,   3,   2,
                            1,   1,   8,   5,   1,   7,   2, 492,   2,  11,  54,   4,  21,
                            2,   3,  19,   1,   1,   8,   2,   5, 119,   7,  55])->在 LiDAR点云数据中，某物体被检测到的点的数量
    'num_radar_pts':array([4, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0,
                            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0])
    'valid_flag':array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                        True])->表示某个数据项目是否有效
    'bboxes2d':[6个array([[]])]->对应于六个相机中物体的2d边界框位置
    'bboxes3d_cams':[]空列表
    'labels2d':[6个array([7, 7, 8, 8, 7, 0, 0, 7, 0, 7]),array([8, 8, 7]), array([7, 7, 7])...]->对应六个相机中物体的标签，与bboxes2d相匹配
    'centers2d':[6个array([[  53.218273,  469.2417  ],...,[1615.7109  ,  546.354   ]], dtype=float32)....]->对应六个相机中物体的中心坐标
    'depths'：[6个array([])]->表示六个相机中物体离摄像头的深度信息
    'bboxes_ignore'：[6个空array([])]->表示六个相机中需要忽略的边框
    'visibilities'：[6个['2', '2', '4', '4', '1', '4', '1', '2', '1', '4']]->表示六个相机中各物体的可见性
    'location': 'singapore-onenorth'
    'scene_name': 'scene-0003'
    'gt_fut_traj'：->表示真实的未来轨迹
    'gt_fut_traj_mask'：->真实未来轨迹的掩码
    
val['infos'][0]为字典，其有27个键值对:
    'gt_velocity':与val_ego的关系为[x,y]_val_ego->[-y,x]_val
    没有'location','scene_name','gt_fut_traj','gt_fut_traj_mask'
"""
from torch.utils.data import Dataset
import pickle
train_ego_dir='/SSD_DISK/datasets/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl'#28130帧--700scene
train_dir='/SSD_DISK/datasets/nuscenes/nuscenes2d_temporal_infos_train.pkl'
val_ego_dir='/SSD_DISK/datasets/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl'#6019帧--150scene
val_dir='/SSD_DISK/datasets/nuscenes/nuscenes2d_temporal_infos_val.pkl'
with open(val_ego_dir, 'rb') as f:
    val_ego = pickle.load(f)
val_info=list(sorted(val_ego['infos'],key=lambda x:x['timestamp']))
print(val_info[0]['sweeps'])




"""T帧的BEV sequence:由不同的通道不同的颜色描绘segmented elements,C:19,10 for depth,3 for bbox,3 for road maps,3 for camera pose embeddings"""