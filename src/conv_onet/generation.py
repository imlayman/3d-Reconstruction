import torch
import torch.optim as optim
from torch import autograd,nn
import numpy as np
from tqdm import trange, tqdm
import trimesh
from src.utils import libmcubes
from src.common import make_3d_grid, normalize_coord, add_key, coord2index
from src.utils.libsimplify import simplify_mesh
from src.utils.libmise import MISE
import time
import math
###
from skimage import measure
import open3d as o3d

counter = 0


class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=1000000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 input_type = None,
                 vol_info = None,
                 vol_bound = None,
                 simplify_nfaces=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size # 每次评估的点数量，用于减小内存消耗
        self.refinement_step = refinement_step
        self.threshold = threshold # 判定点是否被占用的阈值
        self.device = device
        self.resolution0 = resolution0 # 初始分辨率
        self.upsampling_steps = upsampling_steps # 上采样步数
        self.with_normals = with_normals # 是否估计法线
        self.input_type = input_type # 输入类型（例如体素或点云）
        self.padding = padding # MISE 算法中使用的填充
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces # 最终网格面简化后的面数量
        
        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data:输入数据张量,包含生成网格的必要信息
            return_stats:是否返回统计信息（如时间消耗）
        '''
        self.model.eval() # 将模型置于评估模式
        device = self.device
        stats_dict = {}
        # print(data)
        inputs = data.get('inputs', torch.empty(1, 0)).to(device) # 获取输入数据，并移动到设备上
        kwargs = {}

        t0 = time.time()

        # obtain features for all crops
        if self.vol_bound is not None:
            self.get_crop_bound(inputs)
            c_plane = self.encode_crop(inputs, device)
            c_final = c_plane
        else: # 如果不需要裁剪，对整个体积进行处理
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            # inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            t0 = time.time()
            with torch.no_grad(): # 在评估模式下，不需要计算梯度
                #c = self.model.encode_inputs(inputs) ### original 
               
                '''
                c_plane, point_feature = self.model.encoder(inputs, 0) ###  
                # else:
                #     #print('####see pi size:', pi.size())
                #     print('####see pi shape:', pi.shape)
                #     c = self.model.encoder(pi, n, c) ###?

                c = self.model.decode(inputs, c_plane, 0, point_feature) ###
                c_final, _ = self.model.encoder(inputs, 1, c, c_plane)
                '''

                point_feature=None
                if isinstance(self.model, nn.DataParallel):
                    c_plane = self.model.module.encoder(inputs)
                else:
                    c_plane = self.model.encoder(inputs)
                
                c_final = c_plane
        # 记录编码输入数据所用时间
        stats_dict['time (encode inputs)'] = time.time() - t0

        # 从潜在空间中生成网格
        mesh = self.generate_from_latent(c_final, inputs, stats_dict=stats_dict, **kwargs) ### add inputs for poco mc try

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh


    def generate_from_latent(self, c_final, inputs, stats_dict={}, **kwargs):
        ''' 从潜在编码中生成网格
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        inputs = inputs[0].cpu().numpy() #.detach().cpu().numpy() ###
        #print('inputs shape======',inputs.shape) #(10000,3)
        # 将阈值转换为对数几率形式
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # 如果不需要上采样，则直接生成初始分辨率的网格
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,) * 3, (0.5,) * 3, (nx,) * 3
            )

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            # input_points (10000,3)
            # 对输入点计算边界最小值和最大值
            input_points = inputs
            bmin = input_points.min()
            bmax = input_points.max()

            ########################## hard-code paramters for now #########################
            step = None
            resolution = 256
            padding = 1
            dilation_size = 2
            device = self.device
            num_pts = 50000
            out_value = 1
            mc_value = 0
            return_volume = False
            refine_iter = 10
            simplification_target = None
            refine_threshold = None
            ###############################################################################


            if step is None:
                step = (bmax - bmin) / (resolution - 1)  # 0.0039886895348044005 # 计算网格步长
                resolutionX = resolution  # 256
                resolutionY = resolution  # 256
                resolutionZ = resolution  # 256
            else:
                bmin = input_points.min(axis=0)
                bmax = input_points.max(axis=0)
                resolutionX = math.ceil((bmax[0] - bmin[0]) / step)
                resolutionY = math.ceil((bmax[1] - bmin[1]) / step)
                resolutionZ = math.ceil((bmax[2] - bmin[2]) / step)

            bmin_pad = bmin - padding * step
            bmax_pad = bmax + padding * step

            pts_ids = (input_points - bmin) / step + padding
            pts_ids = pts_ids.astype(np.int)  # (10000,3) # 计算点在网格中的索引位置

            # 创建体积，初始化为 NaN
            volume = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), np.nan,
                             dtype=np.float64)
            mask_to_see = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding),
                                  True, dtype=bool)
            while (pts_ids.shape[0] > 0):

                # print("Pts", pts_ids.shape)

                # 创建掩码，用于标记哪些位置有有效点
                mask = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False,
                               dtype=bool)
                mask[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] = True

                # dilation
                for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
                    xc = int(pts_ids[i, 0])
                    yc = int(pts_ids[i, 1])
                    zc = int(pts_ids[i, 2])
                    mask[max(0, xc - dilation_size):xc + dilation_size,
                    max(0, yc - dilation_size):yc + dilation_size,
                    max(0, zc - dilation_size):zc + dilation_size] = True

                # 获取有效点的坐标
                valid_points_coord = np.argwhere(mask).astype(np.float32)
                valid_points = valid_points_coord * step + bmin_pad
                #print('valid_points===',valid_points.shape)

                # 获取每个有效点的占用预测值
                z = []
                near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float, device=device)
                for pnts in tqdm(torch.split(near_surface_samples_torch, num_pts, dim=0), ncols=100, disable=True):

                    ### our decoder
                    occ_hat = self.eval_points(pnts, c_final, **kwargs).cpu().numpy()
                    occ_hat_pos = torch.tensor(occ_hat) #[0,1]
                    occ_hat_neg = occ_hat - 1 #[-1,0]
                    outputs = -(occ_hat_pos + occ_hat_neg) #[-1,1]
                    z.append(outputs)

                z = np.concatenate(z, axis=0)
                z = z.astype(np.float64)

                # update the volume
                volume[mask] = z

                # create the masks
                mask_pos = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding),
                                   False, dtype=bool)
                mask_neg = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding),
                                   False, dtype=bool)

                # dilation
                for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
                    xc = int(pts_ids[i, 0])
                    yc = int(pts_ids[i, 1])
                    zc = int(pts_ids[i, 2])
                    mask_to_see[xc, yc, zc] = False
                    if volume[xc, yc, zc] <= 0:
                        mask_neg[max(0, xc - dilation_size):xc + dilation_size,
                        max(0, yc - dilation_size):yc + dilation_size,
                        max(0, zc - dilation_size):zc + dilation_size] = True
                    if volume[xc, yc, zc] >= 0:
                        mask_pos[max(0, xc - dilation_size):xc + dilation_size,
                        max(0, yc - dilation_size):yc + dilation_size,
                        max(0, zc - dilation_size):zc + dilation_size] = True

                # get the new points

                new_mask = (mask_neg & (volume >= 0) & mask_to_see) | (mask_pos & (volume <= 0) & mask_to_see)
                pts_ids = np.argwhere(new_mask).astype(np.int)

            volume[0:padding, :, :] = out_value
            volume[-padding:, :, :] = out_value
            volume[:, 0:padding, :] = out_value
            volume[:, -padding:, :] = out_value
            volume[:, :, 0:padding] = out_value
            volume[:, :, -padding:] = out_value

            # volume[np.isnan(volume)] = out_value
            maxi = volume[~np.isnan(volume)].max()
            mini = volume[~np.isnan(volume)].min()

            if not (maxi > mc_value and mini < mc_value):
                return None

            if return_volume:
                return volume

            # compute the marching cubes
            verts, faces, _, _ = measure.marching_cubes(
                volume=volume.copy(),
                level=mc_value,
            )

            # removing the nan values in the vertices
            values = verts.sum(axis=1)
            o3d_verts = o3d.utility.Vector3dVector(verts)
            o3d_faces = o3d.utility.Vector3iVector(faces)
            mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
            mesh.remove_vertices_by_mask(np.isnan(values))
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            if refine_iter > 0:

                dirs = verts - np.floor(verts)
                dirs = (dirs > 0).astype(dirs.dtype)

                mask = np.logical_and(dirs.sum(axis=1) > 0, dirs.sum(axis=1) < 2)
                v = verts[mask]
                dirs = dirs[mask]

                # initialize the two values (the two vertices for mc grid)
                v1 = np.floor(v)
                v2 = v1 + dirs

                # get the predicted values for both set of points
                v1 = v1.astype(int)
                v2 = v2.astype(int)
                preds1 = volume[v1[:, 0], v1[:, 1], v1[:, 2]]
                preds2 = volume[v2[:, 0], v2[:, 1], v2[:, 2]]

                # get the coordinates in the real coordinate system
                v1 = v1.astype(np.float32) * step + bmin_pad
                v2 = v2.astype(np.float32) * step + bmin_pad

                # tmp mask
                mask_tmp = np.logical_and(
                    np.logical_not(np.isnan(preds1)),
                    np.logical_not(np.isnan(preds2))
                )
                v = v[mask_tmp]
                dirs = dirs[mask_tmp]
                v1 = v1[mask_tmp]
                v2 = v2[mask_tmp]
                mask[mask] = mask_tmp

                # initialize the vertices
                verts = verts * step + bmin_pad
                v = v * step + bmin_pad

                # iterate for the refinement step
                for iter_id in tqdm(range(refine_iter), ncols=50, disable=True):

                    preds = []
                    pnts_all = torch.tensor(v, dtype=torch.float, device=device)
                    for pnts in tqdm(torch.split(pnts_all, num_pts, dim=0), ncols=100, disable=True):
                        occ_hat = self.eval_points(pnts, c_final, **kwargs).cpu().numpy()
                        occ_hat_pos = torch.tensor(occ_hat)  # [0,1]
                        occ_hat_neg = occ_hat - 1  # [-1,0]
                        outputs = -(occ_hat_pos + occ_hat_neg)  # [-1,1]
                        preds.append(outputs)


                    preds = np.concatenate(preds, axis=0)

                    mask1 = (preds * preds1) > 0
                    v1[mask1] = v[mask1]
                    preds1[mask1] = preds[mask1]

                    mask2 = (preds * preds2) > 0
                    v2[mask2] = v[mask2]
                    preds2[mask2] = preds[mask2]

                    v = (v2 + v1) / 2

                    verts[mask] = v

                    # keep only the points that needs to be refined
                    if refine_threshold is not None:
                        mask_vertices = (np.linalg.norm(v2 - v1, axis=1) > refine_threshold)
                        # print("V", mask_vertices.sum() , "/", v.shape[0])
                        v = v[mask_vertices]
                        preds1 = preds1[mask_vertices]
                        preds2 = preds2[mask_vertices]
                        v1 = v1[mask_vertices]
                        v2 = v2[mask_vertices]
                        mask[mask] = mask_vertices

                        if v.shape[0] == 0:
                            break
                        # print("V", v.shape[0])

            else:
                verts = verts * step + bmin_pad

            o3d_verts = o3d.utility.Vector3dVector(verts)
            o3d_faces = o3d.utility.Vector3iVector(faces)
            mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
            if simplification_target is not None and simplification_target > 0:
                mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, simplification_target)

            return mesh

    def generate_mesh_sliding(self, data, return_stats=True):
        ''' Generates the output mesh in sliding-window manner.
            Adapt for real-world scale.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        # acquire the boundary for every crops
        self.get_crop_bound(inputs)

        nx = self.resolution0
        n_crop = self.vol_bound['n_crop']
        n_crop_axis = self.vol_bound['axis_n_crop']

        # occupancy in each direction
        r = nx * 2**self.upsampling_steps
        occ_values = np.array([]).reshape(r,r,0)
        occ_values_y = np.array([]).reshape(r,0,r*n_crop_axis[2])
        occ_values_x = np.array([]).reshape(0,r*n_crop_axis[1],r*n_crop_axis[2])
        for i in trange(n_crop):
            # encode the current crop
            vol_bound = {}
            vol_bound['query_vol'] = self.vol_bound['query_vol'][i]
            vol_bound['input_vol'] = self.vol_bound['input_vol'][i]
            c = self.encode_crop(inputs, device, vol_bound=vol_bound)

            bb_min = self.vol_bound['query_vol'][i][0]
            bb_max = bb_min + self.vol_bound['query_crop_size']

            if self.upsampling_steps == 0:
                t = (bb_max - bb_min)/nx # inteval
                pp = np.mgrid[bb_min[0]:bb_max[0]:t[0], bb_min[1]:bb_max[1]:t[1], bb_min[2]:bb_max[2]:t[2]].reshape(3, -1).T
                pp = torch.from_numpy(pp).to(device)
                values = self.eval_points(pp, c, vol_bound=vol_bound, **kwargs).detach().cpu().numpy()
                values = values.reshape(nx, nx, nx)
            else:
                mesh_extractor = MISE(self.resolution0, self.upsampling_steps, threshold)
                points = mesh_extractor.query()
                while points.shape[0] != 0:
                    pp = points / mesh_extractor.resolution
                    pp = pp * (bb_max - bb_min) + bb_min
                    pp = torch.from_numpy(pp).to(self.device)

                    values = self.eval_points(pp, c, vol_bound=vol_bound, **kwargs).detach().cpu().numpy()
                    values = values.astype(np.float64)
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()
                
                values = mesh_extractor.to_dense()
                # MISE consider one more voxel around boundary, remove
                values = values[:-1, :-1, :-1]

            # concatenate occ_value along every axis
            # along z axis
            occ_values = np.concatenate((occ_values, values), axis=2)
            # along y axis
            if (i+1) % n_crop_axis[2] == 0: 
                occ_values_y = np.concatenate((occ_values_y, occ_values), axis=1)
                occ_values = np.array([]).reshape(r, r, 0)
            # along x axis
            if (i+1) % (n_crop_axis[2]*n_crop_axis[1]) == 0:
                occ_values_x = np.concatenate((occ_values_x, occ_values_y), axis=0)
                occ_values_y = np.array([]).reshape(r, 0,r*n_crop_axis[2])

        value_grid = occ_values_x    
        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def get_crop_bound(self, inputs):
        ''' Divide a scene into crops, get boundary for each crop

        Args:
            inputs (dict): input point cloud
        '''
        query_crop_size = self.vol_bound['query_crop_size']
        input_crop_size = self.vol_bound['input_crop_size']
        lb_query_list, ub_query_list = [], []
        lb_input_list, ub_input_list = [], []
        
        lb = inputs.min(axis=1).values[0].cpu().numpy() - 0.01
        ub = inputs.max(axis=1).values[0].cpu().numpy() + 0.01
        lb_query = np.mgrid[lb[0]:ub[0]:query_crop_size,\
                    lb[1]:ub[1]:query_crop_size,\
                    lb[2]:ub[2]:query_crop_size].reshape(3, -1).T
        ub_query = lb_query + query_crop_size
        center = (lb_query + ub_query) / 2
        lb_input = center - input_crop_size/2
        ub_input = center + input_crop_size/2
        # number of crops alongside x,y, z axis
        self.vol_bound['axis_n_crop'] = np.ceil((ub - lb)/query_crop_size).astype(int)
        # total number of crops
        num_crop = np.prod(self.vol_bound['axis_n_crop'])
        self.vol_bound['n_crop'] = num_crop
        self.vol_bound['input_vol'] = np.stack([lb_input, ub_input], axis=1)
        self.vol_bound['query_vol'] = np.stack([lb_query, ub_query], axis=1)
        
    def encode_crop(self, inputs, device, vol_bound=None):
        ''' Encode a crop to feature volumes

        Args:
            inputs (dict): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        '''
        if vol_bound == None:
            vol_bound = self.vol_bound

        index = {}
        for fea in self.vol_bound['fea_type']:
            # crop the input point cloud
            mask_x = (inputs[:, :, 0] >= vol_bound['input_vol'][0][0]) &\
                    (inputs[:, :, 0] < vol_bound['input_vol'][1][0])
            mask_y = (inputs[:, :, 1] >= vol_bound['input_vol'][0][1]) &\
                    (inputs[:, :, 1] < vol_bound['input_vol'][1][1])
            mask_z = (inputs[:, :, 2] >= vol_bound['input_vol'][0][2]) &\
                    (inputs[:, :, 2] < vol_bound['input_vol'][1][2])
            mask = mask_x & mask_y & mask_z
            
            p_input = inputs[mask]
            if p_input.shape[0] == 0: # no points in the current crop
                p_input = inputs.squeeze()
                ind = coord2index(p_input.clone(), vol_bound['input_vol'], reso=self.vol_bound['reso'], plane=fea)
                if fea == 'grid':
                    ind[~mask] = self.vol_bound['reso']**3
                else:
                    ind[~mask] = self.vol_bound['reso']**2
            else:
                ind = coord2index(p_input.clone(), vol_bound['input_vol'], reso=self.vol_bound['reso'], plane=fea)
            index[fea] = ind.unsqueeze(0)
            input_cur = add_key(p_input.unsqueeze(0), index, 'points', 'index', device=device)
        
        with torch.no_grad():
            if isinstance(self.model, nn.DataParallel):
                c = self.model.module.encoder(input_cur)
            else:
                c = self.model.encoder(input_cur)
        return c

    def eval_points(self, p, c_info=None, point_feature=None, n=None, N=None, vol_bound=None, **kwargs): ### add n, N
        ''' 评估输入点的占用值，返回它们是否属于三维物体内部的概率

        Args:
            p (tensor): points 
            c (tensor): encoded feature volumes
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
                pi = pi.unsqueeze(0).to(self.device)
                chunk_size = 5000
                pi_chunks = torch.split(pi, chunk_size, 1)
                with torch.no_grad():
                    c_planes,p,c_points = c_info
                    if isinstance(self.model, nn.DataParallel):
                        p_r = self.model.module.decode(c_planes,pi,p,c_points,logits=True)
                    else:
                        p_r = self.model.decode(c_planes,pi,p,c_points,logits=True)
                    occ_hat = p_r.probs
                occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat



    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        
        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() * self.resolution0*2**self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else: 
            # Normalize to bounding box
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = box_size * (vertices - 0.5)
        
        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None


        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)
        


        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
