#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import cv2
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render
import sys
from hoi_scene import HOIDataset, GaussianModel
from hoi_optimizer import HOIOptimizer
from utils.general_utils import safe_state
from utils.visualize_utils import visualize_imgs, visualize_human_mesh_contact, visualize_obj_mesh_contact
import uuid
import imageio
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import fov2focal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from smpl.smpl_numpy import SMPL
import smplx
import open3d as o3d
import random
import json
from scipy.ndimage import distance_transform_edt

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips

loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

def transform_obj(obj,r,trans,s):
    query_center = np.mean(obj,axis=0)
    # print(query_center)
    obj= obj - query_center
    obj= obj * s
    obj = np.matmul(obj,r)
    obj= obj + trans+query_center
    return obj

def save_result_hoi(save_dir, h_pose, o_pose,obj_path):
    h_pose = h_pose.detach().cpu().numpy().tolist()

    param=json.load(open(obj_path+'/smplx_parameters.json'))
    obj_mesh= o3d.io.read_triangle_mesh(obj_path+'/obj_pcd_h_align.obj')
    h_mesh=o3d.io.read_triangle_mesh(obj_path+'/h_mesh.obj')
    obj_verts_org=np.asarray(obj_mesh.vertices)
    cam_trans=param['cam_trans']
    ro = o_pose['ro']
    transl = o_pose['transl']
    scale = o_pose['scale']



    obj_verts = obj_verts_org + cam_trans

    # cam_transl=np.asarray(cam_trans)
    exp_verts = transform_obj(obj_verts, ro, transl, scale)
    exp_verts -= cam_trans

    obj_mesh.vertices = o3d.utility.Vector3dVector(exp_verts)
    o3d.io.write_triangle_mesh(save_dir + '/obj_mesh.obj', obj_mesh)
    o3d.io.write_triangle_mesh(save_dir + '/h_mesh.obj', h_mesh)


    with open(save_dir + '/h_pose.json', 'w') as f:
        json.dump(h_pose, f)
    with open(save_dir + '/o_pose.json', 'w') as f:
        json.dump(o_pose, f)


def get_img_from_cam(cam, world_points):
    Tr = cam.T
    R = cam.R
    K = cam.K

    cam_points = np.matmul(world_points, R)
    cam_points += Tr.T
    imgvec = np.dot(cam_points, K.T)
    img_points = np.zeros((imgvec.shape[0], 2))

    for index in range(imgvec.shape[0]):
        vec = imgvec[index]
        z = vec[2]

        img_points[index][0] = vec[0] / z
        img_points[index][1] = vec[1] / z

    return img_points.astype(np.int32)


def calculate_center(mask):
    num_valid_pixels = mask.sum()
    if num_valid_pixels > 0:
        y_indices, x_indices = torch.meshgrid(torch.arange(mask.size(0)), torch.arange(mask.size(1)), indexing='ij')
        x_indices = x_indices.cuda()
        y_indices = y_indices.cuda()
        center_y = (y_indices * mask).sum() / num_valid_pixels
        center_x = (x_indices * mask).sum() / num_valid_pixels
        return center_y, center_x
    else:
        return None, None


def calculate_size(mask):
    return mask.sum()


def size_centre_loss(mask1, mask2):
    centery_1, centerx_1 = calculate_center(mask1)
    centery_2, centerx_2 = calculate_center(mask2)
    # print(centery_1, centerx_1,centery_2, centerx_2)
    if centery_2 is None or centerx_2 is None or centery_1 is None or centerx_1 is None:
        return 0, 0
    size1 = calculate_size(mask1)
    size2 = calculate_size(mask2)
    center_loss = ((centery_1 - centery_2) ** 2 + (centerx_1 - centerx_2) ** 2) ** 0.5
    size_loss = torch.abs(size1 - size2)
    return center_loss, size_loss


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
              save_dir, contact):
    first_iter = 0
    tb_writer, log_file = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)

    scene = HOIDataset(dataset, gaussians)

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    gif_frames_dir = os.path.join(save_dir, "gif_frames")
    os.makedirs(gif_frames_dir, exist_ok=True)
    for existing_file in os.listdir(gif_frames_dir):
        if existing_file.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(gif_frames_dir, existing_file))
    frame_paths = []

    elapsed_time = 0
    
    # 早停机制相关变量
    hoi_phase_started = False  # 是否进入 HOI 优化阶段
    prev_contact_loss = None   # 上一次的 contact loss
    convergence_epsilon = 0.01  # 收敛阈值（相对变化率）
    patience = 5               # 连续多少次变化小于阈值则停止
    patience_counter = 0       # 当前连续小变化计数
    hoi_converged = False      # HOI 优化是否已收敛
    max_hoi_iterations = 200   # HOI 阶段最大迭代次数（允许更多迭代）
    min_hoi_iterations = 30    # HOI 阶段最少迭代次数（确保充分优化）
    hoi_iteration_count = 0    # HOI 阶段迭代计数
    hoi_start_iteration = 100  # HOI 阶段开始的迭代数
    
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Start timer
        start_time = time.time()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        ## zero grad
        gaussians.optimizer.zero_grad(set_to_none=True)

        viewpoint_cam = viewpoint_stack[0]

        pipe.debug = True
        render_pkg = render(iteration, viewpoint_cam, gaussians, pipe, background)

        image, alpha, viewspace_point_tensor, visibility_filter, radii, image_o, image_h, alpha_o, alpha_h, \
            depth_h, depth_o, obj_pose, h_pose = (
            render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"],
            render_pkg["render_o"], render_pkg["render_h"],
            render_pkg["render_alpha_o"], render_pkg["render_alpha_h"], render_pkg['depth_h'],
            render_pkg['depth_o'], render_pkg['obj_pose'], render_pkg['h_param'])

        # Store current render to reuse when assembling the animation at the end of training.
        frame_rgb = np.clip(image.permute(1, 2, 0).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        frame_path = os.path.join(gif_frames_dir, f"frame_{iteration:06d}.png")
        imageio.imwrite(frame_path, frame_rgb)
        frame_paths.append(frame_path)

        Ll1,Ll1_h,Ll1_o, mask_loss,mask_loss_h, mask_loss_o, ssim_loss, ssim_loss_h, ssim_loss_o, lpips_loss, lpips_loss_h, lpips_loss_o = (
            None, None, None, None, None, None, None, None, None, None, None, None)

        # gaussian Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image_h = viewpoint_cam.original_image_h.cuda()
        gt_image_o = viewpoint_cam.original_image_o.cuda()

        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()

        bkgd_mask_o = viewpoint_cam.bkgd_mask_o.cuda()

        bkgd_mask_h = viewpoint_cam.bkgd_mask_h.cuda()

        centre_loss, size_loss = size_centre_loss(alpha_o.squeeze(0), bkgd_mask_o.squeeze(0))
        alpha_o = alpha_o.masked_fill(~bkgd_mask_o.bool(), 0)
        image_o = image_o.masked_fill(~bkgd_mask_o.bool(), 0)

        bound_mask = viewpoint_cam.bound_mask.cuda()
        Ll1 = l1_loss(image.permute(1, 2, 0)[bound_mask[0] == 1], gt_image.permute(1, 2, 0)[bound_mask[0] == 1])
        Ll1_o = l1_loss(image_o.permute(1, 2, 0)[bound_mask[0] == 1], gt_image_o.permute(1, 2, 0)[bound_mask[0] == 1])
        Ll1_h = l1_loss(image_h.permute(1, 2, 0)[bound_mask[0] == 1], gt_image_h.permute(1, 2, 0)[bound_mask[0] == 1])
        mask_loss = l2_loss(alpha[bound_mask == 1], bkgd_mask[bound_mask == 1])
        mask_loss_o = l2_loss(alpha_o[bound_mask == 1], bkgd_mask_o[bound_mask == 1])
        mask_loss_h = l2_loss(alpha_h[bound_mask == 1], bkgd_mask_h[bound_mask == 1])

        # crop the object region
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_pred_o = image_o[:, y:y + h, x:x + w].unsqueeze(0)
        img_pred_h = image_h[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt_o = gt_image_o[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt_h = gt_image_h[:, y:y + h, x:x + w].unsqueeze(0)

        #ssim loss
        ssim_loss = ssim(img_pred, img_gt)
        ssim_loss_o = ssim(img_pred_o, img_gt_o)
        ssim_loss_h = ssim(img_pred_h, img_gt_h)
        # lipis loss
        lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)
        lpips_loss_o = loss_fn_vgg(img_pred_o, img_gt_o).reshape(-1)
        lpips_loss_h = loss_fn_vgg(img_pred_h, img_gt_h).reshape(-1)


        gaussian_loss = (Ll1 * 0.3 + Ll1_o + Ll1_h +
                            0.05 * mask_loss + 0.1 * mask_loss_o + 0.1 * mask_loss_h
                            + 0.005 * (1.0 - ssim_loss) + 0.01 * (1.0 - ssim_loss_o) +
                            0.01 * (1.0 - ssim_loss_h) + 0.005 * lpips_loss +
                            0.01 * lpips_loss_o + 0.01 * lpips_loss_h)

        if iteration < 100:
            loss = gaussian_loss + size_loss * 0.01 + 0.01 * centre_loss
            loss.backward()
            
            if iteration == 99:
                # 可视化人体和物体的接触区域
                # 获取3D顶点坐标 (人体 + 物体)
                print("Visualizing contact regions...")
                means3D = gaussians.get_xyz  # 所有点的3D坐标
                
                # 获取人体顶点 (变形后的SMPLX顶点)
                dst_posevec = viewpoint_cam.smpl_param['poses'][:, 3:]
                pose_out = gaussians.pose_decoder(dst_posevec)
                correct_Rs = pose_out['Rs']
                _, means3D_h, _,_,_ = gaussians.coarse_deform_c2source(
                    viewpoint_cam.big_pose_world_vertex[None],
                    viewpoint_cam.smpl_param,
                    viewpoint_cam.big_pose_smpl_param,
                    viewpoint_cam.big_pose_world_vertex[None], 
                    lbs_weights=None,
                    correct_Rs=correct_Rs,
                    return_transl=False)
                means3D_h = means3D_h.squeeze(0)
                
                # 获取物体顶点 (变换后的物体点云)
                means3D_o = means3D[10475:, :]
                ro = gaussians.get_transform_obj
                transo = gaussians.get_transl_obj
                scaleo = gaussians.get_scale_obj
                means3D_o = gaussians.obj_transform_gs(means3D_o, ro, transo, scaleo)
                
                # 合并人体和物体顶点
                pcds = torch.cat([means3D_h, means3D_o], dim=0)

                # 获取接触分数
                contact_scores = gaussians.contact
                
                if contact is not None:
                    contact_scores = contact
                
                # 获取物体网格面片
                obj_faces = viewpoint_cam.face_sim
                
                # 调用可视化函数
                visualize_human_mesh_contact(pcds, contact_scores, save_dir)
                visualize_obj_mesh_contact(pcds, obj_faces, contact_scores, save_dir)


        ## init hoi loss
        contact_loss, depth_loss, collision_loss = None, None, None
        
        # 进入 HOI 优化阶段的条件：iteration >= 100 且尚未收敛
        if iteration >= 100 and not hoi_converged:
            hoi_phase_started = True
            hoi_iteration_count += 1

            hoi_optim = HOIOptimizer(gaussians, viewpoint_cam, render_pkg)
            
            if contact is not None:
                # use gt contact area from `contact.json`
                print("use Net contact.")
                hoi_optim.pc.contact = contact
            
            hoi_loss_dict, hoi_loss_weights = hoi_optim(opt)

            if hoi_loss_dict['loss_contact'] is not None:
                contact_loss = hoi_loss_dict['loss_contact'] * hoi_loss_weights['lw_contact']
            else :
                contact_loss = None

            if hoi_loss_dict['loss_depth'] is not None:
                depth_loss = hoi_loss_dict['loss_depth'] * hoi_loss_weights['lw_depth']
            else:
                depth_loss = None

            if hoi_loss_dict['loss_collision'] is not None:
                collision_loss = hoi_loss_dict['loss_collision'] * hoi_loss_weights['lw_collision']
            else:
                collision_loss = None

            if contact_loss != 0 and contact_loss is not None:
                contact_loss.backward(retain_graph=True)

            if depth_loss != 0 and depth_loss != None:
                depth_loss.backward(retain_graph=True)

            if collision_loss != 0 and collision_loss is not None:
                collision_loss.backward(retain_graph=True)

            loss = gaussian_loss + size_loss * 0.01 + 0.01 * centre_loss
            print(f"Iteration {iteration}: Total Loss: {loss.item():.6f}, Gaussian Loss: {gaussian_loss.item():.6f}, Contact Loss: {contact_loss.item() if contact_loss is not None else 'N/A'}, Depth Loss: {depth_loss.item() if depth_loss is not None else 'N/A'}, Collision Loss: {collision_loss.item() if collision_loss is not None else 'N/A'}")
            loss.backward()

            for name, param in gaussians.get_named_parameters().items():
                if ((name == 'scale_obj') or (name == 'x_angle') or (name == 'y_angle') or (name == 'z_angle')):
                    param.grad = None
            
            # 收敛检测：基于 contact_loss 的变化
            # 只有达到最小迭代次数后才允许早停
            if contact_loss is not None and prev_contact_loss is not None and hoi_iteration_count >= min_hoi_iterations:
                loss_change = prev_contact_loss - contact_loss.item()
                threshold = convergence_epsilon * prev_contact_loss
                if loss_change < 0 or loss_change < threshold:
                    patience_counter += 1
                    print(f"  [Convergence Check] Loss change {loss_change:.6f} < threshold {threshold:.6f}, patience: {patience_counter}/{patience}")
                else:
                    patience_counter = 0  # 重置计数器
                
                # 检查是否满足早停条件
                if patience_counter >= patience:
                    hoi_converged = True
                    print(f"  [Early Stop] Converged after {hoi_iteration_count} HOI iterations!")
                    print(f"  [Early Stop] HOI optimization stopped at iteration {iteration}")
            elif hoi_iteration_count < min_hoi_iterations:
                # 还未达到最小迭代次数，只记录当前状态
                if contact_loss is not None and prev_contact_loss is not None:
                    loss_change = abs(prev_contact_loss - contact_loss.item())
                    print(f"  [Warmup] HOI iteration {hoi_iteration_count}/{min_hoi_iterations} (min required), loss change: {loss_change:.6f}")
            
            # 更新上一次的 contact loss
            if contact_loss is not None:
                prev_contact_loss = contact_loss.item()
            
            # 防止无限循环：达到最大迭代次数也停止
            if hoi_iteration_count >= max_hoi_iterations:
                hoi_converged = True
                print(f"  [Max Iterations] Reached max HOI iterations ({max_hoi_iterations}), stopping.")
            
            # 保存结果：在收敛时或达到最大迭代时
            if hoi_converged:
                object_path = f"{args.data_path}/{dataset.file_name}/"
                save_result_hoi(save_dir, h_pose, obj_pose, object_path)
                print(f"  [Early Stop] Saved HOI results to {save_dir}")

        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        if (iteration in testing_iterations):
            print("[Elapsed time]: ", elapsed_time)

        iter_end.record()
        torch.cuda.synchronize()  # 等待 GPU 事件完成

        with torch.no_grad():
            if iteration < 100:
                Ll1_loss_for_log = Ll1.item()
                mask_loss_for_log = mask_loss.item()
                ssim_loss_for_log = ssim_loss.item()
                lpips_loss_for_log = lpips_loss.item()
                contact_loss_for_log=None
                depth_loss_for_log=None
                collision_loss_for_log=None
            if iteration >= hoi_start_iteration and not hoi_converged:
                Ll1_loss_for_log = Ll1.item()
                mask_loss_for_log = mask_loss.item()
                ssim_loss_for_log = ssim_loss.item()
                lpips_loss_for_log = lpips_loss.item()
                contact_loss_for_log = contact_loss.item() if contact_loss is not None else None
                depth_loss_for_log = depth_loss.item() if depth_loss is not None else None
                collision_loss_for_log = collision_loss.item() if collision_loss is not None else None

            if iteration % 5 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}",
                                          "mask Loss": f"{mask_loss_for_log:.{2}f}",
                                          "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}",
                                          "contact": contact_loss_for_log,
                                          "depth": depth_loss_for_log, "collision": collision_loss_for_log
                                          })
                progress_bar.update(5)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, log_file, iteration, Ll1, Ll1_o, Ll1_h, mask_loss, mask_loss_o, mask_loss_h, ssim_loss,
                            ssim_loss_o, ssim_loss_h, lpips_loss, lpips_loss_o, lpips_loss_h, contact_loss, depth_loss, collision_loss,
                            iter_start.elapsed_time(iter_end))

            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
        # Start timer
        start_time = time.time()
        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)
        
        # 如果 HOI 阶段已收敛，跳出循环
        if hoi_converged:
            print(f"\n[Training] Exiting training loop at iteration {iteration} due to HOI convergence.")
            break

    # Close log file
    if log_file:
        log_file.close()
    
    if frame_paths:
        gif_path = os.path.join(save_dir, "training_10hz.gif")
        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        print(f"[GIF] Saved training GIF with {len(frame_paths)} frames to {gif_path}")


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", args.exp_name)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    
    # Create training log file
    log_file = open(os.path.join(args.model_path, "training_log.txt"), 'w')
    log_file.write("Iteration,Ll1_Loss,Ll1_Loss_O,Ll1_Loss_H,Mask_Loss,Mask_Loss_O,Mask_Loss_H,SSIM,SSIM_O,SSIM_H,LPIPS,LPIPS_O,LPIPS_H,Contact_Loss,Depth_Loss,Collision_Loss,Elapsed_Time\n")
    
    return tb_writer, log_file


def training_report(tb_writer, log_file, iteration, Ll1, Ll1_o, Ll1_h, mask, mask_o, mask_h, ssim, ssim_o, ssim_h, lpips, lpips_o,
                    lpips_h, contact_loss, depth_loss, collision_loss, elapsed):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item() if Ll1 is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', ssim.item() if ssim is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/lpips_loss', lpips.item() if lpips is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_o', Ll1_o.item() if Ll1_o is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss_o', ssim_o.item() if ssim_o is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/lpips_loss_o', lpips_o.item() if lpips_o is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_h', Ll1_h.item() if Ll1_h is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss_h', ssim_h.item() if ssim_h is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/lpips_loss_h', lpips_h.item() if lpips_h is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/mask', mask.item() if mask is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/mask_o', mask_o.item() if mask_o is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/mask_h', mask_h.item() if mask_h is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/contact_loss', contact_loss.item() if contact_loss is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item() if depth_loss is not None else 0, iteration)
        tb_writer.add_scalar('train_loss_patches/collision_loss', collision_loss.item() if collision_loss is not None else 0, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    # Write to log file
    if log_file:
        log_file.write(f"{iteration},{Ll1.item() if Ll1 is not None else 0:.6f},"
                      f"{Ll1_o.item() if Ll1_o is not None else 0:.6f},"
                      f"{Ll1_h.item() if Ll1_h is not None else 0:.6f},"
                      f"{mask.item() if mask is not None else 0:.6f},"
                      f"{mask_o.item() if mask_o is not None else 0:.6f},"
                      f"{mask_h.item() if mask_h is not None else 0:.6f},"
                      f"{ssim.item() if ssim is not None else 0:.6f},"
                      f"{ssim_o.item() if ssim_o is not None else 0:.6f},"
                      f"{ssim_h.item() if ssim_h is not None else 0:.6f},"
                      f"{lpips.item() if lpips is not None else 0:.6f},"
                      f"{lpips_o.item() if lpips_o is not None else 0:.6f},"
                      f"{lpips_h.item() if lpips_h is not None else 0:.6f},"
                      f"{contact_loss.item() if contact_loss is not None else 0:.6f},"
                      f"{depth_loss.item() if depth_loss is not None else 0:.6f},"
                      f"{collision_loss.item() if collision_loss is not None else 0:.6f},"
                      f"{elapsed:.6f}\n")
        log_file.flush()  # 立即写入文件

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_200, 2_000, 3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_200, 2_000, 3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--use_contact_gt", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    save_dir = os.path.join("output/",  args.exp_name, args.file_name)
    os.makedirs(save_dir, exist_ok=True)
    
    contact = None
    # if args.use_contact_gt:
    if True:
        contact_path = f"{args.data_path}/{args.file_name}/contact.json"
        with open(contact_path, 'r') as f:
            contact = json.load(f)
        contact = torch.tensor(contact, dtype=torch.bool, device="cuda").unsqueeze(1)
        print(f"Using ground-truth contact area from {contact_path}, with dim {contact.shape}")

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
              save_dir, contact)
    # All done
    print("\nTraining complete.")

