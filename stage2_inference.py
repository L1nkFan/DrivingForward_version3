"""
缂備焦顨忛崗娑氳姳閳哄懏鈷撻柤鍛婎問閸炰粙鏌熼幁鎺戝闁诡喖锕幊妯侯潩椤撶喐瀚?- DrivingForward MF濠碘槅鍨埀顒€纾涵鈧梺鍛婃煟閸斿秹鍩€椤戣法顦︾紓鍌涙尭铻ｉ柍鈺佸暙閼靛綊鏌?闂佽　鍋撴い鏍ㄧ☉閻︻噣鏌涘Δ浣诡槷PU闂佽浜介崝搴ㄥ箖婵犲洦鏅€光偓閸愵亞鐐曢梺鍛婂灥閹碱偅鎱ㄩ埡鍛畱濞撴埃鍋撻柣婵愬櫍瀵剟顢涘Ο宄颁壕濞达絿顢婇々顐︽煛鐏炵偓灏繛瀛橈耿瀹曟捁绠涚€Ｑ冧壕濞达絽鎲＄花姘舵煛閸滀礁鐏＄紒?
婵炶揪缍€濞夋洟寮妶澶婃閻熸瑥瀚妴?
python stage2_inference.py --config_file=configs/nuscenes/phase2_training.yaml \
                           --res_flow_net_path=path/to/res_flow_net_epoch_X.pth \
                           --output_dir=./stage2_outputs
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Ensure repository and vendored packnet_sfm are importable without extra PYTHONPATH.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PACKNET_ROOT = os.path.join(PROJECT_ROOT, 'external', 'packnet_sfm')
for _p in (PROJECT_ROOT, PACKNET_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 闁荤姳绀佹晶浠嬫偪閸℃瑦鍏滄い鏃囧亹閺嗕即鏌熼顒€妫弨钘夆槈?torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

# 濠电儑缍€椤曆勬叏閻愭番浜滈柛锔诲幗缁愭鎮规笟顖氱仩缂?sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2_trainer import Stage2Model
from stage2_modules import RigidFlowCalculator, ResFlowNet, DynamicGaussianGenerator
from stage2_modules.rigid_flow import batch_warp_image_with_flow
from dataset import construct_dataset
from torch.utils.data import DataLoader


def parse_args():
    """闁荤喐鐟辩徊楣冩倵娴犲宸濋柟瀛樺笚婵垽鎮跺☉妯垮鐎殿噮鍓熷?""
    parser = argparse.ArgumentParser(description='Stage2 Inference for DrivingForward MF Mode')
    parser.add_argument('--config_file',
                        default='./configs/nuscenes/phase2_training.yaml',
                        type=str,
                        help='config yaml file')
    parser.add_argument('--res_flow_net_path',
                        default='',
                        type=str,
                        help='path to trained res_flow_net checkpoint')
    parser.add_argument('--checkpoint_path',
                        default='',
                        type=str,
                        help='alias of --res_flow_net_path')
    parser.add_argument('--output_dir',
                        default='./stage2_outputs',
                        type=str,
                        help='output directory for inference results')
    parser.add_argument('--batch_size',
                        default=1,
                        type=int,
                        help='batch size for inference')
    parser.add_argument('--num_workers',
                        default=4,
                        type=int,
                        help='number of data loading workers')
    parser.add_argument('--save_visualizations',
                        action='store_true',
                        help='save visualization images')
    parser.add_argument('--save_flow',
                        action='store_true',
                        help='save optical flow maps')
    parser.add_argument('--save_gaussians',
                        action='store_true',
                        help='save gaussian parameters')
    parser.add_argument('--save_raw_npz',
                        action='store_true',
                        help='save raw arrays (render/gt/residual flow mag) for quantitative comparison')
    parser.add_argument('--max_batches',
                        default=-1,
                        type=int,
                        help='maximum number of batches to run; -1 means full dataset')
    parser.add_argument('--num_samples',
                        default=-1,
                        type=int,
                        help='alias control: convert to max_batches by batch_size when max_batches < 0')
    return parser.parse_args()


def load_config(config_file):
    """闂佸憡姊绘慨鎯归崶顒佺厐鐎广儱娲ㄩ弸鍌炴煛閸屾碍鐭楁繛?""
    import yaml
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


class Stage2InferenceModel(torch.nn.Module):
    """Stage-2 inference model for DrivingForward."""

    def __init__(self, cfg, rank=0):
        super(Stage2InferenceModel, self).__init__()
        self.cfg = cfg
        self.rank = rank
        self.read_config(cfg)

        # 闂佸憡甯楃换鍌烇綖閹版澘绀岄柡宥庡亽閸庢垵鈽夐幘顖氫壕闂傚倸鍟抽崺鏍敊瀹€鍕闁煎鍊楀▔銏㈢磽閸愨晛鐏╃紒?        self.models = self.prepare_frozen_models(cfg, rank)

        # 闂佸憡甯楃换鍌烇綖閹版澘绀岄柡宥庡亽閸庢垵霉濠婂啰鐒告俊顖欑閳绘捇宕归浣风窔瀹?        self.stage2_modules = self.prepare_stage2_modules(cfg, rank)

        # 闂佸憡甯楃换鍌烇綖閹版澘绀岄柡宓啯顓绘繛杈剧稻濞叉繈顢橀崫銉т笉婵°倓姹叉笟鈧畷?        self.view_rendering, self.pose = self.init_geometry(cfg, rank)

        # 闂佸憡姊绘慨鎯归崶顒€绾ч柛鎰靛枛濞?        self.load_weights()

        # 闁荤姳绀佹晶浠嬫偪閸℃鈻斿Δ锕佹硶濡叉垵霉閼测晛顣奸懚鈺冣偓?        self.eval()

    def read_config(self, cfg):
        """闁荤姴娲╅褑銇愰崶顒佺厐鐎广儱娲ㄩ弸?""
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def prepare_frozen_models(self, cfg, rank):
        """闂佸憡鍨靛Λ妤吽囬鍌滅當妞ゆ垼娉曢閬嶆⒒閸愵厼鐓愭い鏂跨焸瀹曟﹢鎳￠妶鍥ㄥ皾缂傚倸鍟崹鍦垝?""
        from network import PoseNetwork, DepthNetwork
        from models.gaussian import GaussianNetwork

        models = {}
        models['pose_net'] = PoseNetwork(cfg).cuda(rank)
        models['depth_net'] = DepthNetwork(cfg).cuda(rank)
        if self.gaussian:
            models['gs_net'] = GaussianNetwork(
                rgb_dim=3,
                depth_dim=1,
                num_cams=self.num_cams,
                cross_view_num_heads=getattr(self, 'cross_view_num_heads', 4),
                enable_cross_view_fusion=getattr(self, 'enable_cross_view_fusion', True),
            ).cuda(rank)
        return models

    def prepare_stage2_modules(self, cfg, rank):
        """闂佸憡鍨靛Λ妤吽囬鍌滅當妞ゆ垼娉曢惂宀勬⒒閸愵厼鐓愭い鏂胯嫰铻ｉ柍銉ㄦ珪閸?""
        modules = {}
        modules['rigid_flow_calc'] = RigidFlowCalculator(
            height=self.height,
            width=self.width
        ).cuda(rank)
        modules['res_flow_net'] = ResFlowNet(
            num_cams=self.num_cams,
            base_channels=64
        ).cuda(rank)
        modules['dynamic_gaussian'] = DynamicGaussianGenerator(
            height=self.height,
            width=self.width
        ).cuda(rank)
        return modules

    def init_geometry(self, cfg, rank):
        """闂佸憡甯楃换鍌烇綖閹版澘绀岄柡宓啯顓绘繛杈剧稻濞叉繈顢橀崫銉т笉婵°倓姹叉笟鈧畷?""
        from models.geometry import ViewRendering, Pose
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose

    def load_weights(self):
        """闂佸憡姊绘慨鎯归崶顒€绠ラ柍褜鍓熷鍨緞瀹€鈧粔鍦磽娴ｅ湱鐭嬫繛鐓庣墦閺?""
        # 闂佸憡姊绘慨鎯归崶鈺冪當妞ゆ垼娉曢閬嶆⒒閸愵厼鐓愭い鏂跨焸瀵爼宕橀鍏碱仧
        stage1_weights_path = self.stage1_weights_path

        # PoseNet
        pose_net_path = os.path.join(stage1_weights_path, 'pose_net.pth')
        if os.path.exists(pose_net_path):
            self.models['pose_net'].load_state_dict(
                torch.load(pose_net_path, map_location=f'cuda:{self.rank}'))
            print(f"Loaded PoseNet from {pose_net_path}")

        # DepthNet
        depth_net_path = os.path.join(stage1_weights_path, 'depth_net.pth')
        if os.path.exists(depth_net_path):
            self.models['depth_net'].load_state_dict(
                torch.load(depth_net_path, map_location=f'cuda:{self.rank}'))
            print(f"Loaded DepthNet from {depth_net_path}")

        # GaussianNet
        if self.gaussian:
            gs_candidates = ['gs_net.pth', 'gaussian_net.pth']
            gs_net_path = None
            for name in gs_candidates:
                candidate = os.path.join(stage1_weights_path, name)
                if os.path.exists(candidate):
                    gs_net_path = candidate
                    break
            if gs_net_path is not None:
                self.models['gs_net'].load_state_dict(
                    torch.load(gs_net_path, map_location=f'cuda:{self.rank}'))
                print(f"Loaded GaussianNet from {gs_net_path}")

        # 闂佸憡鍔樺畷鐢靛垝閵娧呯當妞ゆ垼娉曢閬嶆⒒閸愵厼鐓愭い鏂跨灱缁辨棃骞嬮悩鍨礋
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        # 闂佸憡姊绘慨鎯归崲顡玸FlowNet闂佸搫顦崯鎾闯?(婵炲濮存鎼佸箠閳╁啰顩烽柕鍫濈墱閺€浠嬫煕濞嗗繐鈧綊寮?
        res_flow_net_path = getattr(self, 'res_flow_net_path', None)
        if res_flow_net_path and os.path.exists(res_flow_net_path):
            self.stage2_modules['res_flow_net'].load_state_dict(
                torch.load(res_flow_net_path, map_location=f'cuda:{self.rank}'))
            print(f"Loaded ResFlowNet from {res_flow_net_path}")

        # 闂佸憡鍔樺畷鐢靛垝閵娧呯當妞ゆ垼娉曢惂宀勬⒒閸愵厼鐓愭い鏂挎湰濞煎繘宕ㄩ鐐愭ɑ淇婇妞诲亾閾忣偄浠?        self.stage2_modules['rigid_flow_calc'].eval()
        self.stage2_modules['dynamic_gaussian'].eval()

    @torch.no_grad()
    def forward(self, inputs):
        """
        闂佸憡鎸哥粔鎾箖濠婂嫬顕遍柣妯挎珪鐏?- 闂佽浜介崝搴ㄥ箖婵犲伣鐔煎灳瀹曞洨顢?
        Args:
            inputs: 闁哄鐗婇幐鎼佸矗閸℃稑鏋侀柣妤€鐗嗙粊锕傛倵濞戞瑯娈旈柛姘ｅ亾闂佹寧绋戦懟顖溾偓鍨耿瀹曘儵鎮?1闂佸憡绮岄?1闂佸搫鍟冲▔娑㈠春閵忋倖鍎嶉柛鏇ㄥ亝缁傚牓鏌?
        Returns:
            outputs: 闁哄鐗婇幐鎼佸吹椤撶姭鍋撳☉娆樻當闁告埃鍋撻梺鎸庣☉閼活垳鈧灚锕㈠畷銉╊敃閿濆牜娲梺鍝勬湰閹哥霉濮椻偓瀹曟捁绠涚€Ｑ冧壕濞达絿顭堢敮銊︾箾缂堢娀妾烽柍褜鍏涚欢姘舵偟椤曗偓瀵剟顢涘顒侇棟闂佽桨鐒︽竟鍡涙偤?        """
        outputs = {}

        # 1. 缂備焦顨忛崗娑氱博閹绢喗鈷撻柤鍛婎問閸? 婵炶揪绲界粔鏉戝礂闂佸憡绮岄張顒傛崲娴ｈ鍎熼柨鏇炲亞閺嗘洘绻?        stage1_outputs = self.compute_stage1_outputs(inputs)
        outputs.update(stage1_outputs)

        # 2. 缂備焦顨忛崗娑氳姳閳哄懏鈷撻柤鍛婎問閸? 闂佺绻愰ˇ鎵矈閿旂偓濯奸柨娑樺閺?        stage2_outputs = self.compute_stage2_outputs(inputs, outputs)
        outputs.update(stage2_outputs)

        # 3. 濠电偞鎸稿鍫曟偂鐎ｎ喖妫橀柟宄扮焾濞兼帡鎮?        rendered_output = self.render_novel_view(inputs, outputs, stage2_outputs)
        outputs.update(rendered_output)

        return outputs

    def compute_stage1_outputs(self, inputs):
        """闁荤姳绶ょ槐鏇㈡偩閼姐倗绠旀い鎴ｆ硶椤忛亶姊婚崘顓炵厫妞ゆ柨鏈蹇涘箻閸愬弶鐦?""
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        # 婵☆偅婢樼€氼垶顢橀崫銉т笉婵°倐鍋撴い锕€顭峰畷锝夊磼閻愨晛浜鹃柛鈩冪懅閸欐劙姊?        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])

        # 婵炶揪绲界粔鏉戝礂婵☆偅婢樼€氼厾绮?        pose_pred = self.predict_pose(inputs)

        # 濠电儑绲藉畷顒傗偓纭呮珪閿涙劙宕熼鍌樺仦
        depth_feats = self.predict_depth(inputs)

        # 闂佽桨鐒﹀娆撳箖鎼淬垺缍囬柟鎯у暱濮?        for cam in range(self.num_cams):
            outputs[('cam', cam)].update(pose_pred[('cam', cam)])
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])

        # 闁荤姳绶ょ槐鏇㈡偩鐠囩潿搴ㄥ础閻愬樊鍞洪梺?        self.compute_depth_maps(inputs, outputs)

        # 闁荤姳绶ょ槐鏇㈡偩缂佹﹩娈楁俊顖滅帛閻掑潡鏌℃担鍝勵暭鐎?        if self.gaussian:
            for cam in range(self.num_cams):
                self.get_gaussian_data(inputs, outputs, cam)

        return outputs

    def predict_pose(self, inputs):
        """婵☆偅婢樼€氼厾绮婄€涙ɑ濯寸€广儱鎲?""
        net = self.models['pose_net']
        pose = self.pose.compute_pose(net, inputs)
        return pose

    def predict_depth(self, inputs):
        """婵☆偅婢樼€氼厾绮婇弶鎳酣宕￠悙宸敽"""
        net = self.models['depth_net']
        depth_feats = net(inputs)
        return depth_feats

    def compute_depth_maps(self, inputs, outputs):
        """闁荤姳绶ょ槐鏇㈡偩鐠囩潿搴ㄥ础閻愬樊鍞洪梺?""
        source_scale = 0
        for cam in range(self.num_cams):
            ref_K = inputs[('K', source_scale)][:, cam, ...]
            for scale in self.scales:
                disp = outputs[('cam', cam)][('disp', scale)]
                outputs[('cam', cam)][('depth', 0, scale)] = self.to_depth(disp, ref_K)
                if self.novel_view_mode == 'MF':
                    disp_last = outputs[('cam', cam)][('disp', -1, scale)]
                    outputs[('cam', cam)][('depth', -1, scale)] = self.to_depth(disp_last, ref_K)
                    disp_next = outputs[('cam', cam)][('disp', 1, scale)]
                    outputs[('cam', cam)][('depth', 1, scale)] = self.to_depth(disp_next, ref_K)

    def to_depth(self, disp_in, K_in):
        """闁荤喐鐟ュΛ妤€螣婵犲啯濮滄い鎺嶈兌缁犳帡骞?""
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        disp_range = max_disp - min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1 / disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2) / self.focal_length_scale

    def get_gaussian_data(self, inputs, outputs, cam):
        """闁荤姳绶ょ槐鏇㈡偩缂佹﹩娈楁俊顖滅帛閻掑潡鏌℃担鍝勵暭鐎?""
        from models.gaussian import depth2pc, rotate_sh
        from einops import rearrange

        bs, _, height, width = inputs[('color', 0, 0)][:, cam, ...].shape
        zfar = self.max_depth
        znear = 0.01

        if self.novel_view_mode == 'MF':
            for frame_id in [0, -1, 1]:
                outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...] if frame_id == 0 else \
                    torch.matmul(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)], inputs['extrinsics_inv'][:, cam, ...])
                outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...] if frame_id == 0 else \
                    torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)]))

                outputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(
                    outputs[('cam', cam)][('depth', frame_id, 0)],
                    outputs[('cam', cam)][('e2c_extr', frame_id, 0)],
                    inputs[('K', 0)][:, cam, ...]
                )
                valid = outputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
                outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)

                rot_maps, scale_maps, opacity_maps, sh_maps = self.models['gs_net'](
                    inputs[('color', frame_id, 0)][:, cam, ...],
                    outputs[('cam', cam)][('depth', frame_id, 0)],
                    outputs[('cam', cam)][('img_feat', frame_id, 0)]
                )

                c2w_rotations = rearrange(
                    outputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3],
                    "k i j -> k () () () i j"
                )
                sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])

                outputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps
                outputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps
                outputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps
                outputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_maps

    def compute_stage2_outputs(self, inputs, outputs):
        """闁荤姳绶ょ槐鏇㈡偩閼姐倗绠旀い鎴ｆ硶閻у矂姊婚崘顓炵厫妞ゆ柨鏈蹇涘箻閸愬弶鐦?""
        F_rigid_t_minus_1_list = []
        F_rigid_t_plus_1_list = []
        mask_t_minus_1_list = []
        mask_t_plus_1_list = []

        for cam in range(self.num_cams):
            depth_t_minus_1 = outputs[('cam', cam)][('depth', -1, 0)]
            depth_t_plus_1 = outputs[('cam', cam)][('depth', 1, 0)]

            T_t_minus_1_to_t = outputs[('cam', cam)][('cam_T_cam', 0, -1)]
            T_t_plus_1_to_t = outputs[('cam', cam)][('cam_T_cam', 0, 1)]

            K = inputs[('K', 0)][:, cam, ...]

            F_rigid_t_minus_1, mask_t_minus_1 = self.stage2_modules['rigid_flow_calc'](
                depth_t_minus_1, K, T_t_minus_1_to_t
            )
            F_rigid_t_plus_1, mask_t_plus_1 = self.stage2_modules['rigid_flow_calc'](
                depth_t_plus_1, K, T_t_plus_1_to_t
            )

            F_rigid_t_minus_1_list.append(F_rigid_t_minus_1)
            F_rigid_t_plus_1_list.append(F_rigid_t_plus_1)
            mask_t_minus_1_list.append(mask_t_minus_1)
            mask_t_plus_1_list.append(mask_t_plus_1)

        F_rigid_t_minus_1 = torch.stack(F_rigid_t_minus_1_list, dim=1)
        F_rigid_t_plus_1 = torch.stack(F_rigid_t_plus_1_list, dim=1)
        mask_t_minus_1 = torch.stack(mask_t_minus_1_list, dim=1)
        mask_t_plus_1 = torch.stack(mask_t_plus_1_list, dim=1)

        I_t_minus_1 = inputs[('color', -1, 0)]
        I_t_plus_1 = inputs[('color', 1, 0)]
        I_t = inputs[('color', 0, 0)]

        warped_t_minus_1 = batch_warp_image_with_flow(I_t_minus_1, F_rigid_t_minus_1)
        warped_t_plus_1 = batch_warp_image_with_flow(I_t_plus_1, F_rigid_t_plus_1)

        F_residual_t_minus_1 = self.stage2_modules['res_flow_net'](
            warped_t_minus_1, I_t, F_rigid_t_minus_1
        )
        F_residual_t_plus_1 = self.stage2_modules['res_flow_net'](
            warped_t_plus_1, I_t, F_rigid_t_plus_1
        )

        F_total_t_minus_1 = F_rigid_t_minus_1 + F_residual_t_minus_1
        F_total_t_plus_1 = F_rigid_t_plus_1 + F_residual_t_plus_1

        return {
            'F_rigid_t_minus_1': F_rigid_t_minus_1,
            'F_rigid_t_plus_1': F_rigid_t_plus_1,
            'F_residual_t_minus_1': F_residual_t_minus_1,
            'F_residual_t_plus_1': F_residual_t_plus_1,
            'F_total_t_minus_1': F_total_t_minus_1,
            'F_total_t_plus_1': F_total_t_plus_1,
            'mask_t_minus_1': mask_t_minus_1,
            'mask_t_plus_1': mask_t_plus_1,
        }

    def render_novel_view(self, inputs, outputs, stage2_outputs):
        """濠电偞鎸稿鍫曟偂鐎ｎ喖妫橀柟宄扮焾濞兼帡鎮?""
        from models.gaussian import pts2render

        rendered_imgs = []
        for cam in range(self.num_cams):
            rendered_img = pts2render(
                inputs=inputs,
                outputs=outputs,
                cam_num=self.num_cams,
                novel_cam=cam,
                novel_frame_id=0,
                bg_color=[1.0, 1.0, 1.0],
                mode=self.novel_view_mode
            )
            rendered_imgs.append(rendered_img)

        rendered_I_t = torch.stack(rendered_imgs, dim=1)

        return {'rendered_I_t': rendered_I_t}


def save_image(tensor, path):
    """婵烇絽娲︾换鍌炴偤閵娾晛鐐婇柟顖嗗啫澹杢ensor闂佸憡甯炴晶妤呭几閸愨晝顩?""
    # 缂佺虎鍙庨崰鏇犳崲濮濈ざnsor闂侀潻鑵归埀顒佸綃U婵?    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 闁哄鍎愰崜姘暦閸欏鈻旂紒銊ュmpy濡ょ姷鍋為崕濂告儍閻旂厧鏋侀悗闈涙啞濡﹪骞?    if tensor.dim() == 4:
        # [B, C, H, W] -> 婵烇絽娲︾换鍌炴偤閵娧呯當妞ゆ垼娉曢鍗炩槈閹垮啩绨奸柣妤呬憾瀵?        tensor = tensor[0]

    # [C, H, W] -> [H, W, C]
    img_np = tensor.permute(1, 2, 0).numpy()

    # 闁荤喍妞掔粈浣圭珶閳ь剟鏌涢幒鏃€鐝?, 1]闂佽偐鍘ч崯顐⒚?    img_np = np.clip(img_np, 0, 1)

    # 闁哄鍎愰崜姘暦閸欏鈻旂紒銊ョ┘nt8
    img_np = (img_np * 255).astype(np.uint8)

    # 婵烇絽娲︾换鍌炴偤?    Image.fromarray(img_np).save(path)


def save_flow_visualization(flow, path):
    """婵烇絽娲︾换鍌炴偤閵娾晛绀傚璺侯儑閵堬箓鏌涘▎妯虹仴妞ゎ偄顑夊畷?""
    import matplotlib.pyplot as plt

    if flow.is_cuda:
        flow = flow.cpu()

    if flow.dim() == 4:
        flow = flow[0]

    # [2, H, W] -> [H, W, 2]
    flow_np = flow.permute(1, 2, 0).numpy()

    # 闁荤姳绶ょ槐鏇㈡偩婵犳艾绀傚璺侯儑閵堬附顨ラ悙鍙夊櫣閻?    flow_mag = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)

    # 闂佸憡鐟崹鐢革綖鐎ｎ喖绀?    plt.figure(figsize=(10, 8))
    plt.imshow(flow_mag, cmap='jet')
    plt.colorbar()
    plt.title('Optical Flow Magnitude')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_raw_npz_arrays(outputs, inputs, path):
    """
    Save raw arrays for quantitative comparison across checkpoints.
    """
    payload = {}

    rendered = outputs['rendered_I_t'].detach().cpu().numpy().astype(np.float16)
    gt = inputs[('color', 0, 0)].detach().cpu().numpy().astype(np.float16)

    payload['rendered'] = rendered
    payload['gt'] = gt

    if 'F_residual_t_minus_1' in outputs:
        mag_m1 = torch.linalg.norm(outputs['F_residual_t_minus_1'], dim=2)  # [B, N, H, W]
        payload['residual_mag_t_minus_1'] = mag_m1.detach().cpu().numpy().astype(np.float16)
    if 'F_residual_t_plus_1' in outputs:
        mag_p1 = torch.linalg.norm(outputs['F_residual_t_plus_1'], dim=2)  # [B, N, H, W]
        payload['residual_mag_t_plus_1'] = mag_p1.detach().cpu().numpy().astype(np.float16)
    if 'F_total_t_minus_1' in outputs:
        total_m1 = torch.linalg.norm(outputs['F_total_t_minus_1'], dim=2)  # [B, N, H, W]
        payload['total_mag_t_minus_1'] = total_m1.detach().cpu().numpy().astype(np.float16)
    if 'F_total_t_plus_1' in outputs:
        total_p1 = torch.linalg.norm(outputs['F_total_t_plus_1'], dim=2)  # [B, N, H, W]
        payload['total_mag_t_plus_1'] = total_p1.detach().cpu().numpy().astype(np.float16)

    np.savez_compressed(path, **payload)


def main():
    """婵炴垶鎹侀褔宕甸柆宥呮瀬?""
    args = parse_args()

    if args.max_batches < 0 and args.num_samples > 0:
        args.max_batches = int(np.ceil(args.num_samples / max(args.batch_size, 1)))

    # 闂佸憡姊绘慨鎯归崶顒佺厐鐎广儱娲ㄩ弸?    cfg = load_config(args.config_file)

    res_flow_net_path = args.res_flow_net_path or args.checkpoint_path
    if not res_flow_net_path:
        raise ValueError("Please provide --res_flow_net_path (or --checkpoint_path).")
    cfg['training']['res_flow_net_path'] = res_flow_net_path

    print("=" * 80)
    print("Stage2 Inference for DrivingForward MF Mode")
    print("=" * 80)
    print(f"Config file: {args.config_file}")
    print(f"ResFlowNet path: {res_flow_net_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # 闂佸憡甯楃粙鎴犵磽閹惧瓨缍囬柟鎯у暱濮ｅ鏌ｉ埡濠傛灈缂?    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_visualizations:
        (output_dir / 'images').mkdir(exist_ok=True)
    if args.save_flow:
        (output_dir / 'flow').mkdir(exist_ok=True)
    if args.save_gaussians:
        (output_dir / 'gaussians').mkdir(exist_ok=True)
    if args.save_raw_npz:
        (output_dir / 'raw').mkdir(exist_ok=True)

    # 闁荤姳绀佹晶浠嬫偪閸℃瑦濯奸柟顖嗗本校
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 闂佸憡甯楃粙鎴犵磽閹炬緞鐔煎灳瀹曞洠鍋?    print("Loading model...")
    model = Stage2InferenceModel(cfg, rank=0)
    model = model.to(device)

    # 闂佸憡鍨靛Λ妤吽囬鍕瀬闁绘鐗嗙粊?    print("Preparing dataset...")
    eval_augmentation = {
        'image_shape': (int(cfg['training']['height']), int(cfg['training']['width'])),
        'jittering': (0.0, 0.0, 0.0, 0.0),
        'crop_train_borders': (),
        'crop_eval_borders': ()
    }

    eval_dataset = construct_dataset(cfg, 'eval', **eval_augmentation)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Dataset size: {len(eval_dataset)}")

    # 闂佽浜介崝搴ㄥ箖?    print("Starting inference...")
    model.eval()

    _NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename', 'token']

    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(eval_dataloader, desc='Inference')):
            # 闂佽桨鑳舵晶妤€鐣垫担铏圭煋閻犲洦褰冭闂佸憡甯楁刊浠嬵敊閺囩喎绶?            for key, ipt in inputs.items():
                if key not in _NO_DEVICE_KEYS:
                    if 'context' in key or 'ego_pose' in key:
                        inputs[key] = [ipt[k].float().to(device) for k in range(len(inputs[key]))]
                    else:
                        inputs[key] = ipt.float().to(device)

            # 闂佸憡鎸哥粔鎾箖濠婂嫬顕遍柣妯挎珪鐏?            outputs = model(inputs)

            # 婵烇絽娲︾换鍌炴偤閵娧呯＜闁规儳顕禍?            if args.save_visualizations:
                # 婵烇絽娲︾换鍌炴偤閵娿儯鈧帡宕ｆ径灞藉脯闂佹悶鍎查崕鎶藉磿?                rendered = outputs['rendered_I_t']
                for cam_idx in range(rendered.shape[1]):
                    save_path = output_dir / 'images' / f'batch_{batch_idx:04d}_cam_{cam_idx}_rendered.png'
                    save_image(rendered[:, cam_idx], str(save_path))

                # 婵烇絽娲︾换鍌炴偤閳╁尡闂佹悶鍎查崕鎶藉磿?                gt = inputs[('color', 0, 0)]
                for cam_idx in range(gt.shape[1]):
                    save_path = output_dir / 'images' / f'batch_{batch_idx:04d}_cam_{cam_idx}_gt.png'
                    save_image(gt[:, cam_idx], str(save_path))

            if args.save_flow:
                # 婵烇絽娲︾换鍌炴偤閵娾晛绀傚璺侯儑閵堬箓鏌涘▎妯虹仴妞ゎ偄顑夊畷?                for direction, key in [('t-1_to_t', 'F_total_t_minus_1'), ('t+1_to_t', 'F_total_t_plus_1')]:
                    flow = outputs[key]
                    for cam_idx in range(flow.shape[1]):
                        save_path = output_dir / 'flow' / f'batch_{batch_idx:04d}_cam_{cam_idx}_{direction}.png'
                        save_flow_visualization(flow[:, cam_idx], str(save_path))

            if args.save_gaussians:
                # 婵烇絽娲︾换鍌炴偤閵婏富娈楁俊顖滅帛閻掑潡鏌涘▎蹇撯偓褰掑汲?                gaussian_data = {}
                for cam in range(cfg['model']['num_cams']):
                    cam_data = {}
                    for frame_id in [0, -1, 1]:
                        cam_data[f'xyz_{frame_id}'] = outputs[('cam', cam)][('xyz', frame_id, 0)].cpu().numpy()
                        cam_data[f'rot_{frame_id}'] = outputs[('cam', cam)][('rot_maps', frame_id, 0)].cpu().numpy()
                        cam_data[f'scale_{frame_id}'] = outputs[('cam', cam)][('scale_maps', frame_id, 0)].cpu().numpy()
                        cam_data[f'opacity_{frame_id}'] = outputs[('cam', cam)][('opacity_maps', frame_id, 0)].cpu().numpy()
                    gaussian_data[f'cam_{cam}'] = cam_data

                save_path = output_dir / 'gaussians' / f'batch_{batch_idx:04d}_gaussians.npz'
                np.savez_compressed(str(save_path), **gaussian_data)

            if args.save_raw_npz:
                save_path = output_dir / 'raw' / f'batch_{batch_idx:04d}_raw.npz'
                save_raw_npz_arrays(outputs, inputs, str(save_path))

            if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
                print(f"Reached max_batches={args.max_batches}, stopping early.")
                break

    print("=" * 80)
    print(f"Inference completed. Results saved to {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
