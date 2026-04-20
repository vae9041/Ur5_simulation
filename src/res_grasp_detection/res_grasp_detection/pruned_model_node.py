#!/usr/bin/env python3

from collections import OrderedDict, deque
import glob
import os
import sys
import time

import cv2
import numpy as np
import rclpy
import torch
import torch.nn as nn
import torchvision.transforms as T
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops.misc import FrozenBatchNorm2d

if not hasattr(np, '_core'):
    class _CoreModule:
        multiarray = np.core.multiarray
        umath = np.core.umath
        numerictypes = np.core.numerictypes

    np._core = _CoreModule()
    sys.modules['numpy._core'] = np._core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core.numerictypes'] = np.core.numerictypes


def build_resnet18_faster_rcnn(num_classes: int) -> FasterRCNN:
    backbone = resnet_fpn_backbone(
        backbone_name='resnet18',
        weights=None,
        trainable_layers=3,
    )
    return FasterRCNN(backbone, num_classes=num_classes)


def replace_frozen_batchnorm_with_batchnorm2d(module: nn.Module) -> int:
    replaced = 0

    def recurse(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, FrozenBatchNorm2d):
                num_features = int(child.weight.shape[0])
                batch_norm = nn.BatchNorm2d(num_features, eps=child.eps, affine=True, track_running_stats=True)
                batch_norm.to(device=child.weight.device, dtype=child.weight.dtype)
                batch_norm.weight.data.copy_(child.weight)
                batch_norm.bias.data.copy_(child.bias)
                batch_norm.running_mean.data.copy_(child.running_mean)
                batch_norm.running_var.data.copy_(child.running_var)
                batch_norm.eval()
                setattr(parent, name, batch_norm)
                replaced += 1
            else:
                recurse(child)

    recurse(module)
    return replaced


class _BackboneWithUniformFPN(nn.Module):
    def __init__(self, body: nn.Module, fpn: nn.Module, projectors: nn.ModuleDict, out_channels: int):
        super().__init__()
        self.body = body
        self.fpn = fpn
        self.fpn_output_projectors = projectors
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> OrderedDict:
        features = self.fpn(self.body(x))
        return OrderedDict((key, self.fpn_output_projectors[key](value)) for key, value in features.items())


def checkpoint_requires_structured_loader(checkpoint) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    structured_keys = {
        'structured_pruning_meta',
        'prune_scope',
        'backbone_out_channels_after_prune',
        'replaced_frozen_bn',
    }
    return any(key in checkpoint for key in structured_keys)


def load_structured_pruned_state_dict(model: FasterRCNN, checkpoint, device: torch.device) -> None:
    structured_pruning_meta = checkpoint.get('structured_pruning_meta') if isinstance(checkpoint, dict) else None
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError('checkpoint must contain model_state_dict or be a state dict')

    if structured_pruning_meta and structured_pruning_meta.get('fpn_uniform_wrapper'):
        backbone = model.backbone
        if not isinstance(backbone, _BackboneWithUniformFPN):
            dtype = next(backbone.parameters()).dtype
            uniform_channels = int(structured_pruning_meta['uniform_c'])
            projectors = nn.ModuleDict()
            for level in structured_pruning_meta['fpn_levels']:
                key = level['key']
                input_channels = int(level['in_ch'])
                if input_channels == uniform_channels:
                    projectors[key] = nn.Identity()
                else:
                    projector = nn.Conv2d(input_channels, uniform_channels, kernel_size=1, bias=False)
                    projectors[key] = projector.to(device=device, dtype=dtype)
            model.backbone = _BackboneWithUniformFPN(backbone.body, backbone.fpn, projectors, uniform_channels).to(
                device=device,
                dtype=dtype,
            )

    fpn = model.backbone.fpn
    for index in range(len(fpn.layer_blocks)):
        weight_key = f'backbone.fpn.layer_blocks.{index}.0.weight'
        bias_key = f'backbone.fpn.layer_blocks.{index}.0.bias'
        if weight_key not in state_dict:
            continue
        weight = state_dict[weight_key]
        if weight.ndim != 4:
            continue
        out_channels, in_channels, kernel_height, kernel_width = weight.shape
        has_bias = bias_key in state_dict
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_height, kernel_width),
            stride=1,
            padding=((kernel_height - 1) // 2, (kernel_width - 1) // 2),
            bias=has_bias,
        ).to(device=device, dtype=weight.dtype)
        fpn.layer_blocks[index] = nn.Sequential(conv)

    reconstructed_backbone_out_channels = None

    if 'rpn.head.conv.0.0.weight' in state_dict and 'rpn.head.cls_logits.weight' in state_dict:
        backbone_out_channels = int(state_dict['rpn.head.conv.0.0.weight'].shape[0])
        num_anchors = int(state_dict['rpn.head.cls_logits.weight'].shape[0])
        model.rpn.head = RPNHead(backbone_out_channels, num_anchors).to(
            device=device,
            dtype=state_dict['rpn.head.conv.0.0.weight'].dtype,
        )
        reconstructed_backbone_out_channels = backbone_out_channels

    if 'roi_heads.box_head.fc6.weight' in state_dict and 'roi_heads.box_predictor.cls_score.weight' in state_dict:
        fc6_input_features = int(state_dict['roi_heads.box_head.fc6.weight'].shape[1])
        representation_size = int(state_dict['roi_heads.box_head.fc6.weight'].shape[0])
        num_classes = int(state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0])
        model.roi_heads.box_head = TwoMLPHead(fc6_input_features, representation_size).to(
            device=device,
            dtype=state_dict['roi_heads.box_head.fc6.weight'].dtype,
        )
        model.roi_heads.box_predictor = FastRCNNPredictor(representation_size, num_classes).to(
            device=device,
            dtype=state_dict['roi_heads.box_predictor.cls_score.weight'].dtype,
        )

    if structured_pruning_meta and structured_pruning_meta.get('fpn_uniform_wrapper'):
        reconstructed_backbone_out_channels = int(structured_pruning_meta['uniform_c'])
    elif reconstructed_backbone_out_channels is None:
        layer_block_weight_key = 'backbone.fpn.layer_blocks.0.0.weight'
        if layer_block_weight_key in state_dict:
            reconstructed_backbone_out_channels = int(state_dict[layer_block_weight_key].shape[0])

    if reconstructed_backbone_out_channels is not None:
        model.backbone.out_channels = reconstructed_backbone_out_channels

    model.load_state_dict(state_dict, strict=True)


def resolve_default_model_path() -> str:
    package_share_directory = get_package_share_directory('res_grasp_detection')
    pruned_model_directory = os.path.join(
        package_share_directory,
        'Pruned_Model',
    )
    model_directory = os.path.join(
        package_share_directory,
        'Model',
    )
    candidate_paths = [
        os.path.join(pruned_model_directory, 'structured_pruned.pth'),
        os.path.join(pruned_model_directory, 'final_pruned_finetuned.pth'),
        os.path.join(pruned_model_directory, 'best_pruned_model.pth'),
        os.path.join(pruned_model_directory, 'pruned_model.pth'),
        os.path.join(model_directory, 'structured_pruned.pth'),
        os.path.join(model_directory, 'best_pruned_model.pth'),
        os.path.join(model_directory, 'pruned_model.pth'),
    ]
    candidate_paths.extend(sorted(glob.glob(os.path.join(pruned_model_directory, '*structured*.pth'))))
    candidate_paths.extend(sorted(glob.glob(os.path.join(pruned_model_directory, '*pruned*.pth'))))
    candidate_paths.extend(sorted(glob.glob(os.path.join(model_directory, '*structured*.pth'))))
    candidate_paths.extend(sorted(glob.glob(os.path.join(model_directory, '*pruned*.pth'))))
    candidate_paths.append(os.path.join(model_directory, 'final_pruned_finetuned.pth'))
    candidate_paths.append(os.path.join(model_directory, 'best_model.pth'))

    seen_paths = set()
    for candidate_path in candidate_paths:
        if candidate_path in seen_paths:
            continue
        seen_paths.add(candidate_path)
        if os.path.exists(candidate_path):
            return candidate_path
    return candidate_paths[0]


class PrunedModelNode(Node):
    def __init__(self):
        super().__init__('pruned_model_node')

        default_model_path = resolve_default_model_path()

        self.declare_parameter('topics.rgb', '/camera/image_raw')
        self.declare_parameter('topics.depth', '/camera/depth/image_raw')
        self.declare_parameter('topics.image_annotated', '/grasp_detection/image_annotated')
        self.declare_parameter('topics.object_position_camera_frame', '/grasp_detection/object_position_camera_frame')
        self.declare_parameter('camera.frame_id', 'camera_optical_link')
        self.declare_parameter('model.path', default_model_path)
        self.declare_parameter('model.num_classes', 2)
        self.declare_parameter('model.score_threshold', 0.7)
        self.declare_parameter('camera.fx', 528.433756558705)
        self.declare_parameter('camera.fy', 528.433756558705)
        self.declare_parameter('camera.cx', 320.5)
        self.declare_parameter('camera.cy', 240.5)
        self.declare_parameter('performance.window_size', 30)
        self.declare_parameter('performance.log_every_n_frames', 10)

        rgb_topic = str(self.get_parameter('topics.rgb').value)
        depth_topic = str(self.get_parameter('topics.depth').value)
        annotated_topic = str(self.get_parameter('topics.image_annotated').value)
        position_topic = str(self.get_parameter('topics.object_position_camera_frame').value)
        self.camera_frame_id = str(self.get_parameter('camera.frame_id').value)
        self.model_path = str(self.get_parameter('model.path').value)
        self.num_classes = int(self.get_parameter('model.num_classes').value)
        self.score_threshold = float(self.get_parameter('model.score_threshold').value)
        self.fx = float(self.get_parameter('camera.fx').value)
        self.fy = float(self.get_parameter('camera.fy').value)
        self.cx = float(self.get_parameter('camera.cx').value)
        self.cy = float(self.get_parameter('camera.cy').value)
        self.performance_window_size = max(1, int(self.get_parameter('performance.window_size').value))
        self.performance_log_every_n_frames = max(1, int(self.get_parameter('performance.log_every_n_frames').value))

        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        self.image_pub = self.create_publisher(Image, annotated_topic, 10)
        self.camera_position_pub = self.create_publisher(PointStamped, position_topic, 10)

        self.bridge = CvBridge()
        self.transform = T.ToTensor()
        self.latest_depth_image = None
        self.latest_depth_frame = None
        self.latest_depth_stamp = None
        self.latest_depth_encoding = None
        self.inference_times_ms = deque(maxlen=self.performance_window_size)
        self.processed_frames = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        self.get_logger().info(f'Loading model from: {self.model_path}')

        self.model = build_resnet18_faster_rcnn(self.num_classes)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if checkpoint_requires_structured_loader(checkpoint):
            replaced_batch_norm_layers = replace_frozen_batchnorm_with_batchnorm2d(self.model)
            self.get_logger().info(f'Replaced FrozenBatchNorm2d layers for structured checkpoint: {replaced_batch_norm_layers}')
            load_structured_pruned_state_dict(self.model, checkpoint, self.device)
        else:
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            load_result = self.model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                self.get_logger().warn(f'Missing checkpoint keys: {load_result.missing_keys}')
            if load_result.unexpected_keys:
                self.get_logger().warn(f'Unexpected checkpoint keys: {load_result.unexpected_keys}')
        self.model.to(self.device)
        self.model.eval()

        self.get_logger().info(
            f'Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}'
        )
        self.get_logger().info('Pruned ResNet-18 Faster R-CNN model loaded and ready!')

    def run_timed_inference(self, input_tensor):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions = self.model([input_tensor])[0]
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start_time) * 1000.0
        return predictions, inference_ms

    def update_inference_stats(self, inference_ms: float):
        self.processed_frames += 1
        self.inference_times_ms.append(inference_ms)
        average_inference_ms = sum(self.inference_times_ms) / len(self.inference_times_ms)
        average_fps = 1000.0 / average_inference_ms if average_inference_ms > 0.0 else 0.0
        if self.processed_frames == 1 or self.processed_frames % self.performance_log_every_n_frames == 0:
            self.get_logger().info(
                f'Inference performance: current={inference_ms:.2f} ms, avg={average_inference_ms:.2f} ms, avg_fps={average_fps:.2f}, frames={self.processed_frames}'
            )
        return average_inference_ms, average_fps

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_encoding = msg.encoding
            self.latest_depth_frame = msg.header.frame_id
            self.latest_depth_stamp = msg.header.stamp
            shape = getattr(self.latest_depth_image, 'shape', None)
            self.get_logger().info(
                f"Depth image received (frame='{self.latest_depth_frame}', encoding='{self.latest_depth_encoding}', shape={shape})"
            )
        except Exception as e:
            self.get_logger().error(f'Depth image conversion failed: {e}')

    def rgb_callback(self, msg):
        if self.latest_depth_image is None:
            self.get_logger().warn('Waiting for depth image...')
            return

        try:
            if self.latest_depth_stamp is not None:
                rgb_t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                depth_t = float(self.latest_depth_stamp.sec) + float(self.latest_depth_stamp.nanosec) * 1e-9
                if abs(rgb_t - depth_t) > 0.2:
                    self.get_logger().warn(
                        f'Depth/RGB timestamps too far apart (rgb={rgb_t:.3f}, depth={depth_t:.3f}); skipping frame'
                    )
                    return
        except Exception:
            pass

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            input_tensor = self.transform(cv_image).to(self.device)

            predictions, inference_ms = self.run_timed_inference(input_tensor)
            average_inference_ms, average_fps = self.update_inference_stats(inference_ms)

            best_score = 0.0
            best_box = None

            for box, score in zip(predictions['boxes'], predictions['scores']):
                score_value = float(score.item())
                if score_value > self.score_threshold and score_value > best_score:
                    best_score = score_value
                    best_box = box.int().tolist()

            if best_box is None:
                return

            x1, y1, x2, y2 = best_box
            self.get_logger().info(
                f'Selected bbox score={best_score:.3f} box=({x1}, {y1}) -> ({x2}, {y2})'
            )
            bbox_cx, bbox_cy = (x1 + x2) // 2, (y1 + y2) // 2

            h, w = self.latest_depth_image.shape[:2]
            if bbox_cx < 0 or bbox_cx >= w or bbox_cy < 0 or bbox_cy >= h:
                self.get_logger().warn(
                    f'Projection pixel ({bbox_cx},{bbox_cy}) outside depth image bounds {w}x{h}'
                )
                return

            depth_raw = float(self.latest_depth_image[bbox_cy, bbox_cx])
            if not np.isfinite(depth_raw) or depth_raw <= 0.0:
                self.get_logger().warn(
                    f'Invalid depth at projection pixel ({bbox_cx},{bbox_cy}) = {depth_raw}'
                )
                return

            depth_val = depth_raw
            if self.latest_depth_encoding == '16UC1':
                depth_val = depth_val / 1000.0
            elif depth_val > 50.0:
                depth_val = depth_val / 1000.0

            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(cv_image, (bbox_cx, bbox_cy), 5, (0, 0, 255), -1)
            cv2.putText(
                cv_image,
                f'Depth: {depth_val:.2f}m',
                (bbox_cx, bbox_cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                cv_image,
                f'Infer: {inference_ms:.1f} ms ({average_fps:.1f} FPS avg)',
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                cv_image,
                f'Avg latency: {average_inference_ms:.1f} ms',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            z_cam = depth_val
            x_cam = ((bbox_cx - self.cx) * z_cam) / self.fx
            y_cam = ((bbox_cy - self.cy) * z_cam) / self.fy

            self.get_logger().info(
                f'Projection pixel: ({bbox_cx}, {bbox_cy}), Depth: {z_cam:.3f}m'
            )
            self.get_logger().info(
                f'Camera optical frame: X={x_cam:.3f} Y={y_cam:.3f} Z={z_cam:.3f}'
            )

            camera_point = PointStamped()
            camera_point.header.stamp = self.get_clock().now().to_msg()
            camera_point.header.frame_id = self.camera_frame_id
            camera_point.point.x = float(x_cam)
            camera_point.point.y = float(y_cam)
            camera_point.point.z = float(z_cam)
            self.camera_position_pub.publish(camera_point)

            annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Image processing failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    node = PrunedModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
