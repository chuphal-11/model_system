"""
Neural Nexus — Model Loader
===============================
Handles loading custom YOLOv7 models that contain non-standard layers
(MP, SPPCSPC, RepConv, IDetect, ImplicitA, ImplicitM, etc.).
Falls back to ultralytics YOLO for standard models (yolov8n.pt).
"""

import os
import sys
import types
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def autopad(k, p=None):
    """Compute auto-padding for Conv layers."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution: conv + BN + activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class MP(nn.Module):
    """Max Pooling layer."""
    def __init__(self, k=2):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    """Spatial Pyramid Pooling layer."""
    def __init__(self, k=3, s=1):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class SPPCSPC(nn.Module):
    """Spatial Pyramid Pooling - Cross Stage Partial Connection."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    """Reparameterizable Convolution (used in YOLOv7)."""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1)
                                 if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


class ImplicitA(nn.Module):
    """Implicit addition layer."""
    def __init__(self, channel, mean=0., std=0.02):
        super().__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    """Implicit multiplication layer."""
    def __init__(self, channel, mean=1., std=0.02):
        super().__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        return self.implicit * x


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class IDetect(nn.Module):
    """YOLOv7 Detect head with implicit layers."""
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Detect(nn.Module):
    """Standard YOLO Detect head."""
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Contract(nn.Module):
    """Contract width-height into channels."""
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * s * s, H // s, W // s)


class Expand(nn.Module):
    """Expand channels into width-height."""
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, s, s, C // s // s, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(N, C // s // s, H * s, W * s)


_CUSTOM_CLASSES = {
    "MP": MP, "SP": SP, "Conv": Conv, "SPPCSPC": SPPCSPC,
    "RepConv": RepConv, "ImplicitA": ImplicitA, "ImplicitM": ImplicitM,
    "Bottleneck": Bottleneck, "C3": C3, "SPPF": SPPF,
    "IDetect": IDetect, "Detect": Detect, "Concat": Concat,
    "Contract": Contract, "Expand": Expand,
}


def _patch_modules():
    """
    Inject all custom YOLOv7 classes into the module namespaces that
    torch.load expects (models.common, models.yolo).
    """
    try:
        import yolov5
    except ImportError:
        pass

    if "models" not in sys.modules:
        models_mod = types.ModuleType("models")
        models_mod.__path__ = []
        sys.modules["models"] = models_mod
    else:
        models_mod = sys.modules["models"]

    if "models.common" not in sys.modules:
        common_mod = types.ModuleType("models.common")
        sys.modules["models.common"] = common_mod
    else:
        common_mod = sys.modules["models.common"]

    if "models.yolo" not in sys.modules:
        yolo_mod = types.ModuleType("models.yolo")
        sys.modules["models.yolo"] = yolo_mod
    else:
        yolo_mod = sys.modules["models.yolo"]

    if "models.experimental" not in sys.modules:
        exp_mod = types.ModuleType("models.experimental")
        sys.modules["models.experimental"] = exp_mod
    else:
        exp_mod = sys.modules["models.experimental"]

    try:
        from yolov5.models import common as real_common
        for attr in dir(real_common):
            if not attr.startswith("__"):
                try:
                    setattr(common_mod, attr, getattr(real_common, attr))
                except Exception:
                    pass
    except ImportError:
        pass

    try:
        from yolov5.models import yolo as real_yolo
        for attr in dir(real_yolo):
            if not attr.startswith("__"):
                try:
                    setattr(yolo_mod, attr, getattr(real_yolo, attr))
                except Exception:
                    pass
    except ImportError:
        pass

    try:
        from yolov5.models import experimental as real_exp
        for attr in dir(real_exp):
            if not attr.startswith("__"):
                try:
                    setattr(exp_mod, attr, getattr(real_exp, attr))
                except Exception:
                    pass
    except ImportError:
        pass

    for name, cls in _CUSTOM_CLASSES.items():
        for mod in (common_mod, yolo_mod, exp_mod):
            if not hasattr(mod, name):
                setattr(mod, name, cls)

    if not hasattr(common_mod, "autopad"):
        common_mod.autopad = autopad
    if not hasattr(yolo_mod, "autopad"):
        yolo_mod.autopad = autopad

    models_mod.common = common_mod
    models_mod.yolo = yolo_mod
    models_mod.experimental = exp_mod

    _patch_utils_modules()

    logger.debug("Patched models.common, models.yolo, models.experimental "
                 "with custom YOLOv7 classes")


def _patch_utils_modules():
    """Patch utils.* modules that torch.load may need to resolve."""
    try:
        from yolov5 import utils as yolov5_utils
        submodules = ["general", "torch_utils", "autoanchor", "plots",
                      "metrics", "dataloaders", "augmentations"]
        for submod_name in submodules:
            full_name = f"utils.{submod_name}"
            if full_name not in sys.modules:
                try:
                    import importlib
                    real = importlib.import_module(f"yolov5.utils.{submod_name}")
                    shim = types.ModuleType(full_name)
                    for attr in dir(real):
                        if not attr.startswith("__"):
                            try:
                                setattr(shim, attr, getattr(real, attr))
                            except Exception:
                                pass
                    sys.modules[full_name] = shim
                except (ImportError, AttributeError):
                    pass
    except ImportError:
        pass


class YOLOv7ModelWrapper:
    """
    Wrapper around a loaded YOLOv7 checkpoint that provides a simple
    inference interface.
    """

    def __init__(self, model, class_names, device="cpu", conf_threshold=0.25,
                 img_size=640):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.model.to(device)
        self.model.eval()

        self.stride = int(self.model.stride.max()) if hasattr(self.model, "stride") else 32

    def __call__(self, frame):
        """
        Run inference on a single frame (numpy BGR image).

        Returns:
            list of dicts with bbox, class_name, class_id, confidence
        """
        img = letterbox(frame, self.img_size, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

        pred = non_max_suppression(pred, self.conf_threshold, 0.45)

        detections = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4],
                                         frame.shape).round()
                for *xyxy, conf, cls_id in det:
                    cls_id = int(cls_id)
                    class_name = (self.class_names[cls_id]
                                  if cls_id < len(self.class_names)
                                  else f"class_{cls_id}")
                    detections.append({
                        "bbox": [int(xyxy[0]), int(xyxy[1]),
                                 int(xyxy[2]), int(xyxy[3])],
                        "class_name": class_name,
                        "class_id": cls_id,
                        "confidence": float(conf),
                    })
        return detections


def letterbox(im, new_shape=640, color=(114, 114, 114), auto=False,
              scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2
    dh /= 2

    import cv2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
                        max_det=300):
    """Run Non-Maximum Suppression on inference results."""
    import torchvision

    output = []
    for xi, x in enumerate(prediction):
        if x.shape[-1] == 5:
            mask = x[:, 4] > conf_thres
            x = x[mask]
            if x.shape[0] == 0:
                output.append(torch.zeros((0, 6), device=x.device))
                continue
            box = _xywh2xyxy(x[:, :4])
            conf = x[:, 4:5]
            cls = torch.zeros_like(conf)
            x = torch.cat([box, conf, cls], dim=1)
        else:
            x[:, 5:] *= x[:, 4:5]
            box = _xywh2xyxy(x[:, :4])
            conf, cls = x[:, 5:].max(1, keepdim=True)
            x = torch.cat([box, conf, cls], dim=1)
            x = x[conf.view(-1) > conf_thres]

        if x.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=x.device))
            continue

        boxes = x[:, :4]
        scores = x[:, 4]
        c = x[:, 5:6] * 4096
        boxes_offset = boxes + c
        keep = torchvision.ops.nms(boxes_offset, scores, iou_thres)
        keep = keep[:max_det]
        output.append(x[keep])

    return output


def _xywh2xyxy(x):
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescale boxes from img1_shape to img0_shape."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img0_shape[0])
    return boxes


def load_custom_model(model_path, class_names, device="cpu",
                      conf_threshold=0.25):
    """
    Load a custom YOLOv7 .pt model with all required custom classes.

    Args:
        model_path: Path to the .pt file
        class_names: List of class name strings
        device: 'cpu' or 'cuda'
        conf_threshold: Confidence threshold for detections

    Returns:
        YOLOv7ModelWrapper instance, or None if loading fails.
    """
    _patch_modules()

    try:
        logger.info(f"Loading custom model: {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt:
            model = ckpt["model"].float()
            if hasattr(model, "fuse"):
                try:
                    model = model.fuse()
                except Exception:
                    pass
        else:
            model = ckpt

        model.eval()

        for m in model.modules():
            cls_name = type(m).__name__
            if cls_name in ("IDetect", "Detect") and hasattr(m, "grid"):
                m.grid = [torch.zeros(1)] * m.nl
                if 'anchor_grid' in dict(m.named_buffers()):
                    del m._buffers['anchor_grid']
                m.anchor_grid = [torch.zeros(1)] * m.nl
                if hasattr(m, 'dynamic'):
                    m.dynamic = True

        wrapper = YOLOv7ModelWrapper(
            model=model,
            class_names=class_names,
            device=device,
            conf_threshold=conf_threshold,
        )
        logger.info(f"  ✓ Loaded {os.path.basename(model_path)} "
                     f"({len(class_names)} classes: {class_names})")
        return wrapper

    except Exception as e:
        logger.error(f"  ✗ Failed to load {os.path.basename(model_path)}: {e}")
        return None


def load_person_detector(model_path, device="cpu", conf_threshold=0.30):
    """
    Load the YOLOv8n person detector using the ultralytics library.

    Returns:
        A callable that takes a frame and returns person detections.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.to(device)
        logger.info(f"  ✓ Loaded person detector: {os.path.basename(model_path)} (device={device})")

        def detect_persons(frame):
            results = model(frame, verbose=False, conf=conf_threshold, device=device)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:
                        xyxy = box.xyxy[0].cpu().numpy()
                        detections.append({
                            "bbox": [int(xyxy[0]), int(xyxy[1]),
                                     int(xyxy[2]), int(xyxy[3])],
                            "class_name": "person",
                            "class_id": 0,
                            "confidence": float(box.conf[0]),
                        })
            return detections

        return detect_persons

    except Exception as e:
        logger.error(f"  ✗ Failed to load person detector: {e}")
        return None
