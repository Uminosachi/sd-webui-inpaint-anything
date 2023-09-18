import torch
from torchvision.ops.boxes import box_iou


def nms(bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(-scores).to(bboxes.device)
    indices = torch.arange(bboxes.shape[0]).to(bboxes.device)
    keep = torch.ones_like(indices, dtype=torch.bool).to(bboxes.device)
    for i in indices:
        if keep[i]:
            bbox = bboxes[order[i]]
            iou = box_iou(bbox[None, ...], (bboxes[order[i + 1:]]) * keep[i + 1:][..., None])
            overlapped = torch.nonzero(iou > iou_threshold)
            keep[overlapped + i + 1] = 0
    return order[keep]
