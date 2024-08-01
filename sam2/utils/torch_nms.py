import torch
from torchvision.ops.boxes import box_iou


def nms(bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(-scores)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())

        if order.numel() == 1:
            break

        ious = box_iou(bboxes[i].unsqueeze(0), bboxes[order[1:]])[0]
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, device=bboxes.device)
