import torch
from config import device


def calculate_segmentation_percentages(segmented_images):
    """
    :param segmented_images:
    :return:
    """
    from config import CLASS_ORDER

    # TODO: Use original segmentation (can be negative) or soft (0-1)?

    size = segmented_images.size()
    nbr_pixels = size[2]*size[3]

    segmentation_percentages = []
    for img_idx in range(size[0]):
        percentages = {}
        for class_idx in range(size[1]):
            # class_percentage = np.sum(segmented_images[img_idx, class_idx, :, :]) / nbr_pixels
            class_percentage = segmented_images[img_idx, class_idx, :, :].sum().item() / nbr_pixels * 100
            percentages[CLASS_ORDER[class_idx]] = class_percentage

        segmentation_percentages.append(percentages)

    return segmentation_percentages


def calculate_segmentation_percentage_error(predicted_percentages, real_percentages):
    """
    :param predicted_percentages:
    :param real_percentages:
    :return:
    """

    # TODO: The real percentage values are saved to a tensor as the "transpose" of how we calculate it....

    # init error per type to zero
    batch_percentage_error = {k: 0 for k in real_percentages.keys()}

    # loop over all images and add error to every type
    total_batch_error = 0
    nbr_images = len(predicted_percentages)
    for img_idx in range(nbr_images):
        for class_name in real_percentages.keys():
            error = abs(predicted_percentages[img_idx][class_name] - real_percentages[class_name][img_idx])
            batch_percentage_error[class_name] += error
            total_batch_error += error

    batch_percentage_error = {k: v/nbr_images for k, v in batch_percentage_error.items()}   # TODO: use mean??
    return batch_percentage_error, total_batch_error


def correct_mask_bitmaps_for_crop(bitmaps):
    from config import CLASS_ORDER

    size = bitmaps.size()
    class_names = CLASS_ORDER[1:]
    nbr_pixels = size[1]*size[2]

    bitmap_percentages = {}
    for class_idx, class_name in enumerate(class_names):
        class_percentage = torch.sum(torch.sum(bitmaps == (class_idx + 1), dim=2), dim=1).float() / nbr_pixels
        bitmap_percentages[class_name] = class_percentage.tolist()

    return bitmap_percentages


def ratio_loss_function(class_fractions, bitmaps):
    bitmap_fraction = torch.zeros(size=class_fractions.shape).to(device)
    for j in range(4):
        bitmap_fraction[:, j] = torch.sum(torch.sum(bitmaps == j, dim=2), dim=1).float() / (bitmaps.shape[1]*bitmaps.shape[2])

    loss = torch.mean((class_fractions - bitmap_fraction)**2)
    return loss
