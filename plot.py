import matplotlib.pyplot as plt


def show_image(base, name, predicted=None):

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from scipy.ndimage.interpolation import rotate
    import numpy as np

    # load original image
    image_path = base + '/Images/' + name + '.jpg'
    img = mpimg.imread(image_path)

    # load mask
    mask_path = base + '/Masks/all/' + name + '.png'
    mask = mpimg.imread(mask_path)

    # prediction
    # predicted = rotate(mask, angle=2)
    # predicted = mask + 5
    predicted = mask

    diff = predicted - mask

    fig = plt.figure()
    ax_mask = fig.add_subplot(2, 2, 1)
    ax_pred = fig.add_subplot(2, 2, 2)
    ax_diff = fig.add_subplot(2, 2, 3)
    ax_raw = fig.add_subplot(2, 2, 4)

    ax_mask.imshow(img)
    ax_mask.imshow(mask, alpha=0.4)
    ax_mask.title.set_text('Mask')

    ax_pred.imshow(img)
    ax_pred.imshow(predicted, alpha=0.4)
    ax_pred.title.set_text('Predicted')

    ax_diff.imshow(img)
    ax_diff.imshow(diff, alpha=0.4)
    ax_diff.title.set_text('Diff')

    ax_raw.imshow(img)
    ax_raw.title.set_text('Original')

    plt.show()


def get_images(original, mask, predicted):

    from matplotlib import colors
    # cmap = plt.get_cmap('Set1', 4)

    # TODO: Only plot when value over 0 at mask?

    cmap = colors.ListedColormap(['green', 'red', 'purple', 'blue'])
    bounds = [-0.1, 0.1, 1.1, 2.1, 3.1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    original = original.to('cpu').numpy()[0].transpose([1, 2, 0])
    mask = mask.to('cpu').numpy()[0]
    predicted = predicted.to('cpu').numpy()[0]

    diff = predicted - mask

    fig = plt.figure()
    ax_mask = fig.add_subplot(2, 2, 1)
    ax_pred = fig.add_subplot(2, 2, 2)
    ax_diff = fig.add_subplot(2, 2, 3)
    ax_raw = fig.add_subplot(2, 2, 4)

    ax_mask.imshow(original)
    ax_mask.imshow(mask, alpha=0.4, cmap=cmap, norm=norm)
    ax_mask.title.set_text('Mask')

    ax_pred.imshow(original)
    ax_pred.imshow(predicted, alpha=0.4, cmap=cmap, norm=norm)
    ax_pred.title.set_text('Predicted')

    ax_diff.imshow(original)
    ax_diff.imshow(diff, alpha=0.4, cmap=cmap, norm=norm)
    ax_diff.title.set_text('Diff')

    ax_raw.imshow(original)
    ax_raw.title.set_text('Original')

    return fig


def draw_class_bitmaps(mask, prediction, image):
    fig = plt.figure(figsize=(12, 5))
    from config import CLASS_ORDER
    import numpy as np

    plt.subplot(2, 3, 1)
    plt.imshow(image.transpose((1, 2, 0)))
    plt.title('Image')

    plt.subplot(2, 3, 2)
    plt.imshow(mask)
    plt.clim(0, 3)
    plt.colorbar()
    plt.title('annotated mask')

    plt.subplot(2, 3, 3)
    plt.imshow(np.argmax(prediction, axis=0))
    plt.clim(0, 3)
    plt.colorbar()
    plt.title('predicted mask')

    for i in range(1, 4):
        plt.subplot(2, 3, i+3)
        plt.imshow(prediction[i])
        plt.clim(0, 1)
        plt.colorbar()
        plt.title(CLASS_ORDER[i])

    plt.tight_layout()

    return fig


if __name__ == '__main__':
    from main import load_model
    import torch
    import torch.nn.functional as F
    from dataprep import GLOBHEDataset, ToTensor
    model, device = load_model('unet_2019-09-29_1251.pth')
    with torch.no_grad():
        train_dataset = GLOBHEDataset('data', 'train')

        sample = train_dataset[8]
        sample = ToTensor()(sample)

        mask = sample['bitmap'].cpu().numpy()

        image_tensor = sample['image'].to(device).unsqueeze(0)
        out_bitmap, out_percentages = model(image_tensor)
        output_soft = F.softmax(out_bitmap, dim=1)

        draw_class_bitmaps(mask, output_soft.detach().numpy()[0], sample['image'].detach().numpy())
        plt.show()


