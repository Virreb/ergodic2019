

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

    import matplotlib.pyplot as plt

    original = original.numpy()
    mask = mask.numpy()
    predicted = predicted.numpy()

    diff = predicted - mask

    fig = plt.figure()
    ax_mask = fig.add_subplot(2, 2, 1)
    ax_pred = fig.add_subplot(2, 2, 2)
    ax_diff = fig.add_subplot(2, 2, 3)
    ax_raw = fig.add_subplot(2, 2, 4)

    ax_mask.imshow(original)
    ax_mask.imshow(mask, alpha=0.4)
    ax_mask.title.set_text('Mask')

    ax_pred.imshow(original)
    ax_pred.imshow(predicted, alpha=0.4)
    ax_pred.title.set_text('Predicted')

    ax_diff.imshow(original)
    ax_diff.imshow(diff, alpha=0.4)
    ax_diff.title.set_text('Diff')

    ax_raw.imshow(original)
    ax_raw.title.set_text('Original')

    return fig

