from imports import *


def plot_filters_multi_channel(t, path=""):

    # get the number of kernels
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis("off")
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    if path != "":
        plt.savefig(path)
    plt.show()


def display_images(
    images: [Image.Image],
    columns=5,
    width=20,
    height=8,
    max_images=15,
    label_wrap_length=50,
    label_font_size=8,
):

    if not images:
        print("No images to display.")
        return

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images = images[0:max_images]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.axis("off")
        plt.tight_layout()

        plt.imshow(image)

        if hasattr(image, "filename"):
            title = image.filename
            if title.endswith("/"):
                title = title[0:-1]
            title = os.path.basename(title)
            title = textwrap.wrap(title, label_wrap_length)
            title = "\n".join(title)
            plt.title(title, fontsize=label_font_size)


def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return net


def start_logger(path):
    os.makedirs(path, exist_ok=True)
    import logging

    Log_Format = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(
        filename=os.path.join(path, "log.log"),
        filemode="a",
        format=Log_Format,
        level=logging.INFO,
    )
    logger = logging.getLogger()
    return logger


def make_cf(cf_y, cf_pred, cycle, epoch, path=""):
    plt.figure(figsize=(7, 7))
    cf_matrix = confusion_matrix(cf_y, cf_pred)
    df_cm = pd.DataFrame(
        cf_matrix,
        index=[i for i in ["Same", "Different"]],
        columns=[i for i in ["Same", "Different"]],
    )
    sn.heatmap(df_cm, annot=True, cbar=False, cmap="Blues", fmt="d")
    plt.suptitle(f"Epoch {cycle+1} x {epoch+1}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if path != "":
        plt.savefig(os.path.join(path, f"cm_{cycle+1}x{epoch+1}.png"))
    plt.show()


def plot_lr(lr, cycle, epoch, steps, path=""):
    lr = np.array(lr).squeeze(1)
    sn.lineplot(x=list(range(len(lr))), y=lr)
    plt.suptitle(
        f"Epoch {cycle+1} x {epoch+1}\nlr_min = {np.min(lr)}  lr_max = {np.max(lr)}"
    )
    plt.xlabel("Steps (epoch * steps)")
    plt.ylabel("lr")
    if path != "":
        plt.savefig(os.path.join(path, f"lr_{cycle+1}x{epoch+1}.png"))
    plt.show()


def plot_losses(tr_loss, te_loss, cycle, epoch, path=""):
    plt.plot(tr_loss, label="Train")
    plt.plot(te_loss, label="Test")
    plt.suptitle(f"Losses\nEpoch {cycle+1} x {epoch+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if path != "":
        plt.savefig(os.path.join(path, f"loss_{cycle+1}x{epoch+1}.png"))
    plt.show()


def plot_acc(tr_acc, te_acc, cycle, epoch, path=""):
    plt.plot(tr_acc, label="Train")
    plt.plot(te_acc, label="Test")
    plt.suptitle(f"Accuracy\nEpoch {cycle+1} x {epoch+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if path != "":
        plt.savefig(os.path.join(path, f"acc_{cycle+1}x{epoch+1}.png"))
    plt.show()
