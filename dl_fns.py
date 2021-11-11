from imports import *
import skimage

def label_func(path):
    split_name = path.stem.split("_")
    return 0 if split_name[-1] == split_name[-2] else 1

def get_img_tuple_no_noise(path):
    pair = Image.open(path)

    label = label_func(Path(path))
    orientation = os.path.basename(path).split('_')[-3]

    width, height = pair.size

    if orientation == 'normal':
        left1, top1, right1, bottom1 = width - width//4, 0, width, height//4
        left2, top2, right2, bottom2 = 0, height - height//4, width//4, height
    else:
        left1, top1, right1, bottom1 = 0, 0, width//4, height//4
        left2, top2, right2, bottom2 = width - width//4, height - height//4, width, height

    im1 = pair.crop((left1, top1, right1, bottom1)).resize((224,224))
    im2 = pair.crop((left2, top2, right2, bottom2)).resize((224,224))
    im3 = Image.new('RGB', (224, 224), (125,125,125))

    return (ToTensor()(PILImage(im1)), ToTensor()(PILImage(im2)), ToTensor()(PILImage(im3)), label)

def get_img_tuple_noise(path):
    pair = Image.open(path)

    label = label_func(Path(path))
    orientation = os.path.basename(path).split('_')[-3]

    width, height = pair.size

    if orientation == 'normal':
        left1, top1, right1, bottom1 = width - width//4, 0, width, height//4
        left2, top2, right2, bottom2 = 0, height - height//4, width//4, height
    else:
        left1, top1, right1, bottom1 = 0, 0, width//4, height//4
        left2, top2, right2, bottom2 = width - width//4, height - height//4, width, height

    im1 = pair.crop((left1, top1, right1, bottom1)).resize((224,224))
    im2 = pair.crop((left2, top2, right2, bottom2)).resize((224,224))
    im3 = Image.new('RGB', (224, 224), (125,125,125))
    im3 = Image.fromarray(np.uint8(skimage.util.random_noise(skimage.img_as_float(im3), mode='s&p',amount = 1) * 255)) 

    return (ToTensor()(PILImage(im1)), ToTensor()(PILImage(im2)), ToTensor()(PILImage(im3)), label)

class ImageTuple(fastuple):
    @classmethod
    def create(cls, fns): return cls(fns)

    def show(self, ctx=None, **kwargs):
        t1,t2,t3 = self
        if not isinstance(t1, Tensor) or not isinstance(t2, Tensor) or t1.shape != t2.shape: return ctx
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), ctx=ctx, **kwargs)

def ImageTupleBlock(): return TransformBlock(type_tfms=ImageTuple.create, batch_tfms=IntToFloatTensor)

def get_tuples_no_noise(files): return [[get_img_tuple_no_noise(f)[0], get_img_tuple_no_noise(f)[1], get_img_tuple_no_noise(f)[2], get_img_tuple_no_noise(f)[3]] for f in files]
def get_tuples_noise(files): return [[get_img_tuple_noise(f)[0], get_img_tuple_noise(f)[1], get_img_tuple_noise(f)[2], get_img_tuple_noise(f)[3]] for f in files]
def get_x(t): return t[:3]
def get_y(t): return t[3]

def make_dls(stim_path, batch_sz = 24, fov_noise = False):
    stim_path = Path(stim_path)
    pairs = glob.glob(os.path.join(stim_path, '*.png'))
    fnames = sorted(Path(s) for s in pairs)
    y = [label_func(item) for item in fnames]

    splitter=TrainTestSplitter(test_size = 0.2, random_state=42, shuffle=True, stratify=y)
    splits = splitter(fnames)
    #splits = RandomSplitter()(fnames)
    if fov_noise:
        siamese = DataBlock(
            blocks=(ImageTupleBlock, CategoryBlock),
            get_items=get_tuples_noise,
            get_x=get_x, get_y=get_y,
            splitter=splitter,
            )
    else:
        siamese = DataBlock(
            blocks=(ImageTupleBlock, CategoryBlock),
            get_items=get_tuples_no_noise,
            get_x=get_x, get_y=get_y,
            splitter=splitter,
            )

    dls = siamese.dataloaders(fnames, bs=batch_sz, seed=seed, shuffle=True, device='cuda')
    # check that train and test splits have balanced classes
    train_test = ['TRAIN','TEST']
    for train_test_id in [0, 1]:
        s = 0
        d = 0
        for item in dls.__getitem__(train_test_id).items:
            #print(label_from_path(item))
            #print(item)
            #print('---')
            if item[3] == 1:
                s += 1
            else:
                d += 1
        print(f'{train_test[train_test_id]} SET (same, diff): {str(s)}, {str(d)}')
    return dls


