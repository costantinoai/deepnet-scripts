from imports import *
from loss_fn import ContrastiveLoss
from util_fns import *
from dl_fns import *

# PARAMS
log = True
# stim_path = r'C:\Users\45027900\Desktop\NeuroFovea_PyTorch-main\metamers'
stim_path = r'C:\Users\45027900\Desktop\cornet\stimuli\samediff'
# stim_path = r'C:\Users\45027900\Desktop\cornet\stimuli\no_transf'
log_dir = r'C:\Users\45027900\Desktop\cornet\siamese_logs'
batch_sz = 672//8
# batch_sz = 48
cycles = 1
epochs = 100
lr_min = 1e-3
weight_decay = 1e-2
fb = True
fov_noise = False
run_id = 'FOV-ONLY'

criterion = nn.CrossEntropyLoss()

# MAKE DATALOADER
# pairs = glob.glob(os.path.join(stim_path, '*.png'))
# display_images([Image.open(i) for i in random.sample(pairs, 10)])

dls = make_dls(stim_path, batch_sz, fov_noise)
# print('\nShowing first batch...')
# dls.show_batch(max_n = 2)
# plt.show()

# train_loader = dls.train
# test_loader = dls.valid

train_loader = dls[0]
test_loader = dls[1]

# INIT NET
class SiameseNetEncoderFB(nn.Module):
    def __init__(self):
        super(SiameseNetEncoderFB, self).__init__()

        # V1 layers
        self.V1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2,
                      padding=7 // 2),  # + self.vfb,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        # V2 layers
        self.V2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        # V4 layers
        self.V4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        # IT layers
        self.IT = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        # head
        self.head = nn.Sequential(
          AdaptiveConcatPool2d(),
          nn.Flatten(),
          nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.Dropout(p=0.25, inplace=False),
          nn.Linear(in_features=1024, out_features=512, bias=False),
          nn.ReLU(inplace=True),
          nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=512, out_features=2, bias=False),
          )

        self.fb = nn.Sequential(
            nn.Conv2d(1024, 3, kernel_size=3, stride=1, padding=221),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
          )

    def forward_once(self, inp):
        x = inp
        v1 = self.V1(x)
        v2 = self.V2(v1)
        v4 = self.V4(v2)
        vIT = self.IT(v4)
        return vIT

    def forward(self, inp):
        inp1 = inp[0]
        inp2 = inp[1]
        fov_inp = inp[2]

        # perihperal 1
        v1_p1 = self.V1(inp1)
        v2_p1 = self.V2(v1_p1)
        v4_p1 = self.V4(v2_p1)
        vIT_p1 = self.IT(v4_p1)

        # perihperal 1
        v1_p2 = self.V1(inp2)
        v2_p2 = self.V2(v1_p2)
        v4_p2 = self.V4(v2_p2)
        vIT_p2 = self.IT(v4_p2)

        out_cat = torch.cat((vIT_p1, vIT_p2), 1)

        # fovea
        fb = self.fb(out_cat)
        try:
            v1_fov = self.V1(fb + fov_inp)
        except:
            v1_fov = self.V1(fov_inp)
        v2_fov = self.V2(v1_fov)
        v4_fov = self.V4(v2_fov)
        vIT_fov = self.IT(v4_fov)

        # out_all = torch.cat((vIT_p1, vIT_p2, vIT_fov), 1)
        out = self.head(vIT_fov)

        return out

net = SiameseNetEncoderFB().cuda()
net = init_weights(net)
net = nn.DataParallel(net)
net.to(device)

params_to_update = net.parameters()
optimizer = optim.Adam(params_to_update, lr=lr_min, weight_decay=weight_decay)

## START TRAIN/TEST
if log:
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    run_name = f'{timestamp}_{run_id}_fb-{fb}_fovnoise-{fov_noise}_CrossEntLoss_{os.path.normpath(stim_path).split(os.sep)[-1]}_adam_bs-{batch_sz}_lr-{lr_min}_{cycles}x{epochs}'
    path = os.path.join(log_dir, run_name)
    logger = start_logger(path)
    shutil.copy(r'C:\Users\45027900\Desktop\cornet\project\main.py', os.path.join(path, 'main.py'))
else:
    path = ''

print('\nTrain/Test started!')
# weights = net.module.V1[0].weight.data.cpu()
# plot_filters_multi_channel(weights, path)

for cycle in range(cycles):
    tr_loss = []
    tr_acc = []
    te_loss = []
    te_acc = []

    for epoch in range(epochs):
        # TRAIN
        net.train()

        tr_running_loss = 0.0
        tr_correct = 0
        tr_total = 0
        start = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            out = net(inputs)
            _, pred = torch.max(out, 1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_running_loss += loss.item()
            tr_total += labels.size(0)
            tr_correct += (pred == labels).sum().item()

        tr_loss.append(tr_running_loss)
        tr_acc.append(100 * tr_correct / tr_total)


        # TEST
        net.eval()

        te_running_loss = 0.0
        te_correct = 0
        te_total = 0
        cf_pred = []
        cf_y = []
        with torch.no_grad():
            for (inputs, labels) in test_loader:
                out = net(inputs)
                _, pred = torch.max(out, 1)
                loss = criterion(out, labels)
                te_running_loss += loss.item()
                te_total += labels.size(0)
                te_correct += (pred == labels).sum().item()
                cf_y += labels.cpu().detach().tolist()
                cf_pred += pred.cpu().detach().tolist()

            te_acc.append(100 * te_correct / te_total)
            te_loss.append(te_running_loss)
            end = time.time() - start
            log_msg = f'%5d / %5d TRAIN/TEST losses: \t %.8f \t %.8f \t\t acc: \t %.2f %% \t %.2f %% \t\t time: {round(end,3)}' % (cycle + 1, epoch + 1, tr_running_loss, te_running_loss, 100 * tr_correct / tr_total , 100 * te_correct / te_total)
            print(log_msg)
            if log:
                logger.info(log_msg)
    make_cf(cf_y, cf_pred, cycle, epoch, path)
    plot_losses(tr_loss, te_loss, cycle, epoch, path)
    plot_acc(tr_acc, te_acc, cycle, epoch, path)
    if log:
        filename = f"{datetime.now().strftime('%d-%b-%Y_%H-%M-%S')}_{cycle+1}x{epoch+1}_trloss-{str(round(tr_running_loss, 5)).split('.')[-1]}_teacc-{str(round(te_correct / te_total, 4)).split('.')[-1]}"
        torch.save(net.state_dict(), os.path.join(path,filename))




