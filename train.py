import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from data.ImageNetData import ImageNetDataset
from model.DCGAN import Discriminator, Generator
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.autograd import Variable

device_ids = [0, 1, 2, 3]
device = torch.device('cuda:0')

BATCH_SIZE = 256

dis_model = torch.nn.DataParallel(Discriminator(), device_ids=device_ids, output_device=device_ids[0])
gen_model = torch.nn.DataParallel(Generator(), device_ids=device_ids, output_device=device_ids[0])
dis_model.to(device)
gen_model.to(device)

gen_optimizer = Adam(gen_model.parameters(), lr=3e-4)
dis_optimizer = Adam(dis_model.parameters(), lr=3e-5)

loss_fn = BCEWithLogitsLoss()
l1_loss_fn = L1Loss()



def train(epoch):
    dataLoader = DataLoader(dataset=ImageNetDataset(is_train=True),
                            shuffle=True,
                            batch_size=BATCH_SIZE)
    for idx, (gray, ori) in enumerate(dataLoader):
        gray = Variable(gray).to(device)
        ori = Variable(ori).to(device)
        fake_ori = gen_model(gray)

        dis_fake_pre = dis_model(torch.cat([gray, fake_ori], dim=1))
        dis_fake_loss = loss_fn(dis_fake_pre, torch.zeros_like(dis_fake_pre, device=device))
        dis_real_pre = dis_model(torch.cat([gray, ori], dim=1))
        dis_real_loss = loss_fn(dis_real_pre, torch.ones_like(dis_real_pre, device=device) * 0.9)

        dis_loss = dis_fake_loss + dis_real_loss

        dis_optimizer.zero_grad()
        dis_loss.backward()
        dis_optimizer.step()

        gen_ori = gen_model(gray)
        dis_gen_pre = dis_model(torch.cat([gray, gen_ori], dim=1))
        gen_dis_loss = loss_fn(dis_gen_pre, torch.ones_like(dis_gen_pre, device=device))
        gen_l1_loss = l1_loss_fn(gen_ori, ori)
        gen_loss = gen_dis_loss + gen_l1_loss * 100

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
        print(epoch, dis_loss.item(), gen_loss.item(), gen_dis_loss.item(), gen_l1_loss.item())


if __name__ == '__main__':
    for i in range(500):
        train(i)
        torch.save(dis_model.state_dict(), "./dis_model_{}".format(i))
        torch.save(gen_model.state_dict(), "./gen_model_{}".format(i))
