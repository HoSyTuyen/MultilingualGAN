import os, time, pickle, networks, utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import os 
import torchvision.models as models
from networkv2.generator import GeneratorOctConv
from networkv2.network import GPPatchMcResDis_yaxing, GPPatchMcResDis, BigGanDiscriminator
from FID.fid_score import calculate_fid_given_paths
from pathlib import Path

from tqdm import tqdm

class HWGAN(object):
    """docstring for HWGAN"""
    def __init__(self, config_path):
        super(HWGAN, self).__init__()
        self.config = utils.read_yaml(config_path)
        self.device = utils.get_device(self.config)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def main(self):
        self.create_folder()
        self.process_config(self.config)
        
        self.train_loader, self.test_loader = self.get_loader()
        if self.config['test_multi_lan_data_root'] != "":
            self.test_multiple_loader = utils.data_load(self.config['test_multi_lan_data_root'], self.config['test_multi_lan_data_sub'], self.transforms, 1, shuffle=True, drop_last=True)

        print("DEVICES = {}".format(self.device))

        self.G = self.build_generator()
        self.D = self.build_discriminator()
        self.RestNet18 = self.build_Resnet18()

        self.G.train()
        self.D.train()
        self.RestNet18.eval()

        self.BCE_loss = nn.BCELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)


        if self.config['optimizer'] == 'Adam':
            self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.config['lrG'], betas=(self.config['beta1'], self.config['beta2']))
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.config['lrD'], betas=(self.config['beta1'], self.config['beta2']))
        elif self.config['optimizer'] == 'SGD':
            self.G_optimizer = optim.SGD(self.G.parameters(), lr=self.config['lrG'], momentum=0.9)
            self.D_optimizer = optim.SGD(self.G.parameters(), lr=self.config['lrG'], momentum=0.9)
        else:
            raise NotImplementedError

        self.G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.G_optimizer, milestones=[self.config['train_epoch'] // 2, self.config['train_epoch'] // 4 * 3], gamma=0.1)
        self.D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.D_optimizer, milestones=[self.config['train_epoch'] // 2, self.config['train_epoch'] // 4 * 3], gamma=0.1)

        if self.config['abstraction'] == True:
            self.abs_stage()
        self.per_stage()

        #self.testing_fid()
        

    def create_folder(self):
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results', 'Reconstruction'))
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results', 'Transfer'))
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results', 'generated_imgs'))
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results', 'paired_imgs'))
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results', 'hw_imgs'))
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results', 'source_imgs'))
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results', 'test_result'))
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results', 'weights'))

    def process_config(self, config):
        utils.save_yaml(config, config['project_name'] + '_results' + '/config.yaml')
        utils.print_config(config)

    def get_loader(self):
        train_loader = utils.data_load(self.config['data_path'], self.config['train_data'], self.transforms, self.config['batch_size'], shuffle=True, drop_last=True)
        test_loader = utils.data_load(self.config['data_path'], self.config['test_data'], self.transforms, 1, shuffle=False, drop_last=True)

        return train_loader, test_loader

    def build_generator(self):
        #Chose Generator
        if self.config['generator_arch'] == 'baseline': 
            G = networks.generator(nf = self.config['ngf'], nb = self.config['nb'])
        elif self.config['generator_arch'] == 'OctConv':
            G = GeneratorOctConv(nf = self.config['ngf'], nb = self.config['nb'])
        else:
            raise NotImplementedError
        if self.config['latest_generator_model'] != '':
            if torch.cuda.is_available():
                G.load_state_dict(torch.load(self.config['latest_generator_model']))
            else:
                G.load_state_dict(torch.load(self.config['latest_generator_model'], map_location=lambda storage, loc: storage))
        G.to(self.device)

        return G

    def build_discriminator(self):
        #Chose Generator
        if self.config['discriminator_arch'] == 'baseline':
            D = networks.discriminator(nf = self.config['ndf'])
        elif self.config['discriminator_arch'] == 'OctConv':
            D = GPPatchMcResDis(n_res_blks = 6, nf = self.config['ndf'])
        elif self.config['discriminator_arch'] == 'BigGan':
            D = BigGanDiscriminator()
        else:
            raise NotImplementedError
        if self.config['latest_discriminator_model'] != '':
            if torch.cuda.is_available():
                D.load_state_dict(torch.load(self.config['latest_discriminator_model']))
            else:
                D.load_state_dict(torch.load(self.config['latest_discriminator_model'], map_location=lambda storage, loc: storage))
        D.to(self.device)

        return D

    def build_Resnet18(self):
        RestNet18 = networks.RestNet18(init_weights=None)
        RestNet18.to(self.device)
        # vgg16 = models.vgg16(pretrained=True).to(self.device)
        # RestNet18 = nn.Sequential(
        #     *list(vgg16.children())[:-1]
        #     )
        return RestNet18

    # def build_Resnet18(self):
    #     #RestNet18 = networks.RestNet18(init_weights=None)
    #     #RestNet18.to(self.device)
    #     model = networks.PerNet_RCNN(32, 1, 37, 256)
    #     if torch.cuda.is_available():
    #         model = model.cnn.cuda()
    #     model_path = "linh_tinh/RCNN_extrac_feature.pkl"
    #     print('loading pretrained model from %s' % model_path)
    #     model.load_state_dict(torch.load(model_path))
    #     return model
            

    def abs_stage(self):
        # Tracking
        abs_train_hint = {}
        abs_train_hint['Recon_loss'] = []
        abs_train_hint['per_epoch_time'] = []
        abs_train_hint['total_time'] = []

        # Training
        print('Abstraction stage start!')
        start_time = time.time()
        for epoch in range(self.config['abs_epoch']):
            epoch_start_time = time.time()
            Recon_losses = []
            for x, _ in tqdm(self.train_loader):
                x_pr = x[:, :, :, x.shape[3]//2:]
                x_pr = x_pr.to(self.device)
                x_pr = F.interpolate(x_pr, (self.config['input_w'], self.config['input_h']))

                # train generator G
                self.G_optimizer.zero_grad()
                G_ = self.G(x_pr)
                Recon_loss = 10*self.L1_loss(G_, x_pr)
                Recon_losses.append(Recon_loss.item())
                abs_train_hint['Recon_loss'].append(Recon_loss.item())

                Recon_loss.backward()
                self.G_optimizer.step()

            per_epoch_time = time.time() - epoch_start_time
            abs_train_hint['per_epoch_time'].append(per_epoch_time)
            print('[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), self.config['abs_epoch'], per_epoch_time, torch.mean(torch.FloatTensor(Recon_losses))))

        total_time = time.time() - start_time
        abs_train_hint['total_time'].append(total_time)

        # Save abs stage progress
        with open(os.path.join(self.config['project_name'] + '_results/weights',  'abs_train_hint.pkl'), 'wb') as f:
            pickle.dump(abs_train_hint, f)

        torch.save(self.G.state_dict(), os.path.join(self.config['project_name'] + '_results/weights',  'abs_generator_param.pkl'))

        # Evaluate
        with torch.no_grad():
            self.G.eval()
            for n, (x, _) in enumerate(self.train_loader):
                x_pr = x[:, :, :, x.shape[3]//2:]
                x_pr = x_pr.to(self.device)
                x_pr = F.interpolate(x_pr, (self.config['input_w'], self.config['input_h']))
                G_recon = self.G(x_pr)
                result = torch.cat((x_pr[0], G_recon[0]), 2)
                path = os.path.join(self.config['project_name'] + '_results/results', 'Reconstruction', self.config['project_name'] + '_train_recon_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                if n == 10:
                    break

            for n, (x, _) in enumerate(self.test_loader):
                x_pr = x[:, :, :, x.shape[3]//2:]
                x_pr = x_pr.to(self.device)
                x_pr = F.interpolate(x_pr, (self.config['input_w'], self.config['input_h']))
                G_recon = self.G(x_pr)
                result = torch.cat((x_pr[0], G_recon[0]), 2)
                path = os.path.join(self.config['project_name'] + '_results/results', 'Reconstruction', self.config['project_name'] + '_test_recon_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                if n == 10:
                    break

    def per_stage(self):
        # Tracking
        train_hist = {}
        train_hist['Disc_loss'] = []
        train_hist['Gen_loss'] = []
        train_hist['Con_loss'] = []
        train_hist['per_epoch_time'] = []
        train_hist['total_time'] = []
        train_hist['FID'] = []

        # Training
        print('Perception stage start!')
        FID_log = open(self.config['project_name'] +'_results' + '/FID_log.txt', 'a')
        maxFID = 0
        start_time = time.time()
        real = torch.ones(self.config['batch_size'], 1, self.config['input_w'] // 4, self.config['input_h'] // 4).to(self.device)
        fake = torch.zeros(self.config['batch_size'], 1, self.config['input_w'] // 4, self.config['input_h'] // 4).to(self.device)
        for epoch in range(self.config['start_epoch'],self.config['train_epoch']):
            epoch_start_time = time.time()
            self.G.train()
            Disc_losses = []
            Gen_losses = []
            Con_losses = []
            for x, _ in tqdm(self.train_loader):
                x_pr = x[:, :, :, x.shape[3]//2:]
                x_hw = x[:, :, :, :x.shape[3]//2]

                x_pr, x_hw = x_pr.to(self.device), x_hw.to(self.device)

                x_pr = F.interpolate(x_pr, (self.config['input_w'], self.config['input_h']))
                x_hw = F.interpolate(x_hw, (self.config['input_w'], self.config['input_h']))

                # train D
                self.D_optimizer.zero_grad()
                D_real = self.D(x_hw)
                D_real_loss = self.BCE_loss(D_real, real)
                G_ = self.G(x_pr)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, fake)        
                Disc_loss = (D_real_loss + D_fake_loss)*self.config['adv_lambda']

                Disc_losses.append(Disc_loss.item())
                train_hist['Disc_loss'].append(Disc_loss.item())

                Disc_loss.backward()
                self.D_optimizer.step()


                # train G
                self.G_optimizer.zero_grad()

                G_ = self.G(x_pr)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, real)*self.config['adv_lambda']

                x_feature = self.RestNet18((x_pr + 1) / 2)
                #real_feature = self.RestNet18((x_hw + 1) / 2)
                G_feature = self.RestNet18((G_ + 1) / 2)
                Con_loss = self.config['con_lambda'] * self.L1_loss(G_feature, x_feature.detach())
                #Con_loss = self.config['con_lambda'] * self.L1_loss(G_feature, real_feature.detach())

                Gen_loss = D_fake_loss + Con_loss

                Gen_losses.append(D_fake_loss.item())
                train_hist['Gen_loss'].append(D_fake_loss.item())
                Con_losses.append(Con_loss.item())
                train_hist['Con_loss'].append(Con_loss.item())

                Gen_loss.backward()
                self.G_optimizer.step()

            self.G_scheduler.step()
            self.D_scheduler.step()


            per_epoch_time = time.time() - epoch_start_time
            train_hist['per_epoch_time'].append(per_epoch_time)
            print(
            '[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % ((epoch + 1), self.config['train_epoch'], per_epoch_time, torch.mean(torch.FloatTensor(Disc_losses)),
                torch.mean(torch.FloatTensor(Gen_losses)), torch.mean(torch.FloatTensor(Con_losses))))
                
            if epoch % self.config['save_epoch'] == 0:
                torch.save(self.G.state_dict(), os.path.join(self.config['project_name'] + '_results/weights', 'generator_epoch{}.pkl'.format(str(epoch))))
                torch.save(self.D.state_dict(), os.path.join(self.config['project_name'] + '_results/weights', 'discriminator_epoch{}.pkl'.format(str(epoch))))
                if self.config['save_result_per_epoch'] == 'True':
                    self.save_img_results(epoch)
                if self.config['test_multi_lan_data_root'] != "":
                    self.test_multi_lan_data(epoch)
            # Sampling
            if epoch % 2 == 1 or epoch == self.config['train_epoch'] - 1:
                # with torch.no_grad():
                #     self.G.eval()
                #     for n, (x, _) in enumerate(self.train_loader):
                #         x_pr = x[:, :, :, x.shape[3]//2:]
                #         x_pr = x_pr.to(self.device)
                #         x_pr = F.interpolate(x_pr, (self.config['input_w'], self.config['input_h']))

                #         G_recon = self.G(x_pr)
                #         result = torch.cat((x_pr[0], G_recon[0]), 2)
                #         path = os.path.join(self.config['project_name'] + '_results/results', 'Transfer', str(epoch+1) + '_epoch_' + self.config['project_name'] + '_train_' + str(n + 1) + '.png')
                #         plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                #         if n == 4:
                #             break

                #     for n, (x, _) in enumerate(self.test_loader):
                #         x_pr = x[:, :, :, x.shape[3]//2:]
                #         x_pr = x_pr.to(self.device)
                #         x_pr = F.interpolate(x_pr, (self.config['input_w'], self.config['input_h']))

                #         G_recon = self.G(x_pr)
                #         result = torch.cat((x_pr[0], G_recon[0]), 2)
                #         path = os.path.join(self.config['project_name'] + '_results/results', 'Transfer', str(epoch+1) + '_epoch_' + self.config['project_name'] + '_test_' + str(n + 1) + '.png')
                #         plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                #         if n == 4:
                #             break

                torch.save(self.G.state_dict(), os.path.join(self.config['project_name'] + '_results/weights', 'generator_latest.pkl'))
                torch.save(self.D.state_dict(), os.path.join(self.config['project_name'] + '_results/weights', 'discriminator_latest.pkl'))
            
            FID_score = self.testing_fid()
            FID_log.writelines(str(epoch) + '\t' + str(FID_score) + '\n')
            train_hist['FID'].append(FID_score)
            if FID_score > maxFID:
                maxFID = FID_score
                torch.save(self.G.state_dict(), os.path.join(self.config['project_name'] + '_results/weights', 'generator_best.pkl'))
                torch.save(self.D.state_dict(), os.path.join(self.config['project_name'] + '_results/weights', 'discriminator_best.pkl'))

        total_time = time.time() - start_time
        train_hist['total_time'].append(total_time)

        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), self.config['train_epoch'], total_time))
        print("Training finish!... save training results")

        # Save abs perception progress
        torch.save(self.G.state_dict(), os.path.join(self.config['project_name'] + '_results/weights',  'generator_param.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.config['project_name'] + '_results/weights',  'discriminator_param.pkl'))
        with open(os.path.join(self.config['project_name'] + '_results/weights',  'train_hist.pkl'), 'wb') as f:
            pickle.dump(train_hist, f)
        FID_log.close()

    def testing_fid(self):
        # """ Calculate FID """
        print("Generating results and calculating FID")
        is_save_hwImg = not (len(os.path.join(self.config['project_name'] + '_results/results/hw_imgs')) == 0)
        for n, (x, _) in tqdm(enumerate(self.test_loader)):
            x_hw = x[:, :, :, :x.shape[3]//2]
            x_pr = x[:, :, :, x.shape[3]//2:]
            x_pr = x_pr.to(self.device)
            x_pr = F.interpolate(x_pr, (self.config['input_w'], self.config['input_h']))

            G_recon = self.G(x_pr)
            #paired = torch.cat((x_pr[0], G_recon[0]), 2)
            
            path_output = os.path.join(self.config['project_name'] + '_results/results', 'generated_imgs', str(n + 1) + '.png')
            plt.imsave(path_output, (G_recon[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
            #path_pair = os.path.join(self.config['project_name'] + '_results/results', 'paired_imgs', str(n + 1) + '.png')
            
            path_hw = os.path.join(self.config['project_name'] + '_results/results', 'hw_imgs', str(n + 1) + '.png')
            if is_save_hwImg:
                plt.imsave(path_hw, (x_hw[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
            #path_src = os.path.join(self.config['project_name'] + '_results/results', 'source_imgs', str(n + 1) + '.png')

            #plt.imsave(path_pair, (paired.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
            #plt.imsave(path_src, (x_pr[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
        fid_value = calculate_fid_given_paths(path_A=str(Path(path_output).parent), path_B=str(Path(path_hw).parent),
                                            batch_size=1,
                                            cuda=True,
                                            dims=2048)
        print('FID: ', fid_value)
        return fid_value
    
    def save_img_results(self, epoch):
        print("Generating results at epoch", epoch)
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results/test_result', str(epoch)))
        for n, (x, _) in tqdm(enumerate(self.test_loader)):
            x_pr = x[:, :, :, x.shape[3]//2:]
            x_pr = x_pr.to(self.device)
            G_recon = self.G(x_pr)
            path_output = os.path.join(self.config['project_name'] + '_results/results/test_result', str(epoch), str(n + 1) + '.png')
            plt.imsave(path_output, (G_recon[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)
    
    def test_multi_lan_data(self, epoch):
        if epoch == 0:
            utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results', 'test_multiple_result'))
            #return
        utils.check_and_make_folfer(os.path.join(self.config['project_name'] + '_results/results/test_multiple_result', str(epoch)))
        for n, (x, _) in tqdm(enumerate(self.test_multiple_loader)):
            x_pr = x
            x_pr = x_pr.to(self.device)
            G_recon = self.G(x_pr)
            path_output = os.path.join(self.config['project_name'] + '_results/results/test_multiple_result', str(epoch), str(n + 1) + '.png')
            plt.imsave(path_output, (G_recon[0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)

Trainer = HWGAN('config.yaml')
Trainer.main()
