import torch
import torch.nn as nn
import os
import time
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from skimage.color import lab2rgb
from itertools import islice
import itertools

from model import TPN, PCN
from data_loader import *
from util import *
from ciede2000 import CIEDE2000

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Build the model.
        self.build_model(args.mode)

    def prepare_dict(self):
        input_dict = Dictionary()
        src_path = os.path.join('./data/hexcolor_vf/all_names.pkl')
        with open(src_path, 'rb') as f:
            text_data = pickle.load(f)
            f.close()

        print("Loading %s palette names..." % len(text_data))
        print("Making text dictionary...")

        for i in range(len(text_data)):
            input_dict.index_elements(text_data[i])
        return input_dict

    def prepare_data(self, images, palettes, always_give_global_hint, add_L):
        batch = images.size(0)
        imsize = images.size(3)

        inputs, labels = process_image(images, batch, imsize)
        if add_L:
            for_global = process_palette_lab(palettes, batch)
            global_hint = process_global_lab(for_global, batch, always_give_global_hint)
        else:
            for_global = process_palette_ab(palettes, batch)
            global_hint = process_global_ab(for_global, batch, always_give_global_hint)

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        global_hint = (global_hint).expand(-1, -1, imsize, imsize).to(self.device)
        return inputs, labels, global_hint


    def build_model(self, mode):

        if mode == 'train_TPN':
            # Data loader.
            self.input_dict = self.prepare_dict()
            self.train_loader, _ = t2p_loader(self.args.batch_size, self.input_dict)

            # Load pre-trained GloVe embeddings.
            emb_file = os.path.join('./data', 'Color-Hex-vf.pth')
            if os.path.isfile(emb_file):
                W_emb = torch.load(emb_file)
            else:
                W_emb = load_pretrained_embedding(self.input_dict.word2index,
                                                  '../data/glove.840B.300d.txt',
                                                  300)
                W_emb = torch.from_numpy(W_emb)
                torch.save(W_emb, emb_file)
            W_emb = W_emb.to(self.device)

            # Generator and discriminator.
            self.encoder = TPN.EncoderRNN(self.input_dict.n_words, self.args.hidden_size,
                                      self.args.n_layers, self.args.dropout_p, W_emb).to(self.device)
            self.G = TPN.AttnDecoderRNN(self.input_dict, self.args.hidden_size,
                                    self.args.n_layers, self.args.dropout_p).to(self.device)
            self.D = TPN.Discriminator(15, self.args.hidden_size).to(self.device)

            # Initialize weights.
            self.encoder.apply(init_weights_normal)
            self.G.apply(init_weights_normal)
            self.D.apply(init_weights_normal)

            # Optimizer.
            self.G_parameters = list(self.encoder.parameters()) + list(self.G.parameters())
            self.g_optimizer = torch.optim.Adam(self.G_parameters,
                                                lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.d_optimizer = torch.optim.Adam(self.D.parameters(),
                                                lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))

        elif mode == 'train_PCN':
            # Data loader.
            self.train_loader, imsize = p2c_loader(self.args.dataset, self.args.batch_size, 0)

            # Generator and discriminator.
            self.G = PCN.UNet(imsize, self.args.add_L).to(self.device)
            self.D = PCN.Discriminator(self.args.add_L, imsize).to(self.device)

            # Optimizer.
            self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.g_scheduler = scheduler.ReduceLROnPlateau(g_optimizer, 'min', patience=5, factor=0.1)
            self.d_scheduler = scheduler.ReduceLROnPlateau(d_optimizer, 'min', patience=5, factor=0.1)

        elif mode == 'test_TPN' or 'test_text2colors':
            # Data loader.
            self.input_dict = self.prepare_dict()

            # Load pre-trained GloVe embeddings.
            emb_file = os.path.join('./data', 'Color-Hex-vf.pth')
            if os.path.isfile(emb_file):
                W_emb = torch.load(emb_file)
            else:
                W_emb = load_pretrained_embedding(self.input_dict.word2index,
                                                  '../data/glove.840B.300d.txt',
                                                  300)
                W_emb = torch.from_numpy(W_emb)
                torch.save(W_emb, emb_file)
            W_emb = W_emb.to(self.device)

            # Data loader.
            #self.test_loader, self.imsize = test_loader(self.args.dataset, self.args.batch_size, self.input_dict)
            _, self.test_loader = t2p_loader(self.args.batch_size, self.input_dict)
            # Load the trained generators.
            self.encoder = TPN.EncoderRNN(self.input_dict.n_words, self.args.hidden_size,
                                          self.args.n_layers, self.args.dropout_p, W_emb).to(self.device)
            self.G_TPN = TPN.AttnDecoderRNN(self.input_dict, self.args.hidden_size,
                                        self.args.n_layers, self.args.dropout_p).to(self.device)

    def load_model(self, mode, resume_epoch):
        print('Loading the trained model from epoch {}...'.format(resume_epoch))
        if mode == 'train_TPN':
            encoder_path = os.path.join(self.args.text2pal_dir, '{}_G_encoder.ckpt'.format(resume_epoch))
            G_path = os.path.join(self.args.text2pal_dir, '{}_G_decoder.ckpt'.format(resume_epoch))
            D_path = os.path.join(self.args.text2pal_dir, '{}_D.ckpt'.format(resume_epoch))
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

        elif mode == 'train_PCN':
            G_path = os.path.join(self.args.pal2color_dir, '{}_G.ckpt'.format(resume_epoch))
            D_path = os.path.join(self.args.pal2color_dir, '{}_D.ckpt'.format(resume_epoch))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

        elif mode == 'test_TPN' or 'test_text2colors':
            encoder_path = os.path.join(self.args.text2pal_dir, '{}_G_encoder.ckpt'.format(resume_epoch))
            G_TPN_path = os.path.join(self.args.text2pal_dir, '{}_G_decoder.ckpt'.format(resume_epoch))
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
            self.G_TPN.load_state_dict(torch.load(G_TPN_path, map_location=lambda storage, loc: storage))


    def train_TPN(self):
        # Loss function.
        criterion_GAN = nn.BCELoss()
        criterion_smoothL1 = nn.SmoothL1Loss()

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.args.resume_epoch:
            start_epoch = self.args.resume_epoch
            self.load_model(self.args.mode, self.args.resume_epoch)

        self.encoder.train()
        self.G.train()
        self.D.train()

        print('Start training...')
        start_time = time.time()
        for epoch in range(start_epoch, self.args.num_epochs):
            for batch_idx, (txt_embeddings, real_palettes) in enumerate(self.train_loader):

                # Compute text input size (without zero padding).
                batch_size = txt_embeddings.size(0)
                nonzero_indices = list(torch.nonzero(txt_embeddings)[:, 0])
                each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]

                # Prepare training data.
                txt_embeddings = txt_embeddings.to(self.device)
                real_palettes = real_palettes.to(self.device).float()

                # Prepare labels for the BCE loss.
                real_labels = torch.ones(batch_size).to(self.device)
                fake_labels = torch.zeros(batch_size).to(self.device)

                # Prepare input and output variables.
                palette = torch.FloatTensor(batch_size, 3).zero_().to(self.device)
                fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(self.device)

                # Condition for the generator.
                encoder_hidden = self.encoder.init_hidden(batch_size).to(self.device)
                encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings, encoder_hidden)

                # Generate color palette.
                for i in range(5):
                    palette, decoder_context, decoder_hidden = self.G(palette,
                                                                      decoder_hidden.squeeze(0),
                                                                      encoder_outputs,
                                                                      each_input_size,
                                                                      i)
                    fake_palettes[:, 3 * i:3 * (i+1)] = palette

                # Condition for the discriminator.
                each_input_size = torch.FloatTensor(each_input_size).to(self.device)
                each_input_size = each_input_size.unsqueeze(1).expand(batch_size, self.G.hidden_size)
                encoder_outputs = torch.sum(encoder_outputs, 0)
                encoder_outputs = torch.div(encoder_outputs, each_input_size)

                # =============================== Train the discriminator =============================== #
                # Compute BCE loss using real palettes.
                real = self.D(real_palettes, encoder_outputs)
                d_loss_real = criterion_GAN(real, real_labels)

                # Compute BCE loss using fake palettes.
                fake = self.D(fake_palettes, encoder_outputs)
                d_loss_fake = criterion_GAN(fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake

                # Backprop and optimize.
                self.d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                self.d_optimizer.step()

                # ================================ Train the generator ================================= #
                # Compute BCE loss (fool the discriminator).
                fake = self.D(fake_palettes, encoder_outputs)
                g_loss_GAN = criterion_GAN(fake, real_labels)

                # Compute smooth L1 loss.
                g_loss_smoothL1 = criterion_smoothL1(fake_palettes, real_palettes)

                # Compute KL loss.
                kl_loss = KL_loss(mu, logvar)

                g_loss = g_loss_GAN + g_loss_smoothL1 * self.args.lambda_sL1 + kl_loss * self.args.lambda_KL

                # Backprop and optimize.
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

            # For debugging. Save training output.
            if (epoch+1) % self.args.sample_interval == 0:
                for x in range(5):  # saving 5 samples
                    fig1, axs1 = plt.subplots(nrows=1, ncols=5)
                    input_text = ''
                    for idx in txt_embeddings[x]:
                        if idx.item() == 0: break
                        input_text += self.input_dict.index2word[idx.item()] + " "
                    axs1[0].set_title(input_text)
                    for k in range(5):
                        lab = np.array([fake_palettes.data[x][3*k],
                                        fake_palettes.data[x][3*k+1],
                                        fake_palettes.data[x][3*k+2]], dtype='float64')
                        rgb = lab2rgb_1d(lab)
                        axs1[k].imshow([[rgb]])
                        axs1[k].axis('off')

                    fig1.savefig(os.path.join(self.args.train_sample_dir,
                                              'epoch{}_sample{}.jpg'.format(epoch+1, x+1)))
                    plt.close()
                print('Saved train sample...')

            if (epoch+1) % self.args.log_interval == 0:
                elapsed_time = time.time() - start_time
                print('Elapsed time [{:.4f}], Iteration [{}/{}], '
                      'd_loss: {:.6f}, g_loss: {:.6f}'.format(
                       elapsed_time, (epoch+1), self.args.num_epochs,
                       d_loss.item(), g_loss.item()))

            if (epoch+1) % self.args.save_interval == 0:
                torch.save(self.encoder.state_dict(),
                           os.path.join(self.args.text2pal_dir, '{}_G_encoder.ckpt'.format(epoch+1)))
                torch.save(self.G.state_dict(),
                           os.path.join(self.args.text2pal_dir, '{}_G_decoder.ckpt'.format(epoch+1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.args.text2pal_dir, '{}_D.ckpt'.format(epoch+1)))
                print('Saved model checkpoints...')



    def test_TPN(self):
        # Load model.
        if self.args.resume_epoch:
            self.load_model(self.args.mode, self.args.resume_epoch)

        print('Start testing...')

        lab_dict = {}

        #for batch_idx, (txt_embeddings, real_palettes) in islice(enumerate(self.test_loader), 0, 2):
        for batch_idx, (txt_embeddings, real_palettes) in enumerate(self.test_loader):
            if txt_embeddings.size(0) != self.args.batch_size:
                break

            # Compute text input size (without zero padding).
            batch_size = txt_embeddings.size(0)
            nonzero_indices = list(torch.nonzero(txt_embeddings)[:, 0])
            each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]

            # Prepare test data.
            txt_embeddings = txt_embeddings.to(self.device)
            real_palettes = real_palettes.to(self.device).float()

            # Generate multiple palettes from same text input.
            for num_gen in range(10):

                # Prepare input and output variables.
                palette = torch.FloatTensor(batch_size, 3).zero_().to(self.device)
                fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(self.device)

                # ============================== Text-to-Palette ==============================#
                # Condition for the generator.
                encoder_hidden = self.encoder.init_hidden(batch_size).to(self.device)
                encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings, encoder_hidden)

                # Generate color palette.
                for i in range(5):
                    palette, decoder_context, decoder_hidden = self.G_TPN(palette,
                                                                             decoder_hidden.squeeze(0),
                                                                             encoder_outputs,
                                                                             each_input_size,
                                                                             i)
                    fake_palettes[:, 3 * i:3 * (i + 1)] = palette

                # ================================ Save Results ================================#
                for x in range(self.args.batch_size):
                    # Input text.
                    input_text = ''
                    for idx in txt_embeddings[x]:
                        if idx.item() == 0: break
                        input_text += self.input_dict.index2word[idx.item()] + ' '

                    # Save palette generation results.
                    fig1, axs1 = plt.subplots(nrows=2, ncols=5)
                    axs1[0][0].set_title(input_text + 'generated {}'.format(num_gen + 1))
                    fake_pallete = []
                    for k in range(5):
                        lab_fake = np.array([fake_palettes.data[x][3 * k],
                                        fake_palettes.data[x][3 * k + 1],
                                        fake_palettes.data[x][3 * k + 2]], dtype='float64')
                        fake_pallete.append(lab_fake)
                        rgb = lab2rgb_1d(lab_fake)
                        axs1[0][k].imshow([[rgb]])
                        axs1[0][k].axis('off')

                    axs1[1][0].set_title(input_text + 'real')
                    real_pallete = []
                    for k in range(5):
                        lab_real = np.array([real_palettes.data[x][3 * k],
                                        real_palettes.data[x][3 * k + 1],
                                        real_palettes.data[x][3 * k + 2]], dtype='float64')
                        real_pallete.append(lab_real)
                        rgb = lab2rgb_1d(lab_real)
                        axs1[1][k].imshow([[rgb]])
                        axs1[1][k].axis('off')
                    if not input_text in lab_dict.keys():
                        lab_dict[input_text] = {'real': [],
                                                'fake': []}
                    lab_dict[input_text]['real'].append(real_pallete)
                    lab_dict[input_text]['fake'].append(fake_pallete)

                    fig1.savefig(os.path.join(self.args.test_sample_dir, self.args.mode,
                                              '{}_palette{}.jpg'.format(self.args.batch_size*batch_idx+x+1,
                                                                        num_gen+1)))
                    print('Saved data [{}], input text [{}], test sample [{}]'.format(
                          self.args.batch_size*batch_idx+x+1, input_text, num_gen+1))

        diversity_diff_list = []
        gt_diversity_diff_list = []
        multimodality_diff_list = []

        for text, palletes in lab_dict.items():
            real_palletes_list = palletes['real']
            fake_palletes_list =  palletes['fake']
            for fake_pallete in fake_palletes_list:
                for a, b in itertools.combinations(fake_pallete, 2):
                    diversity_diff_list.append(abs(CIEDE2000(a, b)))
            for real_pallete in real_palletes_list:
                for a, b in itertools.combinations(real_pallete, 2):
                    gt_diversity_diff_list.append(abs(CIEDE2000(a, b)))
            for a, b in itertools.combinations(fake_palletes_list, 2):
                for color_index in range(len(a)):
                    multimodality_diff_list.append(abs(CIEDE2000(a[color_index], b[color_index])))

        print('Mean color difference (diversity)', np.array(diversity_diff_list).mean())
        print('STD color difference (diversity)', np.array(diversity_diff_list).std())

        print('Ground truth Mean color difference (diversity)', np.array(gt_diversity_diff_list).mean())
        print('Ground truth STD color difference (diversity)', np.array(gt_diversity_diff_list).std())

        plt.figure()
        plt.hist(diversity_diff_list, bins=100, density=True)
        plt.title('Color difference (diversity)')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.savefig('CIEDE2000-diversity-%s.png' % (time.time()))

        plt.figure()
        plt.hist(gt_diversity_diff_list, bins=100, density=True)
        plt.title('Color difference (Ground truth diversity)')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.savefig('CIEDE2000-gt-diversity-%s.png' % (time.time()))

        print('Mean color difference (multimodality)', np.array(multimodality_diff_list).mean())
        print('STD color difference (multimodality)', np.array(multimodality_diff_list).std())
        plt.figure()
        plt.hist(multimodality_diff_list, bins=100, density=True)
        plt.title('Color difference (multimodality)')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.savefig('CIEDE2000-multimodality-%s.png' % (time.time()))